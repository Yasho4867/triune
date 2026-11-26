import math
from typing import Any

import torch

import lucid_machines.model.config as defaults
from lucid_machines.model import MoE_FFN

from .muon import Muon

try:
    from bitsandbytes.optim import AdamW8bit

    HAS_8BIT = True
except ImportError:
    HAS_8BIT = False
    AdamW8bit = None


from lucid_machines.ops.centroid_steer import (
    batched_qr,
    polar_stiefel_retraction,
    subspace_projection,
)
from lucid_machines.ops.centroid_steer import (
    batched_subspace_projection as batched_fast_subspace_projection,
)


def fast_subspace_projection(
    grad: torch.Tensor,
    rank: int,
    niter: int = 2,
    cond_threshold: float = 1e4,
    eps_shift: float = 1e-7,
    orth_tol: float = 1e-4,
) -> tuple[torch.Tensor, str]:
    """High-performance CUDA/CPU Subspace Iteration via condition-monitored QR with Householder fallback.

    Replaces slow cuSOLVER svd_lowrank with direct Tensor Core GEMM + condition-monitored
    Gram orthogonalization and Householder QR fallback.
    """
    m, n = grad.shape
    side = "left" if m <= n else "right"
    proj = subspace_projection(
        grad,
        rank,
        niter=niter,
        cond_threshold=cond_threshold,
        eps_shift=eps_shift,
        orth_tol=orth_tol,
    )
    return proj, side


batched_qr = batched_qr
polar_stiefel_retraction = polar_stiefel_retraction


class _SyncedParamGroup(dict):
    """Dictionary subclass wrapping outer param_groups so that lr_scheduler updates
    immediately cascade to all internal optimizer tiers (Base AdamW, Muon, Expert Centroid)."""

    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._optimizer = optimizer

    def __setitem__(self, key, value):
        old_val = self.get(key, None)
        super().__setitem__(key, value)
        if key == "lr" and getattr(self, "_optimizer", None) is not None:
            if getattr(self._optimizer, "_in_set_lr", False):
                return
            if hasattr(self._optimizer, "base_optimizer") and old_val != value:
                self._optimizer.set_lr(value)


class CentroidSteerOptimizer(torch.optim.Optimizer):
    """LucidMachines CentroidSteer 3-Tier Partitioned Optimizer.

    Partitions model parameters into three specialized optimization tiers:
    1. Tier 1 (Muon): Newton-Schulz 5th-order orthogonalization for 2D hidden linear weights.
    2. Tier 2 (CentroidSteer): Low-rank triangular subspace projection along router centroids for MoE routed experts.
    3. Tier 3 (Base AdamW): Fused AdamW or 8-bit AdamW for 1D parameters, layer norms, biases, and vocabulary embeddings.

    Parameter Role Overrides (`param_roles`):
        Pass a dictionary mapping parameter names (str) or parameter tensors to target roles:
        - "muon", "orthogonal": Routes 2D parameters to Tier 1 Muon.
        - "expert", "centroid", "subspace": Routes 2D+ parameters to Tier 2 CentroidSteer.
        - "base", "adamw", "embedding", "vocab", "head", "router", "1d", "norm": Routes parameters to Tier 3 Base AdamW.
        Explicit `param_roles` possess highest precedence and unconditionally override all automatic module-type,
        naming, tied-weight, and aspect-ratio heuristics.
    """

    batched_qr = staticmethod(batched_qr)

    def __init__(
        self,
        model=None,
        lr=defaults.LR,
        betas=defaults.BETAS,
        weight_decay=defaults.WEIGHT_DECAY,
        rank=defaults.GALORE_RANK,
        update_gap=defaults.GALORE_UPDATE_GAP,
        steer_scale=defaults.STEER_SCALE,
        steer_update_gap=getattr(defaults, "STEER_UPDATE_GAP", 10),
        expert_lr=None,
        expert_betas=defaults.GALORE_BETAS,
        expert_wd=defaults.GALORE_WEIGHT_DECAY,
        use_muon=getattr(defaults, "USE_MUON", True),
        muon_lr: float | None = None,
        muon_momentum=getattr(defaults, "MUON_MOMENTUM", 0.95),
        muon_weight_decay=getattr(defaults, "MUON_WEIGHT_DECAY", 0.01),
        muon_ns_steps=getattr(defaults, "MUON_NS_STEPS", 3),
        muon_rms_target: float | None = None,
        svd_device="auto",
        use_triangular_core=getattr(defaults, "USE_TRIANGULAR_CORE", True),
        use_8bit_base: bool = getattr(defaults, "USE_8BIT_BASE", False),
        subspace_stride: int = 1,
        adaptive_subspace_refresh: bool = False,
        energy_threshold: float = 0.50,
        min_subspace_gap: int = 5,
        max_subspace_gap: int = 50,
        param_roles: dict[str | torch.nn.Parameter, str] | None = None,
        core_mask_mode: str = "full",
        steer_mode: str = "upgraded",
        num_steer_dirs: int = getattr(defaults, "CENTROID_SUBSPACE_DIRS", 4),
        use_rms_steer: bool = False,
        *,
        params=None,
    ):
        if model is None and params is not None:
            model = params
        if model is None:
            raise ValueError("CentroidSteerOptimizer requires 'model' or 'params'")
        if steer_mode == "random_subspace":
            steer_mode = "random_same_mem"
        if steer_mode not in ("upgraded", "legacy", "random_same_mem"):
            raise ValueError(f"Unknown steer_mode={steer_mode!r}; expected 'upgraded', 'legacy', or 'random_same_mem'")
        if expert_lr is None:
            expert_lr = lr
        if muon_lr is None:
            muon_lr = min(0.02, max(0.001, lr * 2.0))
        defaults_dict = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        initial_group = _SyncedParamGroup(
            self, {"params": [dummy_param], "lr": lr, "betas": betas, "weight_decay": weight_decay}
        )
        super().__init__([initial_group], defaults_dict)
        self.param_groups = [initial_group]
        self.model = model
        self.rank = rank
        self.update_gap = update_gap
        self.step_count = 0
        self.steer_scale = steer_scale
        self.steer_update_gap = steer_update_gap
        self.steer_mode = steer_mode
        self.num_steer_dirs = max(1, int(num_steer_dirs))
        self.use_rms_steer = bool(use_rms_steer)
        self.expert_lr = expert_lr
        self.initial_expert_lr = expert_lr
        self.expert_betas = expert_betas
        self.expert_wd = expert_wd
        self.use_muon = use_muon
        self.svd_device = svd_device
        self.initial_base_lr = lr
        self.initial_muon_lr = muon_lr
        self._last_synced_lr = lr
        self.use_triangular_core = use_triangular_core
        if core_mask_mode not in ("full", "triu"):
            raise ValueError(f"Unknown core_mask_mode={core_mask_mode!r}; expected 'full' or 'triu'")
        self.core_mask_mode = core_mask_mode
        self.use_8bit_base = use_8bit_base
        self.subspace_stride = max(1, int(subspace_stride))
        self.adaptive_subspace_refresh = adaptive_subspace_refresh
        self.energy_threshold = energy_threshold
        self.min_subspace_gap = min_subspace_gap
        self.max_subspace_gap = max_subspace_gap
        self.param_roles = param_roles or {}
        self._triangular_buffers = {}
        self._qr_telemetry_gpu: torch.Tensor | None = None
        self.qr_telemetry = {
            "total_qr_calls": 0,
            "cholesky_fast_path_survivals": 0,
            "householder_fallbacks": 0,
            "primary_fallback_reasons": {
                "degenerate_columns": 0,
                "cholesky_non_positive_definite": 0,
                "condition_ratio_exceeded": 0,
                "orthogonality_tolerance_exceeded": 0,
            },
            "overlapping_triggers": {
                "degenerate_columns": 0,
                "cholesky_non_positive_definite": 0,
                "condition_ratio_exceeded": 0,
                "orthogonality_tolerance_exceeded": 0,
            },
        }

        self.non_expert_params = []
        self.muon_params = []
        self.layer_groups = []
        seen_param_ids = set()

        # Build unified explicit parameter role dictionary
        explicit_param_roles: dict[int, str] = {}
        if self.param_roles:
            import fnmatch

            named_map = dict(model.named_parameters()) if isinstance(model, torch.nn.Module) else {}
            for k_role, r_val in self.param_roles.items():
                r_val_norm = str(r_val).lower().strip()
                if isinstance(k_role, str):
                    if "*" in k_role or "?" in k_role or "[" in k_role:
                        matched_any = False
                        for p_name, p_tensor in named_map.items():
                            if fnmatch.fnmatch(p_name, k_role):
                                explicit_param_roles[id(p_tensor)] = r_val_norm
                                matched_any = True
                        if not matched_any:
                            raise KeyError(f"param_roles wildcard pattern '{k_role}' matched zero model parameters.")
                    elif k_role in named_map:
                        explicit_param_roles[id(named_map[k_role])] = r_val_norm
                    else:
                        # Reject ambiguous suffix matches while providing clear error diagnosis
                        matching_suffixes = [p_name for p_name in named_map if p_name.endswith(f".{k_role}")]
                        if len(matching_suffixes) > 1:
                            raise ValueError(
                                f"param_roles key '{k_role}' is ambiguous and matches multiple parameters: {matching_suffixes}. "
                                f"Use exact full parameter names or explicit wildcards (e.g. '*{k_role}')."
                            )
                        elif len(matching_suffixes) == 1:
                            explicit_param_roles[id(named_map[matching_suffixes[0]])] = r_val_norm
                        else:
                            raise KeyError(
                                f"param_roles key '{k_role}' is not an exact parameter name. "
                                f"Use exact full parameter names or explicit wildcards (e.g. '*{k_role}')."
                            )
                elif isinstance(k_role, (torch.nn.Parameter, torch.Tensor)):
                    explicit_param_roles[id(k_role)] = r_val_norm

        if isinstance(model, torch.nn.Module):
            # Precedence Rule 1: User-specified param_roles completely override ALL automatic heuristics
            if explicit_param_roles:
                for name, param in model.named_parameters():
                    if id(param) in explicit_param_roles and param.requires_grad:
                        role = explicit_param_roles[id(param)]
                        seen_param_ids.add(id(param))
                        if role in ("muon", "orthogonal") and param.dim() == 2:
                            self.muon_params.append(param)
                        elif role in ("expert", "centroid", "subspace") and param.dim() >= 2:
                            self.layer_groups.append(
                                {
                                    "module": None,
                                    "expert_idx": None,
                                    "param": param,
                                    "projection": None,
                                    "projection_side": "left" if param.shape[0] <= param.shape[1] else "right",
                                    "proj_step": 0,
                                    "state": {"momentum": None, "variance": None, "step": 0},
                                }
                            )
                        else:
                            # ("adamw", "base", "embedding", "vocab", "1d", "norm")
                            self.non_expert_params.append(param)

            # 1. Module-type detection for embeddings (nn.Embedding, nn.EmbeddingBag) and routers
            router_param_ids = set()
            embedding_param_ids = set()

            for mod_name, module in model.named_modules():
                if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                    for p in module.parameters():
                        if id(p) not in seen_param_ids:
                            embedding_param_ids.add(id(p))
                elif isinstance(module, MoE_FFN) and hasattr(module, "router"):
                    for p in module.router.parameters():
                        if id(p) not in seen_param_ids:
                            router_param_ids.add(id(p))
                elif (
                    module.__class__.__name__ in ("GumbelSoftmaxRouter", "Router", "TopKRouter")
                    or "router" in mod_name.lower()
                    or mod_name.endswith(".router")
                ):
                    for p in module.parameters():
                        if id(p) not in seen_param_ids:
                            router_param_ids.add(id(p))

            # 2. Check for tied weights with input embeddings
            input_embed_data_ptrs = set()
            for mod in model.modules():
                if isinstance(mod, (torch.nn.Embedding, torch.nn.EmbeddingBag)) and hasattr(mod, "weight"):
                    input_embed_data_ptrs.add(mod.weight.data_ptr())

            # 3. Naming heuristics and aspect-ratio safeguards
            for name, param in model.named_parameters():
                if id(param) in seen_param_ids:
                    continue

                # Tied weight check
                if param.data_ptr() in input_embed_data_ptrs:
                    embedding_param_ids.add(id(param))
                    continue

                name_lower = name.lower()
                if any(
                    k in name_lower
                    for k in (
                        "embed",
                        "wte",
                        "wpe",
                        "exit_head",
                        "final_head",
                        "lm_head",
                        "classifier",
                        "output_head",
                        "token_table",
                    )
                ):
                    embedding_param_ids.add(id(param))
                    continue

                # Large vocabulary aspect-ratio safeguard: matrices where max dim >= 1000 and ratio >= 4.0
                if param.dim() == 2:
                    d_max = max(param.shape[0], param.shape[1])
                    d_min = min(param.shape[0], param.shape[1])
                    if d_max >= 1000 and (d_max / max(d_min, 1)) >= 4.0:
                        embedding_param_ids.add(id(param))
                        continue

            # 1. Count MoE routed experts
            num_expert_groups = 0
            counting_seen = set()
            for name, module in model.named_modules():
                if isinstance(module, MoE_FFN):
                    for expert_idx, expert in enumerate(module.experts):
                        for subname, submod in expert.named_modules():
                            if hasattr(submod, "weight") and isinstance(submod.weight, torch.nn.Parameter):
                                p = submod.weight
                                if p.requires_grad and p.dim() >= 2 and id(p) not in counting_seen:
                                    counting_seen.add(id(p))
                                    num_expert_groups += 1

            group_idx = 0

            # 1. Populate MoE routed experts (with Centroid Steering)
            for name, module in model.named_modules():
                if isinstance(module, MoE_FFN):
                    for expert_idx, expert in enumerate(module.experts):
                        expert_params = []
                        for subname, submod in expert.named_modules():
                            if hasattr(submod, "weight") and isinstance(submod.weight, torch.nn.Parameter):
                                p = submod.weight
                                if p.requires_grad and p.dim() >= 2 and id(p) not in seen_param_ids:
                                    seen_param_ids.add(id(p))
                                    expert_params.append((subname, p))
                        for pos_idx, (subname, p) in enumerate(expert_params):
                            if (
                                hasattr(module, "dim")
                                and hasattr(module, "hidden_dim")
                                and module.dim != module.hidden_dim
                                and p.ndim >= 2
                            ):
                                steer_role = "input" if p.shape[1] == module.dim else "hidden"
                            else:
                                steer_role = "input" if pos_idx == 0 else "hidden"
                            self.layer_groups.append(
                                {
                                    "module": module,
                                    "expert_idx": expert_idx,
                                    "expert": expert,
                                    "steer_role": steer_role,
                                    "param": p,
                                    "projection": None,
                                    "projection_side": "left" if p.shape[0] <= p.shape[1] else "right",
                                    "proj_step": 0,
                                    "state": {"momentum": None, "variance": None, "step": 0},
                                }
                            )
                            group_idx += 1

            # 2. Populate remaining 2D parameters
            for name, param in model.named_parameters():
                if param.requires_grad and param.dim() >= 2 and id(param) not in seen_param_ids:
                    seen_param_ids.add(id(param))
                    if self.use_muon and id(param) not in router_param_ids and id(param) not in embedding_param_ids:
                        # Hidden 2D linear weight (Attention GLA, Shared Expert, Exit Heads) -> Muon
                        self.muon_params.append(param)
                    elif (
                        not self.use_muon and id(param) not in router_param_ids and id(param) not in embedding_param_ids
                    ):
                        # Standard GaLore fallback
                        self.layer_groups.append(
                            {
                                "module": None,
                                "expert_idx": None,
                                "param": param,
                                "projection": None,
                                "projection_side": "left" if param.shape[0] <= param.shape[1] else "right",
                                "proj_step": 0,
                                "state": {"momentum": None, "variance": None, "step": 0},
                            }
                        )
                        group_idx += 1
                    else:
                        # Router head or token embedding -> Base Optimizer (AdamW)
                        self.non_expert_params.append(param)

            # 3. 1D parameters (RMSNorm, biases) -> Base Optimizer (AdamW)
            for name, param in model.named_parameters():
                if param.requires_grad and id(param) not in seen_param_ids:
                    seen_param_ids.add(id(param))
                    self.non_expert_params.append(param)
        else:
            # Iterable of parameters or parameter groups
            for p in model:
                if isinstance(p, dict) and "params" in p:
                    param_candidates = p["params"]
                else:
                    param_candidates = [p]
                for param in param_candidates:
                    if not isinstance(param, (torch.nn.Parameter, torch.Tensor)):
                        continue
                    if param.requires_grad and id(param) not in seen_param_ids:
                        seen_param_ids.add(id(param))
                        if id(param) in explicit_param_roles:
                            role = explicit_param_roles[id(param)]
                            if role in ("muon", "orthogonal") and param.dim() == 2:
                                self.muon_params.append(param)
                            elif role in ("expert", "centroid", "subspace") and param.dim() >= 2:
                                self.layer_groups.append(
                                    {
                                        "module": None,
                                        "expert_idx": None,
                                        "param": param,
                                        "projection": None,
                                        "projection_side": "left" if param.shape[0] <= param.shape[1] else "right",
                                        "proj_step": 0,
                                        "state": {"momentum": None, "variance": None, "step": 0},
                                    }
                                )
                            else:
                                self.non_expert_params.append(param)
                            continue

                        if self.use_muon and param.dim() == 2:
                            # Vocabulary embedding tables (e.g. [vocab_size, dim]) must stay on base AdamW
                            is_vocab_embedding = (
                                max(param.shape[0], param.shape[1]) >= 1000
                                and max(param.shape[0], param.shape[1]) / min(param.shape[0], param.shape[1]) >= 4.0
                            )
                            if is_vocab_embedding:
                                self.non_expert_params.append(param)
                            else:
                                self.muon_params.append(param)
                        elif param.dim() >= 2:
                            self.layer_groups.append(
                                {
                                    "module": None,
                                    "expert_idx": None,
                                    "param": param,
                                    "projection": None,
                                    "projection_side": "left" if param.shape[0] <= param.shape[1] else "right",
                                    "proj_step": 0,
                                    "state": {"momentum": None, "variance": None, "step": 0},
                                }
                            )
                        else:
                            self.non_expert_params.append(param)

        if self.use_muon and len(self.muon_params) > 0:
            self.muon_optimizer = Muon(
                self.muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                ns_steps=muon_ns_steps,
                rms_target=muon_rms_target,
            )
        else:
            self.muon_optimizer = None

        if len(self.non_expert_params) > 0:
            if (
                self.use_8bit_base
                and HAS_8BIT
                and AdamW8bit is not None
                and torch.cuda.is_available()
                and any(p.is_cuda for p in self.non_expert_params)
            ):
                self.base_optimizer = AdamW8bit(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)
            elif torch.cuda.is_available() and any(p.is_cuda for p in self.non_expert_params):
                try:
                    self.base_optimizer = torch.optim.AdamW(
                        self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay, fused=True
                    )
                except Exception:
                    self.base_optimizer = torch.optim.AdamW(
                        self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay
                    )
            elif HAS_8BIT and AdamW8bit is not None and any(p.is_cuda for p in self.non_expert_params):
                self.base_optimizer = AdamW8bit(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)
            else:
                self.base_optimizer = torch.optim.AdamW(
                    self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay
                )
        else:
            self.base_optimizer = None

    def zero_grad(self, set_to_none=True):
        if self.base_optimizer is not None:
            self.base_optimizer.zero_grad(set_to_none=set_to_none)
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        for group in self.layer_groups:
            p = group["param"]
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    def set_lr(self, lr, expert_lr=None):
        if getattr(self, "_in_set_lr", False):
            return
        self._in_set_lr = True
        try:
            self._last_synced_lr = lr
            if hasattr(self, "param_groups") and self.param_groups:
                self.param_groups[0]["lr"] = lr
            if self.base_optimizer is not None:
                for pg in self.base_optimizer.param_groups:
                    pg["lr"] = lr
            if expert_lr is not None:
                self.expert_lr = expert_lr
            elif hasattr(self, "initial_expert_lr") and self.initial_expert_lr is not None:
                ratio = lr / max(1e-12, self.initial_base_lr)
                self.expert_lr = self.initial_expert_lr * ratio
            else:
                self.expert_lr = lr
            if self.muon_optimizer is not None:
                ratio = lr / max(1e-12, self.initial_base_lr)
                for pg in self.muon_optimizer.param_groups:
                    pg["lr"] = self.initial_muon_lr * ratio
        finally:
            self._in_set_lr = False

    def sync_lr_from_param_groups(self):
        """Automatically synchronizes learning rate changes from outer PyTorch lr_schedulers to internal tiers."""
        if hasattr(self, "param_groups") and self.param_groups:
            current_outer_lr = self.param_groups[0].get("lr", self.initial_base_lr)
            if getattr(self, "_last_synced_lr", None) != current_outer_lr:
                self.set_lr(current_outer_lr)

    @torch.no_grad()
    def step_parameters(self, params: Any, is_global_step_end: bool = False):
        """Executes optimizer step on a specific subset of parameters (e.g. single layer or root components)."""
        self.sync_lr_from_param_groups()
        param_list = list(params) if not isinstance(params, list) else params
        param_ids = {id(p) for p in param_list if p.grad is not None}
        if not param_ids:
            if is_global_step_end:
                self.step_count += 1
            return

        # 1. Step matching parameters in base_optimizer (AdamW / 8-bit AdamW)
        if self.base_optimizer is not None:
            for group in self.base_optimizer.param_groups:
                matching = [p for p in group["params"] if id(p) in param_ids]
                if matching:
                    orig_params = group["params"]
                    group["params"] = matching
                    try:
                        self.base_optimizer.step()
                    finally:
                        group["params"] = orig_params

        # 2. Step matching parameters in muon_optimizer
        if self.muon_optimizer is not None:
            for group in self.muon_optimizer.param_groups:
                matching = [p for p in group["params"] if id(p) in param_ids]
                if matching:
                    orig_params = group["params"]
                    group["params"] = matching
                    try:
                        self.muon_optimizer.step()
                    finally:
                        group["params"] = orig_params

        # 3. Step matching expert parameters in layer_groups (Centroid + GaLore)
        expert_lr = self.expert_lr
        expert_beta1, expert_beta2 = self.expert_betas
        expert_wd = self.expert_wd

        active_groups = []
        for group in self.layer_groups:
            p = group["param"]
            if id(p) in param_ids and p.grad is not None:
                active_groups.append(group)

        if not active_groups:
            if is_global_step_end:
                self.step_count += 1
            return

        with torch.amp.autocast(device_type=active_groups[0]["param"].device.type, enabled=False):
            return self._step_active_groups_unautocast(
                active_groups,
                expert_lr=expert_lr,
                expert_beta1=expert_beta1,
                expert_beta2=expert_beta2,
                expert_wd=expert_wd,
                is_global_step_end=is_global_step_end,
            )

    @torch.no_grad()
    def _step_active_groups_unautocast(
        self,
        active_groups: list[dict[str, Any]],
        *,
        expert_lr: float,
        expert_beta1: float,
        expert_beta2: float,
        expert_wd: float,
        is_global_step_end: bool = False,
    ):
        if self.use_triangular_core:
            if getattr(self, "steer_mode", "upgraded") in ("upgraded", "random_same_mem"):
                return self._step_triangular_direct_sum_v3(active_groups, is_global_step_end=is_global_step_end)
            # Universal Triangular I-Matrix Engine (Legacy v2 mode)
            # Condenses all parameter matrices into (N, rank+1, rank+1) Upper Triangles
            # using dual isometric I-bases (Q_L^T Q_L = I, Q_R^T Q_R = I)
            N = len(active_groups)
            device = active_groups[0]["param"].device
            dtype = active_groups[0]["param"].dtype

            if not hasattr(self, "_triangular_buffers"):
                self._triangular_buffers = {}
            buf_key = (N, device, dtype)
            if buf_key not in self._triangular_buffers:
                self._triangular_buffers[buf_key] = {
                    "batch_mom": torch.zeros(N, self.rank + 1, self.rank + 1, device=device, dtype=torch.float32),
                    "batch_var": torch.zeros(N, self.rank + 1, self.rank + 1, device=device, dtype=torch.float32),
                    "R_batch": torch.zeros(N, self.rank + 1, self.rank + 1, device=device, dtype=dtype),
                    "batch_delta_R": torch.zeros(N, self.rank + 1, self.rank + 1, device=device, dtype=dtype),
                    "step_vals": torch.zeros(N, 1, 1, device=device, dtype=torch.float32),
                    "steer_step_vals": torch.zeros(N, 1, 1, device=device, dtype=torch.float32),
                }
            buf = self._triangular_buffers[buf_key]
            batch_mom = buf["batch_mom"]
            batch_var = buf["batch_var"]
            R_batch = buf["R_batch"]
            batch_delta_R = buf["batch_delta_R"]
            step_vals = buf["step_vals"]
            steer_step_vals = buf["steer_step_vals"]
            R_batch.zero_()
            batch_delta_R.zero_()

            if not hasattr(self, "_bucket_buffers"):
                self._bucket_buffers = {}

            # Group active_groups by (m, n, r, steer_role) so each bucket has uniform geometry & steering axis
            raw_buckets = {}
            for g in active_groups:
                m, n = g["param"].shape
                r = min(self.rank, min(m, n))
                s_role = g.get("steer_role", "input")
                key = (m, n, r, s_role)
                if key not in raw_buckets:
                    raw_buckets[key] = []
                raw_buckets[key].append(g)

            shape_buckets = []
            cursor = 0
            for key, g_list in raw_buckets.items():
                K = len(g_list)
                shape_buckets.append((key, cursor, cursor + K, g_list))
                cursor += K

            def _batched_qr(Y_mat: torch.Tensor, target_dtype: torch.dtype = dtype) -> torch.Tensor:
                Q, diag = batched_qr(
                    Y_mat,
                    cond_threshold=1e4,
                    eps_shift=1e-7,
                    orth_tol=1e-4,
                    return_diagnostics=True,
                    out_dtype=target_dtype,
                )
                self._record_qr_diagnostics(diag)
                return Q

            bucket_cache = []
            step_list = [0.0] * N
            steer_step_list = [0.0] * N
            is_upgraded = getattr(self, "steer_mode", "upgraded") == "upgraded"
            n_steer_dirs = int(getattr(self, "num_steer_dirs", 4))

            # 1 & 2. Batched Subspace Init/Update, Batched Centroid Steering, and Batched Projection
            for (m, n, r, steer_role), start, end, g_list in shape_buckets:
                K = end - start
                d_small = min(m, n)
                d_large = max(m, n)
                side = "left" if m <= n else "right"
                centroid_on_ql = m > n
                in_dim = n

                def _extract_steering_pair(g):
                    mod = g["module"]
                    e_idx = g["expert_idx"]
                    if mod is None or e_idx is None:
                        return None, None
                    if hasattr(mod, "get_expert_steering_directions"):
                        return mod.get_expert_steering_directions(
                            e_idx,
                            steer_role=steer_role,
                            num_dirs=n_steer_dirs,
                            steer_mode=getattr(self, "steer_mode", "upgraded"),
                            device=device,
                            dtype=dtype,
                        )
                    if getattr(mod, "last_centroids", None) is None or e_idx >= mod.last_centroids.size(0):
                        return None, None
                    c_e = mod.last_centroids[e_idx].to(device=device, dtype=dtype)
                    if steer_role == "hidden":
                        if is_upgraded and getattr(mod, "last_hidden_centroids", None) is not None:
                            h_raw = mod.last_hidden_centroids[e_idx]
                            if float(h_raw.float().norm().item()) > 1e-8:
                                h_c = h_raw.to(device=device, dtype=dtype)
                                return h_c, h_c.unsqueeze(-1)
                        exp = g.get("expert")
                        if exp is not None and len(exp) >= 2:
                            with torch.no_grad():
                                h = exp[0](c_e.unsqueeze(0))
                                if len(exp) > 2:
                                    h = exp[1](h)
                                return h.squeeze(0).to(device=device, dtype=dtype), None
                        return None, None
                    return c_e, None

                bkey = (K, m, n, r, steer_role, device, dtype)
                if bkey not in self._bucket_buffers:
                    self._bucket_buffers[bkey] = {
                        "QL_stack": torch.zeros(K, d_small, self.rank + 1, device=device, dtype=dtype),
                        "QR_stack": torch.zeros(K, d_large, self.rank + 1, device=device, dtype=dtype),
                    }
                bbuf = self._bucket_buffers[bkey]
                QL_stack = bbuf["QL_stack"]
                QR_stack = bbuf["QR_stack"]

                G_raw = torch.stack([g["param"].grad.data for g in g_list], dim=0)
                G_stack = G_raw if m <= n else G_raw.transpose(1, 2)  # [K, d_small, d_large]

                init_flags = [g.get("Q_L") is None for g in g_list]
                upd_flags = []
                for b_idx, g in enumerate(g_list):
                    if init_flags[b_idx]:
                        upd_flags.append(False)
                        continue
                    steps_since_proj = self.step_count - g.get("proj_step", 0)
                    time_refresh = steps_since_proj >= self.update_gap
                    energy_refresh = (
                        self.adaptive_subspace_refresh
                        and steps_since_proj >= self.min_subspace_gap
                        and float(g.get("last_energy_ratio", 1.0)) < self.energy_threshold
                    )
                    upd_flags.append(time_refresh or energy_refresh)
                any_refresh = any(init_flags) or any(upd_flags)

                steer_pairs = (
                    [_extract_steering_pair(g) for g in g_list] if self.steer_scale > 0 else [(None, None)] * K
                )

                if any_refresh:
                    # Periodic refresh path (every update_gap steps): performs one boolean vector readback
                    # per shape bucket to filter unrouted zero-gradient experts before batched_qr.
                    # Non-refresh steps (95%–99.5% of training) bypass this block completely with zero host-device syncs.
                    refresh_indices = [b_idx for b_idx in range(K) if init_flags[b_idx] or upd_flags[b_idx]]
                    G_ref_all = G_stack[refresh_indices].float()
                    g_norms = torch.linalg.norm(G_ref_all, dim=(1, 2))
                    nz_flags = (g_norms > 1e-12).tolist()
                    qr_sub_indices = [i for i, nz in enumerate(nz_flags) if nz]
                    qr_bucket_indices = [refresh_indices[i] for i in qr_sub_indices]

                    qr_map: dict[int, int] = {}
                    ql_out: torch.Tensor | None = None
                    qr_out: torch.Tensor | None = None

                    if qr_bucket_indices:
                        G_qr = G_ref_all[qr_sub_indices]
                        has_preset_proj = all(
                            (
                                init_flags[b_idx]
                                and g_list[b_idx].get("projection") is not None
                                and tuple(g_list[b_idx]["projection"].shape) == (d_small, r)
                            )
                            for b_idx in qr_bucket_indices
                        )
                        if has_preset_proj:
                            ql_f32 = torch.stack(
                                [
                                    g_list[b_idx]["projection"].to(device=device, dtype=torch.float32)
                                    for b_idx in qr_bucket_indices
                                ],
                                dim=0,
                            )
                            Z = torch.bmm(G_qr.transpose(1, 2), ql_f32)
                            qr_out = _batched_qr(Z, target_dtype=dtype)
                            ql_out = ql_f32.to(dtype=dtype)
                        else:
                            K_qr = len(qr_bucket_indices)
                            Omega = torch.randn(K_qr, d_large, r, device=device, dtype=torch.float32)
                            G_work = G_qr
                            c_hat_qr: torch.Tensor | None = None
                            if is_upgraded and self.steer_scale > 0:
                                ref_c_vecs = [steer_pairs[b_i][0] for b_i in qr_bucket_indices]
                                if all(c_v is not None and c_v.numel() == in_dim for c_v in ref_c_vecs):
                                    C_qr = torch.stack([c_v.float() for c_v in ref_c_vecs], dim=0)  # [K_qr, in_dim]
                                    c_qr_norm = C_qr.norm(dim=-1, keepdim=True)
                                    c_hat_qr = torch.where(
                                        c_qr_norm > 1e-8, C_qr / c_qr_norm.clamp_min(1e-8), torch.zeros_like(C_qr)
                                    ).unsqueeze(-1)  # [K_qr, in_dim, 1]
                                    if centroid_on_ql:
                                        G_defl = G_qr - torch.bmm(c_hat_qr, torch.bmm(c_hat_qr.transpose(1, 2), G_qr))
                                    else:
                                        G_defl = G_qr - torch.bmm(torch.bmm(G_qr, c_hat_qr), c_hat_qr.transpose(1, 2))
                                    defl_norm = torch.linalg.norm(G_defl, dim=(1, 2), keepdim=True)
                                    G_work = torch.where(defl_norm > 1e-9, G_defl, G_qr)

                                    # Seed leading probe columns with secondary multi-directional activation directions
                                    if r >= 2:
                                        for q_i, b_i in enumerate(qr_bucket_indices):
                                            c_sub_i = steer_pairs[b_i][1]
                                            if c_sub_i is not None and c_sub_i.ndim == 2 and c_sub_i.size(1) >= 2:
                                                k_s = min(c_sub_i.size(1) - 1, max(1, r // 2))
                                                u_sec = c_sub_i[:, 1 : 1 + k_s].float()  # [in_dim, k_s]
                                                if centroid_on_ql:
                                                    v_seed = torch.mm(G_work[q_i].t(), u_sec)  # [d_large, k_s]
                                                else:
                                                    c_h_i = c_hat_qr[q_i]  # [d_large, 1]
                                                    v_seed = u_sec - torch.mm(c_h_i, torch.mm(c_h_i.t(), u_sec))
                                                v_s_norm = v_seed.norm(dim=0, keepdim=True)
                                                valid_s = v_s_norm > 1e-8
                                                Omega[q_i, :, :k_s] = torch.where(
                                                    valid_s, v_seed / v_s_norm.clamp_min(1e-8), Omega[q_i, :, :k_s]
                                                )

                            Y = torch.bmm(G_work, Omega)
                            if c_hat_qr is not None and centroid_on_ql:
                                Y_d = Y - torch.bmm(c_hat_qr, torch.bmm(c_hat_qr.transpose(1, 2), Y))
                                Y = torch.where(torch.linalg.norm(Y_d, dim=1, keepdim=True) > 1e-9, Y_d, Y)
                            ql_f32 = _batched_qr(Y, target_dtype=torch.float32)
                            Z = torch.bmm(G_work.transpose(1, 2), ql_f32)
                            if c_hat_qr is not None and not centroid_on_ql:
                                Z_d = Z - torch.bmm(c_hat_qr, torch.bmm(c_hat_qr.transpose(1, 2), Z))
                                Z = torch.where(torch.linalg.norm(Z_d, dim=1, keepdim=True) > 1e-9, Z_d, Z)
                            qr_f32 = _batched_qr(Z, target_dtype=torch.float32)
                            Y = torch.bmm(G_work, qr_f32)
                            if c_hat_qr is not None and centroid_on_ql:
                                Y_d = Y - torch.bmm(c_hat_qr, torch.bmm(c_hat_qr.transpose(1, 2), Y))
                                Y = torch.where(torch.linalg.norm(Y_d, dim=1, keepdim=True) > 1e-9, Y_d, Y)
                            ql_out = _batched_qr(Y, target_dtype=dtype)
                            qr_out = qr_f32.to(dtype=dtype)
                        qr_map = {b_idx: i for i, b_idx in enumerate(qr_bucket_indices)}

                    aligned_proj_step = (self.step_count // max(1, self.update_gap)) * self.update_gap
                    for b_idx, g in enumerate(g_list):
                        idx = start + b_idx
                        st = g["state"]
                        if init_flags[b_idx] or upd_flags[b_idx]:
                            if b_idx in qr_map and ql_out is not None and qr_out is not None:
                                q_i = qr_map[b_idx]
                                QL_stack[b_idx].zero_()
                                QL_stack[b_idx, :, :r].copy_(ql_out[q_i])
                                QR_stack[b_idx].zero_()
                                QR_stack[b_idx, :, :r].copy_(qr_out[q_i])
                                g["_has_nonzero_grad"] = True
                            elif init_flags[b_idx]:
                                QL_stack[b_idx].zero_()
                                QL_stack[b_idx, :r, :r].copy_(torch.eye(r, device=device, dtype=dtype))
                                QR_stack[b_idx].zero_()
                                QR_stack[b_idx, :r, :r].copy_(torch.eye(r, device=device, dtype=dtype))
                                g["_has_nonzero_grad"] = False
                            else:
                                if g["Q_L"].data_ptr() != QL_stack[b_idx].data_ptr():
                                    QL_stack[b_idx].copy_(g["Q_L"])
                                    QR_stack[b_idx].copy_(g["Q_R"])
                                g["_has_nonzero_grad"] = False
                            g["Q_L"] = QL_stack[b_idx]
                            g["Q_R"] = QR_stack[b_idx]
                            g["projection"] = QL_stack[b_idx, :, :r]
                            g["projection_side"] = side
                            g["proj_step"] = (
                                aligned_proj_step if not self.adaptive_subspace_refresh else self.step_count
                            )
                            batch_mom[idx].zero_()
                            batch_var[idx].zero_()
                            st["momentum"] = batch_mom[idx]
                            st["variance"] = batch_var[idx]
                            st["subspace_step"] = 0
                            st["steer_step"] = 0
                            if init_flags[b_idx]:
                                st["step"] = 0
                        else:
                            if g["Q_L"].data_ptr() != QL_stack[b_idx].data_ptr():
                                QL_stack[b_idx].copy_(g["Q_L"])
                                QR_stack[b_idx].copy_(g["Q_R"])
                                g["Q_L"] = QL_stack[b_idx]
                                g["Q_R"] = QR_stack[b_idx]
                                g["projection"] = QL_stack[b_idx, :, :r]
                            if st["momentum"].data_ptr() != batch_mom[idx].data_ptr():
                                batch_mom[idx].copy_(st["momentum"])
                                batch_var[idx].copy_(st["variance"])
                                st["momentum"] = batch_mom[idx]
                                st["variance"] = batch_var[idx]
                else:
                    for b_idx, g in enumerate(g_list):
                        idx = start + b_idx
                        st = g["state"]
                        if g["Q_L"].data_ptr() != QL_stack[b_idx].data_ptr():
                            QL_stack[b_idx].copy_(g["Q_L"])
                            QR_stack[b_idx].copy_(g["Q_R"])
                            g["Q_L"] = QL_stack[b_idx]
                            g["Q_R"] = QR_stack[b_idx]
                            g["projection"] = QL_stack[b_idx, :, :r]
                        if st["momentum"].data_ptr() != batch_mom[idx].data_ptr():
                            batch_mom[idx].copy_(st["momentum"])
                            batch_var[idx].copy_(st["variance"])
                            st["momentum"] = batch_mom[idx]
                            st["variance"] = batch_var[idx]

                # Batched Centroid Steering with per-group refresh and dual orthonormal vector
                if is_upgraded:
                    group_steer_refresh = [
                        (
                            init_flags[b_idx]
                            or upd_flags[b_idx]
                            or (g.get("last_steer_step") is None)
                            or ((self.step_count - g.get("last_steer_step", 0)) >= self.steer_update_gap)
                        )
                        for b_idx, g in enumerate(g_list)
                    ]
                    needs_steer_refresh = any(group_steer_refresh)
                else:
                    g0 = g_list[0]
                    needs_steer_refresh = (
                        any_refresh
                        or (g0.get("last_steer_step") is None)
                        or ((self.step_count - g0.get("last_steer_step", 0)) >= self.steer_update_gap)
                    )
                    group_steer_refresh = [needs_steer_refresh] * K

                if needs_steer_refresh:
                    # Reset stale 1st/2nd moments on rotated (r+1)-th steering border and reset independent steer_step
                    for b_idx, g in enumerate(g_list):
                        if group_steer_refresh[b_idx]:
                            idx = start + b_idx
                            batch_mom[idx, : r + 1, r].zero_()
                            batch_mom[idx, r, : r + 1].zero_()
                            batch_var[idx, : r + 1, r].zero_()
                            batch_var[idx, r, : r + 1].zero_()
                            if is_upgraded:
                                g["state"]["steer_step"] = 0

                    c_list = [p[0] for p in steer_pairs] if self.steer_scale > 0 else []
                    can_steer = (
                        self.steer_scale > 0
                        and len(c_list) == K
                        and all(c_vec is not None and c_vec.numel() == in_dim for c_vec in c_list)
                    )
                    if can_steer:
                        C_f32 = torch.stack([c_vec.float() for c_vec in c_list], dim=0)
                        C_norm = C_f32.norm(dim=-1, keepdim=True)
                        C_unit = torch.where(C_norm > 1e-8, C_f32 / C_norm.clamp_min(1e-8), torch.zeros_like(C_f32))
                        C = C_unit.to(dtype=dtype).unsqueeze(-1)  # [K, in_dim, 1]
                        QL_r = QL_stack[:, :, :r]
                        QR_r = QR_stack[:, :, :r]
                        if centroid_on_ql:
                            # Input centroid sits on QL (m > n, e.g. e[0] transposed); dual v = G_stack^T c_orth sits on QR
                            c_proj1 = torch.bmm(QL_r, torch.bmm(QL_r.transpose(1, 2), C))
                            c_res1 = C - c_proj1
                            c_proj2 = torch.bmm(QL_r, torch.bmm(QL_r.transpose(1, 2), c_res1))
                            c_res = (c_res1 - c_proj2).squeeze(-1)
                            c_res_norm = c_res.float().norm(dim=-1, keepdim=True)
                            c_orth = torch.where(
                                c_res_norm > 1e-8,
                                (c_res.float() / c_res_norm.clamp_min(1e-8)).to(dtype=dtype),
                                torch.zeros_like(c_res),
                            )
                            if is_upgraded:
                                for b_i, g in enumerate(g_list):
                                    last_c = g.get("last_c_orth")
                                    if (
                                        last_c is not None
                                        and last_c.shape == c_orth[b_i].shape
                                        and last_c.device == c_orth.device
                                    ):
                                        if float(torch.dot(c_orth[b_i].float(), last_c.float()).item()) < 0.0:
                                            c_orth[b_i].neg_()
                                    g["last_c_orth"] = c_orth[b_i].detach().clone()

                            v = torch.bmm(G_stack.transpose(1, 2), c_orth.unsqueeze(-1))
                            v_proj1 = torch.bmm(QR_r, torch.bmm(QR_r.transpose(1, 2), v))
                            v_res1 = v - v_proj1
                            v_proj2 = torch.bmm(QR_r, torch.bmm(QR_r.transpose(1, 2), v_res1))
                            v_res = (v_res1 - v_proj2).squeeze(-1)
                            v_res_norm = v_res.float().norm(dim=-1, keepdim=True)
                            steer_mask = (c_res_norm.squeeze(-1) > 1e-8) & (v_res_norm.squeeze(-1) > 1e-8)
                            ql_vec = torch.where(steer_mask.unsqueeze(-1), c_orth, torch.zeros_like(c_orth))
                            qr_vec = torch.where(
                                steer_mask.unsqueeze(-1),
                                (v_res.float() / v_res_norm.clamp_min(1e-8)).to(dtype=dtype),
                                torch.zeros_like(v_res),
                            )
                        else:
                            # Input centroid sits on QR (m <= n, e[2]); dual u = G_stack v_orth sits on QL
                            v_proj1 = torch.bmm(QR_r, torch.bmm(QR_r.transpose(1, 2), C))
                            v_res1 = C - v_proj1
                            v_proj2 = torch.bmm(QR_r, torch.bmm(QR_r.transpose(1, 2), v_res1))
                            v_res = (v_res1 - v_proj2).squeeze(-1)
                            v_res_norm = v_res.float().norm(dim=-1, keepdim=True)
                            v_orth = torch.where(
                                v_res_norm > 1e-8,
                                (v_res.float() / v_res_norm.clamp_min(1e-8)).to(dtype=dtype),
                                torch.zeros_like(v_res),
                            )
                            if is_upgraded:
                                for b_i, g in enumerate(g_list):
                                    last_c = g.get("last_c_orth")
                                    if (
                                        last_c is not None
                                        and last_c.shape == v_orth[b_i].shape
                                        and last_c.device == v_orth.device
                                    ):
                                        if float(torch.dot(v_orth[b_i].float(), last_c.float()).item()) < 0.0:
                                            v_orth[b_i].neg_()
                                    g["last_c_orth"] = v_orth[b_i].detach().clone()

                            u = torch.bmm(G_stack, v_orth.unsqueeze(-1))
                            u_proj1 = torch.bmm(QL_r, torch.bmm(QL_r.transpose(1, 2), u))
                            u_res1 = u - u_proj1
                            u_proj2 = torch.bmm(QL_r, torch.bmm(QL_r.transpose(1, 2), u_res1))
                            u_res = (u_res1 - u_proj2).squeeze(-1)
                            u_res_norm = u_res.float().norm(dim=-1, keepdim=True)
                            steer_mask = (v_res_norm.squeeze(-1) > 1e-8) & (u_res_norm.squeeze(-1) > 1e-8)
                            ql_vec = torch.where(
                                steer_mask.unsqueeze(-1),
                                (u_res.float() / u_res_norm.clamp_min(1e-8)).to(dtype=dtype),
                                torch.zeros_like(u_res),
                            )
                            qr_vec = torch.where(steer_mask.unsqueeze(-1), v_orth, torch.zeros_like(v_orth))

                        for b_i, g in enumerate(g_list):
                            if group_steer_refresh[b_i]:
                                QL_stack[b_i, :, r].copy_(ql_vec[b_i])
                                QR_stack[b_i, :, r].copy_(qr_vec[b_i])
                                g["steer_applied"] = bool(g.get("_has_nonzero_grad", True)) and bool(
                                    steer_mask[b_i].item()
                                )
                                g["last_steer_step"] = self.step_count
                    else:
                        for b_i, g in enumerate(g_list):
                            if group_steer_refresh[b_i]:
                                QL_stack[b_i, :, r].zero_()
                                QR_stack[b_i, :, r].zero_()
                                g["steer_applied"] = False
                                g["last_steer_step"] = self.step_count

                # Batched isometric projection into R_batch (uncorrupted by steer_scale before AdamW)
                QL_act = QL_stack[:, :, : r + 1]
                QR_act = QR_stack[:, :, : r + 1]
                temp_proj = torch.bmm(G_stack, QR_act)
                S_bmm = torch.bmm(QL_act.transpose(1, 2), temp_proj)

                if self.adaptive_subspace_refresh:
                    s_energies = (S_bmm[:, :r, :r].float() ** 2).sum(dim=(-2, -1))
                    g_energies = (G_stack.float() ** 2).sum(dim=(-2, -1)).clamp_min(1e-12)
                    ratios = s_energies / g_energies
                    for b_idx, g in enumerate(g_list):
                        g["last_energy_ratio"] = ratios[b_idx]

                R_batch[start:end, : r + 1, : r + 1].copy_(
                    torch.triu(S_bmm) if self.core_mask_mode == "triu" else S_bmm
                )

                for b_idx, g in enumerate(g_list):
                    idx = start + b_idx
                    sub_step = g["state"].get("subspace_step", 0) + 1
                    g["state"]["subspace_step"] = sub_step
                    if is_upgraded:
                        str_step = g["state"].get("steer_step", 0) + 1
                    else:
                        str_step = sub_step
                    g["state"]["steer_step"] = str_step
                    g["state"]["step"] += 1
                    step_list[idx] = float(sub_step)
                    steer_step_list[idx] = float(str_step)

                bucket_cache.append(
                    (
                        m,
                        n,
                        r,
                        start,
                        end,
                        g_list,
                        QL_act,
                        QR_act,
                        QL_stack,
                        QR_stack,
                        G_stack,
                        S_bmm,
                        steer_pairs,
                        centroid_on_ql,
                    )
                )

            first_sub = step_list[0]
            if all(s == first_sub for s in step_list):
                step_vals.fill_(first_sub)
            else:
                step_vals.copy_(torch.tensor(step_list, device=device, dtype=torch.float32).view(N, 1, 1))

            first_str = steer_step_list[0]
            if all(s == first_str for s in steer_step_list):
                steer_step_vals.fill_(first_str)
            else:
                steer_step_vals.copy_(torch.tensor(steer_step_list, device=device, dtype=torch.float32).view(N, 1, 1))

            # 3. Unified Fused Triton Core AdamW Step with PyTorch Vectorized Fallback
            from lucid_machines.kernels.fast_triangular import fast_triangular_adamw

            if all(b[2] == self.rank for b in bucket_cache):
                fast_triangular_adamw(
                    R_batch,
                    batch_mom,
                    batch_var,
                    batch_delta_R,
                    step_vals,
                    beta1=expert_beta1,
                    beta2=expert_beta2,
                    eps=1e-8,
                    mask_mode=self.core_mask_mode,
                    steer_step_vals=steer_step_vals if is_upgraded else None,
                    rms_reg=0.35 if is_upgraded else 0.0,
                )
            else:
                for _, _, r_b, b_start, b_end, *_ in bucket_cache:
                    dr_b = r_b + 1
                    fast_triangular_adamw(
                        R_batch[b_start:b_end, :dr_b, :dr_b],
                        batch_mom[b_start:b_end, :dr_b, :dr_b],
                        batch_var[b_start:b_end, :dr_b, :dr_b],
                        batch_delta_R[b_start:b_end, :dr_b, :dr_b],
                        step_vals[b_start:b_end],
                        beta1=expert_beta1,
                        beta2=expert_beta2,
                        eps=1e-8,
                        mask_mode=self.core_mask_mode,
                        steer_step_vals=steer_step_vals[b_start:b_end] if is_upgraded else None,
                        rms_reg=0.35 if is_upgraded else 0.0,
                    )

            # 4. Batched BMM Reconstruction & Fused Multi-Tensor Weight Update
            beta_wd = 1.0 - expert_lr * expert_wd if expert_wd != 0 else 1.0
            alpha_upd = -expert_lr
            for (
                m,
                n,
                r,
                start,
                end,
                g_list,
                QL_act,
                QR_act,
                QL_stack,
                QR_stack,
                G_stack,
                S_bmm,
                steer_pairs,
                centroid_on_ql,
            ) in bucket_cache:
                delta_stack = batch_delta_R[start:end, : r + 1, : r + 1]
                live_ortho_updates: dict[int, torch.Tensor] = {}
                if self.steer_scale > 0:
                    steer_rows = [b_i for b_i, g in enumerate(g_list) if g.get("steer_applied", False)]
                    if steer_rows:
                        if not is_upgraded:
                            s_idx = torch.tensor(steer_rows, dtype=torch.long, device=delta_stack.device)
                            scale_mult = 1.0 + self.steer_scale
                            delta_stack[s_idx, : r + 1, r] = delta_stack[s_idx, : r + 1, r] * scale_mult
                            if self.core_mask_mode == "full":
                                delta_stack[s_idx, r, :r] = delta_stack[s_idx, r, :r] * scale_mult
                        else:
                            # Upgraded multi-directional contrastive & Fisher-alignment-weighted steering
                            for b_i in steer_rows:
                                g = g_list[b_i]
                                mod = g.get("module")
                                e_idx = g.get("expert_idx")
                                util_mult = 1.0
                                if (
                                    mod is not None
                                    and hasattr(mod, "expert_load_ratio")
                                    and e_idx is not None
                                    and e_idx < mod.expert_load_ratio.size(0)
                                ):
                                    load_r = float(mod.expert_load_ratio[e_idx].item())
                                    util_mult = min(1.5, max(0.5, 1.0 + 0.5 * (1.0 - load_r)))

                                g_tot_norm = float(G_stack[b_i].float().norm().item())
                                border_s = S_bmm[b_i, : r + 1, r].float().norm().item()
                                if self.core_mask_mode == "full":
                                    border_s = math.hypot(border_s, S_bmm[b_i, r, :r].float().norm().item())
                                expected_slice = g_tot_norm / max(1.0, math.sqrt(float(r + 1)))
                                align_gain = min(1.5, max(0.15, border_s / max(1e-12, expected_slice)))
                                eff_steer = float(self.steer_scale) * util_mult * align_gain
                                scale_mult = 1.0 + eff_steer

                                delta_stack[b_i, : r + 1, r] = delta_stack[b_i, : r + 1, r] * scale_mult
                                if self.core_mask_mode == "full":
                                    delta_stack[b_i, r, :r] = delta_stack[b_i, r, :r] * scale_mult

                                c_sub = steer_pairs[b_i][1]
                                if c_sub is not None and c_sub.ndim == 2 and c_sub.size(1) >= 1 and g_tot_norm > 1e-12:
                                    Q_in = QL_act[b_i] if centroid_on_ql else QR_act[b_i]  # [in_dim, r+1]
                                    W_c = torch.mm(Q_in.transpose(0, 1), c_sub)  # [r+1, k_c]
                                    sub_boost = 0.5 * eff_steer
                                    if centroid_on_ql:
                                        proj_d = torch.mm(W_c, torch.mm(W_c.transpose(0, 1), delta_stack[b_i]))
                                    else:
                                        proj_d = torch.mm(torch.mm(delta_stack[b_i], W_c), W_c.transpose(0, 1))
                                    if self.core_mask_mode == "triu":
                                        proj_d = torch.triu(proj_d)
                                    delta_stack[b_i].add_(proj_d, alpha=sub_boost)

                                    # Live out-of-core activation subspace drift correction between update_gap refreshes
                                    C_perp = c_sub.float() - torch.mm(Q_in.float(), W_c.float())
                                    if float(C_perp.norm().item()) > 1e-5:
                                        G_b_f32 = G_stack[b_i].float()
                                        if centroid_on_ql:
                                            G_perp = torch.mm(C_perp.transpose(0, 1), G_b_f32)  # [k_c, d_large]
                                            dW_perp = torch.mm(G_perp.transpose(0, 1), C_perp.transpose(0, 1))  # [m, n]
                                        else:
                                            G_perp = torch.mm(G_b_f32, C_perp)  # [d_small, k_c]
                                            dW_perp = torch.mm(G_perp, C_perp.transpose(0, 1))  # [m, n]
                                        g_perp_norm = float(G_perp.norm().item())
                                        dw_perp_norm = float(dW_perp.norm().item())
                                        if g_perp_norm > 1e-10 and dw_perp_norm > 1e-10:
                                            rho_perp = min(1.0, g_perp_norm / max(1e-12, g_tot_norm))
                                            live_ortho_updates[b_i] = (
                                                dW_perp.to(dtype=dtype),
                                                dw_perp_norm,
                                                float(self.steer_scale) * util_mult * rho_perp,
                                            )

                if m <= n:
                    temp_recon = torch.bmm(QL_act, delta_stack)
                    delta_W = torch.bmm(temp_recon, QR_act.transpose(1, 2))
                else:
                    temp_recon = torch.bmm(QR_act, delta_stack.transpose(1, 2))
                    delta_W = torch.bmm(temp_recon, QL_act.transpose(1, 2))

                if live_ortho_updates:
                    k_scale = math.sqrt(max(1.0, float(n_steer_dirs)) / max(1.0, float(r + 1)))
                    for b_i, (dw_p, dw_p_norm, gain_p) in live_ortho_updates.items():
                        core_norm = float(delta_W[b_i].float().norm().item())
                        if core_norm > 1e-10:
                            alpha_p = gain_p * k_scale * (core_norm / max(1e-12, dw_p_norm))
                            delta_W[b_i].add_(dw_p, alpha=alpha_p)

                params_b = [g["param"].data for g in g_list]
                if expert_wd != 0:
                    torch._foreach_mul_(params_b, beta_wd)
                torch._foreach_add_(params_b, list(delta_W.unbind(0)), alpha=alpha_upd)

                if N < len(self.layer_groups):
                    # Subset / Layer-Streaming mode: isolate per-group state tensors before CPU offload
                    for b_idx, g in enumerate(g_list):
                        idx = start + b_idx
                        g["Q_L"] = QL_stack[b_idx].clone()
                        g["Q_R"] = QR_stack[b_idx].clone()
                        g["projection"] = g["Q_L"][:, :r]
                        g["state"]["momentum"] = batch_mom[idx].clone()
                        g["state"]["variance"] = batch_var[idx].clone()

            if is_global_step_end:
                self.step_count += 1
            return

        # Ensure all active groups have initialized projection bases
        for group in active_groups:
            p = group["param"]
            grad = p.grad.data
            m, n = grad.shape
            rank = min(self.rank, m, n)
            state = group["state"]

            if group.get("proj_basis") is None:
                target_dim = m if m <= n else n
                group["projection_side"] = "left" if m <= n else "right"
                basis = torch.zeros(target_dim, rank + 1, device=grad.device, dtype=grad.dtype)
                init_p = torch.empty(target_dim, rank, device=grad.device, dtype=torch.float32)
                torch.nn.init.orthogonal_(init_p)
                basis[:, :rank] = init_p.to(dtype=grad.dtype)
                group["projection"] = basis[:, :rank]
                group["proj_basis"] = basis
                state["momentum"] = None
                state["variance"] = None
                state["step"] = 0
                group["last_c_orth"] = None
                group["proj_step"] = self.step_count

        # Group active groups into shape buckets to execute Batched Tensor Core GEMMs (BMM)
        buckets = {}
        for group in active_groups:
            p = group["param"]
            key = (p.shape, p.device, p.dtype)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(group)

        for (shape, device, dtype), bucket_groups in buckets.items():
            B = len(bucket_groups)
            m, n = shape
            rank = min(self.rank, m, n)
            side = bucket_groups[0].get("projection_side", "left")

            # Periodic subspace update check
            upd_groups = [g for g in bucket_groups if (self.step_count - g["proj_step"]) >= self.update_gap]
            if upd_groups:
                if len(upd_groups) > 1:
                    stacked_grads = torch.stack([g["param"].grad.data for g in upd_groups], dim=0)
                    new_projs, new_side, diag_history = batched_fast_subspace_projection(
                        stacked_grads, rank, niter=2, return_diagnostics=True
                    )
                    for d in diag_history:
                        self._record_qr_diagnostics(d)
                    for i, g in enumerate(upd_groups):
                        new_proj = new_projs[i]
                        new_proj_fp32 = new_proj.float()
                        old_proj = g["projection"].to(device=device, dtype=torch.float32)
                        old_side = g.get("projection_side", "left")
                        st = g["state"]
                        if (
                            st["momentum"] is not None
                            and st["variance"] is not None
                            and old_side == new_side
                            and old_proj.shape == new_proj_fp32.shape
                        ):
                            T_mat = old_proj.T @ new_proj_fp32
                            T_sq = T_mat.pow(2)
                            mom_fp32 = st["momentum"].to(device=device, dtype=torch.float32)
                            var_fp32 = st["variance"].to(device=device, dtype=torch.float32)
                            if new_side == "left" and mom_fp32.size(0) >= rank:
                                mom_r = T_mat.T @ mom_fp32[:rank, :]
                                row_energy = T_sq.sum(dim=0, keepdim=True).T.clamp(0.0, 1.0)
                                var_mean = var_fp32[:rank, :].mean(dim=0, keepdim=True)
                                var_r = (T_sq.T @ var_fp32[:rank, :]) + (1.0 - row_energy) * var_mean
                                mom_fp32 = torch.cat([mom_r, mom_fp32[rank:, :]], dim=0)
                                var_fp32 = torch.cat([var_r, var_fp32[rank:, :]], dim=0)
                            elif new_side == "right" and mom_fp32.size(1) >= rank:
                                mom_r = mom_fp32[:, :rank] @ T_mat
                                col_energy = T_sq.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
                                var_mean = var_fp32[:, :rank].mean(dim=1, keepdim=True)
                                var_r = (var_fp32[:, :rank] @ T_sq) + var_mean * (1.0 - col_energy)
                                mom_fp32 = torch.cat([mom_r, mom_fp32[:, rank:]], dim=1)
                                var_fp32 = torch.cat([var_r, var_fp32[:, rank:]], dim=1)
                            st["momentum"] = mom_fp32.to(dtype=dtype)
                            st["variance"] = var_fp32.to(dtype=dtype)
                        else:
                            st["momentum"] = None
                            st["variance"] = None
                            st["step"] = 0
                        g["projection_side"] = new_side
                        target_dim = m if new_side == "left" else n
                        basis = torch.zeros(target_dim, rank + 1, device=device, dtype=dtype)
                        basis[:, :rank] = new_proj.to(device=device, dtype=dtype)
                        g["projection"] = basis[:, :rank]
                        g["proj_basis"] = basis
                        g["proj_step"] = self.step_count
                else:
                    for g in upd_groups:
                        new_proj, new_side = fast_subspace_projection(g["param"].grad.data, rank, niter=2)
                        new_proj_fp32 = new_proj.float()
                        old_proj = g["projection"].to(device=device, dtype=torch.float32)
                        old_side = g.get("projection_side", "left")
                        st = g["state"]
                        if (
                            st["momentum"] is not None
                            and st["variance"] is not None
                            and old_side == new_side
                            and old_proj.shape == new_proj_fp32.shape
                        ):
                            T_mat = old_proj.T @ new_proj_fp32
                            T_sq = T_mat.pow(2)
                            mom_fp32 = st["momentum"].to(device=device, dtype=torch.float32)
                            var_fp32 = st["variance"].to(device=device, dtype=torch.float32)
                            if new_side == "left" and mom_fp32.size(0) >= rank:
                                mom_r = T_mat.T @ mom_fp32[:rank, :]
                                row_energy = T_sq.sum(dim=0, keepdim=True).T.clamp(0.0, 1.0)
                                var_mean = var_fp32[:rank, :].mean(dim=0, keepdim=True)
                                var_r = (T_sq.T @ var_fp32[:rank, :]) + (1.0 - row_energy) * var_mean
                                mom_fp32 = torch.cat([mom_r, mom_fp32[rank:, :]], dim=0)
                                var_fp32 = torch.cat([var_r, var_fp32[rank:, :]], dim=0)
                            elif new_side == "right" and mom_fp32.size(1) >= rank:
                                mom_r = mom_fp32[:, :rank] @ T_mat
                                col_energy = T_sq.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
                                var_mean = var_fp32[:, :rank].mean(dim=1, keepdim=True)
                                var_r = (var_fp32[:, :rank] @ T_sq) + var_mean * (1.0 - col_energy)
                                mom_fp32 = torch.cat([mom_r, mom_fp32[:, rank:]], dim=1)
                                var_fp32 = torch.cat([var_r, var_fp32[:, rank:]], dim=1)
                            st["momentum"] = mom_fp32.to(dtype=dtype)
                            st["variance"] = var_fp32.to(dtype=dtype)
                        else:
                            st["momentum"] = None
                            st["variance"] = None
                            st["step"] = 0
                        g["projection_side"] = new_side
                        target_dim = m if new_side == "left" else n
                        basis = torch.zeros(target_dim, rank + 1, device=device, dtype=dtype)
                        basis[:, :rank] = new_proj.to(device=device, dtype=dtype)
                        g["projection"] = basis[:, :rank]
                        g["proj_basis"] = basis
                        g["proj_step"] = self.step_count

            # Centroid steering setup for each group in bucket
            for g in bucket_groups:
                proj = g["projection"].to(device=device, dtype=dtype)
                module = g["module"]
                expert_idx = g["expert_idx"]
                centroids = module.last_centroids if module is not None else None
                steer_applied = False

                if (
                    centroids is not None
                    and expert_idx is not None
                    and expert_idx < centroids.size(0)
                    and self.steer_scale > 0
                ):
                    c = centroids[expert_idx]
                    target_dim = m if side == "left" else n
                    if c.size(0) != target_dim:
                        if (
                            getattr(self, "steer_mode", "upgraded") == "upgraded"
                            and getattr(module, "last_hidden_centroids", None) is not None
                            and expert_idx < module.last_hidden_centroids.size(0)
                            and module.last_hidden_centroids.size(1) == target_dim
                            and float(module.last_hidden_centroids[expert_idx].float().norm().item()) > 1e-8
                        ):
                            c_projected = module.last_hidden_centroids[expert_idx].to(
                                device=proj.device, dtype=proj.dtype
                            )
                        else:
                            expert = module.experts[expert_idx]
                            with torch.no_grad():
                                c_in = c.unsqueeze(0).to(device=proj.device, dtype=proj.dtype)
                                h = expert[0](
                                    c_in.to(
                                        dtype=expert[0].weight.dtype if hasattr(expert[0], "weight") else proj.dtype
                                    )
                                )
                                if len(expert) > 2:
                                    h = expert[1](h)
                                c_projected = h.squeeze(0).to(device=proj.device, dtype=proj.dtype)
                    else:
                        c_projected = c.to(device=proj.device, dtype=proj.dtype)

                    c_norm = c_projected.norm()
                    if c_norm > 1e-8:
                        c_hat = (c_projected / c_norm).to(device=proj.device, dtype=proj.dtype)
                        c_proj = proj @ (proj.T @ c_hat)
                        c_res = c_hat - c_proj
                        c_res_norm = c_res.norm()
                        if c_res_norm > 1e-8:
                            c_orth = (c_res / c_res_norm).to(device=proj.device, dtype=proj.dtype)
                            last_c = g.get("last_c_orth")
                            if last_c is not None and last_c.shape == c_orth.shape and last_c.device == c_orth.device:
                                if torch.dot(c_orth.view(-1), last_c.view(-1)) < 0:
                                    c_orth = -c_orth
                            g["last_c_orth"] = c_orth.detach()
                            if hasattr(module, "expert_load_ratio") and expert_idx < module.expert_load_ratio.size(0):
                                load_r = module.expert_load_ratio[expert_idx].to(device=proj.device, dtype=proj.dtype)
                                util_mult = torch.clamp(1.0 + (1.0 - load_r), min=0.3, max=2.0)
                            else:
                                util_mult = torch.tensor(1.0, device=proj.device, dtype=proj.dtype)
                            residual_mult = torch.clamp(c_res_norm, min=0.0, max=1.0)
                            effective_steer = (self.steer_scale * util_mult * residual_mult).to(dtype=proj.dtype)
                            basis = g["proj_basis"]
                            basis[:, rank].copy_(c_orth)
                            g["proj_unit"] = basis
                            proj_recon = basis.clone()
                            proj_recon[:, rank].mul_(effective_steer)
                            g["proj_recon"] = proj_recon
                            steer_applied = True

                if not steer_applied:
                    basis = g["proj_basis"]
                    basis[:, rank].zero_()
                    g["proj_unit"] = basis
                    g["proj_recon"] = basis
                    g["last_c_orth"] = None

            if B > 1:
                # Batched Tensor Core GEMMs (BMM) across all expert matrices simultaneously
                batch_grads = torch.stack([g["param"].grad.data for g in bucket_groups], dim=0)
                batch_units = torch.stack([g["proj_unit"] for g in bucket_groups], dim=0)
                batch_recons = torch.stack([g["proj_recon"] for g in bucket_groups], dim=0)

                if side == "left":
                    batch_g_lr = torch.bmm(batch_units.transpose(1, 2), batch_grads)
                else:
                    batch_g_lr = torch.bmm(batch_grads, batch_units)

                for idx, g in enumerate(bucket_groups):
                    st = g["state"]
                    st["step"] += 1
                    g_lr_item = batch_g_lr[idx]
                    if st["momentum"] is None:
                        st["momentum"] = g_lr_item.clone()
                        st["variance"] = g_lr_item.pow(2).clone()
                    else:
                        if st["momentum"].device != g_lr_item.device:
                            st["momentum"] = st["momentum"].to(g_lr_item.device)
                            st["variance"] = st["variance"].to(g_lr_item.device)
                        st["momentum"].mul_(expert_beta1).add_(g_lr_item, alpha=1.0 - expert_beta1)
                        st["variance"].mul_(expert_beta2).addcmul_(g_lr_item, g_lr_item, value=1.0 - expert_beta2)

                batch_mom = torch.stack([g["state"]["momentum"] for g in bucket_groups], dim=0)
                batch_var = torch.stack([g["state"]["variance"] for g in bucket_groups], dim=0)
                step_vals = torch.tensor(
                    [g["state"]["step"] for g in bucket_groups], device=device, dtype=torch.float32
                ).view(B, 1, 1)
                bias_c1 = (1.0 - (expert_beta1**step_vals)).to(dtype=dtype)
                bias_c2 = (1.0 - (expert_beta2**step_vals)).clamp_min(1e-7).to(dtype=dtype)
                batch_m_hat = batch_mom / bias_c1
                batch_v_hat = batch_var / bias_c2
                batch_delta_lr = (batch_m_hat / (batch_v_hat.sqrt() + 1e-8)).to(dtype=batch_recons.dtype)

                if side == "left":
                    batch_dfull = torch.bmm(batch_recons, batch_delta_lr)
                else:
                    batch_dfull = torch.bmm(batch_delta_lr, batch_recons.transpose(1, 2))

                for idx, g in enumerate(bucket_groups):
                    p = g["param"]
                    dfull = batch_dfull[idx].to(dtype=p.dtype)
                    if expert_wd != 0:
                        p.data.mul_(1.0 - expert_lr * expert_wd)
                    p.data.sub_(dfull.reshape(p.shape), alpha=expert_lr)
            else:
                g = bucket_groups[0]
                p = g["param"]
                grad = p.grad.data
                st = g["state"]
                proj_unit = g["proj_unit"]
                proj_recon = g["proj_recon"]

                if side == "left":
                    g_lr = proj_unit.T @ grad
                else:
                    g_lr = grad @ proj_unit

                st["step"] += 1
                if st["momentum"] is None:
                    st["momentum"] = g_lr.clone()
                    st["variance"] = g_lr.pow(2).clone()
                else:
                    if st["momentum"].device != g_lr.device:
                        st["momentum"] = st["momentum"].to(g_lr.device)
                        st["variance"] = st["variance"].to(g_lr.device)
                    st["momentum"].mul_(expert_beta1).add_(g_lr, alpha=1.0 - expert_beta1)
                    st["variance"].mul_(expert_beta2).addcmul_(g_lr, g_lr, value=1.0 - expert_beta2)

                step = st["step"]
                m_hat = st["momentum"] / (1.0 - expert_beta1**step)
                v_hat = st["variance"] / (1.0 - expert_beta2**step)
                delta_lr = (m_hat / (v_hat.sqrt() + 1e-8)).to(dtype=proj_recon.dtype)

                if side == "left":
                    delta_full = proj_recon @ delta_lr
                else:
                    delta_full = delta_lr @ proj_recon.T

                delta_full = delta_full.to(dtype=p.dtype)
                if expert_wd != 0:
                    p.data.mul_(1.0 - expert_lr * expert_wd)
                p.data.sub_(delta_full.reshape(p.shape), alpha=expert_lr)

        if is_global_step_end:
            self.step_count += 1

    @torch.no_grad()
    def _step_triangular_direct_sum_v3(self, active_groups: list[dict[str, Any]], is_global_step_end: bool = False):
        """CentroidSteer v3: Exact Orthogonal Direct-Sum G = G_parallel (+) G_perp with Grassmann Parallel Transport."""
        N = len(active_groups)
        device = active_groups[0]["param"].device
        dtype = active_groups[0]["param"].dtype
        expert_lr = self.expert_lr
        expert_beta1, expert_beta2 = self.expert_betas
        expert_wd = self.expert_wd
        n_steer_dirs = int(getattr(self, "num_steer_dirs", 4))

        if not hasattr(self, "_triangular_buffers"):
            self._triangular_buffers = {}
        buf_key = ("v3", N, device, dtype)
        if buf_key not in self._triangular_buffers:
            self._triangular_buffers[buf_key] = {
                "batch_mom": torch.zeros(N, self.rank, self.rank, device=device, dtype=torch.float32),
                "batch_var": torch.zeros(N, self.rank, self.rank, device=device, dtype=torch.float32),
                "R_batch": torch.zeros(N, self.rank, self.rank, device=device, dtype=dtype),
                "batch_delta_R": torch.zeros(N, self.rank, self.rank, device=device, dtype=dtype),
                "step_vals": torch.zeros(N, 1, 1, device=device, dtype=torch.float32),
            }
        buf = self._triangular_buffers[buf_key]
        batch_mom = buf["batch_mom"]
        batch_var = buf["batch_var"]
        R_batch = buf["R_batch"]
        batch_delta_R = buf["batch_delta_R"]
        step_vals = buf["step_vals"]
        R_batch.zero_()
        batch_delta_R.zero_()

        if not hasattr(self, "_bucket_buffers"):
            self._bucket_buffers = {}

        raw_buckets: dict[tuple[int, int, int, str], list[dict[str, Any]]] = {}
        for g in active_groups:
            m, n = g["param"].shape
            r = min(self.rank, min(m, n))
            s_role = g.get("steer_role", "input")
            key = (m, n, r, s_role)
            if key not in raw_buckets:
                raw_buckets[key] = []
            raw_buckets[key].append(g)

        MAX_BUCKET_ELEMS = 32_000_000
        shape_buckets = []
        cursor = 0
        for key, full_g_list in raw_buckets.items():
            m_k, n_k, _, _ = key
            chunk_k = max(1, MAX_BUCKET_ELEMS // max(1, m_k * n_k))
            for chunk_idx, offset in enumerate(range(0, len(full_g_list), chunk_k)):
                g_chunk = full_g_list[offset : offset + chunk_k]
                K = len(g_chunk)
                shape_buckets.append((key, chunk_idx, cursor, cursor + K, g_chunk))
                cursor += K

        def _batched_qr(Y_mat: torch.Tensor, target_dtype: torch.dtype = dtype) -> torch.Tensor:
            Q, diag = batched_qr(
                Y_mat,
                cond_threshold=1e4,
                eps_shift=1e-7,
                orth_tol=1e-4,
                return_diagnostics=True,
                out_dtype=target_dtype,
            )
            self._record_qr_diagnostics(diag)
            return Q

        bucket_cache = []
        step_list = [0.0] * N

        for (m, n, r, steer_role), chunk_idx, start, end, g_list in shape_buckets:
            K = end - start
            d_out, d_in = m, n
            d_small = min(m, n)
            d_large = max(m, n)
            side = "left" if m <= n else "right"
            centroid_on_ql = m > n
            k_c = min(n_steer_dirs, d_in)

            bkey = ("v3", chunk_idx, K, m, n, r, steer_role, device, dtype)
            if bkey not in self._bucket_buffers:
                self._bucket_buffers[bkey] = {
                    "QL_stack": torch.zeros(K, d_small, self.rank + 1, device=device, dtype=dtype),
                    "QR_stack": torch.zeros(K, d_large, self.rank + 1, device=device, dtype=dtype),
                }
            bbuf = self._bucket_buffers[bkey]
            QL_stack = bbuf["QL_stack"]
            QR_stack = bbuf["QR_stack"]

            G_raw = torch.stack([g["param"].grad.data for g in g_list], dim=0)  # [K, d_out, d_in]
            G_f32 = G_raw.float()

            # 1. Subspace C_e Refresh & Grassmann/Stiefel Parallel Transport (C_old -> C_new)
            active_s_idx: list[int] = []
            if self.steer_scale > 0:
                for b_i, g in enumerate(g_list):
                    mod = g.get("module")
                    e_idx = g.get("expert_idx")
                    if mod is None or e_idx is None:
                        g["steer_applied"] = False
                        continue
                    st = g["state"]
                    needs_c_ref = (
                        g.get("C_e") is None
                        or g.get("last_steer_step") is None
                        or (self.step_count - g.get("last_steer_step", 0)) >= self.steer_update_gap
                    )
                    if needs_c_ref:
                        c_new: torch.Tensor | None = None
                        if self.steer_mode == "upgraded":
                            if hasattr(mod, "get_expert_steering_directions"):
                                _, sub_mat = mod.get_expert_steering_directions(
                                    e_idx,
                                    steer_role=steer_role,
                                    num_dirs=k_c,
                                    steer_mode="upgraded",
                                    device=device,
                                    dtype=torch.float32,
                                )
                            else:
                                sub_mat = None
                            if sub_mat is not None and sub_mat.ndim == 2 and float(sub_mat.norm().item()) > 1e-8:
                                if sub_mat.size(1) >= k_c:
                                    c_cand = sub_mat[:, :k_c].float()
                                else:
                                    k_eff = sub_mat.size(1)
                                    eye_pad = torch.eye(d_in, k_c, device=device, dtype=torch.float32)
                                    c_cand = torch.cat([sub_mat.float(), eye_pad[:, : k_c - k_eff]], dim=1)
                                gram_err = float(
                                    (torch.mm(c_cand.t(), c_cand) - torch.eye(k_c, device=device, dtype=torch.float32))
                                    .norm()
                                    .item()
                                )
                                if gram_err > 1e-5:
                                    c_new = _batched_qr(c_cand.unsqueeze(0), target_dtype=torch.float32).squeeze(0)
                                else:
                                    c_new = c_cand
                        else:
                            # Memory-matched random subspace control ("random_same_mem")
                            r_raw = torch.randn(d_in, k_c, device=device, dtype=torch.float32)
                            q_rand, r_rand = torch.linalg.qr(r_raw, mode="reduced")
                            r_diag = torch.where(r_rand.diagonal() < 0.0, -1.0, 1.0).unsqueeze(0)
                            c_new = q_rand * r_diag

                        if c_new is not None:
                            c_old = g.get("C_e")
                            if c_old is not None and c_old.shape == c_new.shape:
                                c_old_f = c_old.to(device=device, dtype=torch.float32)
                                dots = (c_old_f * c_new).sum(dim=0, keepdim=True)
                                c_new = c_new * torch.where(dots < 0.0, -1.0, 1.0)
                                if st.get("M_c") is not None and st.get("V_c") is not None:
                                    T = torch.mm(c_old_f.t(), c_new)
                                    t_sq = T * T
                                    col_energy = t_sq.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
                                    m_old = st["M_c"].to(device=device, dtype=torch.float32)
                                    v_old = st["V_c"].to(device=device, dtype=torch.float32)
                                    v_mean = v_old.mean(dim=1, keepdim=True)
                                    st["M_c"] = torch.mm(m_old, T)
                                    st["V_c"] = (torch.mm(v_old, t_sq) + v_mean * (1.0 - col_energy)).clamp_min_(0.0)
                            else:
                                st["M_c"] = torch.zeros(d_out, k_c, device=device, dtype=torch.float32)
                                st["V_c"] = torch.zeros(d_out, k_c, device=device, dtype=torch.float32)
                                st["transport_step"] = 0
                            st["steer_step"] = 0
                            st.setdefault("transport_step", 0)
                            g["C_e"] = c_new
                            g["last_steer_step"] = self.step_count
                            g["steer_applied"] = True
                        else:
                            g["steer_applied"] = g.get("C_e") is not None
                    else:
                        g["steer_applied"] = g.get("C_e") is not None

                    if g["steer_applied"]:
                        active_s_idx.append(b_i)
            else:
                for g in g_list:
                    g["steer_applied"] = False

            # 2. Exact Orthogonal Direct-Sum Decomposition: G = G_parallel (+) G_perp
            C_act: torch.Tensor | None = None
            U_c_s_cache: torch.Tensor | None = None
            if active_s_idx:
                C_act = torch.stack(
                    [g_list[b_i]["C_e"].to(device=device, dtype=torch.float32) for b_i in active_s_idx], dim=0
                )
                G_act_f32 = G_f32[active_s_idx]
                G_c_act = torch.bmm(G_act_f32, C_act)  # [K_s, d_out, k_c]
                G_par_act = torch.bmm(G_c_act, C_act.transpose(1, 2))  # [K_s, d_out, d_in]
                G_perp_f32 = G_f32.clone()
                G_perp_f32[active_s_idx] = G_act_f32 - G_par_act

                # 3. Direct-Sum Preconditioned FP32 AdamW Update (continuous transport_step + refresh-relative steer_step)
                for b_i in active_s_idx:
                    g_list[b_i]["state"]["steer_step"] = g_list[b_i]["state"].get("steer_step", 0) + 1
                    g_list[b_i]["state"]["transport_step"] = g_list[b_i]["state"].get("transport_step", 0) + 1
                M_c_s = torch.stack(
                    [g_list[b_i]["state"]["M_c"].to(device=device, dtype=torch.float32) for b_i in active_s_idx], dim=0
                )
                V_c_s = torch.stack(
                    [g_list[b_i]["state"]["V_c"].to(device=device, dtype=torch.float32) for b_i in active_s_idx], dim=0
                )
                s_steps = torch.tensor(
                    [g_list[b_i]["state"]["transport_step"] for b_i in active_s_idx],
                    device=device,
                    dtype=torch.float32,
                ).view(-1, 1, 1)
                M_c_s.lerp_(G_c_act, 1.0 - expert_beta1)
                V_c_s.mul_(expert_beta2).addcmul_(G_c_act, G_c_act, value=1.0 - expert_beta2)
                bc1_s = 1.0 - (expert_beta1**s_steps)
                bc2_s = (1.0 - (expert_beta2**s_steps)).clamp_min(1e-7)
                U_c_s = (M_c_s / bc1_s) / ((V_c_s / bc2_s).sqrt() + 1e-8)
                if getattr(self, "use_rms_steer", False):
                    target_rms = math.sqrt(float(d_out * k_c))
                    U_c_s = U_c_s * (target_rms / U_c_s.norm(dim=(1, 2), keepdim=True).clamp_min(1e-8))
                for k_pos, b_i in enumerate(active_s_idx):
                    st = g_list[b_i]["state"]
                    if st["M_c"].device == device and st["M_c"].dtype == torch.float32:
                        st["M_c"].copy_(M_c_s[k_pos])
                        st["V_c"].copy_(V_c_s[k_pos])
                    else:
                        st["M_c"] = M_c_s[k_pos].clone()
                        st["V_c"] = V_c_s[k_pos].clone()
                U_c_s_cache = U_c_s  # [K_s, d_out, k_c]
                del G_act_f32, G_c_act, G_par_act, M_c_s, V_c_s
            else:
                G_perp_f32 = G_f32

            G_stack = (G_perp_f32 if m <= n else G_perp_f32.transpose(1, 2)).to(dtype=dtype)

            # 4. Orthogonal Complement Rank-r Krylov Core (Q_L, Q_R) on G_perp
            init_flags = [g.get("Q_L") is None for g in g_list]
            upd_flags = []
            for b_idx, g in enumerate(g_list):
                if init_flags[b_idx]:
                    upd_flags.append(False)
                    continue
                steps_since_proj = self.step_count - g.get("proj_step", 0)
                time_refresh = steps_since_proj >= self.update_gap
                energy_refresh = (
                    self.adaptive_subspace_refresh
                    and steps_since_proj >= self.min_subspace_gap
                    and float(g.get("last_energy_ratio", 1.0)) < self.energy_threshold
                )
                upd_flags.append(time_refresh or energy_refresh)
            any_refresh = any(init_flags) or any(upd_flags)

            if any_refresh:
                refresh_indices = [b_idx for b_idx in range(K) if init_flags[b_idx] or upd_flags[b_idx]]
                G_ref_all = G_stack[refresh_indices].float()
                g_norms = torch.linalg.norm(G_ref_all, dim=(1, 2))
                nz_flags = (g_norms > 1e-12).tolist()
                qr_sub_indices = [i for i, nz in enumerate(nz_flags) if nz]
                qr_bucket_indices = [refresh_indices[i] for i in qr_sub_indices]

                qr_map: dict[int, int] = {}
                ql_out: torch.Tensor | None = None
                qr_out: torch.Tensor | None = None

                if qr_bucket_indices:
                    G_qr = G_ref_all[qr_sub_indices]
                    has_preset_proj = all(
                        (
                            init_flags[b_idx]
                            and g_list[b_idx].get("projection") is not None
                            and tuple(g_list[b_idx]["projection"].shape) == (d_small, r)
                        )
                        for b_idx in qr_bucket_indices
                    )
                    if has_preset_proj:
                        ql_f32 = torch.stack(
                            [
                                g_list[b_idx]["projection"].to(device=device, dtype=torch.float32)
                                for b_idx in qr_bucket_indices
                            ],
                            dim=0,
                        )
                        Z = torch.bmm(G_qr.transpose(1, 2), ql_f32)
                        qr_out = _batched_qr(Z, target_dtype=dtype)
                        ql_out = ql_f32.to(dtype=dtype)
                    else:
                        K_qr = len(qr_bucket_indices)
                        Omega = torch.randn(K_qr, d_large, r, device=device, dtype=torch.float32)
                        C_qr: torch.Tensor | None = None
                        if active_s_idx and d_in >= r + k_c:
                            if all(g_list[b_i].get("C_e") is not None for b_i in qr_bucket_indices):
                                C_qr = torch.stack(
                                    [
                                        g_list[b_i]["C_e"].to(device=device, dtype=torch.float32)
                                        for b_i in qr_bucket_indices
                                    ],
                                    dim=0,
                                )

                        Y = torch.bmm(G_qr, Omega)
                        if C_qr is not None and centroid_on_ql:
                            Y_d = Y - torch.bmm(C_qr, torch.bmm(C_qr.transpose(1, 2), Y))
                            Y = torch.where(torch.linalg.norm(Y_d, dim=1, keepdim=True) > 1e-9, Y_d, Y)
                        ql_f32 = _batched_qr(Y, target_dtype=torch.float32)
                        Z = torch.bmm(G_qr.transpose(1, 2), ql_f32)
                        if C_qr is not None and not centroid_on_ql:
                            Z_d = Z - torch.bmm(C_qr, torch.bmm(C_qr.transpose(1, 2), Z))
                            Z = torch.where(torch.linalg.norm(Z_d, dim=1, keepdim=True) > 1e-9, Z_d, Z)
                        qr_f32 = _batched_qr(Z, target_dtype=torch.float32)
                        Y = torch.bmm(G_qr, qr_f32)
                        if C_qr is not None and centroid_on_ql:
                            Y_d = Y - torch.bmm(C_qr, torch.bmm(C_qr.transpose(1, 2), Y))
                            Y = torch.where(torch.linalg.norm(Y_d, dim=1, keepdim=True) > 1e-9, Y_d, Y)
                        ql_out = _batched_qr(Y, target_dtype=dtype)
                        qr_out = qr_f32.to(dtype=dtype)
                    qr_map = {b_idx: i for i, b_idx in enumerate(qr_bucket_indices)}

                aligned_proj_step = (self.step_count // max(1, self.update_gap)) * self.update_gap
                for b_idx, g in enumerate(g_list):
                    idx = start + b_idx
                    st = g["state"]
                    if init_flags[b_idx] or upd_flags[b_idx]:
                        if b_idx in qr_map and ql_out is not None and qr_out is not None:
                            q_i = qr_map[b_idx]
                            QL_stack[b_idx].zero_()
                            QL_stack[b_idx, :, :r].copy_(ql_out[q_i])
                            QR_stack[b_idx].zero_()
                            QR_stack[b_idx, :, :r].copy_(qr_out[q_i])
                            g["_has_nonzero_grad"] = True
                        elif init_flags[b_idx]:
                            QL_stack[b_idx].zero_()
                            QL_stack[b_idx, :r, :r].copy_(torch.eye(r, device=device, dtype=dtype))
                            QR_stack[b_idx].zero_()
                            QR_stack[b_idx, :r, :r].copy_(torch.eye(r, device=device, dtype=dtype))
                            g["_has_nonzero_grad"] = False
                        else:
                            if g["Q_L"].data_ptr() != QL_stack[b_idx].data_ptr():
                                QL_stack[b_idx].copy_(g["Q_L"])
                                QR_stack[b_idx].copy_(g["Q_R"])
                            g["_has_nonzero_grad"] = False
                        g["Q_L"] = QL_stack[b_idx]
                        g["Q_R"] = QR_stack[b_idx]
                        g["projection"] = QL_stack[b_idx, :, :r]
                        g["projection_side"] = side
                        g["proj_step"] = aligned_proj_step if not self.adaptive_subspace_refresh else self.step_count
                        batch_mom[idx].zero_()
                        batch_var[idx].zero_()
                        st["momentum"] = batch_mom[idx]
                        st["variance"] = batch_var[idx]
                        st["subspace_step"] = 0
                        if init_flags[b_idx]:
                            st["step"] = 0
                            st.setdefault("steer_step", 0)
                    else:
                        if g["Q_L"].data_ptr() != QL_stack[b_idx].data_ptr():
                            QL_stack[b_idx].copy_(g["Q_L"])
                            QR_stack[b_idx].copy_(g["Q_R"])
                            g["Q_L"] = QL_stack[b_idx]
                            g["Q_R"] = QR_stack[b_idx]
                            g["projection"] = QL_stack[b_idx, :, :r]
                        if st["momentum"].data_ptr() != batch_mom[idx].data_ptr():
                            batch_mom[idx].copy_(st["momentum"])
                            batch_var[idx].copy_(st["variance"])
                            st["momentum"] = batch_mom[idx]
                            st["variance"] = batch_var[idx]
            else:
                for b_idx, g in enumerate(g_list):
                    idx = start + b_idx
                    st = g["state"]
                    if g["Q_L"].data_ptr() != QL_stack[b_idx].data_ptr():
                        QL_stack[b_idx].copy_(g["Q_L"])
                        QR_stack[b_idx].copy_(g["Q_R"])
                        g["Q_L"] = QL_stack[b_idx]
                        g["Q_R"] = QR_stack[b_idx]
                        g["projection"] = QL_stack[b_idx, :, :r]
                    if st["momentum"].data_ptr() != batch_mom[idx].data_ptr():
                        batch_mom[idx].copy_(st["momentum"])
                        batch_var[idx].copy_(st["variance"])
                        st["momentum"] = batch_mom[idx]
                        st["variance"] = batch_var[idx]

            if active_s_idx:
                for b_i in active_s_idx:
                    c_0 = g_list[b_i]["C_e"][:, 0].float()
                    Q_in = (QL_stack[b_i, :, :r] if centroid_on_ql else QR_stack[b_i, :, :r]).float()
                    Q_out = (QR_stack[b_i, :, :r] if centroid_on_ql else QL_stack[b_i, :, :r]).float()
                    u_in = c_0 - torch.mv(Q_in, torch.mv(Q_in.t(), c_0))
                    u_in = u_in - torch.mv(Q_in, torch.mv(Q_in.t(), u_in))
                    u_in_norm = float(u_in.norm().item())
                    if u_in_norm <= 1e-6:
                        for col_try in range(min(r + 2, Q_in.size(0))):
                            e_try = torch.zeros_like(c_0)
                            e_try[col_try] = 1.0
                            u_in = e_try - torch.mv(Q_in, torch.mv(Q_in.t(), e_try))
                            u_in = u_in - torch.mv(Q_in, torch.mv(Q_in.t(), u_in))
                            u_in_norm = float(u_in.norm().item())
                            if u_in_norm > 1e-6:
                                break
                    q_in_f = u_in / max(u_in_norm, 1e-12)
                    v_0 = torch.mv(G_f32[b_i], q_in_f)
                    u_out = v_0 - torch.mv(Q_out, torch.mv(Q_out.t(), v_0))
                    u_out = u_out - torch.mv(Q_out, torch.mv(Q_out.t(), u_out))
                    u_out_norm = float(u_out.norm().item())
                    if u_out_norm <= 1e-6:
                        for col_try in range(min(r + 2, Q_out.size(0))):
                            e_try = torch.zeros(Q_out.size(0), device=device, dtype=torch.float32)
                            e_try[col_try] = 1.0
                            u_out = e_try - torch.mv(Q_out, torch.mv(Q_out.t(), e_try))
                            u_out = u_out - torch.mv(Q_out, torch.mv(Q_out.t(), u_out))
                            u_out_norm = float(u_out.norm().item())
                            if u_out_norm > 1e-6:
                                break
                    q_out_f = u_out / max(u_out_norm, 1e-12)
                    if centroid_on_ql:
                        QL_stack[b_i, :, r].copy_(q_in_f.to(dtype=dtype))
                        QR_stack[b_i, :, r].copy_(q_out_f.to(dtype=dtype))
                    else:
                        QR_stack[b_i, :, r].copy_(q_in_f.to(dtype=dtype))
                        QL_stack[b_i, :, r].copy_(q_out_f.to(dtype=dtype))

            QL_act = QL_stack[:, :, :r]
            QR_act = QR_stack[:, :, :r]
            S_bmm = torch.bmm(QL_act.transpose(1, 2), torch.bmm(G_stack, QR_act))

            if self.adaptive_subspace_refresh:
                s_energies = (S_bmm.float() ** 2).sum(dim=(-2, -1))
                g_energies = (G_stack.float() ** 2).sum(dim=(-2, -1)).clamp_min(1e-12)
                ratios = s_energies / g_energies
                for b_idx, g in enumerate(g_list):
                    g["last_energy_ratio"] = ratios[b_idx]

            R_batch[start:end, :r, :r].copy_(torch.triu(S_bmm) if self.core_mask_mode == "triu" else S_bmm)

            for b_idx, g in enumerate(g_list):
                idx = start + b_idx
                sub_step = g["state"].get("subspace_step", 0) + 1
                g["state"]["subspace_step"] = sub_step
                g["state"]["step"] += 1
                step_list[idx] = float(sub_step)

            bucket_cache.append(
                (m, n, r, start, end, g_list, QL_act, QR_act, QL_stack, QR_stack, active_s_idx, C_act, U_c_s_cache)
            )
            del G_raw, G_f32, G_perp_f32, G_stack, S_bmm

        first_sub = step_list[0]
        if all(s == first_sub for s in step_list):
            step_vals.fill_(first_sub)
        else:
            step_vals.copy_(torch.tensor(step_list, device=device, dtype=torch.float32).view(N, 1, 1))

        from lucid_machines.kernels.fast_triangular import fast_triangular_adamw

        if all(b[2] == self.rank for b in bucket_cache):
            fast_triangular_adamw(
                R_batch,
                batch_mom,
                batch_var,
                batch_delta_R,
                step_vals,
                beta1=expert_beta1,
                beta2=expert_beta2,
                eps=1e-8,
                mask_mode=self.core_mask_mode,
                steer_step_vals=None,
                rms_reg=0.0,
            )
        else:
            for _, _, r_b, b_start, b_end, *_ in bucket_cache:
                fast_triangular_adamw(
                    R_batch[b_start:b_end, :r_b, :r_b],
                    batch_mom[b_start:b_end, :r_b, :r_b],
                    batch_var[b_start:b_end, :r_b, :r_b],
                    batch_delta_R[b_start:b_end, :r_b, :r_b],
                    step_vals[b_start:b_end],
                    beta1=expert_beta1,
                    beta2=expert_beta2,
                    eps=1e-8,
                    mask_mode=self.core_mask_mode,
                    steer_step_vals=None,
                    rms_reg=0.0,
                )

        # 5. Reconstruct Orthogonal Direct-Sum Step: Delta_W = Delta_W_perp + steer_scale * Delta_W_parallel
        beta_wd = 1.0 - expert_lr * expert_wd if expert_wd != 0 else 1.0
        alpha_upd = -expert_lr
        for (
            m,
            n,
            r,
            start,
            end,
            g_list,
            QL_act,
            QR_act,
            QL_stack,
            QR_stack,
            active_s_idx,
            C_act,
            U_c_s_cache,
        ) in bucket_cache:
            delta_stack_f32 = batch_delta_R[start:end, :r, :r].float()
            if m <= n:
                Delta_W_perp = torch.bmm(torch.bmm(QL_act.float(), delta_stack_f32), QR_act.float().transpose(1, 2))
            else:
                Delta_W_perp = torch.bmm(
                    torch.bmm(QR_act.float(), delta_stack_f32.transpose(1, 2)), QL_act.float().transpose(1, 2)
                )

            if active_s_idx and C_act is not None and U_c_s_cache is not None:
                dw_perp_c = torch.bmm(Delta_W_perp[active_s_idx], C_act)
                dW_par_act = torch.bmm(U_c_s_cache, C_act.transpose(1, 2))
                Delta_W_perp[active_s_idx] = (
                    Delta_W_perp[active_s_idx]
                    - torch.bmm(dw_perp_c, C_act.transpose(1, 2))
                    + dW_par_act * float(self.steer_scale)
                )
                del dw_perp_c, dW_par_act

            if dtype == torch.bfloat16:
                # Exact unbiased BF16 stochastic rounding prevents ULP quantization loss on subspace updates
                W_stack = torch.stack([g["param"].data for g in g_list], dim=0).float()
                if expert_wd != 0:
                    W_stack.mul_(beta_wd)
                W_stack.add_(Delta_W_perp, alpha=alpha_upd)
                rand_bits = torch.randint(0, 65536, W_stack.shape, device=device, dtype=torch.int32)
                w_sr = ((W_stack.view(torch.int32) + rand_bits) & -65536).view(torch.float32).to(dtype=torch.bfloat16)
                for b_idx, g in enumerate(g_list):
                    g["param"].data.copy_(w_sr[b_idx])
                del W_stack, rand_bits, w_sr
            else:
                delta_W = Delta_W_perp.to(dtype=dtype)
                params_b = [g["param"].data for g in g_list]
                if expert_wd != 0:
                    torch._foreach_mul_(params_b, beta_wd)
                torch._foreach_add_(params_b, list(delta_W.unbind(0)), alpha=alpha_upd)
                del delta_W

            del delta_stack_f32, Delta_W_perp

            if N < len(self.layer_groups):
                for b_idx, g in enumerate(g_list):
                    idx = start + b_idx
                    g["Q_L"] = QL_stack[b_idx].clone()
                    g["Q_R"] = QR_stack[b_idx].clone()
                    g["projection"] = g["Q_L"][:, :r]
                    g["state"]["momentum"] = batch_mom[idx].clone()
                    g["state"]["variance"] = batch_var[idx].clone()

        if is_global_step_end:
            self.step_count += 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if isinstance(self.model, torch.nn.Module):
            params_to_step = self.model.parameters()
        else:
            params_to_step = self.muon_params + [g["param"] for g in self.layer_groups] + self.non_expert_params
        self.step_parameters(params_to_step, is_global_step_end=True)
        return loss

    def state_dict(self):
        sd = {
            "base_optimizer": self.base_optimizer.state_dict() if self.base_optimizer is not None else None,
            "muon_optimizer": self.muon_optimizer.state_dict() if self.muon_optimizer is not None else None,
            "step_count": self.step_count,
            "expert_lr": float(self.expert_lr),
            "steer_scale": float(self.steer_scale),
            "steer_mode": getattr(self, "steer_mode", "upgraded"),
            "num_steer_dirs": int(getattr(self, "num_steer_dirs", 4)),
            "steer_update_gap": int(getattr(self, "steer_update_gap", 2)),
            "use_rms_steer": bool(getattr(self, "use_rms_steer", False)),
            "param_groups": [{k: v for k, v in pg.items() if k != "_optimizer"} for pg in self.param_groups],
            "_last_synced_lr": getattr(self, "_last_synced_lr", self.initial_base_lr),
            "layer_groups": [
                {
                    "projection": g["projection"],
                    "projection_side": g.get("projection_side", "left"),
                    "proj_step": g["proj_step"],
                    "Q_L": g.get("Q_L"),
                    "Q_R": g.get("Q_R"),
                    "C_e": g.get("C_e").clone() if isinstance(g.get("C_e"), torch.Tensor) else None,
                    "proj_basis": g.get("proj_basis"),
                    "last_c_orth": g.get("last_c_orth"),
                    "last_steer_step": g.get("last_steer_step"),
                    "steer_applied": g.get("steer_applied", False),
                    "last_energy_ratio": g.get("last_energy_ratio"),
                    "state": {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in g["state"].items()},
                }
                for g in self.layer_groups
            ],
        }
        return sd

    def load_state_dict(self, state_dict):
        assert len(state_dict["layer_groups"]) == len(self.layer_groups), (
            f"Layer group count mismatch: saved {len(state_dict['layer_groups'])}, current {len(self.layer_groups)}"
        )
        for saved, cur in zip(state_dict["layer_groups"], self.layer_groups):
            if saved["projection"] is not None:
                rows, cols = cur["param"].shape
                side = saved.get("projection_side", "left")
                if side == "left":
                    expected_shape = (rows, min(self.rank, rows, cols))
                else:
                    expected_shape = (cols, min(self.rank, rows, cols))
                assert tuple(saved["projection"].shape) == expected_shape, (
                    f"Projection shape mismatch: saved {saved['projection'].shape}, expected {expected_shape}"
                )
        if self.base_optimizer is not None and state_dict.get("base_optimizer") is not None:
            self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        if self.muon_optimizer is not None and state_dict.get("muon_optimizer") is not None:
            self.muon_optimizer.load_state_dict(state_dict["muon_optimizer"])
        self.step_count = state_dict.get("step_count", 0)
        if "expert_lr" in state_dict:
            self.expert_lr = state_dict["expert_lr"]
        if "steer_scale" in state_dict:
            self.steer_scale = state_dict["steer_scale"]
        if "steer_mode" in state_dict:
            self.steer_mode = state_dict["steer_mode"]
        if "num_steer_dirs" in state_dict:
            self.num_steer_dirs = int(state_dict["num_steer_dirs"])
        if "steer_update_gap" in state_dict:
            self.steer_update_gap = int(state_dict["steer_update_gap"])
        if "use_rms_steer" in state_dict:
            self.use_rms_steer = bool(state_dict["use_rms_steer"])
        if "_last_synced_lr" in state_dict:
            self._last_synced_lr = state_dict["_last_synced_lr"]
        if "param_groups" in state_dict and len(state_dict["param_groups"]) == len(self.param_groups):
            for cur_pg, saved_pg in zip(self.param_groups, state_dict["param_groups"]):
                for k, v in saved_pg.items():
                    if k != "params":
                        cur_pg[k] = v

        for g, sd in zip(self.layer_groups, state_dict["layer_groups"]):
            target_device = g["param"].device
            target_dtype = g["param"].dtype
            g["projection"] = (
                sd["projection"].to(device=target_device, dtype=target_dtype)
                if sd.get("projection") is not None
                else None
            )
            g["projection_side"] = sd.get("projection_side", "left")
            g["proj_step"] = sd.get("proj_step", 0)
            g["last_steer_step"] = sd.get("last_steer_step")
            g["steer_applied"] = sd.get("steer_applied", False)
            g["last_energy_ratio"] = sd.get("last_energy_ratio")

            # Restore complete isometric bases Q_L, Q_R, and direct-sum subspace C_e
            if sd.get("Q_L") is not None:
                g["Q_L"] = sd["Q_L"].to(device=target_device, dtype=target_dtype)
            else:
                g["Q_L"] = None

            if sd.get("Q_R") is not None:
                g["Q_R"] = sd["Q_R"].to(device=target_device, dtype=target_dtype)
            else:
                g["Q_R"] = None

            if sd.get("C_e") is not None:
                g["C_e"] = sd["C_e"].to(device=target_device, dtype=torch.float32)
            else:
                g["C_e"] = None

            if sd.get("proj_basis") is not None:
                g["proj_basis"] = sd["proj_basis"].to(device=target_device, dtype=target_dtype)
            else:
                g["proj_basis"] = None

            if sd.get("last_c_orth") is not None:
                g["last_c_orth"] = sd["last_c_orth"].to(device=target_device, dtype=target_dtype)
            else:
                g["last_c_orth"] = None

            g["_basis_dirty"] = True

            restored_state = {}
            for k, v in sd.get("state", {}).items():
                if isinstance(v, torch.Tensor):
                    restored_state[k] = v.to(
                        device=target_device,
                        dtype=torch.float32 if k in ("momentum", "variance", "M_c", "V_c") else v.dtype,
                    )
                else:
                    restored_state[k] = v
            g["state"].update(restored_state)

    def _record_qr_diagnostics(self, diag: dict[str, torch.Tensor]) -> None:
        """Accumulates telemetry from a batched_qr call into a GPU-resident tensor without host-device synchronization during training."""
        fb = diag["fallback_mask"]
        K = fb.numel()
        self.qr_telemetry["total_qr_calls"] += K
        stacked = torch.stack(
            [
                fb,
                diag["primary_degen_mask"],
                diag["primary_cholesky_mask"],
                diag["primary_cond_mask"],
                diag["primary_orth_mask"],
                diag["degen_mask"],
                diag["cholesky_fail_mask"],
                diag["cond_fail_mask"],
                diag["orth_fail_mask"],
            ],
            dim=0,
        ).sum(dim=-1, dtype=torch.long)
        if self._qr_telemetry_gpu is None or self._qr_telemetry_gpu.device != stacked.device:
            if self._qr_telemetry_gpu is not None:
                self._flush_qr_telemetry_gpu()
            self._qr_telemetry_gpu = stacked
        else:
            self._qr_telemetry_gpu.add_(stacked)

    def _flush_qr_telemetry_gpu(self) -> None:
        """Materializes accumulated GPU telemetry counters to CPU dictionary via .tolist() at the reporting boundary."""
        if self._qr_telemetry_gpu is None:
            return
        vals = self._qr_telemetry_gpu.tolist()
        self._qr_telemetry_gpu = None
        fallbacks, p_deg, p_chol, p_cond, p_orth, o_deg, o_chol, o_cond, o_orth = [int(v) for v in vals]
        self.qr_telemetry["householder_fallbacks"] += fallbacks
        self.qr_telemetry["cholesky_fast_path_survivals"] = (
            self.qr_telemetry["total_qr_calls"] - self.qr_telemetry["householder_fallbacks"]
        )
        self.qr_telemetry["primary_fallback_reasons"]["degenerate_columns"] += p_deg
        self.qr_telemetry["primary_fallback_reasons"]["cholesky_non_positive_definite"] += p_chol
        self.qr_telemetry["primary_fallback_reasons"]["condition_ratio_exceeded"] += p_cond
        self.qr_telemetry["primary_fallback_reasons"]["orthogonality_tolerance_exceeded"] += p_orth

        self.qr_telemetry["overlapping_triggers"]["degenerate_columns"] += o_deg
        self.qr_telemetry["overlapping_triggers"]["cholesky_non_positive_definite"] += o_chol
        self.qr_telemetry["overlapping_triggers"]["condition_ratio_exceeded"] += o_cond
        self.qr_telemetry["overlapping_triggers"]["orthogonality_tolerance_exceeded"] += o_orth

    def get_qr_telemetry(self) -> dict[str, Any]:
        """Returns live telemetry of QR fast-path survivals vs Householder fallbacks."""
        self._flush_qr_telemetry_gpu()
        total = self.qr_telemetry["total_qr_calls"]
        fallbacks = self.qr_telemetry["householder_fallbacks"]
        survivals = total - fallbacks
        self.qr_telemetry["cholesky_fast_path_survivals"] = survivals
        return {
            "total_qr_calls": total,
            "cholesky_fast_path_survivals": survivals,
            "cholesky_survival_rate_pct": round(100.0 * survivals / max(1, total), 2),
            "householder_fallbacks": fallbacks,
            "householder_fallback_rate_pct": round(100.0 * fallbacks / max(1, total), 2),
            "primary_fallback_reasons": dict(self.qr_telemetry["primary_fallback_reasons"]),
            "overlapping_triggers": dict(self.qr_telemetry["overlapping_triggers"]),
        }

    def get_telemetry(self) -> dict:
        """Returns detailed parameter breakdown, memory compression metrics, and live tier telemetry."""
        muon_count = sum(p.numel() for p in self.muon_params)
        centroid_count = sum(g["param"].numel() for g in self.layer_groups)
        base_count = sum(p.numel() for p in self.non_expert_params)
        total_count = muon_count + centroid_count + base_count

        uncompressed_bytes = total_count * 8
        muon_bytes = muon_count * 4
        centroid_bytes = 0
        for g in self.layer_groups:
            p = g["param"]
            m, n = p.shape
            r = min(self.rank, m, n)
            if self.use_triangular_core:
                if getattr(self, "steer_mode", "upgraded") == "legacy":
                    basis_bytes = (m + n) * (r + 1) * p.element_size()
                    core_bytes = 2 * ((r + 1) * (r + 1)) * 4
                    centroid_bytes += basis_bytes + core_bytes
                else:
                    basis_bytes = (m + n) * r * p.element_size()
                    core_bytes = 2 * (r * r) * 4
                    sub_bytes = 0
                    if self.steer_scale > 0 and g.get("module") is not None:
                        k_c = min(int(getattr(self, "num_steer_dirs", 4)), n)
                        sub_bytes = (n * k_c * 4) + (2 * m * k_c * 4)
                    centroid_bytes += basis_bytes + core_bytes + sub_bytes
            else:
                side = g.get("projection_side", "left")
                proj_elements = (r * n) if side == "left" else (m * r)
                centroid_bytes += proj_elements * 2 * 4

        base_bytes = base_count * (2 if HAS_8BIT and AdamW8bit is not None else 8)
        lucid_bytes = muon_bytes + centroid_bytes + base_bytes
        compression_ratio = (uncompressed_bytes / max(1, lucid_bytes)) if lucid_bytes > 0 else 1.0
        saved_bytes = max(0, uncompressed_bytes - lucid_bytes)
        expert_compression_pct = round((1.0 - (centroid_bytes / max(1, centroid_count * 8))) * 100, 1)

        return {
            "type": "CentroidSteerOptimizer",
            "name": "3-Tier Partitioned Optimizer",
            "step_count": self.step_count,
            "steer_scale": float(self.steer_scale),
            "rank": int(self.rank),
            "update_gap": int(self.update_gap),
            "expert_lr": float(self.expert_lr),
            "base_lr": float(self.base_optimizer.param_groups[0]["lr"])
            if (self.base_optimizer and self.base_optimizer.param_groups)
            else float(self.initial_base_lr),
            "muon_lr": float(self.muon_optimizer.param_groups[0]["lr"])
            if (self.muon_optimizer and self.muon_optimizer.param_groups)
            else float(self.initial_muon_lr),
            "muon_momentum": float(self.muon_optimizer.param_groups[0]["momentum"])
            if (self.muon_optimizer and self.muon_optimizer.param_groups)
            else 0.95,
            "total_params": total_count,
            "tier1_muon": {
                "name": "5th-Order Newton-Schulz Muon",
                "param_count": muon_count,
                "param_pct": round((muon_count / max(1, total_count)) * 100, 1),
                "target": "2D Attention Projections & Shared Weights",
                "vram_saving": "50% (Zero Variance State)",
                "kernel": "lucid_machines/kernels/fast_muon.py",
            },
            "tier2_centroid": {
                "name": (
                    "Steered Triangular Subspace (sCholQR2)"
                    if self.use_triangular_core
                    else "Single-Sided sCholQR2 CentroidSteer"
                ),
                "param_count": centroid_count,
                "param_pct": round((centroid_count / max(1, total_count)) * 100, 1),
                "target": "MoE Routed Experts (Feed-Forward Blocks)",
                "vram_saving": f"{expert_compression_pct}% State Compression",
                "basis_transport": "Active (T = P_old^T @ P_new)",
                "steer_invariant": "P^T c_orth = 0 (sin θ scaling)",
            },
            "tier3_adamw": {
                "name": "Decoupled AdamW / 8-bit AdamW",
                "param_count": base_count,
                "param_pct": round((base_count / max(1, total_count)) * 100, 1),
                "target": "1D RMSNorm, Biases, Router Gates, Embeddings",
                "vram_saving": "Quantized 8-bit" if (HAS_8BIT and AdamW8bit is not None) else "Uncompressed FP32",
            },
            "memory": {
                "uncompressed_state_mb": round(uncompressed_bytes / (1024 * 1024), 2),
                "lucid_state_mb": round(lucid_bytes / (1024 * 1024), 2),
                "triune_state_mb": round(lucid_bytes / (1024 * 1024), 2),
                "saved_state_mb": round(saved_bytes / (1024 * 1024), 2),
                "compression_ratio": round(compression_ratio, 2),
            },
            "qr_telemetry": self.get_qr_telemetry(),
        }


def profile_rank_sensitivity(
    model: torch.nn.Module,
    target_energy: float = 0.85,
    max_rank: int = 32,
    min_rank: int = 4,
) -> dict[str, Any]:
    """Offline Rank Sensitivity Profiler.

    Analyzes the singular value spectrum across all 2D weight matrices in the model,
    calculating the cumulative spectral Frobenius energy:
        E(k) = sum_{i=1}^k sigma_i^2 / sum_{i=1}^d sigma_i^2
    Determines the minimal rank k required to capture >= target_energy of the operator energy.
    """
    layer_profiles = {}
    total_uncompressed_bytes = 0
    total_low_rank_bytes = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                m, n = param.shape
                d_min = min(m, n)
                w_f32 = param.data.float()

                # Compute singular value spectrum
                try:
                    s = torch.linalg.svdvals(w_f32)
                    s_sq = s**2
                    total_energy = s_sq.sum().clamp_min(1e-12)
                    cum_energy = torch.cumsum(s_sq, dim=0) / total_energy

                    # Find minimal k satisfying target_energy
                    idx = (cum_energy >= target_energy).nonzero()
                    if len(idx) > 0:
                        rec_k = int(idx[0].item()) + 1
                    else:
                        rec_k = min(max_rank, d_min)
                except Exception:
                    rec_k = min(max_rank, max(min_rank, d_min // 4))
                    cum_energy = torch.ones(d_min)

                rec_k = max(min_rank, min(max_rank, min(rec_k, d_min)))
                actual_energy = float(cum_energy[rec_k - 1].item()) if rec_k <= len(cum_energy) else 1.0

                uncompressed = m * n
                low_rank = (m + n) * rec_k
                total_uncompressed_bytes += uncompressed * param.element_size()
                total_low_rank_bytes += low_rank * param.element_size()

                layer_profiles[name] = {
                    "shape": [m, n],
                    "recommended_rank": rec_k,
                    "captured_energy": round(actual_energy, 4),
                    "compression_ratio": round(uncompressed / max(1, low_rank), 2),
                }

    return {
        "target_energy": target_energy,
        "max_rank": max_rank,
        "min_rank": min_rank,
        "num_profiled_matrices": len(layer_profiles),
        "total_uncompressed_params": sum(p["shape"][0] * p["shape"][1] for p in layer_profiles.values()),
        "total_low_rank_params": sum(
            (p["shape"][0] + p["shape"][1]) * p["recommended_rank"] for p in layer_profiles.values()
        ),
        "overall_compression_ratio": round(total_uncompressed_bytes / max(1, total_low_rank_bytes), 2),
        "layers": layer_profiles,
    }
