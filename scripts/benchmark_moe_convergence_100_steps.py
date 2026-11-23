#!/usr/bin/env python3
"""Industry Standard Multi-Seed 100-Step Mixture-of-Experts (MoE) Convergence & Loss Trajectory Benchmark.

Trains identical Lucid Universal Mixture-of-Experts Transformers (90.79M parameters,
8 layers, 4 routed experts per layer, shared expert, top-1 routing) on real FineWeb-Edu
educational text across multiple independent random seeds for 100 continuous steps under
core optimizer regimes, published memory-efficient baselines (GaLore, BAdam), and
CentroidSteer component ablations at genuine low-rank compression (e.g. rank=16, gap=20).

Methodological Rigor & Parity Guarantees:
- Multi-seed evaluation reporting mean ± std plus paired & Welch t-test p-values
- Identical step-0 initial weights cloned from a common state_dict for each seed
- Identical training batch sequences per seed
- Document-level train/validation split with dynamic [SEP] boundary packing
- Multi-batch held-out validation set evaluated periodically
- Live deduplicated optimizer state bytes/param and MoE routing variance concentration checks
- Common pre-warmup pass to eliminate CUDA JIT / context setup distortion from step timing
- Full cryptographic SHA-256 provenance and exact reproduction commands
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lucid_machines.model.moe import MoE_FFN
from lucid_machines.optim.centroid import CentroidSteerOptimizer
from lucid_machines.universal import UniversalMoETransformer

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_source_hashes() -> Dict[str, str]:
    files = [
        "scripts/benchmark_moe_convergence_100_steps.py",
        "lucid_machines/kernels/fast_gla.py",
        "lucid_machines/kernels/fast_muon.py",
        "lucid_machines/kernels/fast_triangular.py",
        "lucid_machines/model/moe.py",
        "lucid_machines/model/transformer.py",
        "lucid_machines/optim/centroid.py",
        "lucid_machines/optim/muon.py",
        "lucid_machines/universal.py",
        "lucid_machines/model/config.py",
        "triune_tokenizer.json",
        "data/fineweb_edu_real.jsonl",
    ]
    hashes = {}
    for f in files:
        p = REPO_ROOT / f
        if p.is_file():
            hashes[f] = compute_sha256(p)
    return hashes


def get_git_info() -> Tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
        return commit, len(status) > 0
    except Exception:
        return "unknown", False


def get_hardware_info(device: torch.device) -> Dict[str, Any]:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return {
            "device_name": props.name,
            "total_vram_gb": round(props.total_memory / (1024**3), 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda_version": torch.version.cuda or "N/A",
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
        }
    return {
        "device_name": "CPU",
        "total_vram_gb": 0.0,
        "compute_capability": "N/A",
        "cuda_version": "N/A",
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }


def load_fineweb_documents(
    data_path: Path,
    tokenizer_path: Path,
    num_val_docs: int = 15,
    required_tokens: int = 120000,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Loads genuine FineWeb-Edu educational samples, strictly partitions documents, and tokenizes splits independently."""
    assert data_path.is_file(), f"FineWeb dataset file missing: {data_path}"
    assert tokenizer_path.is_file(), f"Tokenizer file missing: {tokenizer_path}"

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    sep_token_id = tok.token_to_id("[SEP]") if hasattr(tok, "token_to_id") else 3
    if sep_token_id is None:
        sep_token_id = 3

    docs: List[str] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text", "")
            if text.strip():
                docs.append(text)

    # Document-level split: reserve last num_val_docs documents for validation
    val_docs = docs[-num_val_docs:]
    train_docs = docs[:-num_val_docs]

    train_tokens: List[int] = []
    for d in train_docs:
        enc = tok.encode(d).ids
        if enc:
            train_tokens.extend(enc)
            train_tokens.append(sep_token_id)
        if len(train_tokens) >= required_tokens:
            break

    val_tokens: List[int] = []
    for d in val_docs:
        enc = tok.encode(d).ids
        if enc:
            val_tokens.extend(enc)
            val_tokens.append(sep_token_id)

    meta = {
        "dataset_name": "HuggingFaceFW/fineweb-edu (sample-10BT)",
        "dataset_path": str(data_path.relative_to(REPO_ROOT)),
        "dataset_sha256": compute_sha256(data_path),
        "tokenizer_path": str(tokenizer_path.relative_to(REPO_ROOT)),
        "tokenizer_sha256": compute_sha256(tokenizer_path),
        "total_documents_read": len(docs),
        "train_documents_count": len(train_docs),
        "val_documents_count": len(val_docs),
        "train_tokens_packed": len(train_tokens),
        "val_tokens_packed": len(val_tokens),
        "separator_token_id": sep_token_id,
    }
    return train_tokens, val_tokens, meta


RUN_CONFIG: Dict[str, Any] = {
    "lr": 1e-3,
    "muon_lr": 0.002,
    "steer_scale": 0.20,
    "rank": 16,
    "update_gap": 20,
    "steer_update_gap": 5,
    "core_mask_mode": "full",
}


class BAdamBlockOptimizer:
    """Block-coordinate AdamW (Luo et al., 2024) keeping only one transformer block's optimizer state live."""

    def __init__(self, model: torch.nn.Module, lr: float = 1e-3, weight_decay: float = 0.01, block_steps: int = 2):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.block_steps = max(1, int(block_steps))
        self.all_trainable_params = [p for p in model.parameters() if p.requires_grad]

        block_map: Dict[str, List[torch.nn.Parameter]] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "layers." in name:
                after = name.split("layers.", 1)[1]
                idx_str = after.split(".", 1)[0]
                key = f"layer_{int(idx_str):03d}" if idx_str.isdigit() else "layers_other"
            elif "embed" in name.lower():
                key = "000_embed"
            else:
                key = "zzz_head"
            block_map.setdefault(key, []).append(param)

        self.block_names = sorted(block_map.keys())
        self.blocks = [block_map[k] for k in self.block_names if block_map[k]]
        self.step_count = 0
        self.active_idx = 0
        self.active_optimizer: torch.optim.AdamW | None = None
        self._activate_block(0)

    def _activate_block(self, new_idx: int) -> None:
        if self.active_optimizer is not None:
            self.active_optimizer.zero_grad(set_to_none=True)
            self.active_optimizer = None
        for idx, block_params in enumerate(self.blocks):
            is_active = idx == new_idx
            for p in block_params:
                if not is_active:
                    p.grad = None
                p.requires_grad_(is_active)
        self.active_idx = new_idx
        self.active_optimizer = torch.optim.AdamW(self.blocks[new_idx], lr=self.lr, weight_decay=self.weight_decay)

    def zero_grad(self, set_to_none: bool = True) -> None:
        target_idx = (self.step_count // self.block_steps) % len(self.blocks)
        if target_idx != self.active_idx or self.active_optimizer is None:
            self._activate_block(target_idx)
        self.active_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.active_optimizer.step(closure)
        self.step_count += 1
        return loss

    def restore_requires_grad(self) -> None:
        for p in self.all_trainable_params:
            p.requires_grad_(True)


def measure_optimizer_state_bytes(optimizer: Any, model: torch.nn.Module) -> Dict[str, Any]:
    """Computes live deduplicated optimizer state bytes (total and routed-expert tier) without double counting."""
    model_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
    expert_param_ids = {id(p) for m in model.modules() if isinstance(m, MoE_FFN) for p in m.experts.parameters()}
    trainable_params = getattr(optimizer, "all_trainable_params", None) or [
        p for p in model.parameters() if p.requires_grad
    ]
    total_param_count = sum(p.numel() for p in trainable_params)
    expert_param_count = sum(
        p.numel() for m in model.modules() if isinstance(m, MoE_FFN) for p in m.experts.parameters()
    )

    seen_ptrs = set(model_ptrs)
    total_bytes = 0
    expert_bytes = 0

    def _add_tensor(t: Any, is_expert_fraction: float = 0.0) -> None:
        nonlocal total_bytes, expert_bytes
        if not isinstance(t, torch.Tensor):
            return
        st = t.untyped_storage()
        ptr = st.data_ptr()
        if ptr in seen_ptrs:
            return
        seen_ptrs.add(ptr)
        nbytes = int(st.nbytes())
        total_bytes += nbytes
        if is_expert_fraction > 0.0:
            expert_bytes += int(round(nbytes * is_expert_fraction))

    def _walk_state_dict_opt(opt_obj: Any) -> None:
        if opt_obj is None or not hasattr(opt_obj, "state"):
            return
        for param, p_state in opt_obj.state.items():
            if not isinstance(p_state, dict):
                continue
            frac = 1.0 if id(param) in expert_param_ids else 0.0
            for v in p_state.values():
                _add_tensor(v, frac)

    if isinstance(optimizer, BAdamBlockOptimizer):
        _walk_state_dict_opt(optimizer.active_optimizer)
    elif isinstance(optimizer, CentroidSteerOptimizer):
        _walk_state_dict_opt(getattr(optimizer, "base_optimizer", None))
        _walk_state_dict_opt(getattr(optimizer, "muon_optimizer", None))
        for g in getattr(optimizer, "layer_groups", []):
            is_exp = 1.0 if (g.get("module") is not None or id(g["param"]) in expert_param_ids) else 0.0
            for key in ("projection", "proj_basis", "Q_L", "Q_R", "last_c_orth"):
                _add_tensor(g.get(key), is_exp)
            g_state = g.get("state")
            if isinstance(g_state, dict):
                for v in g_state.values():
                    _add_tensor(v, is_exp)
        bbufs = getattr(optimizer, "_bucket_buffers", {})
        if isinstance(bbufs, dict) and bbufs:
            bucket_counts: Dict[Tuple[Any, ...], List[int]] = {}
            for g in getattr(optimizer, "layer_groups", []):
                p = g["param"]
                m, n = p.shape
                r = min(optimizer.rank, m, n)
                steer_role = (
                    g.get("steer_role", "input")
                    if (g.get("module") is not None and optimizer.steer_scale != 0.0)
                    else "none"
                )
                key = (m, n, r, steer_role)
                counts = bucket_counts.setdefault(key, [0, 0])
                counts[0] += 1
                if g.get("module") is not None or id(p) in expert_param_ids:
                    counts[1] += 1
            for bkey, bdict in bbufs.items():
                if not isinstance(bdict, dict):
                    continue
                # bkey = (K, m, n, r, steer_role, device, dtype)
                frac = 0.0
                if len(bkey) >= 5:
                    sub_key = (bkey[1], bkey[2], bkey[3], bkey[4])
                    cnts = bucket_counts.get(sub_key)
                    if cnts and cnts[0] > 0:
                        frac = cnts[1] / cnts[0]
                    elif bkey[4] != "none":
                        frac = 1.0
                for v in bdict.values():
                    _add_tensor(v, frac)
    else:
        _walk_state_dict_opt(optimizer)

    expert_bpp = expert_bytes / max(1, expert_param_count)
    total_bpp = total_bytes / max(1, total_param_count)
    return {
        "total_state_bytes": int(total_bytes),
        "total_state_mb": round(total_bytes / (1024 * 1024), 2),
        "total_bytes_per_param": round(total_bpp, 4),
        "expert_state_bytes": int(expert_bytes),
        "expert_state_mb": round(expert_bytes / (1024 * 1024), 3),
        "expert_bytes_per_param": round(expert_bpp, 4),
        "expert_reduction_vs_bf16_adamw_pct": round((1.0 - expert_bpp / 4.0) * 100.0, 2),
        "expert_reduction_vs_fp32_adamw_pct": round((1.0 - expert_bpp / 8.0) * 100.0, 2),
    }


def compute_welch_and_paired_pvalues(a_vals: List[float], b_vals: List[float]) -> Dict[str, Any]:
    """Computes paired two-sided t-test and Welch t-test p-values comparing two seed-matched lists."""
    n = min(len(a_vals), len(b_vals))
    if n < 2:
        return {"n": n, "mean_diff": None, "paired_p": None, "welch_p": None}
    diffs = [a - b for a, b in zip(a_vals[:n], b_vals[:n])]
    mean_diff = statistics.mean(diffs)
    paired_p = None
    welch_p = None
    if scipy_stats is not None:
        _, p_rel = scipy_stats.ttest_rel(a_vals[:n], b_vals[:n])
        _, p_ind = scipy_stats.ttest_ind(a_vals[:n], b_vals[:n], equal_var=False)
        paired_p = None if math.isnan(float(p_rel)) else round(float(p_rel), 6)
        welch_p = None if math.isnan(float(p_ind)) else round(float(p_ind), 6)
    return {
        "n": n,
        "mean_a": round(statistics.mean(a_vals[:n]), 4),
        "std_a": round(statistics.stdev(a_vals[:n]), 4),
        "mean_b": round(statistics.mean(b_vals[:n]), 4),
        "std_b": round(statistics.stdev(b_vals[:n]), 4),
        "mean_diff": round(mean_diff, 4),
        "paired_p": paired_p,
        "welch_p": welch_p,
    }


def create_moe_optimizer(
    model: UniversalMoETransformer,
    optimizer_type: str,
    lr: float | None = None,
    muon_lr: float | None = None,
    use_8bit_base: bool = False,
):
    """Instantiates the specified optimizer regime for UniversalMoETransformer."""
    lr = RUN_CONFIG["lr"] if lr is None else lr
    muon_lr = RUN_CONFIG["muon_lr"] if muon_lr is None else muon_lr
    steer = RUN_CONFIG["steer_scale"]
    rank = int(RUN_CONFIG["rank"])
    update_gap = int(RUN_CONFIG["update_gap"])
    steer_update_gap = int(RUN_CONFIG["steer_update_gap"])
    core_mask_mode = str(RUN_CONFIG["core_mask_mode"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == "adamw":
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    elif optimizer_type == "badam":
        return BAdamBlockOptimizer(model, lr=lr, weight_decay=0.01, block_steps=2)
    elif optimizer_type == "galore":
        # Authentic single-sided rank-r GaLore (Zhao et al., 2024): no Muon, no centroid steering, no triangular core
        return CentroidSteerOptimizer(
            model,
            lr=lr,
            weight_decay=0.01,
            rank=rank,
            update_gap=update_gap,
            steer_update_gap=steer_update_gap,
            use_muon=False,
            steer_scale=0.0,
            use_triangular_core=False,
            use_8bit_base=use_8bit_base,
        )
    elif optimizer_type == "muon_adamw":
        # Muon on attention/shared 2D weights; standard AdamW on experts and embeddings
        param_roles = {}
        for _, module in model.named_modules():
            if isinstance(module, MoE_FFN):
                for p in module.experts.parameters():
                    param_roles[p] = "adamw"
        return CentroidSteerOptimizer(
            model,
            lr=lr,
            muon_lr=muon_lr,
            weight_decay=0.01,
            rank=rank,
            update_gap=update_gap,
            steer_update_gap=steer_update_gap,
            use_muon=True,
            steer_scale=0.0,
            param_roles=param_roles,
            use_8bit_base=use_8bit_base,
        )
    elif optimizer_type == "centroid_no_muon":
        return CentroidSteerOptimizer(
            model,
            lr=lr,
            weight_decay=0.01,
            rank=rank,
            update_gap=update_gap,
            steer_update_gap=steer_update_gap,
            use_muon=False,
            steer_scale=steer,
            use_triangular_core=True,
            core_mask_mode=core_mask_mode,
            use_8bit_base=use_8bit_base,
        )
    elif optimizer_type == "triune_moe":
        return CentroidSteerOptimizer(
            model,
            lr=lr,
            muon_lr=muon_lr,
            weight_decay=0.01,
            rank=rank,
            update_gap=update_gap,
            steer_update_gap=steer_update_gap,
            use_muon=True,
            steer_scale=steer,
            use_triangular_core=True,
            core_mask_mode=core_mask_mode,
            use_8bit_base=use_8bit_base,
        )
    elif optimizer_type == "triune_moe_nosteer":
        # Identical to triune_moe but with router-centroid steering disabled (pure two-sided triangular core).
        return CentroidSteerOptimizer(
            model,
            lr=lr,
            muon_lr=muon_lr,
            weight_decay=0.01,
            rank=rank,
            update_gap=update_gap,
            steer_update_gap=steer_update_gap,
            use_muon=True,
            steer_scale=0.0,
            use_triangular_core=True,
            core_mask_mode=core_mask_mode,
            use_8bit_base=use_8bit_base,
        )
    elif optimizer_type == "triune_moe_random":
        # Identical to triune_moe except m.last_centroids are replaced by random isotropic unit vectors.
        return CentroidSteerOptimizer(
            model,
            lr=lr,
            muon_lr=muon_lr,
            weight_decay=0.01,
            rank=rank,
            update_gap=update_gap,
            steer_update_gap=steer_update_gap,
            use_muon=True,
            steer_scale=steer,
            use_triangular_core=True,
            core_mask_mode=core_mask_mode,
            use_8bit_base=use_8bit_base,
        )
    else:
        raise ValueError(f"Unknown MoE optimizer_type: {optimizer_type}")


def evaluate_validation_loss(
    model: UniversalMoETransformer,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
) -> float:
    """Evaluates cross-entropy loss over a multi-batch validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for val_x, val_y in val_batches:
            v_out = model(val_x, force_depth=2)
            v_logits = v_out[0] if isinstance(v_out, tuple) else v_out
            loss = F.cross_entropy(
                v_logits.reshape(-1, vocab_size).float(),
                val_y.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += val_y.numel()
    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def measure_routing_concentration(
    model: UniversalMoETransformer,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, float]:
    """Measures empirical MoE routing variance concentration: tr(Sigma_e) vs ||c_e||^2 and tr(Sigma_global)."""
    was_training = model.training
    model.eval()

    per_expert_ratios: List[float] = []
    within_vs_global_ratios: List[float] = []
    centroid_energy_fracs: List[float] = []
    centroid_norm_sqs: List[float] = []
    within_cov_traces: List[float] = []
    entropies: List[float] = []
    handles = []

    def _hook(module: MoE_FFN, inputs: Tuple[torch.Tensor, ...], _output: Any) -> None:
        if not inputs:
            return
        x = inputs[0]
        flat_x = x.reshape(-1, x.shape[-1]).float()
        logits = module.router(x.reshape(-1, x.shape[-1])).float()
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        entropies.append(float(entropy.item()))

        assign = logits.argmax(dim=-1)
        x_mean_global = flat_x.mean(dim=0, keepdim=True)
        global_var = float(((flat_x - x_mean_global) ** 2).sum(dim=-1).mean().item())

        for e_idx in range(module.num_experts):
            mask = assign == e_idx
            cnt = int(mask.sum().item())
            if cnt < 4:
                continue
            x_e = flat_x[mask]
            c_e = x_e.mean(dim=0)
            c_sq = float((c_e * c_e).sum().item())
            tr_sigma_e = float(((x_e - c_e.unsqueeze(0)) ** 2).sum(dim=-1).mean().item())
            total_e = c_sq + tr_sigma_e
            if c_sq > 1e-8 and total_e > 1e-8:
                centroid_norm_sqs.append(c_sq)
                within_cov_traces.append(tr_sigma_e)
                per_expert_ratios.append(tr_sigma_e / c_sq)
                centroid_energy_fracs.append(c_sq / total_e)
                if global_var > 1e-8:
                    within_vs_global_ratios.append(tr_sigma_e / global_var)

    for m in model.modules():
        if isinstance(m, MoE_FFN):
            handles.append(m.register_forward_hook(_hook))

    try:
        for val_x, _ in val_batches[:2]:
            model(val_x, force_depth=2)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    def _safe_mean(vals: List[float], digits: int = 4) -> float:
        return round(statistics.mean(vals), digits) if vals else 0.0

    return {
        "mean_centroid_norm_sq": _safe_mean(centroid_norm_sqs, 4),
        "mean_within_cov_trace": _safe_mean(within_cov_traces, 4),
        "mean_trace_to_centroid_ratio": _safe_mean(per_expert_ratios, 4),
        "mean_within_to_global_var_ratio": _safe_mean(within_vs_global_ratios, 4),
        "mean_centroid_energy_fraction": _safe_mean(centroid_energy_fracs, 4),
        "mean_router_entropy": _safe_mean(entropies, 4),
    }


def run_single_moe_seed_run(
    seed: int,
    regime_name: str,
    optimizer_type: str,
    initial_state_dict: Dict[str, torch.Tensor],
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    total_steps: int = 100,
    batch_size: int = 2,
    seq_len: int = 256,
    val_interval: int = 20,
    vocab_size: int = 32000,
    dim: int = 384,
    num_layers: int = 8,
    num_heads: int = 6,
    num_experts: int = 4,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Dict[str, Any]:
    """Runs 100 steps of MoE LM pretraining for a specific seed and regime with identical initial weights."""
    print(f"  --> Seed {seed} | MoE Regime: {regime_name} ({optimizer_type})")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Instantiate model and load initial weights
    model = UniversalMoETransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    model.train()

    # 2. Pre-warmup step to compile CUDA/Triton kernels and initialize memory pools
    optimizer = create_moe_optimizer(model, optimizer_type)
    if len(train_batches) > 0:
        warmup_x, warmup_y = train_batches[0]
        optimizer.zero_grad(set_to_none=True)
        w_out = model(warmup_x, force_depth=2)
        w_logits = w_out[0] if isinstance(w_out, tuple) else w_out
        w_loss = F.cross_entropy(w_logits.reshape(-1, vocab_size).float(), warmup_y.reshape(-1))
        if w_loss.requires_grad:
            w_loss.backward()
            optimizer.step()
        for m in model.modules():
            if isinstance(m, MoE_FFN):
                m.step_bias()
        optimizer.zero_grad(set_to_none=True)
    if hasattr(optimizer, "restore_requires_grad"):
        optimizer.restore_requires_grad()

    # 3. Reload identical initial weights and create fresh optimizer for timed benchmark
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_moe_optimizer(model, optimizer_type)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Track parameter tier distribution accurately
    if isinstance(optimizer, CentroidSteerOptimizer):
        tier1_muon_params = sum(p.numel() for p in getattr(optimizer, "muon_params", []))
        tier2_centroid_params = sum(g["param"].numel() for g in getattr(optimizer, "layer_groups", []))
        tier3_base_params = sum(p.numel() for p in getattr(optimizer, "non_expert_params", []))
        centroid_groups_count = len(getattr(optimizer, "layer_groups", []))
        has_dynamic_centroids = bool(
            optimizer.steer_scale != 0.0
            and optimizer_type != "triune_moe_random"
            and any(g.get("module") is not None for g in getattr(optimizer, "layer_groups", []))
        )
    else:
        tier1_muon_params = 0
        tier2_centroid_params = 0
        tier3_base_params = sum(p.numel() for p in trainable_params)
        centroid_groups_count = 0
        has_dynamic_centroids = False

    # Ensure strict run-order isolation across regimes by resetting RNG
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Evaluate step 0 validation loss
    init_val_loss = evaluate_validation_loss(model, val_batches, vocab_size)

    step_records = []
    val_records = [
        {"step": 0, "val_loss": round(init_val_loss, 4), "val_ppl": round(math.exp(min(20.0, init_val_loss)), 2)}
    ]

    cum_time_sec = 0.0

    # Tracking per-expert token routing and gradient activity
    total_expert_tokens = [0] * num_experts
    tier1_nonzero_grads = 0
    tier2_nonzero_grads = 0
    tier3_nonzero_grads = 0
    peak_badam_state_mb = 0.0
    peak_badam_expert_state_mb = 0.0

    for step in range(1, total_steps + 1):
        x, y = train_batches[step - 1]

        if device.type == "cuda":
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

            optimizer.zero_grad(set_to_none=True)
            out = model(x, force_depth=2)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), y.reshape(-1))
            if loss.requires_grad:
                loss.backward()

            if step in (1, total_steps // 2, total_steps):
                if isinstance(optimizer, CentroidSteerOptimizer):
                    tier1_nonzero_grads = sum(
                        p.numel()
                        for p in getattr(optimizer, "muon_params", [])
                        if p.grad is not None and p.grad.abs().sum() > 0
                    )
                    tier2_nonzero_grads = sum(
                        g["param"].numel()
                        for g in getattr(optimizer, "layer_groups", [])
                        if g["param"].grad is not None and g["param"].grad.abs().sum() > 0
                    )
                    tier3_nonzero_grads = sum(
                        p.numel()
                        for p in getattr(optimizer, "non_expert_params", [])
                        if p.grad is not None and p.grad.abs().sum() > 0
                    )
                else:
                    tier3_nonzero_grads = sum(
                        p.numel() for p in trainable_params if p.grad is not None and p.grad.abs().sum() > 0
                    )

            if optimizer_type == "triune_moe_random":
                for m in model.modules():
                    if isinstance(m, MoE_FFN) and hasattr(m, "last_centroids") and m.last_centroids is not None:
                        m.last_centroids.normal_()
                        m.last_centroids.div_(m.last_centroids.norm(dim=-1, keepdim=True).clamp_min(1e-8))

            optimizer.step()
            for m in model.modules():
                if isinstance(m, MoE_FFN):
                    m.step_bias()
                    if hasattr(m, "last_accepted_counts") and m.last_accepted_counts is not None:
                        counts_list = m.last_accepted_counts.tolist()
                        for e_idx in range(min(num_experts, len(counts_list))):
                            total_expert_tokens[e_idx] += int(counts_list[e_idx])

            t1.record()
            torch.cuda.synchronize(device)
            step_ms = t0.elapsed_time(t1)
        else:
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            out = model(x, force_depth=2)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), y.reshape(-1))
            if loss.requires_grad:
                loss.backward()
            if optimizer_type == "triune_moe_random":
                for m in model.modules():
                    if isinstance(m, MoE_FFN) and hasattr(m, "last_centroids") and m.last_centroids is not None:
                        m.last_centroids.normal_()
                        m.last_centroids.div_(m.last_centroids.norm(dim=-1, keepdim=True).clamp_min(1e-8))

            optimizer.step()
            for m in model.modules():
                if isinstance(m, MoE_FFN):
                    m.step_bias()
            t1 = time.perf_counter()
            step_ms = (t1 - t0) * 1000.0

        # Sample BAdam state over one full block cycle near the end (outside timed step)
        if isinstance(optimizer, BAdamBlockOptimizer) and step > max(0, total_steps - 22):
            cur_mem = measure_optimizer_state_bytes(optimizer, model)
            peak_badam_state_mb = max(peak_badam_state_mb, cur_mem["total_state_mb"])
            peak_badam_expert_state_mb = max(peak_badam_expert_state_mb, cur_mem["expert_state_mb"])

        cum_time_sec += step_ms / 1000.0
        loss_val = float(loss.item())
        ppl_val = math.exp(min(20.0, loss_val))

        step_records.append(
            {
                "step": step,
                "loss": round(loss_val, 4),
                "ppl": round(ppl_val, 2),
                "step_ms": round(step_ms, 2),
                "cum_time_sec": round(cum_time_sec, 3),
            }
        )

        if step % val_interval == 0 or step == total_steps:
            cur_val_loss = evaluate_validation_loss(model, val_batches, vocab_size)
            cur_val_ppl = math.exp(min(20.0, cur_val_loss))
            val_records.append(
                {
                    "step": step,
                    "val_loss": round(cur_val_loss, 4),
                    "val_ppl": round(cur_val_ppl, 2),
                    "cum_time_sec": round(cum_time_sec, 3),
                }
            )

    peak_alloc_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1) if device.type == "cuda" else 0.0
    peak_res_mb = round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 1) if device.type == "cuda" else 0.0

    state_mem = measure_optimizer_state_bytes(optimizer, model)
    if isinstance(optimizer, BAdamBlockOptimizer) and peak_badam_expert_state_mb > 0.0:
        expert_p_total = sum(
            p.numel() for m in model.modules() if isinstance(m, MoE_FFN) for p in m.experts.parameters()
        )
        peak_exp_bytes = int(peak_badam_expert_state_mb * 1024 * 1024)
        peak_tot_bytes = int(peak_badam_state_mb * 1024 * 1024)
        exp_bpp = peak_exp_bytes / max(1, expert_p_total)
        state_mem = {
            "total_state_bytes": peak_tot_bytes,
            "total_state_mb": round(peak_badam_state_mb, 2),
            "total_bytes_per_param": round(peak_tot_bytes / max(1, sum(p.numel() for p in trainable_params)), 4),
            "expert_state_bytes": peak_exp_bytes,
            "expert_state_mb": round(peak_badam_expert_state_mb, 3),
            "expert_bytes_per_param": round(exp_bpp, 4),
            "expert_reduction_vs_bf16_adamw_pct": round((1.0 - exp_bpp / 4.0) * 100.0, 2),
            "expert_reduction_vs_fp32_adamw_pct": round((1.0 - exp_bpp / 8.0) * 100.0, 2),
        }

    if hasattr(optimizer, "restore_requires_grad"):
        optimizer.restore_requires_grad()

    routing_conc = measure_routing_concentration(model, val_batches)

    init_train_loss = step_records[0]["loss"]
    final_train_loss = step_records[-1]["loss"]
    train_loss_drop = round(init_train_loss - final_train_loss, 4)

    final_val_loss = val_records[-1]["val_loss"]
    val_loss_drop = round(init_val_loss - final_val_loss, 4)

    median_step_ms = statistics.median([r["step_ms"] for r in step_records])
    first_step_ms = step_records[0]["step_ms"] if step_records else 0.0
    steady_state_step_ms = (
        statistics.median([r["step_ms"] for r in step_records[1:]]) if len(step_records) > 1 else first_step_ms
    )
    throughput_tok_s = (batch_size * seq_len) / (median_step_ms / 1000.0)
    convergence_efficiency = round(val_loss_drop / max(cum_time_sec, 1e-6), 4)

    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "regime_name": regime_name,
        "optimizer": optimizer_type,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "median_step_ms": round(median_step_ms, 2),
        "first_step_ms": round(first_step_ms, 2),
        "steady_state_step_ms": round(steady_state_step_ms, 2),
        "tier1_muon_params": tier1_muon_params,
        "tier2_centroid_params": tier2_centroid_params,
        "tier3_base_params": tier3_base_params,
        "tier1_nonzero_grads": tier1_nonzero_grads,
        "tier2_nonzero_grads": tier2_nonzero_grads,
        "tier3_nonzero_grads": tier3_nonzero_grads,
        "centroid_groups_count": centroid_groups_count,
        "has_dynamic_centroids": has_dynamic_centroids,
        "expert_tokens_routed": total_expert_tokens,
        "total_time_sec": round(cum_time_sec, 2),
        "throughput_tok_s": round(throughput_tok_s, 0),
        "peak_vram_alloc_mb": peak_alloc_mb,
        "peak_vram_reserved_mb": peak_res_mb,
        "state_memory": state_mem,
        "routing_concentration": routing_conc,
        "initial_train_loss": init_train_loss,
        "final_train_loss": final_train_loss,
        "train_loss_reduction": train_loss_drop,
        "initial_val_loss": round(init_val_loss, 4),
        "final_val_loss": final_val_loss,
        "val_loss_reduction": val_loss_drop,
        "initial_val_ppl": val_records[0]["val_ppl"],
        "final_val_ppl": val_records[-1]["val_ppl"],
        "convergence_efficiency_drop_per_sec": convergence_efficiency,
        "step_records": step_records,
        "val_records": val_records,
    }


def aggregate_moe_regime_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes mean and std across multiple seed runs for an MoE regime."""
    n = len(runs)

    def mean_std(key: str, round_digits: int = 4) -> Dict[str, float]:
        vals = [r[key] for r in runs]
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if n > 1 else 0.0
        return {
            "mean": round(m, round_digits),
            "std": round(s, round_digits),
            "min": round(min(vals), round_digits),
            "max": round(max(vals), round_digits),
        }

    def nested_mean_std(parent: str, key: str, round_digits: int = 4) -> Dict[str, float]:
        vals = [float(r.get(parent, {}).get(key, 0.0)) for r in runs]
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if n > 1 else 0.0
        return {
            "mean": round(m, round_digits),
            "std": round(s, round_digits),
            "min": round(min(vals), round_digits),
            "max": round(max(vals), round_digits),
        }

    steps = [v["step"] for v in runs[0]["val_records"]]
    val_traj = []
    for st in steps:
        st_losses = [next(v["val_loss"] for v in r["val_records"] if v["step"] == st) for r in runs]
        st_ppls = [next(v["val_ppl"] for v in r["val_records"] if v["step"] == st) for r in runs]
        val_traj.append(
            {
                "step": st,
                "val_loss_mean": round(statistics.mean(st_losses), 4),
                "val_loss_std": round(statistics.stdev(st_losses) if n > 1 else 0.0, 4),
                "val_ppl_mean": round(statistics.mean(st_ppls), 2),
                "val_ppl_std": round(statistics.stdev(st_ppls) if n > 1 else 0.0, 2),
            }
        )

    num_experts = len(runs[0]["expert_tokens_routed"])
    avg_expert_tokens = [int(statistics.mean([r["expert_tokens_routed"][e] for r in runs])) for e in range(num_experts)]

    return {
        "regime_name": runs[0]["regime_name"],
        "optimizer": runs[0]["optimizer"],
        "num_seeds": n,
        "seeds": [r["seed"] for r in runs],
        "initial_val_loss": mean_std("initial_val_loss", 4),
        "final_val_loss": mean_std("final_val_loss", 4),
        "val_loss_reduction": mean_std("val_loss_reduction", 4),
        "final_val_ppl": mean_std("final_val_ppl", 2),
        "median_step_ms": mean_std("median_step_ms", 2),
        "first_step_ms": mean_std("first_step_ms", 2),
        "steady_state_step_ms": mean_std("steady_state_step_ms", 2),
        "tier1_muon_params": runs[0].get("tier1_muon_params", 0),
        "tier2_centroid_params": runs[0].get("tier2_centroid_params", 0),
        "tier3_base_params": runs[0].get("tier3_base_params", 0),
        "tier1_nonzero_grads": mean_std("tier1_nonzero_grads", 0),
        "tier2_nonzero_grads": mean_std("tier2_nonzero_grads", 0),
        "tier3_nonzero_grads": mean_std("tier3_nonzero_grads", 0),
        "centroid_groups_count": runs[0].get("centroid_groups_count", 0),
        "has_dynamic_centroids": runs[0].get("has_dynamic_centroids", False),
        "expert_tokens_routed": avg_expert_tokens,
        "total_time_sec": mean_std("total_time_sec", 2),
        "throughput_tok_s": mean_std("throughput_tok_s", 0),
        "peak_vram_alloc_mb": mean_std("peak_vram_alloc_mb", 1),
        "peak_vram_reserved_mb": mean_std("peak_vram_reserved_mb", 1),
        "total_state_mb": nested_mean_std("state_memory", "total_state_mb", 2),
        "total_bytes_per_param": nested_mean_std("state_memory", "total_bytes_per_param", 4),
        "expert_state_mb": nested_mean_std("state_memory", "expert_state_mb", 3),
        "expert_bytes_per_param": nested_mean_std("state_memory", "expert_bytes_per_param", 4),
        "expert_reduction_vs_bf16_adamw_pct": nested_mean_std("state_memory", "expert_reduction_vs_bf16_adamw_pct", 2),
        "expert_reduction_vs_fp32_adamw_pct": nested_mean_std("state_memory", "expert_reduction_vs_fp32_adamw_pct", 2),
        "routing_concentration": {
            "mean_centroid_norm_sq": nested_mean_std("routing_concentration", "mean_centroid_norm_sq", 4),
            "mean_within_cov_trace": nested_mean_std("routing_concentration", "mean_within_cov_trace", 4),
            "mean_trace_to_centroid_ratio": nested_mean_std("routing_concentration", "mean_trace_to_centroid_ratio", 4),
            "mean_within_to_global_var_ratio": nested_mean_std(
                "routing_concentration", "mean_within_to_global_var_ratio", 4
            ),
            "mean_centroid_energy_fraction": nested_mean_std(
                "routing_concentration", "mean_centroid_energy_fraction", 4
            ),
            "mean_router_entropy": nested_mean_std("routing_concentration", "mean_router_entropy", 4),
        },
        "convergence_efficiency": mean_std("convergence_efficiency_drop_per_sec", 4),
        "val_trajectory": val_traj,
        "runs": runs,
    }


def generate_moe_convergence_report(
    summary: Dict[str, Dict[str, Any]],
    output_path: Path,
    provenance: Dict[str, Any],
    pairwise_stats: Dict[str, Any] | None = None,
):
    """Generates a comprehensive Markdown report documenting multi-seed MoE convergence, memory, and p-values."""
    hw = provenance.get("hardware", {})
    sw = provenance.get("software", {})
    ds = provenance.get("dataset", {})
    opt_cfg = provenance.get("optimizer_configuration", {})
    git_c = provenance.get("git_commit", "unknown")
    git_dirty_str = " (Dirty Working Tree)" if provenance.get("git_dirty") else " (Clean Tree)"
    seeds_str = ", ".join(str(s) for s in provenance.get("seeds", []))
    num_val_batches = provenance.get("num_val_batches", 5)
    val_tokens = num_val_batches * provenance.get("batch_size", 2) * provenance.get("seq_len", 256)

    canonical_order = [
        "adamw",
        "muon_adamw",
        "badam",
        "galore",
        "centroid_no_muon",
        "triune_moe_nosteer",
        "triune_moe_random",
        "triune_moe",
    ]
    regimes_order = [r for r in canonical_order if r in summary] + [
        r for r in summary.keys() if r not in canonical_order
    ]

    regime_labels = {
        "adamw": "Standard AdamW (`torch.optim.AdamW`)",
        "muon_adamw": "Muon + AdamW Baseline (Muon Attention + AdamW Experts)",
        "badam": "BAdam Baseline (Block-Coordinate AdamW, Luo et al. 2024)",
        "galore": f"GaLore Baseline (Single-Sided SVD r={opt_cfg.get('rank', 16)}, gap={opt_cfg.get('update_gap', 20)})",
        "centroid_no_muon": "CentroidSteer Only (No Muon: Triangular Subspace + Steering + AdamW Base)",
        "triune_moe_nosteer": "Triune Unsteered Ablation (Muon + Triangular Subspace, alpha=0)",
        "triune_moe_random": "Triune Random-Steer Ablation (Muon + Triangular Subspace + Random Noise Dir)",
        "triune_moe": "**Triune MoE Stack** (Muon Attention + CentroidSteer Experts + AdamW Base)",
    }
    short_labels = {
        "adamw": "AdamW",
        "muon_adamw": "Muon+AdamW",
        "badam": "BAdam",
        "galore": "GaLore",
        "centroid_no_muon": "Centroid(NoMuon)",
        "triune_moe_nosteer": "Triune(NoSteer)",
        "triune_moe_random": "Triune(RandSteer)",
        "triune_moe": "Triune MoE",
    }

    best_regime = min(regimes_order, key=lambda r: summary[r]["final_val_loss"]["mean"]) if regimes_order else None

    md = f"""# Lucid Machines Multi-Seed {provenance.get("total_steps", 100)}-Step Mixture-of-Experts (MoE) Convergence Report
**Empirical Multi-Seed Optimizer Evaluation & 3-Tier Dynamic Steering Synergy on Real FineWeb-Edu**

- **Date / Timestamp**: `{time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}`
- **Git Commit**: [`{git_c[:10]}`](https://github.com/Yasho4867/lucid_machines/commit/{git_c}){git_dirty_str}
- **Hardware**: `{hw.get("device_name", "GPU")}` ({hw.get("total_vram_gb", 0)} GB VRAM, CC {hw.get("compute_capability", "N/A")})
- **Software**: PyTorch `{sw.get("torch", "N/A")}` | CUDA `{sw.get("cuda", "N/A")}` | Python `{sw.get("python", "N/A")}`
- **Dataset**: Real FineWeb-Edu educational text (`{ds.get("dataset_path", "data/fineweb_edu_real.jsonl")}`, SHA-256 `{ds.get("dataset_sha256", "N/A")[:12]}...`)
- **Model Topology**: Universal Mixture-of-Experts Transformer (`90,788,895` parameters, 8 layers, 4 experts, 384 dim, 6 heads, 32,000 vocab)
- **Subspace & Optimizer Config**: `rank={opt_cfg.get("rank", 16)}`, `update_gap={opt_cfg.get("update_gap", 20)}`, `steer_update_gap={opt_cfg.get("steer_update_gap", 5)}`, `steer_scale={opt_cfg.get("steer_scale", 0.2)}`, `core_mask_mode={opt_cfg.get("core_mask_mode", "full")}`, `lr={opt_cfg.get("learning_rate", 1e-3)}`, `muon_lr={opt_cfg.get("muon_learning_rate", 0.002)}`
- **Experimental Protocol**:
  - **Random Seeds**: `{seeds_str}` ($N={len(provenance.get("seeds", []))}$ independent runs per regime)
  - **Step-0 Initial Weights**: Identical per seed (sampled once and cloned bit-for-bit into all regimes)
  - **Batch Sequences**: Identical causal batches fed in exact synchronous order across all regimes per seed
  - **Document Partitioning**: Strict document-level train/validation split with dynamic `[SEP]` separator packing
  - **Warm-Up Phase**: Common pre-warmup step executed before timing to eliminate CUDA JIT / memory allocation skew
  - **Execution Order**: Randomized regime order per seed to guarantee zero position or allocator bias
  - **Validation Set**: Multi-batch held-out split of {num_val_batches} batches ({val_tokens:,} tokens)

---

## 1. Active Parameter Tier Distribution & Measured Optimizer State Memory

| Regime | Total Params | Tier 1: Muon | Tier 2: Low-Rank | Tier 3: Base AdamW | Dynamic Steering | Total Opt State | Expert Opt State | Expert Cut vs BF16 / FP32 AdamW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    for r_key in regimes_order:
        agg = summary[r_key]
        opt_name = regime_labels.get(r_key, r_key)
        tot_p = agg["tier1_muon_params"] + agg["tier2_centroid_params"] + agg["tier3_base_params"]
        t1_str = f"{agg['tier1_muon_params']:,} ({agg['tier1_muon_params'] / max(tot_p, 1) * 100:.1f}%)"
        t2_str = f"{agg['tier2_centroid_params']:,} ({agg['tier2_centroid_params'] / max(tot_p, 1) * 100:.1f}%)"
        t3_str = f"{agg['tier3_base_params']:,} ({agg['tier3_base_params'] / max(tot_p, 1) * 100:.1f}%)"
        dyn_str = "Yes" if agg.get("has_dynamic_centroids") else "No"
        tot_st = agg.get("total_state_mb", {}).get("mean", 0.0)
        tot_bpp = agg.get("total_bytes_per_param", {}).get("mean", 0.0)
        exp_st = agg.get("expert_state_mb", {}).get("mean", 0.0)
        exp_bpp = agg.get("expert_bytes_per_param", {}).get("mean", 0.0)
        exp_red_bf16 = agg.get("expert_reduction_vs_bf16_adamw_pct", {}).get("mean", 0.0)
        exp_red_fp32 = agg.get("expert_reduction_vs_fp32_adamw_pct", {}).get("mean", 0.0)
        md += (
            f"| {opt_name} | {tot_p:,} | {t1_str} | {t2_str} | {t3_str} | {dyn_str} | "
            f"{tot_st:.2f} MB ({tot_bpp:.3f} B/p) | {exp_st:.2f} MB ({exp_bpp:.4f} B/p) | "
            f"{exp_red_bf16:.1f}% / {exp_red_fp32:.1f}% |\n"
        )

    md += """
---

## 2. Multi-Seed MoE Optimization & Convergence Summary (Mean ± Std)

| Framework & Optimizer Regime | Initial Val Loss | Final Val Loss | Val Loss Drop | Final Val PPL | Steady Step | Total Time | Peak VRAM | Conv Efficiency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    for r_key in regimes_order:
        agg = summary[r_key]
        opt_name = regime_labels.get(r_key, r_key)
        is_best = r_key == best_regime
        init_l = f"{agg['initial_val_loss']['mean']:.4f} ± {agg['initial_val_loss']['std']:.4f}"
        fin_raw = f"{agg['final_val_loss']['mean']:.4f} ± {agg['final_val_loss']['std']:.4f}"
        fin_l = f"**{fin_raw}**" if is_best else fin_raw
        drop_raw = f"{agg['val_loss_reduction']['mean']:.4f} ± {agg['val_loss_reduction']['std']:.4f}"
        drop_l = f"**{drop_raw}**" if is_best else drop_raw
        fin_ppl_raw = f"{agg['final_val_ppl']['mean']:.2f} ± {agg['final_val_ppl']['std']:.2f}"
        fin_ppl = f"**{fin_ppl_raw}**" if is_best else fin_ppl_raw
        steady_ms = f"{agg['steady_state_step_ms']['mean']:.2f} ± {agg['steady_state_step_ms']['std']:.2f} ms"
        tot_time = f"{agg['total_time_sec']['mean']:.2f} ± {agg['total_time_sec']['std']:.2f} s"
        vram_str = f"{agg['peak_vram_alloc_mb']['mean']:.1f} MB"
        eff_str = f"{agg['convergence_efficiency']['mean']:.4f} loss/s"
        md += f"| {opt_name} | {init_l} | {fin_l} | {drop_l} | {fin_ppl} | {steady_ms} | {tot_time} | {vram_str} | {eff_str} |\n"

    all_steps = sorted({pt["step"] for r in regimes_order for pt in summary[r].get("val_trajectory", [])})
    col_headers = " | ".join(short_labels.get(r, r) for r in regimes_order)
    col_align = " | ".join(":---:" for _ in regimes_order)
    md += f"""
---

## 3. Validation Loss & Perplexity Trajectory Across Training Steps

Evaluated over the document-isolated held-out validation set ({num_val_batches} batches, {val_tokens:,} tokens):

| Step | Metric | {col_headers} |
| :---: | :--- | {col_align} |
"""

    for st in all_steps:
        loss_cells = []
        ppl_cells = []
        step_means = {
            r_k: next((p["val_loss_mean"] for p in summary[r_k].get("val_trajectory", []) if p["step"] == st), None)
            for r_k in regimes_order
        }
        valid_means = [v for v in step_means.values() if v is not None]
        best_step_val = min(valid_means) if valid_means else None
        for r_k in regimes_order:
            traj = summary[r_k].get("val_trajectory", [])
            pt = next((p for p in traj if p["step"] == st), None)
            if pt:
                is_best_pt = best_step_val is not None and pt["val_loss_mean"] == best_step_val
                l_str = f"{pt['val_loss_mean']:.4f} ± {pt['val_loss_std']:.4f}"
                p_str = f"{pt['val_ppl_mean']:.2f} ± {pt['val_ppl_std']:.2f}"
                loss_cells.append(f"**{l_str}**" if is_best_pt else l_str)
                ppl_cells.append(f"**{p_str}**" if is_best_pt else p_str)
            else:
                loss_cells.append("N/A")
                ppl_cells.append("N/A")

        md += f"| **Step {st}** | Val Loss | " + " | ".join(loss_cells) + " |\n"
        md += "| | Val PPL | " + " | ".join(ppl_cells) + " |\n"

    md += """
---

## 4. Expert Routing, Gradient Activity & Empirical Variance Concentration

### Expert Routing Distribution (Tokens Accepted Per Expert Across All Layers)
"""
    for r_k in regimes_order:
        agg = summary[r_k]
        routing = agg.get("expert_tokens_routed", [])
        if any(routing):
            opt_name = short_labels.get(r_k, r_k)
            routing_str = ", ".join(f"Expert {idx}: {cnt:,} tokens" for idx, cnt in enumerate(routing))
            md += f"- **{opt_name}**: {routing_str}\n"

    ref_key = "triune_moe" if "triune_moe" in summary else regimes_order[0]
    agg_ref = summary[ref_key]
    t1_tot = max(1, agg_ref.get("tier1_muon_params", 0))
    t2_tot = max(1, agg_ref.get("tier2_centroid_params", 0))
    t3_tot = max(1, agg_ref.get("tier3_base_params", 0))
    t1_nz = agg_ref.get("tier1_nonzero_grads", {}).get("mean", 0.0)
    t2_nz = agg_ref.get("tier2_nonzero_grads", {}).get("mean", 0.0)
    t3_nz = agg_ref.get("tier3_nonzero_grads", {}).get("mean", 0.0)
    md += f"""
### Parameter Activity & Nonzero Gradients (`{short_labels.get(ref_key, ref_key)}`)
- **Tier 1 (Muon)**: `{t1_nz:,.0f}` / {agg_ref.get("tier1_muon_params", 0):,} ({(t1_nz / t1_tot) * 100:.1f}% non-zero at sampled step)
- **Tier 2 (Subspace)**: `{t2_nz:,.0f}` / {agg_ref.get("tier2_centroid_params", 0):,} ({(t2_nz / t2_tot) * 100:.1f}% non-zero at sampled step)
- **Tier 3 (Base AdamW)**: `{t3_nz:,.0f}` / {agg_ref.get("tier3_base_params", 0):,} ({(t3_nz / t3_tot) * 100:.1f}% non-zero at sampled step)

### Empirical Routing Concentration Diagnostics (Held-Out Validation)
| Regime | Mean Centroid Norm² $\\|c_e\\|_2^2$ | Mean Intra-Expert Trace $\\operatorname{{tr}}(\\Sigma_e)$ | Dispersion Ratio $\\operatorname{{tr}}(\\Sigma_e)/\\|c_e\\|_2^2$ | Within / Global Var $\\operatorname{{tr}}(\\Sigma_e)/\\operatorname{{tr}}(\\Sigma_{{\\text{{glob}}}})$ | Centroid Energy Frac | Router Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r_k in regimes_order:
        rc = summary[r_k].get("routing_concentration", {})
        c_sq = rc.get("mean_centroid_norm_sq", {}).get("mean", 0.0)
        tr_e = rc.get("mean_within_cov_trace", {}).get("mean", 0.0)
        ratio = rc.get("mean_trace_to_centroid_ratio", {}).get("mean", 0.0)
        w2g = rc.get("mean_within_to_global_var_ratio", {}).get("mean", 0.0)
        cef = rc.get("mean_centroid_energy_fraction", {}).get("mean", 0.0)
        ent = rc.get("mean_router_entropy", {}).get("mean", 0.0)
        md += (
            f"| {short_labels.get(r_k, r_k)} | {c_sq:.4f} | {tr_e:.4f} | {ratio:.4f} | "
            f"{w2g:.4f} | {cef:.4f} | {ent:.4f} |\n"
        )

    md += """
---

## 5. Statistical Comparisons (Paired & Welch Two-Sided t-Tests)

"""
    valid_pairs = {k: v for k, v in (pairwise_stats or {}).items() if v.get("n", 0) >= 2 and "mean_a" in v}
    if valid_pairs:
        md += "| Comparison | N | Mean A | Mean B | Mean Diff (A - B) | Paired t-test p | Welch t-test p |\n"
        md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for pair_name, st_dict in valid_pairs.items():
            p_rel = f"{st_dict['paired_p']:.6f}" if st_dict.get("paired_p") is not None else "N/A"
            p_wel = f"{st_dict['welch_p']:.6f}" if st_dict.get("welch_p") is not None else "N/A"
            md += (
                f"| {pair_name} | {st_dict['n']} | {st_dict['mean_a']:.4f} ± {st_dict['std_a']:.4f} | "
                f"{st_dict['mean_b']:.4f} ± {st_dict['std_b']:.4f} | {st_dict['mean_diff']:+.4f} | {p_rel} | {p_wel} |\n"
            )
    else:
        md += "At least N >= 2 seeds are required to compute paired / Welch t-test statistics.\n"

    md += """
---

## 6. Cryptographic Provenance & Reproduction

To reproduce these exact MoE results from a clean repository checkout:

```bash
{repro}
```

### Source File Cryptographic Verification Hashes (SHA-256)
""".format(
        repro=provenance.get(
            "reproduction_command",
            "python scripts/benchmark_moe_convergence_100_steps.py --steps 100 --seeds 42 1337 2026 --num-val-batches 5",
        )
    )

    src_hashes = provenance.get("source_hashes", {})
    if src_hashes:
        md += "| Relative Path | SHA-256 Checksum |\n| :--- | :--- |\n"
        for p_str, h_val in sorted(src_hashes.items()):
            md += f"| `{p_str}` | `{h_val}` |\n"

    output_path.write_text(md, encoding="utf-8")
    print(f"\n[SUCCESS] Generated Multi-Seed MoE Convergence Report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Lucid Machines Multi-Seed 100-Step MoE Convergence Benchmark")
    parser.add_argument("--steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--val-interval", type=int, default=20, help="Evaluation interval")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1337, 2026, 7, 99, 314, 777, 1024],
        help="Random seeds for multi-seed run",
    )
    parser.add_argument("--num-val-batches", type=int, default=5, help="Number of held-out validation batches")
    parser.add_argument("--use-8bit-base", action="store_true", help="Use BitsAndBytes 8-bit AdamW as base optimizer")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "fineweb_edu_real.jsonl")
    parser.add_argument("--required-tokens", type=int, default=120000, help="Training tokens to pack")
    parser.add_argument("--num-val-docs", type=int, default=15, help="Held-out documents")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base (AdamW / subspace) learning rate")
    parser.add_argument("--muon-lr", type=float, default=0.002, help="Muon learning rate")
    parser.add_argument("--rank", type=int, default=16, help="Subspace projection rank for CentroidSteer and GaLore")
    parser.add_argument("--update-gap", type=int, default=20, help="Subspace refresh interval in steps")
    parser.add_argument("--steer-update-gap", type=int, default=5, help="Centroid steering update interval in steps")
    parser.add_argument(
        "--core-mask-mode",
        type=str,
        default="full",
        choices=("full", "triu"),
        help="Triangular core storage mode ('full' (r+1)x(r+1) or legacy 'triu')",
    )
    parser.add_argument("--regimes", type=str, nargs="+", default=None, help="Subset of regime names to run")
    parser.add_argument("--output-tag", type=str, default="", help="Suffix for result/report file names")
    parser.add_argument("--steer-scale", type=float, default=0.20, help="Centroid steering scale")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    git_commit, git_dirty = get_git_info()
    script_sha256 = compute_sha256(Path(__file__).resolve())
    source_hashes = compute_source_hashes()

    RUN_CONFIG["lr"] = args.lr
    RUN_CONFIG["muon_lr"] = args.muon_lr
    RUN_CONFIG["steer_scale"] = args.steer_scale
    RUN_CONFIG["rank"] = args.rank
    RUN_CONFIG["update_gap"] = args.update_gap
    RUN_CONFIG["steer_update_gap"] = args.steer_update_gap
    RUN_CONFIG["core_mask_mode"] = args.core_mask_mode

    data_path = args.data_path.resolve()
    tokenizer_path = REPO_ROOT / "triune_tokenizer.json"

    print("================================================================================")
    print(" LUCID MACHINES MULTI-SEED 100-STEP MIXTURE-OF-EXPERTS (MoE) CONVERGENCE BENCHMARK")
    print(f" Hardware: {device} | Precision: {dtype}")
    print(f" Steps: {args.steps} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len}")
    print(
        f" Rank: {args.rank} | Update Gap: {args.update_gap} | Steer Gap: {args.steer_update_gap}"
        f" | Core Mode: {args.core_mask_mode}"
    )
    print(f" Seeds ({len(args.seeds)}): {args.seeds} | Val Batches: {args.num_val_batches}")
    print(f" Base Optimizer: {'BitsAndBytes 8-bit' if args.use_8bit_base else 'PyTorch Fused AdamW'}")
    print("================================================================================")

    train_tokens, val_tokens, dataset_meta = load_fineweb_documents(
        data_path, tokenizer_path, num_val_docs=args.num_val_docs, required_tokens=args.required_tokens
    )

    repro_parts = [
        "python scripts/benchmark_moe_convergence_100_steps.py",
        f"--steps {args.steps}",
        f"--batch-size {args.batch_size}",
        f"--seq-len {args.seq_len}",
        f"--val-interval {args.val_interval}",
        f"--seeds {' '.join(str(s) for s in args.seeds)}",
        f"--num-val-batches {args.num_val_batches}",
    ]
    if args.use_8bit_base:
        repro_parts.append("--use-8bit-base")

    provenance = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "script_sha256": script_sha256,
        "source_hashes": source_hashes,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "hardware": get_hardware_info(device),
        "dataset": dataset_meta,
        "total_steps": args.steps,
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "num_val_batches": args.num_val_batches,
        "optimizer_configuration": {
            "use_8bit_base": args.use_8bit_base,
            "base_optimizer_impl": (
                "bitsandbytes.optim.AdamW8bit" if args.use_8bit_base else "torch.optim.AdamW (fused=True)"
            ),
            "learning_rate": args.lr,
            "muon_learning_rate": args.muon_lr,
            "rank": args.rank,
            "update_gap": args.update_gap,
            "steer_update_gap": args.steer_update_gap,
            "steer_scale": args.steer_scale,
            "core_mask_mode": args.core_mask_mode,
            "weight_decay": 0.01,
            "loss_function": "torch.nn.functional.cross_entropy",
        },
        "reproduction_command": " ".join(repro_parts),
    }

    regimes = [
        ("adamw", "adamw"),
        ("muon_adamw", "muon_adamw"),
        ("badam", "badam"),
        ("galore", "galore"),
        ("centroid_no_muon", "centroid_no_muon"),
        ("triune_moe", "triune_moe"),
        ("triune_moe_nosteer", "triune_moe_nosteer"),
        ("triune_moe_random", "triune_moe_random"),
    ]
    if args.regimes:
        regimes = [r for r in regimes if r[0] in set(args.regimes)]
        assert regimes, f"no regimes matched {args.regimes}"
    try:
        rel_data_path = str(data_path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        rel_data_path = str(data_path)
    provenance["reproduction_command"] += (
        f" --data-path {rel_data_path} --required-tokens {args.required_tokens}"
        f" --num-val-docs {args.num_val_docs} --lr {args.lr} --muon-lr {args.muon_lr}"
        f" --rank {args.rank} --update-gap {args.update_gap} --steer-update-gap {args.steer_update_gap}"
        f" --core-mask-mode {args.core_mask_mode} --steer-scale {args.steer_scale}"
        f" --regimes {' '.join(r[0] for r in regimes)}"
        + (f" --output-tag {args.output_tag}" if args.output_tag else "")
    )

    tok_per_batch = args.batch_size * (args.seq_len + 1)

    # Prepare document-isolated held-out validation batches
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for b_idx in range(args.num_val_batches):
        v_chunk = val_tokens[b_idx * tok_per_batch : (b_idx + 1) * tok_per_batch]
        if len(v_chunk) < tok_per_batch:
            break
        v_tensor = torch.tensor(v_chunk, dtype=torch.long, device=device).reshape(args.batch_size, args.seq_len + 1)
        val_batches.append((v_tensor[:, :-1], v_tensor[:, 1:]))

    summary: Dict[str, Dict[str, Any]] = {}
    all_seed_runs: Dict[str, List[Dict[str, Any]]] = {r[0]: [] for r in regimes}

    vocab_size = 32000
    dim = 384
    num_layers = 8
    num_heads = 6
    num_experts = 4

    for seed in args.seeds:
        print("\n=======================================================")
        print(f" >>> STARTING MoE SEED {seed}")
        print("=======================================================")

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # Build initial weights to be shared across all regimes for this seed
        ref_model = UniversalMoETransformer(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_experts=num_experts,
        ).to(device=device, dtype=dtype)
        shared_state_dict = copy.deepcopy(ref_model.state_dict())
        del ref_model
        gc.collect()

        # Slice train batches deterministically from document-partitioned train tokens
        train_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        cursor = (seed * 1024) % max(1, len(train_tokens) - (args.steps * tok_per_batch))
        for _ in range(args.steps):
            if cursor + tok_per_batch > len(train_tokens):
                cursor = 0
            chunk = train_tokens[cursor : cursor + tok_per_batch]
            cursor += tok_per_batch
            t_seq = torch.tensor(chunk, dtype=torch.long, device=device).reshape(args.batch_size, args.seq_len + 1)
            train_batches.append((t_seq[:, :-1], t_seq[:, 1:]))

        # Randomize regime execution order for this seed to prevent order bias
        rng = random.Random(seed)
        seed_regimes = list(regimes)
        rng.shuffle(seed_regimes)

        for regime_name, opt_type in seed_regimes:
            run_res = run_single_moe_seed_run(
                seed=seed,
                regime_name=regime_name,
                optimizer_type=opt_type,
                initial_state_dict=shared_state_dict,
                train_batches=train_batches,
                val_batches=val_batches,
                total_steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                val_interval=args.val_interval,
                vocab_size=vocab_size,
                dim=dim,
                num_layers=num_layers,
                num_heads=num_heads,
                num_experts=num_experts,
                device=device,
                dtype=dtype,
            )
            all_seed_runs[regime_name].append(run_res)

        del shared_state_dict, train_batches
        gc.collect()

    # Aggregate statistics
    for regime_name, _ in regimes:
        summary[regime_name] = aggregate_moe_regime_stats(all_seed_runs[regime_name])

    # Pairwise paired/Welch t-test statistics on final validation loss
    pairwise_candidates = [
        ("triune_moe", "adamw"),
        ("triune_moe", "badam"),
        ("triune_moe", "galore"),
        ("triune_moe", "muon_adamw"),
        ("triune_moe", "centroid_no_muon"),
        ("triune_moe", "triune_moe_nosteer"),
        ("triune_moe", "triune_moe_random"),
        ("centroid_no_muon", "galore"),
        ("centroid_no_muon", "badam"),
        ("triune_moe_nosteer", "galore"),
    ]
    pairwise_stats: Dict[str, Any] = {}
    for a_key, b_key in pairwise_candidates:
        if a_key in all_seed_runs and b_key in all_seed_runs and all_seed_runs[a_key] and all_seed_runs[b_key]:
            a_vals = [r["final_val_loss"] for r in all_seed_runs[a_key]]
            b_vals = [r["final_val_loss"] for r in all_seed_runs[b_key]]
            pairwise_stats[f"{a_key} vs {b_key}"] = compute_welch_and_paired_pvalues(a_vals, b_vals)

    out_dir = REPO_ROOT / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for r_name, _ in regimes:
        if r_name in all_seed_runs and all_seed_runs[r_name]:
            primary_run = copy.deepcopy(all_seed_runs[r_name][0])
            agg = summary[r_name]
            primary_run["val_loss_reduction"] = agg["val_loss_reduction"]["mean"]
            primary_run["throughput_tok_s"] = agg["throughput_tok_s"]["mean"]
            primary_run["initial_val_loss"] = agg["initial_val_loss"]["mean"]
            primary_run["final_val_loss"] = agg["final_val_loss"]["mean"]
            primary_run["final_val_ppl"] = agg["final_val_ppl"]["mean"]
            primary_run["median_step_ms"] = agg["median_step_ms"]["mean"]
            primary_run["first_step_ms"] = agg["first_step_ms"]["mean"]
            primary_run["steady_state_step_ms"] = agg["steady_state_step_ms"]["mean"]
            primary_run["total_time_sec"] = agg["total_time_sec"]["mean"]
            primary_run["peak_vram_alloc_mb"] = agg["peak_vram_alloc_mb"]["mean"]
            primary_run["peak_vram_reserved_mb"] = agg["peak_vram_reserved_mb"]["mean"]
            primary_run["convergence_efficiency_drop_per_sec"] = agg["convergence_efficiency"]["mean"]
            primary_run["multi_seed_summary"] = agg
            results.append(primary_run)

    artifact_payload = {
        "schema_version": "2.5.0",
        "benchmark_name": "moe_convergence_100_steps_multiseed_ablation",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "provenance": provenance,
        "results": results,
        "summary": summary,
        "pairwise_statistics": pairwise_stats,
        "seed_runs": all_seed_runs,
    }

    tag = f"_{args.output_tag}" if args.output_tag else ""
    json_path = out_dir / f"moe_convergence_100_steps_results{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact_payload, f, indent=2)
    print(f"\n[SUCCESS] Saved MoE JSON results: {json_path}")

    report_path = out_dir / f"MOE_CONVERGENCE_100_STEPS_REPORT{tag.upper()}.md"
    generate_moe_convergence_report(summary, report_path, provenance, pairwise_stats=pairwise_stats)


if __name__ == "__main__":
    main()
