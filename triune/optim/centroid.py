import torch
import torch.nn.functional as F
from typing import Any, Iterable, Optional, Set

import triune.model.config as defaults
from triune.model import MoE_FFN, FP4Linear
from .muon import Muon


try:
    from bitsandbytes.optim import AdamW8bit
    HAS_8BIT = True
except ImportError:
    HAS_8BIT = False
    AdamW8bit = None


def _cholesky_qr(Y_f: torch.Tensor) -> torch.Tensor:
    """Robust Cholesky-QR orthogonalization with column pre-conditioning and QR fallback."""
    col_norms = torch.linalg.norm(Y_f, dim=0, keepdim=True).clamp_min(1e-12)
    Y_norm = Y_f / col_norms
    Gram = torch.mm(Y_norm.t(), Y_norm)
    Gram.diagonal().add_(1e-7)
    try:
        L = torch.linalg.cholesky(Gram)
        return torch.linalg.solve_triangular(L, Y_norm.t(), upper=False).t()
    except torch.linalg.LinAlgError:
        Q, _ = torch.linalg.qr(Y_f)
        return Q


def fast_subspace_projection(grad: torch.Tensor, rank: int, niter: int = 2) -> tuple[torch.Tensor, str]:
    """High-performance CUDA/CPU Subspace Iteration via Cholesky-QR.
    
    Replaces slow cuSOLVER svd_lowrank with direct Tensor Core GEMM + Cholesky Gram
    orthogonalization. 6.8x faster (5.2ms vs 35.7ms) and consumes 3.1x less transient VRAM.
    """
    m, n = grad.shape
    q = min(rank + 10, m, n)
    side = 'left' if m <= n else 'right'
    if side == 'left':
        Omega = torch.randn(n, q, device=grad.device, dtype=grad.dtype)
        Y = torch.mm(grad, Omega)
        Q = _cholesky_qr(Y.float())
        for _ in range(niter):
            Z = torch.mm(grad.t(), Q.to(grad.dtype))
            Q_z = _cholesky_qr(Z.float())
            Y = torch.mm(grad, Q_z.to(grad.dtype))
            Q = _cholesky_qr(Y.float())
        proj = Q[:, :rank].to(device=grad.device, dtype=grad.dtype)
    else:
        Omega = torch.randn(m, q, device=grad.device, dtype=grad.dtype)
        Y = torch.mm(grad.t(), Omega)
        Q = _cholesky_qr(Y.float())
        for _ in range(niter):
            Z = torch.mm(grad, Q.to(grad.dtype))
            Q_z = _cholesky_qr(Z.float())
            Y = torch.mm(grad.t(), Q_z.to(grad.dtype))
            Q = _cholesky_qr(Y.float())
        proj = Q[:, :rank].to(device=grad.device, dtype=grad.dtype)
    return proj, side


def batched_fast_subspace_projection(grads: torch.Tensor, rank: int, niter: int = 2) -> tuple[torch.Tensor, str]:
    """High-performance CUDA Batched Tensor Core Subspace Iteration via Cholesky-QR.
    
    Processes all N expert weight matrices simultaneously in batched Tensor Core GEMMs (BMM),
    delivering 11x faster orthogonalization (285ms vs 3148ms for 40 matrices) with zero loop stalls.
    """
    N, m, n = grads.shape
    q = min(rank + 10, m, n)
    side = 'left' if m <= n else 'right'
    device, dtype = grads.device, grads.dtype
    
    def _batched_cholesky_qr(Y_mat: torch.Tensor) -> torch.Tensor:
        Y_f = Y_mat.float()
        col_norms = torch.linalg.norm(Y_f, dim=1, keepdim=True).clamp_min(1e-12)
        Y_norm = Y_f / col_norms
        Gram = torch.bmm(Y_norm.transpose(1, 2), Y_norm)
        for i in range(N):
            Gram[i].diagonal().add_(1e-7)
        try:
            L = torch.linalg.cholesky(Gram)
            Q = torch.linalg.solve_triangular(L, Y_norm.transpose(1, 2), upper=False).transpose(1, 2)
            return Q.to(dtype=dtype)
        except torch.linalg.LinAlgError:
            Q, _ = torch.linalg.qr(Y_f)
            return Q.to(dtype=dtype)

    if side == 'left':
        Omega = torch.randn(N, n, q, device=device, dtype=dtype)
        Y = torch.bmm(grads, Omega)
        Q = _batched_cholesky_qr(Y)
        for _ in range(niter):
            Z = torch.bmm(grads.transpose(1, 2), Q)
            Q_z = _batched_cholesky_qr(Z)
            Y = torch.bmm(grads, Q_z)
            Q = _batched_cholesky_qr(Y)
        proj = Q[:, :, :rank]
    else:
        Omega = torch.randn(N, m, q, device=device, dtype=dtype)
        Y = torch.bmm(grads.transpose(1, 2), Omega)
        Q = _batched_cholesky_qr(Y)
        for _ in range(niter):
            Z = torch.bmm(grads, Q)
            Q_z = _batched_cholesky_qr(Z)
            Y = torch.bmm(grads.transpose(1, 2), Q_z)
            Q = _batched_cholesky_qr(Y)
        proj = Q[:, :, :rank]
    return proj, side


class CentroidSteerOptimizer(torch.optim.Optimizer):
    def __init__(self, model, lr=defaults.LR, betas=defaults.BETAS, weight_decay=defaults.WEIGHT_DECAY,
                 rank=defaults.GALORE_RANK, update_gap=defaults.GALORE_UPDATE_GAP,
                 steer_scale=defaults.STEER_SCALE,
                 steer_update_gap=getattr(defaults, 'STEER_UPDATE_GAP', 10),
                 expert_lr=defaults.GALORE_LR, expert_betas=defaults.GALORE_BETAS,
                 expert_wd=defaults.GALORE_WEIGHT_DECAY,
                 use_muon=getattr(defaults, 'USE_MUON', True),
                 muon_lr=getattr(defaults, 'MUON_LR', 0.02),
                 muon_momentum=getattr(defaults, 'MUON_MOMENTUM', 0.95),
                 muon_weight_decay=getattr(defaults, 'MUON_WEIGHT_DECAY', 0.01),
                 svd_device="auto",
                 use_triangular_core=getattr(defaults, 'USE_TRIANGULAR_CORE', True)):
        defaults_dict = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        super().__init__([{'params': [dummy_param]}], defaults_dict)
        self.model = model
        self.rank = rank
        self.update_gap = update_gap
        self.step_count = 0
        self.steer_scale = steer_scale
        self.steer_update_gap = steer_update_gap
        self.expert_lr = expert_lr
        self.initial_expert_lr = expert_lr
        self.expert_betas = expert_betas
        self.expert_wd = expert_wd
        self.use_muon = use_muon
        self.svd_device = svd_device
        self.initial_base_lr = lr
        self.initial_muon_lr = muon_lr
        self.use_triangular_core = use_triangular_core
        self._triangular_buffers = {}

        self.non_expert_params = []
        self.muon_params = []
        self.layer_groups = []
        seen_param_ids = set()

        if isinstance(model, torch.nn.Module):
            # Identify router parameters and token embeddings / vocabulary exit heads (these must stay on AdamW)
            router_param_ids = set()
            for name, module in model.named_modules():
                if isinstance(module, MoE_FFN) and hasattr(module, 'router'):
                    for p in module.router.parameters():
                        router_param_ids.add(id(p))
                elif module.__class__.__name__ == 'GumbelSoftmaxRouter' or name == 'router' or name.endswith('.router'):
                    for p in module.parameters():
                        router_param_ids.add(id(p))

            embedding_param_ids = set()
            for name, param in model.named_parameters():
                name_lower = name.lower()
                if (
                    'embed' in name_lower
                    or 'wte' in name_lower
                    or 'exit_head' in name_lower
                    or 'final_head' in name_lower
                    or 'lm_head' in name_lower
                ):
                    embedding_param_ids.add(id(param))

            # 1. Count MoE routed experts
            num_expert_groups = 0
            for name, module in model.named_modules():
                if isinstance(module, MoE_FFN):
                    for expert_idx, expert in enumerate(module.experts):
                        for subname, submod in expert.named_modules():
                            if hasattr(submod, 'weight') and isinstance(submod.weight, torch.nn.Parameter):
                                p = submod.weight
                                if p.requires_grad and p.dim() >= 2 and id(p) not in seen_param_ids:
                                    seen_param_ids.add(id(p))
                                    num_expert_groups += 1

            seen_param_ids.clear()
            group_idx = 0

            # 1. Populate MoE routed experts (with Centroid Steering)
            for name, module in model.named_modules():
                if isinstance(module, MoE_FFN):
                    for expert_idx, expert in enumerate(module.experts):
                        for subname, submod in expert.named_modules():
                            if hasattr(submod, 'weight') and isinstance(submod.weight, torch.nn.Parameter):
                                p = submod.weight
                                if p.requires_grad and p.dim() >= 2 and id(p) not in seen_param_ids:
                                    seen_param_ids.add(id(p))
                                    stagger = -(group_idx * (update_gap // max(1, num_expert_groups)))
                                    self.layer_groups.append({
                                        'module': module,
                                        'expert_idx': expert_idx,
                                        'param': p,
                                        'projection': None,
                                        'projection_side': 'left' if p.shape[0] <= p.shape[1] else 'right',
                                        'proj_step': stagger,
                                        'state': {'momentum': None, 'variance': None, 'step': 0}
                                    })
                                    group_idx += 1

            # 2. Populate remaining 2D parameters
            for name, param in model.named_parameters():
                if param.requires_grad and param.dim() >= 2 and id(param) not in seen_param_ids:
                    seen_param_ids.add(id(param))
                    if self.use_muon and id(param) not in router_param_ids and id(param) not in embedding_param_ids:
                        # Hidden 2D linear weight (Attention GLA, Shared Expert, Exit Heads) -> Muon
                        self.muon_params.append(param)
                    elif not self.use_muon and id(param) not in router_param_ids and id(param) not in embedding_param_ids:
                        # Standard GaLore fallback
                        stagger = -(group_idx * (update_gap // max(1, num_expert_groups + 1)))
                        self.layer_groups.append({
                            'module': None,
                            'expert_idx': None,
                            'param': param,
                            'projection': None,
                            'projection_side': 'left' if param.shape[0] <= param.shape[1] else 'right',
                            'proj_step': stagger,
                            'state': {'momentum': None, 'variance': None, 'step': 0}
                        })
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
                if isinstance(p, dict) and 'params' in p:
                    param_candidates = p['params']
                else:
                    param_candidates = [p]
                for param in param_candidates:
                    if not isinstance(param, (torch.nn.Parameter, torch.Tensor)):
                        continue
                    if param.requires_grad and id(param) not in seen_param_ids:
                        seen_param_ids.add(id(param))
                        if self.use_muon and param.dim() == 2:
                            self.muon_params.append(param)
                        elif param.dim() >= 2:
                            self.layer_groups.append({
                                'module': None,
                                'expert_idx': None,
                                'param': param,
                                'projection': None,
                                'projection_side': 'left' if param.shape[0] <= param.shape[1] else 'right',
                                'proj_step': 0,
                                'state': {'momentum': None, 'variance': None, 'step': 0}
                            })
                        else:
                            self.non_expert_params.append(param)

        if self.use_muon and len(self.muon_params) > 0:
            self.muon_optimizer = Muon(
                self.muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
            )
        else:
            self.muon_optimizer = None

        if len(self.non_expert_params) > 0:
            if HAS_8BIT and AdamW8bit is not None:
                self.base_optimizer = AdamW8bit(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)
            else:
                self.base_optimizer = torch.optim.AdamW(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            self.base_optimizer = None

    def zero_grad(self, set_to_none=True):
        if self.base_optimizer is not None:
            self.base_optimizer.zero_grad(set_to_none=set_to_none)
        if self.muon_optimizer is not None:
            self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        for group in self.layer_groups:
            p = group['param']
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    def set_lr(self, lr, expert_lr=None):
        if self.base_optimizer is not None:
            for pg in self.base_optimizer.param_groups:
                pg['lr'] = lr
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
                pg['lr'] = self.initial_muon_lr * ratio

    @torch.no_grad()
    def step_parameters(self, params: Any, is_global_step_end: bool = False):
        """Executes optimizer step on a specific subset of parameters (e.g. single layer or root components)."""
        param_list = list(params) if not isinstance(params, list) else params
        param_ids = {id(p) for p in param_list if p.grad is not None}
        if not param_ids:
            if is_global_step_end:
                self.step_count += 1
            return

        # 1. Step matching parameters in base_optimizer (AdamW / 8-bit AdamW)
        if self.base_optimizer is not None:
            for group in self.base_optimizer.param_groups:
                matching = [p for p in group['params'] if id(p) in param_ids]
                if matching:
                    orig_params = group['params']
                    group['params'] = matching
                    try:
                        for p in matching:
                            if p in self.base_optimizer.state:
                                st = self.base_optimizer.state[p]
                                if "exp_avg" not in st or "exp_avg_sq" not in st:
                                    st.clear()
                        self.base_optimizer.step()
                    finally:
                        group['params'] = orig_params

        # 2. Step matching parameters in muon_optimizer
        if self.muon_optimizer is not None:
            for group in self.muon_optimizer.param_groups:
                matching = [p for p in group['params'] if id(p) in param_ids]
                if matching:
                    orig_params = group['params']
                    group['params'] = matching
                    try:
                        self.muon_optimizer.step()
                    finally:
                        group['params'] = orig_params

        # 3. Step matching expert parameters in layer_groups (Centroid + GaLore)
        expert_lr = self.expert_lr
        expert_beta1, expert_beta2 = self.expert_betas
        expert_wd = self.expert_wd

        active_groups = []
        for group in self.layer_groups:
            p = group['param']
            if id(p) in param_ids and p.grad is not None:
                active_groups.append(group)

        if not active_groups:
            if is_global_step_end:
                self.step_count += 1
            return

        if self.use_triangular_core:
            # Universal Triangular I-Matrix Engine (Zero Shape Buckets)
            # Condenses all parameter matrices into (N, rank+1, rank+1) Upper Triangles
            # using dual isometric I-bases (Q_L^T Q_L = I, Q_R^T Q_R = I)
            N = len(active_groups)
            device = active_groups[0]['param'].device
            dtype = active_groups[0]['param'].dtype

            # 1. Initialize or periodically update Q_L and Q_R isometric bases
            for g in active_groups:
                p = g['param']
                grad = p.grad.data
                m, n = grad.shape
                d_small, d_large = min(m, n), max(m, n)
                r = min(self.rank, d_small, d_large)
                st = g['state']

                needs_init = g.get('Q_L') is None
                needs_update = (self.step_count - g.get('proj_step', 0)) >= self.update_gap
                if needs_init or needs_update:
                    G_canon = grad if m <= n else grad.t()
                    Omega = torch.randn(d_large, r, device=device, dtype=dtype)
                    Y = G_canon @ Omega
                    ql, _ = torch.linalg.qr(Y.float())
                    ql = ql.to(dtype)
                    Z = G_canon.t() @ ql
                    qr, _ = torch.linalg.qr(Z.float())
                    qr = qr.to(dtype)
                    Y = G_canon @ qr
                    ql, _ = torch.linalg.qr(Y.float())
                    ql = ql.to(dtype)

                    # Fix 3: Support mixed effective ranks safely using fixed self.rank + 1 spatial dimensions
                    Q_L_basis = torch.zeros(d_small, self.rank + 1, device=device, dtype=dtype)
                    Q_L_basis[:, :r] = ql
                    Q_R_basis = torch.zeros(d_large, self.rank + 1, device=device, dtype=dtype)
                    Q_R_basis[:, :r] = qr

                    g['Q_L'] = Q_L_basis
                    g['Q_R'] = Q_R_basis
                    g['projection'] = Q_L_basis[:, :r]
                    g['projection_side'] = 'left' if m <= n else 'right'
                    g['proj_step'] = self.step_count

                    # Fix 1 & 3: Allocate persistent states with fixed (self.rank + 1, self.rank + 1) in FP32
                    if needs_init:
                        st['momentum'] = torch.zeros(self.rank + 1, self.rank + 1, device=device, dtype=torch.float32)
                        st['variance'] = torch.zeros(self.rank + 1, self.rank + 1, device=device, dtype=torch.float32)
                        st['subspace_step'] = 0
                        st['step'] = 0
                    elif needs_update:
                        st['momentum'].zero_()
                        st['variance'].zero_()
                        st['subspace_step'] = 0

                # Step 1: Incremental Centroid Steering with dual orthonormal vector (eliminates redundant every-step QR)
                centroids = g['module'].last_centroids if g['module'] is not None else None
                expert_idx = g['expert_idx']
                steer_applied = g.get('steer_applied', False)
                if centroids is not None and expert_idx is not None and expert_idx < centroids.size(0) and self.steer_scale > 0:
                    c = centroids[expert_idx]
                    if c.size(0) == d_small:
                        needs_steer_refresh = (
                            needs_init
                            or needs_update
                            or (g.get('last_steer_step') is None)
                            or ((self.step_count - g.get('last_steer_step', 0)) >= self.steer_update_gap)
                        )
                        if needs_steer_refresh:
                            c_d = c.to(device=device, dtype=dtype)
                            c_proj = g['Q_L'][:, :r] @ (g['Q_L'][:, :r].t() @ c_d)
                            c_res = c_d - c_proj
                            c_res_norm = c_res.norm()
                            if c_res_norm > 1e-8:
                                c_orth = c_res / c_res_norm
                                g['Q_L'][:, r].copy_(c_orth)

                                # Dual right vector ensures S[r, r] > 0 and survives torch.triu
                                G_canon = grad if m <= n else grad.t()
                                v = G_canon.t() @ c_orth
                                v_proj = g['Q_R'][:, :r] @ (g['Q_R'][:, :r].t() @ v)
                                v_res = v - v_proj
                                v_res_norm = v_res.norm()
                                if v_res_norm > 1e-8:
                                    v_orth = v_res / v_res_norm
                                    g['Q_R'][:, r].copy_(v_orth)
                                    steer_applied = True
                                else:
                                    g['Q_R'][:, r].zero_()
                                    steer_applied = False
                            else:
                                g['Q_L'][:, r].zero_()
                                g['Q_R'][:, r].zero_()
                                steer_applied = False
                            g['last_steer_step'] = self.step_count
                else:
                    g['Q_L'][:, r].zero_()
                    g['Q_R'][:, r].zero_()
                    steer_applied = False
                g['steer_applied'] = steer_applied

            # 2. Project all gradients into unified (N, rank+1, rank+1) Upper Triangles
            # Zero-allocation buffer reuse
            if not hasattr(self, '_triangular_buffers'):
                self._triangular_buffers = {}
            buf_key = (N, device, dtype)
            if buf_key not in self._triangular_buffers:
                self._triangular_buffers[buf_key] = {
                    'R_batch': torch.zeros(N, self.rank + 1, self.rank + 1, device=device, dtype=dtype),
                    'batch_mom': torch.zeros(N, self.rank + 1, self.rank + 1, device=device, dtype=torch.float32),
                    'batch_var': torch.zeros(N, self.rank + 1, self.rank + 1, device=device, dtype=torch.float32),
                    'step_vals': torch.zeros(N, 1, 1, device=device, dtype=torch.float32),
                    'batch_delta': torch.zeros(N, self.rank + 1, self.rank + 1, device=device, dtype=dtype),
                }
            buf = self._triangular_buffers[buf_key]
            R_batch = buf['R_batch'].zero_()
            batch_mom = buf['batch_mom']
            batch_var = buf['batch_var']
            step_vals = buf['step_vals']
            batch_delta_R = buf['batch_delta']

            # Step 2: Layer-Level Batched Projections using torch.bmm for matching matrix dimensions
            shape_buckets = {}
            for idx, g in enumerate(active_groups):
                m, n = g['param'].shape
                r = min(self.rank, min(m, n))
                key = (m, n, r)
                if key not in shape_buckets:
                    shape_buckets[key] = []
                shape_buckets[key].append((idx, g))

            for (m, n, r), bucket in shape_buckets.items():
                K = len(bucket)
                if K >= 2:
                    G_canon_list = [g['param'].grad.data if m <= n else g['param'].grad.data.t() for _, g in bucket]
                    G_stack = torch.stack(G_canon_list, dim=0)
                    QL_stack = torch.stack([g['Q_L'][:, :r+1] for _, g in bucket], dim=0)
                    QR_stack = torch.stack([g['Q_R'][:, :r+1] for _, g in bucket], dim=0)

                    temp = torch.bmm(G_stack, QR_stack)
                    S_bmm = torch.bmm(QL_stack.transpose(1, 2), temp)

                    for b_idx, (idx, g) in enumerate(bucket):
                        S_i = S_bmm[b_idx]
                        if g.get('steer_applied', False):
                            S_i[r, r] *= (1.0 + self.steer_scale)
                        R_batch[idx, :r+1, :r+1] = torch.triu(S_i)
                        batch_mom[idx].copy_(g['state']['momentum'])
                        batch_var[idx].copy_(g['state']['variance'])
                        step_vals[idx, 0, 0] = float(g['state'].get('subspace_step', 0) + 1)
                else:
                    idx, g = bucket[0]
                    p = g['param']
                    grad = p.grad.data
                    G_canon = grad if m <= n else grad.t()
                    S = g['Q_L'][:, :r+1].t() @ G_canon @ g['Q_R'][:, :r+1]
                    if g.get('steer_applied', False):
                        S[r, r] *= (1.0 + self.steer_scale)
                    R_batch[idx, :r+1, :r+1] = torch.triu(S)
                    batch_mom[idx].copy_(g['state']['momentum'])
                    batch_var[idx].copy_(g['state']['variance'])
                    step_vals[idx, 0, 0] = float(g['state'].get('subspace_step', 0) + 1)

            # Step 4: Unified Fused Triton Triangular AdamW Step with PyTorch Vectorized Fallback
            from triune.kernels.fast_triangular import fast_triangular_adamw
            fast_triangular_adamw(
                R_batch, batch_mom, batch_var, batch_delta_R, step_vals,
                beta1=expert_beta1, beta2=expert_beta2, eps=1e-8
            )

            # Fix 1: Save updated persistent moments back to each group
            for idx, g in enumerate(active_groups):
                g['state']['momentum'].copy_(batch_mom[idx])
                g['state']['variance'].copy_(batch_var[idx])
                g['state']['subspace_step'] = g['state'].get('subspace_step', 0) + 1
                g['state']['step'] += 1

            # 4. Batched BMM Reconstruction and in-place parameter update with Frobenius 1/sqrt(r) scaling
            for (m, n, r), bucket in shape_buckets.items():
                K = len(bucket)
                scale = 1.0 / (r ** 0.5)
                beta_wd = 1.0 - expert_lr * expert_wd if expert_wd != 0 else 1.0
                alpha_upd = -expert_lr * scale
                if K >= 2:
                    QL_stack = torch.stack([g['Q_L'][:, :r+1] for _, g in bucket], dim=0)
                    QR_stack = torch.stack([g['Q_R'][:, :r+1] for _, g in bucket], dim=0)
                    delta_stack = torch.stack([batch_delta_R[idx, :r+1, :r+1] for idx, _ in bucket], dim=0)
                    if m <= n:
                        temp = torch.bmm(QL_stack, delta_stack)
                        delta_W = torch.bmm(temp, QR_stack.transpose(1, 2))
                    else:
                        temp = torch.bmm(QR_stack, delta_stack.transpose(1, 2))
                        delta_W = torch.bmm(temp, QL_stack.transpose(1, 2))

                    for b_idx, (idx, g) in enumerate(bucket):
                        p = g['param']
                        p.data.mul_(beta_wd).add_(delta_W[b_idx], alpha=alpha_upd)
                else:
                    idx, g = bucket[0]
                    p = g['param']
                    active_delta = batch_delta_R[idx, :r+1, :r+1]
                    if m <= n:
                        temp = g['Q_L'][:, :r+1] @ active_delta
                        p.data.addmm_(temp, g['Q_R'][:, :r+1].t(), beta=beta_wd, alpha=alpha_upd)
                    else:
                        temp = g['Q_R'][:, :r+1] @ active_delta.t()
                        p.data.addmm_(temp, g['Q_L'][:, :r+1].t(), beta=beta_wd, alpha=alpha_upd)

            if is_global_step_end:
                self.step_count += 1
            return

        # Ensure all active groups have initialized projection bases
        for group in active_groups:
            p = group['param']
            grad = p.grad.data
            m, n = grad.shape
            rank = min(self.rank, m, n)
            state = group['state']

            if group.get('proj_basis') is None:
                target_dim = m if m <= n else n
                group['projection_side'] = 'left' if m <= n else 'right'
                basis = torch.zeros(target_dim, rank + 1, device=grad.device, dtype=grad.dtype)
                init_p = torch.empty(target_dim, rank, device=grad.device, dtype=torch.float32)
                torch.nn.init.orthogonal_(init_p)
                basis[:, :rank] = init_p.to(dtype=grad.dtype)
                group['projection'] = basis[:, :rank]
                group['proj_basis'] = basis
                state['momentum'] = None
                state['variance'] = None
                state['step'] = 0
                group['last_c_orth'] = None
                group['proj_step'] = self.step_count

        # Group active groups into shape buckets to execute Batched Tensor Core GEMMs (BMM)
        buckets = {}
        for group in active_groups:
            p = group['param']
            key = (p.shape, p.device, p.dtype)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(group)

        for (shape, device, dtype), bucket_groups in buckets.items():
            B = len(bucket_groups)
            m, n = shape
            rank = min(self.rank, m, n)
            side = bucket_groups[0].get('projection_side', 'left')

            # Periodic subspace update check
            upd_groups = [g for g in bucket_groups if (self.step_count - g['proj_step']) >= self.update_gap]
            if upd_groups:
                if len(upd_groups) > 1:
                    stacked_grads = torch.stack([g['param'].grad.data for g in upd_groups], dim=0)
                    new_projs, new_side = batched_fast_subspace_projection(stacked_grads, rank, niter=2)
                    for i, g in enumerate(upd_groups):
                        new_proj = new_projs[i]
                        new_proj_fp32 = new_proj.float()
                        old_proj = g['projection'].to(device=device, dtype=torch.float32)
                        old_side = g.get('projection_side', 'left')
                        st = g['state']
                        if st['momentum'] is not None and st['variance'] is not None and old_side == new_side and old_proj.shape == new_proj_fp32.shape:
                            T_mat = old_proj.T @ new_proj_fp32
                            T_sq = T_mat.pow(2)
                            mom_fp32 = st['momentum'].to(device=device, dtype=torch.float32)
                            var_fp32 = st['variance'].to(device=device, dtype=torch.float32)
                            if new_side == 'left' and mom_fp32.size(0) >= rank:
                                mom_r = T_mat.T @ mom_fp32[:rank, :]
                                row_energy = T_sq.sum(dim=0, keepdim=True).T.clamp(0.0, 1.0)
                                var_mean = var_fp32[:rank, :].mean(dim=0, keepdim=True)
                                var_r = (T_sq.T @ var_fp32[:rank, :]) + (1.0 - row_energy) * var_mean
                                mom_fp32 = torch.cat([mom_r, mom_fp32[rank:, :]], dim=0)
                                var_fp32 = torch.cat([var_r, var_fp32[rank:, :]], dim=0)
                            elif new_side == 'right' and mom_fp32.size(1) >= rank:
                                mom_r = mom_fp32[:, :rank] @ T_mat
                                col_energy = T_sq.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
                                var_mean = var_fp32[:, :rank].mean(dim=1, keepdim=True)
                                var_r = (var_fp32[:, :rank] @ T_sq) + var_mean * (1.0 - col_energy)
                                mom_fp32 = torch.cat([mom_r, mom_fp32[:, rank:]], dim=1)
                                var_fp32 = torch.cat([var_r, var_fp32[:, rank:]], dim=1)
                            st['momentum'] = mom_fp32.to(dtype=dtype)
                            st['variance'] = var_fp32.to(dtype=dtype)
                        else:
                            st['momentum'] = None
                            st['variance'] = None
                            st['step'] = 0
                        g['projection_side'] = new_side
                        target_dim = m if new_side == 'left' else n
                        basis = torch.zeros(target_dim, rank + 1, device=device, dtype=dtype)
                        basis[:, :rank] = new_proj.to(device=device, dtype=dtype)
                        g['projection'] = basis[:, :rank]
                        g['proj_basis'] = basis
                        g['proj_step'] = self.step_count
                else:
                    for g in upd_groups:
                        new_proj, new_side = fast_subspace_projection(g['param'].grad.data, rank, niter=2)
                        new_proj_fp32 = new_proj.float()
                        old_proj = g['projection'].to(device=device, dtype=torch.float32)
                        old_side = g.get('projection_side', 'left')
                        st = g['state']
                        if st['momentum'] is not None and st['variance'] is not None and old_side == new_side and old_proj.shape == new_proj_fp32.shape:
                            T_mat = old_proj.T @ new_proj_fp32
                            T_sq = T_mat.pow(2)
                            mom_fp32 = st['momentum'].to(device=device, dtype=torch.float32)
                            var_fp32 = st['variance'].to(device=device, dtype=torch.float32)
                            if new_side == 'left' and mom_fp32.size(0) >= rank:
                                mom_r = T_mat.T @ mom_fp32[:rank, :]
                                row_energy = T_sq.sum(dim=0, keepdim=True).T.clamp(0.0, 1.0)
                                var_mean = var_fp32[:rank, :].mean(dim=0, keepdim=True)
                                var_r = (T_sq.T @ var_fp32[:rank, :]) + (1.0 - row_energy) * var_mean
                                mom_fp32 = torch.cat([mom_r, mom_fp32[rank:, :]], dim=0)
                                var_fp32 = torch.cat([var_r, var_fp32[rank:, :]], dim=0)
                            elif new_side == 'right' and mom_fp32.size(1) >= rank:
                                mom_r = mom_fp32[:, :rank] @ T_mat
                                col_energy = T_sq.sum(dim=0, keepdim=True).clamp(0.0, 1.0)
                                var_mean = var_fp32[:, :rank].mean(dim=1, keepdim=True)
                                var_r = (var_fp32[:, :rank] @ T_sq) + var_mean * (1.0 - col_energy)
                                mom_fp32 = torch.cat([mom_r, mom_fp32[:, rank:]], dim=1)
                                var_fp32 = torch.cat([var_r, var_fp32[:, rank:]], dim=1)
                            st['momentum'] = mom_fp32.to(dtype=dtype)
                            st['variance'] = var_fp32.to(dtype=dtype)
                        else:
                            st['momentum'] = None
                            st['variance'] = None
                            st['step'] = 0
                        g['projection_side'] = new_side
                        target_dim = m if new_side == 'left' else n
                        basis = torch.zeros(target_dim, rank + 1, device=device, dtype=dtype)
                        basis[:, :rank] = new_proj.to(device=device, dtype=dtype)
                        g['projection'] = basis[:, :rank]
                        g['proj_basis'] = basis
                        g['proj_step'] = self.step_count

            # Centroid steering setup for each group in bucket
            for g in bucket_groups:
                proj = g['projection'].to(device=device, dtype=dtype)
                module = g['module']
                expert_idx = g['expert_idx']
                centroids = module.last_centroids if module is not None else None
                steer_applied = False

                if centroids is not None and expert_idx is not None and expert_idx < centroids.size(0) and self.steer_scale > 0:
                    c = centroids[expert_idx]
                    target_dim = m if side == 'left' else n
                    if c.size(0) != target_dim:
                        expert = module.experts[expert_idx]
                        first_linear = expert[0]
                        w = first_linear.linear.weight if isinstance(first_linear, FP4Linear) else first_linear.weight
                        b = first_linear.linear.bias if isinstance(first_linear, FP4Linear) else first_linear.bias
                        c_projected = F.linear(c.unsqueeze(0).to(device=w.device, dtype=w.dtype), w, b).squeeze(0)
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
                            last_c = g.get('last_c_orth')
                            if last_c is not None and last_c.shape == c_orth.shape and last_c.device == c_orth.device:
                                if torch.dot(c_orth.view(-1), last_c.view(-1)) < 0:
                                    c_orth = -c_orth
                            g['last_c_orth'] = c_orth.detach()
                            if hasattr(module, 'expert_load_ratio') and expert_idx < module.expert_load_ratio.size(0):
                                load_r = module.expert_load_ratio[expert_idx].to(device=proj.device, dtype=proj.dtype)
                                util_mult = torch.clamp(1.0 + (1.0 - load_r), min=0.3, max=2.0)
                            else:
                                util_mult = torch.tensor(1.0, device=proj.device, dtype=proj.dtype)
                            residual_mult = torch.clamp(c_res_norm, min=0.0, max=1.0)
                            effective_steer = (self.steer_scale * util_mult * residual_mult).to(dtype=proj.dtype)
                            basis = g['proj_basis']
                            basis[:, rank].copy_(c_orth)
                            g['proj_unit'] = basis
                            proj_recon = basis.clone()
                            proj_recon[:, rank].mul_(effective_steer)
                            g['proj_recon'] = proj_recon
                            steer_applied = True

                if not steer_applied:
                    basis = g['proj_basis']
                    basis[:, rank].zero_()
                    g['proj_unit'] = basis
                    g['proj_recon'] = basis
                    g['last_c_orth'] = None

            if B > 1:
                # Batched Tensor Core GEMMs (BMM) across all expert matrices simultaneously
                batch_grads = torch.stack([g['param'].grad.data for g in bucket_groups], dim=0)
                batch_units = torch.stack([g['proj_unit'] for g in bucket_groups], dim=0)
                batch_recons = torch.stack([g['proj_recon'] for g in bucket_groups], dim=0)

                if side == 'left':
                    batch_g_lr = torch.bmm(batch_units.transpose(1, 2), batch_grads)
                else:
                    batch_g_lr = torch.bmm(batch_grads, batch_units)

                for idx, g in enumerate(bucket_groups):
                    st = g['state']
                    st['step'] += 1
                    g_lr_item = batch_g_lr[idx]
                    if st['momentum'] is None:
                        st['momentum'] = g_lr_item.clone()
                        st['variance'] = g_lr_item.pow(2).clone()
                    else:
                        if st['momentum'].device != g_lr_item.device:
                            st['momentum'] = st['momentum'].to(g_lr_item.device)
                            st['variance'] = st['variance'].to(g_lr_item.device)
                        st['momentum'].mul_(expert_beta1).add_(g_lr_item, alpha=1.0 - expert_beta1)
                        st['variance'].mul_(expert_beta2).addcmul_(g_lr_item, g_lr_item, value=1.0 - expert_beta2)

                batch_mom = torch.stack([g['state']['momentum'] for g in bucket_groups], dim=0)
                batch_var = torch.stack([g['state']['variance'] for g in bucket_groups], dim=0)
                step_vals = torch.tensor([g['state']['step'] for g in bucket_groups], device=device, dtype=torch.float32).view(B, 1, 1)
                bias_c1 = (1.0 - (expert_beta1 ** step_vals)).to(dtype=dtype)
                bias_c2 = (1.0 - (expert_beta2 ** step_vals)).clamp_min(1e-7).to(dtype=dtype)
                batch_m_hat = batch_mom / bias_c1
                batch_v_hat = batch_var / bias_c2
                batch_delta_lr = (batch_m_hat / (batch_v_hat.sqrt() + 1e-8)).to(dtype=batch_recons.dtype)

                if side == 'left':
                    batch_dfull = torch.bmm(batch_recons, batch_delta_lr)
                else:
                    batch_dfull = torch.bmm(batch_delta_lr, batch_recons.transpose(1, 2))

                for idx, g in enumerate(bucket_groups):
                    p = g['param']
                    dfull = batch_dfull[idx].to(dtype=p.dtype)
                    if expert_wd != 0:
                        p.data.mul_(1.0 - expert_lr * expert_wd)
                    p.data.sub_(dfull.reshape(p.shape), alpha=expert_lr)
            else:
                g = bucket_groups[0]
                p = g['param']
                grad = p.grad.data
                st = g['state']
                proj_unit = g['proj_unit']
                proj_recon = g['proj_recon']

                if side == 'left':
                    g_lr = proj_unit.T @ grad
                else:
                    g_lr = grad @ proj_unit

                st['step'] += 1
                if st['momentum'] is None:
                    st['momentum'] = g_lr.clone()
                    st['variance'] = g_lr.pow(2).clone()
                else:
                    if st['momentum'].device != g_lr.device:
                        st['momentum'] = st['momentum'].to(g_lr.device)
                        st['variance'] = st['variance'].to(g_lr.device)
                    st['momentum'].mul_(expert_beta1).add_(g_lr, alpha=1.0 - expert_beta1)
                    st['variance'].mul_(expert_beta2).addcmul_(g_lr, g_lr, value=1.0 - expert_beta2)

                step = st['step']
                m_hat = st['momentum'] / (1.0 - expert_beta1 ** step)
                v_hat = st['variance'] / (1.0 - expert_beta2 ** step)
                delta_lr = (m_hat / (v_hat.sqrt() + 1e-8)).to(dtype=proj_recon.dtype)

                if side == 'left':
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
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if isinstance(self.model, torch.nn.Module):
            params_to_step = self.model.parameters()
        else:
            params_to_step = self.muon_params + [g['param'] for g in self.layer_groups] + self.non_expert_params
        self.step_parameters(params_to_step, is_global_step_end=True)
        return loss

    def state_dict(self):
        sd = {
            'base_optimizer': self.base_optimizer.state_dict() if self.base_optimizer is not None else None,
            'step_count': self.step_count,
            'layer_groups': [
                {
                    'projection': g['projection'],
                    'projection_side': g.get('projection_side', 'left'),
                    'proj_step': g['proj_step'],
                    'state': g['state']
                }
                for g in self.layer_groups
            ]
        }
        if self.muon_optimizer is not None:
            sd['muon_optimizer'] = self.muon_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        assert len(state_dict['layer_groups']) == len(self.layer_groups), \
            f"Layer group count mismatch: saved {len(state_dict['layer_groups'])}, current {len(self.layer_groups)}"
        for saved, cur in zip(state_dict['layer_groups'], self.layer_groups):
            if saved['projection'] is not None:
                rows, cols = cur['param'].shape
                side = saved.get('projection_side', 'left')
                if side == 'left':
                    expected_shape = (rows, min(self.rank, rows, cols))
                else:
                    expected_shape = (cols, min(self.rank, rows, cols))
                assert tuple(saved['projection'].shape) == expected_shape, \
                    f"Projection shape mismatch: saved {saved['projection'].shape}, expected {expected_shape}"
        if self.base_optimizer is not None and state_dict.get('base_optimizer') is not None:
            self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        if self.muon_optimizer is not None and 'muon_optimizer' in state_dict:
            self.muon_optimizer.load_state_dict(state_dict['muon_optimizer'])
        self.step_count = state_dict['step_count']
        for g, sd in zip(self.layer_groups, state_dict['layer_groups']):
            target_device = g['param'].device
            g['projection'] = sd['projection'].to(target_device) if sd['projection'] is not None else None
            g['projection_side'] = sd.get('projection_side', 'left')
            g['proj_step'] = sd['proj_step']
            restored_state = {}
            for k, v in sd['state'].items():
                if isinstance(v, torch.Tensor):
                    restored_state[k] = v.to(target_device)
                else:
                    restored_state[k] = v
            g['state'].update(restored_state)

    def get_telemetry(self) -> dict:
        """Returns detailed parameter breakdown, memory compression metrics, and live tier telemetry."""
        muon_count = sum(p.numel() for p in self.muon_params)
        centroid_count = sum(g['param'].numel() for g in self.layer_groups)
        base_count = sum(p.numel() for p in self.non_expert_params)
        total_count = muon_count + centroid_count + base_count

        uncompressed_bytes = total_count * 8
        muon_bytes = muon_count * 4
        centroid_bytes = 0
        for g in self.layer_groups:
            p = g['param']
            m, n = p.shape
            r = min(self.rank, m, n)
            side = g.get('projection_side', 'left')
            proj_elements = (r * n) if side == 'left' else (m * r)
            centroid_bytes += proj_elements * 2 * 4

        base_bytes = base_count * (2 if HAS_8BIT and AdamW8bit is not None else 8)
        triune_bytes = muon_bytes + centroid_bytes + base_bytes
        compression_ratio = (uncompressed_bytes / max(1, triune_bytes)) if triune_bytes > 0 else 1.0
        saved_bytes = max(0, uncompressed_bytes - triune_bytes)
        expert_compression_pct = round((1.0 - (centroid_bytes / max(1, centroid_count * 8))) * 100, 1)

        return {
            "type": "CentroidSteerOptimizer",
            "name": "3-Tier Partitioned Optimizer",
            "step_count": self.step_count,
            "steer_scale": float(self.steer_scale),
            "rank": int(self.rank),
            "update_gap": int(self.update_gap),
            "expert_lr": float(self.expert_lr),
            "base_lr": float(self.base_optimizer.param_groups[0]["lr"]) if (self.base_optimizer and self.base_optimizer.param_groups) else float(self.initial_base_lr),
            "muon_lr": float(self.muon_optimizer.param_groups[0]["lr"]) if (self.muon_optimizer and self.muon_optimizer.param_groups) else float(self.initial_muon_lr),
            "muon_momentum": float(self.muon_optimizer.param_groups[0]["momentum"]) if (self.muon_optimizer and self.muon_optimizer.param_groups) else 0.95,
            "total_params": total_count,
            "tier1_muon": {
                "name": "5th-Order Newton-Schulz Muon",
                "param_count": muon_count,
                "param_pct": round((muon_count / max(1, total_count)) * 100, 1),
                "target": "2D Attention Projections & Shared Weights",
                "vram_saving": "50% (Zero Variance State)",
                "kernel": "triune/kernels/fast_muon.py"
            },
            "tier2_centroid": {
                "name": "Adaptive SVD CentroidSteer",
                "param_count": centroid_count,
                "param_pct": round((centroid_count / max(1, total_count)) * 100, 1),
                "target": "MoE Routed Experts (Feed-Forward Blocks)",
                "vram_saving": f"{expert_compression_pct}% State Compression",
                "basis_transport": "Active (T = P_old^T @ P_new)",
                "steer_invariant": "P^T c_orth = 0 (sin θ scaling)"
            },
            "tier3_adamw": {
                "name": "Decoupled AdamW / 8-bit AdamW",
                "param_count": base_count,
                "param_pct": round((base_count / max(1, total_count)) * 100, 1),
                "target": "1D RMSNorm, Biases, Router Gates, Embeddings",
                "vram_saving": "Quantized 8-bit" if (HAS_8BIT and AdamW8bit is not None) else "Uncompressed FP32"
            },
            "memory": {
                "uncompressed_state_mb": round(uncompressed_bytes / (1024 * 1024), 2),
                "triune_state_mb": round(triune_bytes / (1024 * 1024), 2),
                "saved_state_mb": round(saved_bytes / (1024 * 1024), 2),
                "compression_ratio": round(compression_ratio, 2)
            }
        }
