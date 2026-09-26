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

class CentroidSteerOptimizer(torch.optim.Optimizer):
    def __init__(self, model, lr, betas, weight_decay,
                 rank=defaults.GALORE_RANK, update_gap=defaults.GALORE_UPDATE_GAP,
                 steer_scale=defaults.STEER_SCALE,
                 expert_lr=defaults.GALORE_LR, expert_betas=defaults.GALORE_BETAS,
                 expert_wd=defaults.GALORE_WEIGHT_DECAY,
                 use_muon=getattr(defaults, 'USE_MUON', True),
                 muon_lr=getattr(defaults, 'MUON_LR', 0.02),
                 muon_momentum=getattr(defaults, 'MUON_MOMENTUM', 0.95),
                 muon_weight_decay=getattr(defaults, 'MUON_WEIGHT_DECAY', 0.01)):
        defaults_dict = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        super().__init__([{'params': [dummy_param]}], defaults_dict)
        self.model = model
        self.rank = rank
        self.update_gap = update_gap
        self.step_count = 0
        self.steer_scale = steer_scale
        self.expert_lr = expert_lr
        self.expert_betas = expert_betas
        self.expert_wd = expert_wd
        self.use_muon = use_muon
        self.initial_base_lr = lr
        self.initial_muon_lr = muon_lr

        self.non_expert_params = []
        self.muon_params = []
        self.layer_groups = []
        seen_param_ids = set()

        # Identify router parameters and token embeddings (these must stay on AdamW)
        router_param_ids = set()
        for name, module in model.named_modules():
            if isinstance(module, MoE_FFN) and hasattr(module, 'router'):
                for p in module.router.parameters():
                    router_param_ids.add(id(p))

        embedding_param_ids = set()
        for name, param in model.named_parameters():
            name_lower = name.lower()
            if 'embed' in name_lower or 'wte' in name_lower:
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
                                    'projection_side': 'left',
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
                        'projection_side': 'left',
                        'proj_step': stagger,
                        'state': {'momentum': None, 'variance': None, 'step': 0}
                    })
                    group_idx += 1
                else:
                    # Router head or token embedding -> Base Optimizer (AdamW)
                    self.non_expert_params.append(param)

        # 3. 1D parameters (RMSNorm, biases) -> Base Optimizer (AdamW)
        for name, param in model.named_parameters():
            if id(param) not in seen_param_ids:
                seen_param_ids.add(id(param))
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

        if HAS_8BIT and AdamW8bit is not None:
            self.base_optimizer = AdamW8bit(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            self.base_optimizer = torch.optim.AdamW(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)

    def zero_grad(self, set_to_none=True):
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

    def set_lr(self, lr):
        for pg in self.base_optimizer.param_groups:
            pg['lr'] = lr
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
        for group in self.base_optimizer.param_groups:
            matching = [p for p in group['params'] if id(p) in param_ids]
            if matching:
                orig_params = group['params']
                group['params'] = matching
                try:
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

        for group in self.layer_groups:
            p = group['param']
            if id(p) not in param_ids:
                continue

            if p.grad is None:
                continue

            grad = p.grad.data
            # Fix M-2: removed zero-grad skip — Adam bias correction and momentum decay
            # require processing zero gradients to maintain correct state

            module = group['module']
            expert_idx = group['expert_idx']
            state = group['state']
            m, n = grad.shape

            is_initial = (group['projection'] is None)
            if (self.step_count - group['proj_step']) >= self.update_gap or is_initial:
                svd_device = grad.device
                grad_fp32 = grad.to(device=svd_device, non_blocking=True).float()
                U, S, V = torch.svd_lowrank(grad_fp32, q=min(self.rank + 10, m, n), niter=2)
                rank = min(self.rank, m, n)
                
                if m >= n:
                    group['projection_side'] = 'left'
                    group['projection'] = U[:, :rank].to(device=grad.device, dtype=grad.dtype).contiguous()
                else:
                    group['projection_side'] = 'right'
                    group['projection'] = V[:, :rank].to(device=grad.device, dtype=grad.dtype).contiguous()
                    
                del grad_fp32, U, S, V
                if svd_device.type == "cuda":
                    torch.cuda.empty_cache()
                if not is_initial:
                    group['proj_step'] = self.step_count
                state['momentum'] = None
                state['variance'] = None
                state['step'] = 0

            proj = group['projection'].to(device=grad.device, dtype=grad.dtype)
            side = group.get('projection_side', 'left')

            # Centroid steering
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
                        if hasattr(module, 'expert_load_ratio') and expert_idx < module.expert_load_ratio.size(0):
                            # Fix H-6: Replace .item() with tensor ops to avoid CPU-GPU sync
                            load_r = module.expert_load_ratio[expert_idx]
                            util_mult = torch.clamp(1.0 + (1.0 - load_r), min=0.3, max=2.0)
                        else:
                            util_mult = torch.tensor(1.0, device=proj.device, dtype=proj.dtype)
                        # Fix H-6: Also replace c_res_norm.item()
                        residual_mult = torch.clamp(c_res_norm, min=0.1, max=1.0)
                        effective_steer = self.steer_scale * util_mult * residual_mult
                        proj_aug = torch.cat([proj, (effective_steer * c_orth).unsqueeze(1)], dim=1)
                        steer_applied = True

            if not steer_applied:
                zero_vec = torch.zeros(m if side == 'left' else n, 1, dtype=proj.dtype, device=proj.device)
                proj_aug = torch.cat([proj, zero_vec], dim=1)

            if side == 'left':
                g_lr = proj_aug.T @ grad
            else:
                g_lr = grad @ proj_aug

            state['step'] += 1
            if state['momentum'] is None:
                state['momentum'] = g_lr.clone()
                state['variance'] = g_lr.pow(2).clone()
            else:
                if state['momentum'].device != g_lr.device:
                    state['momentum'] = state['momentum'].to(g_lr.device)
                    state['variance'] = state['variance'].to(g_lr.device)
                state['momentum'] = expert_beta1 * state['momentum'] + (1 - expert_beta1) * g_lr
                state['variance'] = expert_beta2 * state['variance'] + (1 - expert_beta2) * g_lr.pow(2)

            step = state['step']
            m_hat = state['momentum'] / (1 - expert_beta1 ** step)
            v_hat = state['variance'] / (1 - expert_beta2 ** step)
            delta_lr = m_hat / (v_hat.sqrt() + 1e-8)

            if side == 'left':
                delta_full = proj_aug @ delta_lr
            else:
                delta_full = delta_lr @ proj_aug.T

            if expert_wd != 0:
                p.data.mul_(1.0 - expert_lr * expert_wd)  # Fix M-3: decoupled weight decay without .data subtraction
            p.data -= expert_lr * delta_full.reshape(p.shape)

        if is_global_step_end:
            self.step_count += 1
            if torch.cuda.is_available() and (self.step_count <= 2 or self.step_count % 50 == 0):
                torch.cuda.empty_cache()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.step_parameters(self.model.parameters(), is_global_step_end=True)
        return loss

    def state_dict(self):
        sd = {
            'base_optimizer': self.base_optimizer.state_dict(),
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
