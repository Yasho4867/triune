"""On-demand layer activation and routing telemetry for Triune Studio.

Hooks are attached only while a local inference request is running. The most
recent sampled activations are retained on CPU for the read-only visualization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class NeuronScope:
    """Taps into intermediate layer activations and dynamic routing choices."""

    def __init__(self, model: Optional[nn.Module] = None) -> None:
        self.model = model
        self.is_active = False
        self.activations: Dict[int, torch.Tensor] = {}
        self.router_decisions: List[int] = []
        self._hooks: List[Any] = []

    def attach(self, model: Optional[nn.Module] = None) -> None:
        """Attach lightweight forward hooks to monitor neurons across layers."""
        if model is not None and model is not self.model:
            self.detach()
            self.model = model
        if self._hooks or self.model is None:
            return

        layers = getattr(self.model, "layers", None)
        if layers is None and hasattr(self.model, "transformer"):
            layers = getattr(self.model.transformer, "layers", None)

        if layers is not None:
            for idx, layer in enumerate(layers):
                hook = layer.register_forward_hook(self._make_layer_hook(idx))
                self._hooks.append(hook)
        self.is_active = True

    def _make_layer_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if not self.is_active:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
                return
            with torch.no_grad():
                act = tensor[0].detach() if tensor.ndim >= 3 else tensor.detach()
                t_sample = min(act.shape[0], 32)
                d_sample = min(act.shape[-1], 64)
                self.activations[layer_idx] = act[:t_sample, :d_sample].float()

        return hook_fn

    def record_routing(self, selected_tier: int) -> None:
        """Record the active routing tier (0: Reflex, 1: Limbic, 2: Cortex)."""
        self.router_decisions.append(int(selected_tier))
        if len(self.router_decisions) > 64:
            self.router_decisions.pop(0)

    def get_snapshot(self) -> Dict[str, Any]:
        """Get live serialized neuron heatmap and tier stats for Studio (uses native C/AVX2 binning when available)."""
        from triune.ops._c_bridge import c_compute_neuron_heatmap_f32, get_native_c_info

        layer_heatmaps: Dict[int, Any] = {}
        layer_bins: Dict[int, List[float]] = {}
        for l_idx, act in sorted(self.activations.items()):
            layer_heatmaps[l_idx] = act.cpu().numpy().tolist()
            c_bins = c_compute_neuron_heatmap_f32(act, num_bins=64)
            if c_bins is not None:
                layer_bins[l_idx] = c_bins[0]

        active_tier = self.router_decisions[-1] if self.router_decisions else 0
        if self.model is not None and getattr(self.model, "last_depth_choice", None) is not None:
            dc = getattr(self.model, "last_depth_choice")
            if isinstance(dc, int):
                active_tier = dc
            elif isinstance(dc, torch.Tensor) and dc.numel() > 0:
                active_tier = int(dc.view(-1)[0].item())

        tier_names = {0: "Reflex", 1: "Limbic", 2: "Cortex"}
        return {
            "layers": layer_heatmaps,
            "layer_bins": layer_bins,
            "num_layers": int(getattr(self.model, "num_layers", len(layer_bins) or 8)),
            "router_history": self.router_decisions[-16:],
            "active_tier": active_tier,
            "active_tier_name": tier_names.get(active_tier, "Reflex"),
            "native_c_engine": get_native_c_info(),
        }

    def detach(self, *, clear: bool = True) -> None:
        """Remove hooks; optionally retain the last sample for read-only display."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if clear:
            self.activations.clear()
        else:
            # Keep only CPU snapshots so the viewer does not pin GPU allocations
            # between inference requests.
            self.activations = {idx: act.detach().cpu() for idx, act in self.activations.items()}
        self.is_active = False
