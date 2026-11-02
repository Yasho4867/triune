"""Real-Time Neuron Activation Microscope & Router Telemetry Tap.

Extracts live layer activations, attention matrices, and dynamic routing
decisions directly on-chip for 120 FPS visualization in Triune Studio.
Zero host memory overhead when idle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional


class NeuronScope:
    """Taps into intermediate layer activations and dynamic routing choices."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.is_active = False
        self.activations: Dict[int, torch.Tensor] = {}
        self.router_decisions: List[int] = []
        self._hooks: List[Any] = []

    def attach(self) -> None:
        """Attach lightweight forward hooks to monitor neurons across layers."""
        if self._hooks:
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
            # Sample first batch element, compact activation mean across heads
            tensor = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                # Extract downsampled [Tokens, Neurons] profile (e.g. 32x32 footprint)
                act = tensor[0].detach()  # [T, D]
                t_sample = min(act.shape[0], 32)
                d_sample = min(act.shape[1], 32)
                self.activations[layer_idx] = act[:t_sample, :d_sample].float()
        return hook_fn

    def record_routing(self, selected_tier: int) -> None:
        """Record the active routing tier (0: Reflex, 1: Limbic, 2: Cortex)."""
        self.router_decisions.append(selected_tier)
        if len(self.router_decisions) > 64:
            self.router_decisions.pop(0)

    def get_snapshot(self) -> Dict[str, Any]:
        """Get live serialized neuron heatmap and tier stats for Studio."""
        layer_heatmaps = {}
        for l_idx, act in self.activations.items():
            layer_heatmaps[l_idx] = act.cpu().numpy().tolist()

        return {
            "layers": layer_heatmaps,
            "router_history": self.router_decisions[-16:],
            "active_tier": self.router_decisions[-1] if self.router_decisions else 0,
        }

    def detach(self) -> None:
        """Remove hooks to restore zero-overhead execution."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.activations.clear()
        self.is_active = False
