"""Smart VRAM Profiler & Dynamic Layer Offloader for Triune Runtime."""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional
import torch


class VRAMProfiler:
    """Monitors live PyTorch GPU VRAM allocation and warns before OOM spikes."""

    @staticmethod
    def get_vram_stats(device: torch.device | str = "cuda") -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "max_allocated_gb": 0.0,
                "total_gb": 8.0,
                "oom_risk": False,
            }

        dev = torch.device(device)
        allocated = torch.cuda.memory_allocated(dev) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        total = torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
        oom_risk = (allocated / total) >= 0.90 if total > 0 else False

        return {
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "max_allocated_gb": round(max_allocated, 3),
            "total_gb": round(total, 3),
            "allocated": round(allocated, 3),
            "reserved": round(reserved, 3),
            "total": round(total, 3),
            "oom_risk": oom_risk,
        }

    @staticmethod
    def check_oom_risk(threshold_pct: float = 0.90) -> bool:
        stats = VRAMProfiler.get_vram_stats()
        if stats["total_gb"] == 0:
            return False
        used_pct = stats["allocated_gb"] / stats["total_gb"]
        if used_pct >= threshold_pct:
            print(f"⚠️ VRAM OOM Warning: High memory usage ({used_pct * 100:.1f}%)! Offloading recommended.")
            return True
        return False


class AutoOffloader:
    """Automatically offloads inactive model layers to CPU RAM when VRAM spikes."""

    def __init__(self, model: torch.nn.Module | None = None, offload_device: str = "cpu", vram_threshold_gb: float = 7.5):
        self.model = model
        self.offload_device = torch.device(offload_device)
        self.vram_threshold_gb = vram_threshold_gb
        self.original_devices: Dict[str, torch.device] = {}

    def check_and_offload(self) -> bool:
        stats = VRAMProfiler.get_vram_stats()
        if stats["allocated_gb"] >= self.vram_threshold_gb:
            if self.model and hasattr(self.model, "layers"):
                for idx, layer in enumerate(reversed(self.model.layers)):
                    if idx < 4:
                        actual_idx = len(self.model.layers) - 1 - idx
                        self.offload_layer(f"layer_{actual_idx}", layer)
            return True
        return False

    def offload_layer(self, layer_name: str, layer_module: torch.nn.Module) -> None:
        """Move specific layer module weights to CPU RAM."""
        print(f"🔄 Offloading layer [{layer_name}] to CPU RAM...")
        for p in layer_module.parameters():
            if p.device.type != self.offload_device.type:
                self.original_devices[layer_name] = p.device
                p.data = p.data.to(self.offload_device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def restore_layer(self, layer_name: str, layer_module: torch.nn.Module, target_device: torch.device) -> None:
        """Restore layer module weights back to target GPU device."""
        print(f"⚡ Restoring layer [{layer_name}] back to GPU {target_device}...")
        for p in layer_module.parameters():
            p.data = p.data.to(target_device)
