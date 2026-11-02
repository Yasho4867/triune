"""Zero-Overhead CUDA Graph Execution Engine.

Bypasses Python kernel-launch latency by recording static training and inference
DAGs directly into GPU hardware execution graphs. Once captured, execution
runs at bare-metal GPU silicon speed via single-pulse hardware replay.
"""

from __future__ import annotations

import torch
from typing import Callable, Any, Optional


class CUDAGraphEngine:
    """Manages recording and replaying static CUDA execution graphs."""

    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
        self.static_loss: Optional[torch.Tensor] = None
        self.is_captured = False

    def capture_training_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        warmup_steps: int = 3,
    ) -> bool:
        """Capture full forward, backward, and optimizer update into a CUDA graph."""
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return False

        # Allocate static input/target buffers
        self.static_input = sample_input.clone().to(self.device)
        self.static_target = sample_target.clone().to(self.device)

        # Warmup steps to stabilize CUDA caching allocator and dynamic dispatch
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_steps):
                optimizer.zero_grad(set_to_none=True)
                out = model(self.static_input)
                loss = loss_fn(out, self.static_target)
                loss.backward()
                optimizer.step()

        torch.cuda.current_stream().wait_stream(stream)

        # Capture the static execution graph
        self.graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph, stream=stream):
            out = model(self.static_input)
            self.static_loss = loss_fn(out, self.static_target)
            self.static_loss.backward()
            optimizer.step()

        self.is_captured = True
        return True

    def step(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor) -> float:
        """Replay the hardware CUDA graph with new batch inputs at zero CPU latency."""
        if not self.is_captured or self.graph is None or self.static_input is None:
            raise RuntimeError("CUDA Graph has not been captured yet.")

        # Zero-copy copy into static graph input tensors
        self.static_input.copy_(input_tokens, non_blocking=True)
        if self.static_target is not None and target_tokens is not None:
            self.static_target.copy_(target_tokens, non_blocking=True)

        # Single hardware pulse replay
        self.graph.replay()
        return float(self.static_loss.item()) if self.static_loss is not None else 0.0
