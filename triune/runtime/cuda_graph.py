"""Zero-Overhead CUDA Graph Execution Engine.

Bypasses Python kernel-launch latency by recording static training and inference
DAGs directly into GPU hardware execution graphs. Once captured, execution
runs at bare-metal GPU silicon speed via single-pulse hardware replay with
automatic fallback to standard execution for dynamic shapes or non-CUDA devices.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)


class CUDAGraphEngine:
    """Manages recording and replaying static CUDA execution graphs with transparent fallback."""

    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_target: Optional[torch.Tensor] = None
        self.static_loss: Optional[torch.Tensor] = None
        self.static_output: Any = None
        self.is_captured: bool = False

    def invalidate(self) -> None:
        """Invalidate the recorded graph to trigger re-capture or fall back to standard execution."""
        self.graph = None
        self.static_input = None
        self.static_target = None
        self.static_loss = None
        self.static_output = None
        self.is_captured = False

    def can_replay(self, input_tokens: torch.Tensor, target_tokens: Optional[torch.Tensor] = None) -> bool:
        """Check whether incoming tensors match static buffer dimensions for graph replay."""
        if not self.is_captured or self.graph is None or self.static_input is None:
            return False
        if (
            input_tokens.shape != self.static_input.shape
            or input_tokens.dtype != self.static_input.dtype
            or input_tokens.device != self.static_input.device
        ):
            return False
        if target_tokens is not None and self.static_target is not None:
            if (
                target_tokens.shape != self.static_target.shape
                or target_tokens.dtype != self.static_target.dtype
                or target_tokens.device != self.static_target.device
            ):
                return False
        return True

    def capture_training_step(
        self,
        model: torch.nn.Module,
        optimizer: Any,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        loss_fn: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        step_fn: Optional[Callable[[torch.nn.Module, Any, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        warmup_steps: int = 3,
    ) -> bool:
        """Capture full forward, backward, and optimizer update into a CUDA graph with safe fallback."""
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return False

        try:
            # Enable capturable flag on optimizer param groups if supported (e.g. AdamW)
            for group in getattr(optimizer, "param_groups", []):
                if "capturable" in group:
                    group["capturable"] = True

            # Allocate static input/target buffers
            self.static_input = sample_input.clone().to(self.device)
            self.static_target = sample_target.clone().to(self.device)

            # Stabilize CUDA stream settings for graph capture
            if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
                torch.autograd.graph.set_override_stale_capture_stream(True)
            if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
                torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(stream):
                for _ in range(warmup_steps):
                    if step_fn is not None:
                        _ = step_fn(model, optimizer, self.static_input, self.static_target)
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        out = model(self.static_input)
                        if loss_fn is not None:
                            loss = loss_fn(out, self.static_target)
                        else:
                            out_t = out[0] if isinstance(out, tuple) else out
                            loss = out_t.sum()
                        loss.backward()
                        optimizer.step()

            torch.cuda.current_stream().wait_stream(stream)

            # Capture the static execution graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                if step_fn is not None:
                    self.static_loss = step_fn(model, optimizer, self.static_input, self.static_target)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    out = model(self.static_input)
                    if loss_fn is not None:
                        self.static_loss = loss_fn(out, self.static_target)
                    else:
                        out_t = out[0] if isinstance(out, tuple) else out
                        self.static_loss = out_t.sum()
                    self.static_loss.backward()
                    optimizer.step()

            self.is_captured = True
            return True

        except Exception as exc:
            logger.debug("CUDA Graph capture bypassed (falling back to standard execution): %s", exc)
            self.invalidate()
            return False

    def step(
        self,
        input_tokens: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        fallback_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], float]] = None,
    ) -> float:
        """Replay the hardware CUDA graph with new batch inputs at zero CPU latency."""
        if not self.can_replay(input_tokens, target_tokens):
            if fallback_fn is not None:
                return float(fallback_fn(input_tokens, target_tokens))
            raise RuntimeError("CUDA Graph cannot replay with mismatched tensor shapes without a fallback.")

        # Zero-copy copy into static graph input tensors
        self.static_input.copy_(input_tokens, non_blocking=True)
        if self.static_target is not None and target_tokens is not None:
            self.static_target.copy_(target_tokens, non_blocking=True)

        # Single hardware pulse replay
        self.graph.replay()
        return float(self.static_loss.item()) if self.static_loss is not None else 0.0

    def capture_inference_step(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        warmup_steps: int = 3,
    ) -> bool:
        """Capture static inference forward pass into a CUDA graph."""
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return False
        try:
            self.static_input = sample_input.clone().to(self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(warmup_steps):
                    _ = model(self.static_input)
            torch.cuda.current_stream().wait_stream(stream)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream), torch.no_grad():
                self.static_output = model(self.static_input)

            self.is_captured = True
            return True
        except Exception as exc:
            logger.debug("CUDA Graph inference capture bypassed: %s", exc)
            self.invalidate()
            return False

    def inference_step(
        self,
        input_tokens: torch.Tensor,
        fallback_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    ) -> Any:
        """Replay inference graph at bare-metal zero-overhead speed."""
        if not self.can_replay(input_tokens):
            if fallback_fn is not None:
                return fallback_fn(input_tokens)
            raise RuntimeError("Inference CUDA Graph cannot replay with mismatched tensor shapes without a fallback.")
        self.static_input.copy_(input_tokens, non_blocking=True)
        self.graph.replay()
        return self.static_output

