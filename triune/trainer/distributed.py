"""Distributed Data Parallel (DDP) and multi-GPU training orchestration for Triune."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from triune.trainer.engine import TrainingEngine


@dataclass
class DistributedContext:
    is_distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_main_process: bool = True


_DISTRIBUTED_CTX: DistributedContext | None = None


def is_distributed() -> bool:
    """Return True if currently executing inside an initialized distributed process group."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return global rank of the current process (0 if non-distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Return total number of distributed worker processes (1 if non-distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    """Return device/local rank on current physical node (0 if non-distributed)."""
    if not is_distributed():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Return True if current process is rank 0 (responsible for logging & checkpoints)."""
    return get_rank() == 0


def setup_distributed(backend: str | None = None) -> DistributedContext:
    """Initialize distributed process group from environment variables (e.g. torchrun)."""
    global _DISTRIBUTED_CTX
    if _DISTRIBUTED_CTX is not None and _DISTRIBUTED_CTX.is_distributed:
        return _DISTRIBUTED_CTX

    world_size_env = os.environ.get("WORLD_SIZE")
    rank_env = os.environ.get("RANK")

    if not dist.is_available() or world_size_env is None or rank_env is None:
        _DISTRIBUTED_CTX = DistributedContext(
            is_distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main_process=True,
        )
        return _DISTRIBUTED_CTX

    world_size = int(world_size_env)
    rank = int(rank_env)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size <= 1:
        _DISTRIBUTED_CTX = DistributedContext(
            is_distributed=False,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            is_main_process=(rank == 0),
        )
        return _DISTRIBUTED_CTX

    if backend is None:
        if torch.cuda.is_available() and sys.platform != "win32":
            backend = "nccl"
        else:
            backend = "gloo"

    if torch.cuda.is_available() and torch.cuda.device_count() > local_rank:
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    _DISTRIBUTED_CTX = DistributedContext(
        is_distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main_process=(rank == 0),
    )
    return _DISTRIBUTED_CTX


def cleanup_distributed() -> None:
    """Tear down active distributed process group."""
    global _DISTRIBUTED_CTX
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _DISTRIBUTED_CTX = None


def all_reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.AVG) -> torch.Tensor:
    """All-reduce a tensor across all distributed worker ranks."""
    if not is_distributed():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=op)
    return t


def all_reduce_dict(metrics: Dict[str, float | int], op=dist.ReduceOp.AVG) -> Dict[str, float]:
    """Efficiently all-reduce a dictionary of scalar metrics across ranks in a single collective."""
    if not is_distributed():
        return {k: float(v) for k, v in metrics.items()}

    keys = sorted(metrics.keys())
    values = [float(metrics[k]) for k in keys]
    dev = torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")
    t = torch.tensor(values, dtype=torch.float32, device=dev)
    dist.all_reduce(t, op=op)
    return {k: float(v.item()) for k, v in zip(keys, t)}


def wrap_distributed_model(
    model: nn.Module,
    device_ids: list[int] | None = None,
    find_unused_parameters: bool = True,
) -> DDP:
    """Wrap a model in DistributedDataParallel with parameter broadcasting and exit support."""
    if not is_distributed():
        raise RuntimeError("Cannot wrap model in DDP when distributed mode is not initialized.")

    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=False,
    )


class DistributedTrainingEngine(TrainingEngine):
    """Distributed training execution engine orchestrating all-reduce metrics across ranks."""

    def __init__(self, trainer) -> None:
        super().__init__(trainer)
        self.is_distributed = is_distributed()

    def step_once(self, *, advance_step: bool = False) -> dict:
        step_result = super().step_once(advance_step=advance_step)
        if self.is_distributed and isinstance(step_result, dict):
            # All-reduce scalar loss and throughput metrics across all workers
            reduced = all_reduce_dict(
                {
                    "loss": step_result.get("loss", 0.0),
                    "lm_loss": step_result.get("lm", 0.0),
                    "tokens": step_result.get("tokens", 0),
                }
            )
            step_result["global_loss"] = reduced["loss"]
            step_result["global_lm_loss"] = reduced["lm_loss"]
            step_result["global_tokens"] = int(reduced["tokens"])

        return step_result

    def _log_training(self, **kwargs) -> None:
        # Only rank 0 emits console telemetry and logs to external sinks
        if is_main_process():
            super()._log_training(**kwargs)
