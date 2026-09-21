"""Optional experiment logging adapters."""

from __future__ import annotations


class NullLogger:
    run_id = None

    def log(self, values: dict, *, step: int) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbLogger:
    def __init__(self, *, project: str, config: dict, run_id: str | None = None) -> None:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("wandb is not installed; pass --no-wandb to disable experiment logging") from error
        self._wandb = wandb
        self._run = wandb.init(project=project, config=config, id=run_id, resume="allow" if run_id else None)

    @property
    def run_id(self) -> str | None:
        return self._run.id

    def log(self, values: dict, *, step: int) -> None:
        self._wandb.log(values, step=step)

    def finish(self) -> None:
        self._wandb.finish()
