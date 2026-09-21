"""Data-loader iteration utilities."""

from __future__ import annotations


class CyclingDataLoader:
    def __init__(self, loader) -> None:
        if hasattr(loader, "__len__") and len(loader) == 0:
            raise ValueError("DataLoader is empty; cannot cycle an empty dataset")
        self.loader = loader
        self._iterator = iter(loader)

    def next(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            try:
                return next(self._iterator)
            except StopIteration:
                raise RuntimeError("DataLoader is empty; cannot cycle an empty dataset") from None
