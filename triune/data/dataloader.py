"""Data-loader iteration utilities."""

from __future__ import annotations


class CyclingDataLoader:
    def __init__(self, loader) -> None:
        self.loader = loader
        self._iterator = iter(loader)

    def next(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)
