"""Data-loader iteration and asynchronous prefetching utilities."""

from __future__ import annotations

from typing import Any

import torch


class CyclingDataLoader:
    def __init__(self, loader) -> None:
        try:
            if hasattr(loader, "__len__") and len(loader) == 0:
                raise ValueError("DataLoader is empty; cannot cycle an empty dataset")
        except (TypeError, NotImplementedError):
            pass
        self.loader = loader
        self._iterator = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            try:
                return next(self._iterator)
            except StopIteration:
                raise RuntimeError("DataLoader is empty; cannot cycle an empty dataset") from None

    def next(self):
        return self.__next__()


class AsyncDevicePrefetcher:
    """Asynchronous pipelined host-to-device batch prefetcher.

    Uses a dedicated CUDA transfer stream and non-blocking DMA copies to overlap
    batch transfers over PCIe concurrently with forward/backward GPU compute,
    eliminating data-transfer and host-paging latency.
    """

    def __init__(self, loader, device: torch.device | str) -> None:
        self.loader = loader
        self.iterator = iter(loader)
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.stream = torch.cuda.Stream(self.device) if self.is_cuda else None
        self.next_batch: Any = None
        self._preload()

    def _to_device(self, item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return item.to(self.device, non_blocking=True)
        if isinstance(item, tuple):
            return tuple(self._to_device(x) for x in item)
        if isinstance(item, list):
            return [self._to_device(x) for x in item]
        if isinstance(item, dict):
            return {k: self._to_device(v) for k, v in item.items()}
        return item

    def _preload(self) -> None:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.next_batch = None
                return

        if self.is_cuda and self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_batch = self._to_device(batch)
        else:
            self.next_batch = self._to_device(batch)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        if self.is_cuda and self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    def next(self):
        return self.__next__()
