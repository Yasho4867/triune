"""Backward-compatibility shim delegating ``triune`` imports to ``lucid_machines``.

The canonical Python package is ``lucid_machines`` (distribution ``lucid-machines``).
This lightweight shim preserves ``import triune``, ``from triune import ...``, and
``from triune.<submodule> import ...`` without duplicating the codebase.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from types import ModuleType
from typing import Any, Sequence

import lucid_machines as _lucid_machines

warnings.warn(
    "The 'triune' import namespace is deprecated; use 'lucid_machines' instead.",
    DeprecationWarning,
    stacklevel=2,
)


class _TriuneSubmoduleAliasLoader(importlib.abc.Loader):
    """Loader that aliases ``triune.<submod>`` to ``lucid_machines.<submod>``."""

    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        target = importlib.import_module(self._target_name)
        sys.modules[spec.name] = target
        return target

    def exec_module(self, module: ModuleType) -> None:
        return None


class _TriuneSubmoduleAliasFinder(importlib.abc.MetaPathFinder):
    """Meta-path hook forwarding ``triune.<submod>`` imports to ``lucid_machines.<submod>``."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith("triune."):
            return None
        target_name = "lucid_machines." + fullname[len("triune.") :]
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None
        is_pkg = target_spec.submodule_search_locations is not None
        spec = importlib.machinery.ModuleSpec(
            fullname,
            _TriuneSubmoduleAliasLoader(target_name),
            is_package=is_pkg,
        )
        if is_pkg and target_spec.submodule_search_locations is not None:
            spec.submodule_search_locations = list(target_spec.submodule_search_locations)
        return spec


if not any(isinstance(f, _TriuneSubmoduleAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _TriuneSubmoduleAliasFinder())

__all__ = list(getattr(_lucid_machines, "__all__", []))
__path__: list[str] = []


def __getattr__(name: str) -> Any:
    if hasattr(_lucid_machines, name):
        return getattr(_lucid_machines, name)
    # Support submodule attribute access after ``import triune.<submod>`` or direct access
    try:
        submod = importlib.import_module(f"lucid_machines.{name}")
        globals()[name] = submod
        return submod
    except ImportError as exc:
        raise AttributeError(f"module 'triune' has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(dir(_lucid_machines))))
