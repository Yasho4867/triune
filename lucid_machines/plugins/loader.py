"""Plugin loader enforcing Ed25519 signature verification (trusted=False) by default for external plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from lucid_machines.modules.manager import ModuleManager


def load_plugin_nodes(module_dir: str | Path, *, trusted: bool = False) -> List[str]:
    """Discover and register DAG nodes from a plugin directory.

    External directories default to ``trusted=False``, requiring a verified
    Ed25519-signed ``lucid_machines_plugin.json`` (or legacy ``triune_plugin.json``)
    manifest with matching SHA-256 entrypoints.
    """
    return ModuleManager().discover_and_register_nodes(Path(module_dir), trusted=trusted)


def scan_workspace_plugins(workspace_dir: str | Path | None = None, *, trusted: bool = False) -> Dict[str, Any]:
    """Scan workspace plugin directories with ``trusted=False`` by default."""
    return ModuleManager().scan_workspace_plugins(workspace_dir=workspace_dir, trusted=trusted)


__all__ = ["load_plugin_nodes", "scan_workspace_plugins"]
