"""Model inspection entry point with no work performed at import time."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triune.configs.config import build_config
from triune.model import build_model


def inspect_model(config: dict | None = None) -> dict:
    model = build_model(config or build_config())
    first_gla = model.layers[0].attn.gla
    first_expert = model.layers[7].ffn.experts[0]
    first_linear = getattr(first_expert[0], "linear", first_expert[0])
    return {
        "layers": len(model.layers),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "hidden_dim": first_gla.q_proj.in_features,
        "heads": first_gla.heads,
        "expert_in_features": first_linear.in_features,
        "expert_out_features": first_linear.out_features,
    }


def main() -> None:
    for key, value in inspect_model().items():
        print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")


if __name__ == "__main__":
    main()
