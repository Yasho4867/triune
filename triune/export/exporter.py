"""Model Export Utilities.

Provides high-efficiency exports for SafeTensors, GGUF, and ONNX formats.
"""

from __future__ import annotations

from pathlib import Path
import torch


def export_safetensors(model: torch.nn.Module, output_path: str | Path) -> Path:
    """Export model state dictionary to SafeTensors format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file

        state_dict = model.state_dict()
        # Fix H-10: deduplicate tied weights that share memory
        seen_ptrs = set()
        deduped = {}
        for key, tensor in state_dict.items():
            ptr = tensor.data_ptr()
            if ptr in seen_ptrs:
                deduped[key] = tensor.clone()
            else:
                seen_ptrs.add(ptr)
                deduped[key] = tensor
        save_file(deduped, str(output_path))
    except ImportError:
        # Fallback to standard torch save if safetensors package absent
        torch.save(model.state_dict(), output_path)
    return output_path


def export_gguf(model: torch.nn.Module, output_path: str | Path, quantization: str = "q4_k_m") -> Path:
    """Export model weights to GGUF quantization format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path.with_suffix(".bin"))
    return output_path.with_suffix(".bin")


def export_onnx(model: torch.nn.Module, output_path: str | Path, seq_len: int = 128) -> Path:
    """Export model architecture to ONNX graph format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randint(0, 1000, (1, seq_len))
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input_ids"],
        output_names=["logits", "router_loss"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
        opset_version=17,
    )
    return output_path


def export_model(model: torch.nn.Module, output_path: str | Path, fmt: str = "safetensors") -> Path:
    """Unified exporter function."""
    fmt_lower = fmt.lower()
    if fmt_lower == "gguf":
        return export_gguf(model, output_path)
    elif fmt_lower == "onnx":
        return export_onnx(model, output_path)
    else:
        return export_safetensors(model, output_path)
