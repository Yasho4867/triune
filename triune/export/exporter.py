"""Model Export Utilities.

Provides SafeTensors and ONNX exports. GGUF is explicitly unsupported until a
converter for Triune's custom architecture is implemented.
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
        raise RuntimeError("SafeTensors export requires the 'safetensors' package; no file was written.")
    return output_path


def export_lora_safetensors(model: torch.nn.Module, output_path: str | Path) -> Path:
    """Export only LoRA A/B adapter matrices from a model with attached adapters."""
    output_path = Path(output_path)
    adapters = {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in model.named_parameters()
        if name.endswith(("lora_A", "lora_B"))
    }
    if not adapters:
        raise ValueError("No attached LoRA adapter matrices were found; no file was written.")
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("SafeTensors export requires the 'safetensors' package; no file was written.") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(adapters, str(output_path))
    return output_path


def export_gguf(model: torch.nn.Module, output_path: str | Path, quantization: str = "q4_k_m") -> Path:
    """Fail explicitly until Triune's custom architecture has a real GGUF converter."""
    raise NotImplementedError(
        "GGUF export is not implemented for Triune's GLA, hierarchical exits, and routed experts; "
        "the model was not exported."
    )


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
