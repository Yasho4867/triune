"""Model Export Utilities.

Provides SafeTensors and ONNX exports. GGUF is explicitly unsupported until a
converter for Triune's custom architecture is implemented.
"""

from __future__ import annotations

from pathlib import Path
import json
import struct
import torch

_TORCH_TO_SAFETENSORS_DTYPE = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def _save_safetensors_fallback(tensors: dict[str, torch.Tensor], output_path: str | Path) -> Path:
    """Pure-Python SafeTensors serializer compliant with Hugging Face specification."""
    header = {}
    total_bytes = bytearray()
    curr_offset = 0

    for name, tensor in tensors.items():
        t = tensor.detach().to(device="cpu").contiguous()
        dt_str = _TORCH_TO_SAFETENSORS_DTYPE.get(t.dtype)
        if dt_str is None:
            t = t.to(torch.float32)
            dt_str = "F32"
        try:
            raw = t.numpy().tobytes()
        except (TypeError, ValueError):
            raw = bytes(t.view(torch.uint8).numpy())
        length = len(raw)
        header[name] = {
            "dtype": dt_str,
            "shape": list(t.shape),
            "data_offsets": [curr_offset, curr_offset + length],
        }
        total_bytes.extend(raw)
        curr_offset += length

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(total_bytes)
    return out


def export_safetensors(model: torch.nn.Module, output_path: str | Path) -> Path:
    """Export model state dictionary to SafeTensors format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    try:
        from safetensors.torch import save_file
        save_file(deduped, str(output_path))
    except ImportError:
        _save_safetensors_fallback(deduped, output_path)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        save_file(adapters, str(output_path))
    except ImportError:
        _save_safetensors_fallback(adapters, output_path)
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
