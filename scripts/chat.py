"""Interactive chat entry point backed by :mod:`triune.inference`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triune.inference import generate_response, load_checkpoint_model, load_model_for_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a Triune checkpoint or Hugging Face model")
    parser.add_argument("--model", default=None, help="HF model repo ID, local HF directory, or NVFP4 disk cache")
    parser.add_argument("--checkpoint", default="checkpoints_full/best.pt")
    parser.add_argument("--tokenizer_path", default="triune_tokenizer.json")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory or safetensors path to attach")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--speculative", action="store_true", help="Enable Reflex-to-Cortex self-speculative decoding")
    parser.add_argument("--k-draft", type=int, default=3, help="Number of draft tokens per speculative cycle")
    parser.add_argument("--nvfp4", action="store_true", help="Quantize linear blocks to Hadamard-rotated NVFP4")
    parser.add_argument("--disk-stream", action="store_true", help="Stream NVFP4 layers directly from NVMe disk")
    parser.add_argument("--disk-cache", default=None, help="Optional directory for NVFP4 disk cache")
    parser.add_argument("--device", default=None, help="Target device (cuda / cpu)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source = args.model or args.checkpoint
    quantize = "nvfp4" if args.nvfp4 else None
    if args.model is not None or args.nvfp4 or args.disk_stream or Path(source).is_dir():
        model, tokenizer, metadata, device, pad_id, eos_id = load_model_for_inference(
            source,
            args.tokenizer_path,
            device=args.device,
            adapter_path=args.adapter,
            quantize=quantize,
            disk_stream=args.disk_stream,
            disk_cache_dir=args.disk_cache,
        )
    else:
        model, tokenizer, metadata, device, pad_id, eos_id = load_checkpoint_model(
            args.checkpoint, args.tokenizer_path, device=args.device, adapter_path=args.adapter
        )
    if args.adapter:
        print(f"LoRA Adapter attached from: {args.adapter}")
    print(f"Model loaded ({source}). Info: {metadata.get('best_eval_loss', 'ready')}")
    is_triune_model = hasattr(model, "init_cache") or getattr(model, "_is_triune", False)
    if is_triune_model:
        print("Triune chat mode. Commands: reflex / limbic / cortex / auto; exit to quit.")
    else:
        print("Chat mode active (HF/NVFP4). Type 'clear' to reset history or 'exit' to quit.")
    history: list[dict[str, str]] = []
    has_chat_tmpl = bool(hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None))
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            return
        if prompt.lower() == "clear":
            history.clear()
            print("[History cleared]")
            continue
        force_depth = None
        if is_triune_model:
            for command, depth in (("reflex ", 0), ("limbic ", 1), ("cortex ", 2)):
                if prompt.lower().startswith(command):
                    force_depth, prompt = depth, prompt[len(command) :]
                    break
            if prompt.lower().startswith("auto "):
                prompt = prompt[5:]
        if args.speculative and is_triune_model:
            from triune.model.speculative import SelfSpeculativeGenerator

            spec_gen = SelfSpeculativeGenerator(
                model,
                tokenizer,
                k_draft=args.k_draft,
                temperature=args.temperature,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            spec_res = spec_gen.generate(prompt, max_new_tokens=args.max_new_tokens)
            print(
                f"Assistant [Speculative K={args.k_draft}, acc={spec_res['acceptance_rate']:.1%}, speed={spec_res['tokens_per_second']} tok/s]: {spec_res['text']}"
            )
        else:
            effective_prompt = prompt
            prompt_is_formatted = False
            if has_chat_tmpl and not is_triune_model:
                history.append({"role": "user", "content": prompt})
                effective_prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                prompt_is_formatted = True
            response = generate_response(
                model,
                tokenizer,
                effective_prompt,
                device=device,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                force_depth=force_depth,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                prompt_is_formatted=prompt_is_formatted,
            )
            if has_chat_tmpl and not is_triune_model:
                history.append({"role": "assistant", "content": str(response)})
            print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
