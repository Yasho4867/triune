"""Interactive chat entry point backed by :mod:`triune.inference`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triune.inference import generate_response, load_checkpoint_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a Triune checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints_full/best.pt")
    parser.add_argument("--tokenizer_path", default="triune_tokenizer.json")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory or safetensors path to attach")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--speculative", action="store_true", help="Enable Reflex-to-Cortex self-speculative decoding")
    parser.add_argument("--k-draft", type=int, default=3, help="Number of draft tokens per speculative cycle")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model, tokenizer, checkpoint, device, pad_id, eos_id = load_checkpoint_model(
        args.checkpoint, args.tokenizer_path, adapter_path=args.adapter
    )
    if args.adapter:
        print(f"LoRA Adapter attached from: {args.adapter}")
    print(f"Model loaded. Best eval loss: {checkpoint.get('best_eval_loss', 'unknown')}")
    print("Triune chat mode. Commands: reflex / limbic / cortex / auto; exit to quit.")
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            return
        force_depth = None
        for command, depth in (("reflex ", 0), ("limbic ", 1), ("cortex ", 2)):
            if prompt.lower().startswith(command):
                force_depth, prompt = depth, prompt[len(command):]
                break
        if prompt.lower().startswith("auto "):
            prompt = prompt[5:]
        if args.speculative:
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
            print(f"Assistant [Speculative K={args.k_draft}, acc={spec_res['acceptance_rate']:.1%}, speed={spec_res['tokens_per_second']} tok/s]: {spec_res['text']}")
        else:
            response = generate_response(
                model, tokenizer, prompt, device=device, pad_token_id=pad_id, eos_token_id=eos_id,
                force_depth=force_depth, temperature=args.temperature,
                repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens,
            )
            print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
