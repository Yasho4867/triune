"""Continuation rescue training for checkpoints starved at Cortex depth under dynamic exit routing."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_full_model_e2e import load_trained_model  # noqa: E402

from lucid_machines.data.tokenizer import load_tokenizer  # noqa: E402
from lucid_machines.kernels.fast_cross_entropy import fast_cross_entropy  # noqa: E402
from lucid_machines.optim.centroid import CentroidSteerOptimizer  # noqa: E402


def build_packed_blocks(
    docs: list[str],
    tokenizer,
    seq_len: int,
    sep_id: int,
    max_blocks: int | None = None,
) -> torch.Tensor:
    """Packs documents with [SEP] boundary markers into fixed-length causal blocks."""
    stream: list[int] = []
    block_stride = seq_len + 1
    blocks: list[list[int]] = []
    for text in docs:
        ids = tokenizer.encode(text).ids
        if len(ids) < 16:
            continue
        stream.extend(ids)
        stream.append(sep_id)
        while len(stream) >= block_stride:
            blocks.append(stream[:block_stride])
            stream = stream[seq_len:]
            if max_blocks is not None and len(blocks) >= max_blocks:
                return torch.tensor(blocks, dtype=torch.long)
    if not blocks:
        raise RuntimeError("Insufficient tokens to form packed blocks")
    return torch.tensor(blocks, dtype=torch.long)


@torch.no_grad()
def evaluate_all_depths(model, val_blocks: torch.Tensor, device: torch.device, batch_size: int = 8) -> dict:
    """Computes held-out loss and PPL across Reflex, Limbic, and Cortex exits."""
    was_training = model.training
    model.eval()
    totals = {"reflex": 0.0, "limbic": 0.0, "cortex": 0.0}
    total_tokens = 0
    try:
        for chunk in val_blocks.split(batch_size):
            x = chunk[:, :-1].to(device)
            y = chunk[:, 1:].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
                r_logits, l_logits, c_logits, _ = model.forward_all_exits(x, update_stats=False)
            for name, logits in (("reflex", r_logits), ("limbic", l_logits), ("cortex", c_logits)):
                ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
                totals[name] += float(ce.item())
            total_tokens += int(y.numel())
    finally:
        model.train(was_training)

    out = {"val_tokens": total_tokens}
    for name in ("reflex", "limbic", "cortex"):
        loss = totals[name] / max(1, total_tokens)
        out[f"{name}_loss"] = round(loss, 4)
        out[f"{name}_ppl"] = round(math.exp(min(loss, 20.0)), 2)
    return out


@torch.no_grad()
def sample_completions(model, tokenizer, prompts: list[str], device: torch.device, max_new: int = 40) -> dict[str, str]:
    """Generates deterministic greedy Cortex continuations for qualitative inspection."""
    was_training = model.training
    model.eval()
    samples: dict[str, str] = {}
    try:
        for prompt in prompts:
            ids = tokenizer.encode(prompt).ids
            x = torch.tensor([ids], dtype=torch.long, device=device)
            cache = model.init_cache(batch_size=1)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
                logits, cache = model(x, cache=cache, force_depth=2)
                nxt = logits[:, -1:, :].argmax(dim=-1)
                gen = [int(nxt.item())]
                for _ in range(max_new - 1):
                    logits, cache = model(nxt, cache=cache, force_depth=2)
                    nxt = logits[:, -1:, :].argmax(dim=-1)
                    gen.append(int(nxt.item()))
            samples[prompt] = tokenizer.decode(gen)
    finally:
        model.train(was_training)
    return samples


def cosine_lr(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescue under-trained Cortex depth and calibrate exits")
    parser.add_argument("--checkpoint", default="checkpoints/triune_studio_step_36500.pt")
    parser.add_argument("--output-checkpoint", default="checkpoints/triune_studio_step_36500_rescued.pt")
    parser.add_argument("--output-json", default="benchmarks/checkpoint_rescue_results.json")
    parser.add_argument("--data-path", default="data/fineweb_edu_large.jsonl")
    parser.add_argument(
        "--train-data-path",
        default=None,
        help="Optional separate disjoint JSONL training corpus while keeping val strictly from --data-path",
    )
    parser.add_argument("--train-docs", type=int, default=7500)
    parser.add_argument("--val-docs", type=int, default=500)
    parser.add_argument(
        "--resume-stage1",
        default=None,
        help="Optional post-Stage-1 checkpoint to resume into Stage 1.5 / Stage 2 while keeping baseline metrics from --checkpoint",
    )
    parser.add_argument("--stage1-steps", type=int, default=3500)
    parser.add_argument("--stage1b-steps", type=int, default=400)
    parser.add_argument("--stage15-steps", type=int, default=110)
    parser.add_argument("--stage2-steps", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--muon-lr", type=float, default=1.6e-3)
    parser.add_argument("--expert-lr", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer("triune_tokenizer.json")
    sep_id = tokenizer.token_to_id("[SEP]")
    if sep_id is None:
        sep_id = 0

    rows = [json.loads(line)["text"] for line in Path(args.data_path).open(encoding="utf-8")]
    val_texts = rows[args.train_docs : args.train_docs + args.val_docs]
    if args.train_data_path:
        train_texts = [json.loads(line)["text"] for line in Path(args.train_data_path).open(encoding="utf-8")]
        train_desc = f"{args.train_data_path} ({len(train_texts)} docs)"
    else:
        train_texts = rows[: args.train_docs]
        train_desc = f"0:{len(train_texts)}"

    print(f"Packing train docs ({train_desc}) and val docs ({args.train_docs}:{args.train_docs + len(val_texts)})...")
    train_blocks = build_packed_blocks(train_texts, tokenizer, args.seq_len, sep_id)
    val_blocks = build_packed_blocks(val_texts, tokenizer, args.seq_len, sep_id, max_blocks=128)
    print(f"Packed {len(train_blocks)} train blocks and {len(val_blocks)} val blocks of len={args.seq_len}.")

    model, meta = load_trained_model(args.checkpoint, device=device)
    model.float()  # Keep FP32 master weights under BF16 AMP so low-LR updates do not round to zero

    prompts = [
        "In 1969, the Apollo 11 mission",
        "Machine learning models train using gradient",
        "Photosynthesis is the process by which",
        "The European Renaissance was a period",
    ]
    init_eval = evaluate_all_depths(model, val_blocks, device)
    init_samples = sample_completions(model, tokenizer, prompts, device)
    print(f"[Initial] {init_eval}")

    s1_executed = 0
    resumed_tokens = 0
    ckpt_raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    has_opt_state = (
        (not args.resume_stage1) and isinstance(ckpt_raw, dict) and isinstance(ckpt_raw.get("optimizer_state"), dict)
    )
    if args.resume_stage1:
        s1_payload = torch.load(args.resume_stage1, map_location="cpu", weights_only=True)
        model.load_state_dict(s1_payload["model_state"], strict=False)
        model.float()
        s1_executed = int(s1_payload.get("s1_executed", args.stage1_steps))
        resumed_tokens = int(s1_payload.get("new_tokens", 0))
        print(
            f"[Resumed Stage 1] {args.resume_stage1} (s1_steps={s1_executed}, tokens={resumed_tokens}) | val={evaluate_all_depths(model, val_blocks, device)}"
        )
        del s1_payload

    opt_rank = 64 if has_opt_state else 16
    optimizer = CentroidSteerOptimizer(
        model,
        lr=args.lr,
        muon_lr=args.muon_lr,
        expert_lr=args.expert_lr,
        rank=opt_rank,
        update_gap=20,
        steer_update_gap=5,
        steer_scale=0.2,
        core_mask_mode="full",
        use_8bit_base=False,
        weight_decay=0.01,
    )
    if has_opt_state:
        try:
            optimizer.load_state_dict(ckpt_raw["optimizer_state"])
            optimizer.initial_base_lr = float(args.lr)
            optimizer.initial_muon_lr = float(args.muon_lr)
            optimizer.initial_expert_lr = float(args.expert_lr)
            optimizer.set_lr(args.lr)
        except Exception as exc:
            print(f"[warn] Could not resume checkpoint optimizer_state ({exc}); using warm startup.", flush=True)
    del ckpt_raw

    t0 = time.perf_counter()
    new_tokens = resumed_tokens
    num_train = len(train_blocks)
    perm = torch.randperm(num_train)
    cursor = 0
    accum = max(1, int(args.grad_accum))

    def next_batch():
        nonlocal perm, cursor
        if cursor + args.batch_size > num_train:
            perm = torch.randperm(num_train)
            cursor = 0
        idx = perm[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        batch = train_blocks[idx].to(device)
        return batch[:, :-1], batch[:, 1:]

    # Stage 1: Focused Cortex recovery with proportional AdamW / Muon / CentroidSteer LRs
    stage1_tmp_path = Path(args.output_checkpoint).with_name(Path(args.output_checkpoint).stem + "_stage1_tmp.pt")
    if not args.resume_stage1:
        model.train()
        for step in range(args.stage1_steps):
            lr_cur = args.lr * min(1.0, (step + 1) / 15.0)
            optimizer.set_lr(lr_cur)
            optimizer.zero_grad()
            step_loss = 0.0
            for _ in range(accum):
                x, y = next_batch()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.enable_grad():
                    logits, _ = model(x, force_depth=2)
                    loss = fast_cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / accum
                loss.backward()
                step_loss += float(loss.detach().item())
                new_tokens += int(y.numel())
                del logits, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            s1_executed = step + 1
            if (step + 1) % 250 == 0:
                torch.save(
                    {"s1_executed": s1_executed, "new_tokens": new_tokens, "model_state": model.state_dict()},
                    str(stage1_tmp_path),
                )
            if step == 4 or (step + 1) % 150 == 0 or step == args.stage1_steps - 1:
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
                mid = evaluate_all_depths(model, val_blocks, device)
                print(
                    f"[Stage 1 Step {step + 1}/{args.stage1_steps}] elapsed={time.perf_counter() - t0:.1f}s "
                    f"vram={vram_mb:.0f}MB train_loss={step_loss:.4f} | val={mid}",
                    flush=True,
                )
                best_shallow = min(init_eval["reflex_loss"], mid["limbic_loss"], mid["reflex_loss"])
                if (step + 1) >= 200 and mid["cortex_loss"] < best_shallow - 0.025:
                    print(
                        f"[Stage 1] Cortex ({mid['cortex_loss']:.4f}) surpassed Reflex & Limbic baseline ({best_shallow:.4f}) by >0.025 nats at step {step + 1}; advancing to Stage 1.5.",
                        flush=True,
                    )
                    break

    stage1_eval = evaluate_all_depths(model, val_blocks, device)
    print(f"[Post-Stage 1 Eval] {stage1_eval}", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage 1b: Cosine-annealed Cortex convergence (skipped when resuming a Stage 1b checkpoint directly)
    s1b_executed = 0
    if (not args.resume_stage1) and args.stage1b_steps > 0:
        model.train()
        for step in range(args.stage1b_steps):
            lr_cur = cosine_lr(
                step,
                args.stage1b_steps,
                warmup_steps=50,
                lr_max=args.lr * 0.5,
                lr_min=args.lr * 0.05,
            )
            optimizer.set_lr(lr_cur)
            optimizer.zero_grad()
            step_loss = 0.0
            for _ in range(accum):
                x, y = next_batch()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.enable_grad():
                    logits, _ = model(x, force_depth=2)
                    loss = fast_cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)) / accum
                loss.backward()
                step_loss += float(loss.detach().item())
                new_tokens += int(y.numel())
                del logits, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            s1b_executed = step + 1
            if (step + 1) % 100 == 0 or step == args.stage1b_steps - 1:
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
                mid = evaluate_all_depths(model, val_blocks, device)
                print(
                    f"[Stage 1b Anneal Step {step + 1}/{args.stage1b_steps}] elapsed={time.perf_counter() - t0:.1f}s "
                    f"vram={vram_mb:.0f}MB train_loss={step_loss:.4f} | val={mid}",
                    flush=True,
                )

    # Stage 1.5: Frozen-trunk exit-head/norm alignment with per-depth LRs (counters AdamW gradient-scale invariance)
    s15_executed = 0
    if args.stage15_steps > 0:
        for p in model.parameters():
            p.requires_grad_(False)
        reflex_head_params: list[torch.nn.Parameter] = []
        limbic_head_params: list[torch.nn.Parameter] = []
        for attr in ("exit_norm", "exit_head"):
            mod_r = getattr(model.layers[model.reflex_exit_layer], attr, None)
            mod_l = getattr(model.layers[model.limbic_exit_layer], attr, None)
            if mod_r is not None:
                for p in mod_r.parameters():
                    p.requires_grad_(True)
                    reflex_head_params.append(p)
            if mod_l is not None:
                for p in mod_l.parameters():
                    p.requires_grad_(True)
                    limbic_head_params.append(p)
        head_opt = torch.optim.AdamW(
            [
                {"params": reflex_head_params, "lr": 3.8e-4, "weight_decay": 0.01},
                {"params": limbic_head_params, "lr": 8.8e-4, "weight_decay": 0.01},
            ]
        )
        model.eval()
        for step in range(args.stage15_steps):
            head_opt.zero_grad()
            step_loss = 0.0
            for _ in range(accum):
                x, y = next_batch()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.enable_grad():
                    r_logits, l_logits, c_logits, _ = model.forward_all_exits(x, update_stats=False)
                    v_dim = c_logits.size(-1)
                    y_flat = y.reshape(-1)
                    ce_r = fast_cross_entropy(r_logits.reshape(-1, v_dim), y_flat)
                    ce_l = fast_cross_entropy(l_logits.reshape(-1, v_dim), y_flat)
                    p_c = F.softmax(c_logits.detach().float().reshape(-1, v_dim), dim=-1)
                    kl_r = F.kl_div(
                        F.log_softmax(r_logits.float().reshape(-1, v_dim), dim=-1),
                        p_c,
                        reduction="batchmean",
                    )
                    kl_l = F.kl_div(
                        F.log_softmax(l_logits.float().reshape(-1, v_dim), dim=-1),
                        p_c,
                        reduction="batchmean",
                    )
                    loss = (0.50 * ce_r + 0.50 * ce_l + 0.50 * kl_r + 0.50 * kl_l) / accum
                loss.backward()
                step_loss += float(loss.detach().item())
                new_tokens += int(y.numel())
                del r_logits, l_logits, c_logits, loss
            head_opt.step()
            s15_executed = step + 1
            if (step + 1) % 30 == 0 or step == args.stage15_steps - 1:
                mid = evaluate_all_depths(model, val_blocks, device)
                print(
                    f"[Stage 1.5 HeadAlign Step {step + 1}/{args.stage15_steps}] elapsed={time.perf_counter() - t0:.1f}s "
                    f"train_loss={step_loss:.4f} | val={mid}",
                    flush=True,
                )
        for p in model.parameters():
            p.requires_grad_(True)
        del head_opt

    # Stage 2: Gentle joint trunk + Limbic/Cortex co-adaptation (keep L2 exit head anchored from Stage 1.5)
    model.train()
    for attr in ("exit_norm", "exit_head"):
        mod_r = getattr(model.layers[model.reflex_exit_layer], attr, None)
        if mod_r is not None:
            for p in mod_r.parameters():
                p.requires_grad_(False)
    s2_executed = 0
    for step in range(args.stage2_steps):
        lr_cur = cosine_lr(step, args.stage2_steps, warmup_steps=15, lr_max=args.lr * 0.12, lr_min=args.lr * 0.03)
        optimizer.set_lr(lr_cur)
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(accum):
            x, y = next_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.enable_grad():
                r_logits, l_logits, c_logits, _ = model.forward_all_exits(x, update_stats=True)
                v_dim = c_logits.size(-1)
                y_flat = y.reshape(-1)
                ce_r = fast_cross_entropy(r_logits.reshape(-1, v_dim), y_flat)
                ce_l = fast_cross_entropy(l_logits.reshape(-1, v_dim), y_flat)
                ce_c = fast_cross_entropy(c_logits.reshape(-1, v_dim), y_flat)
                p_c = F.softmax(c_logits.detach().float().reshape(-1, v_dim), dim=-1)
                kl_r = F.kl_div(
                    F.log_softmax(r_logits.float().reshape(-1, v_dim), dim=-1),
                    p_c,
                    reduction="batchmean",
                )
                kl_l = F.kl_div(
                    F.log_softmax(l_logits.float().reshape(-1, v_dim), dim=-1),
                    p_c,
                    reduction="batchmean",
                )
                loss = ((0.02 * ce_r + 0.18 * ce_l + 0.80 * ce_c) + (0.02 * kl_r + 0.08 * kl_l)) / accum
            loss.backward()
            step_loss += float(loss.detach().item())
            new_tokens += int(y.numel())
            del r_logits, l_logits, c_logits, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        s2_executed = step + 1
        if step == 4 or (step + 1) % 15 == 0 or step == args.stage2_steps - 1:
            vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
            mid = evaluate_all_depths(model, val_blocks, device)
            print(
                f"[Stage 2 Step {step + 1}/{args.stage2_steps}] elapsed={time.perf_counter() - t0:.1f}s "
                f"vram={vram_mb:.0f}MB train_loss={step_loss:.4f} | val={mid}",
                flush=True,
            )
    for p in model.parameters():
        p.requires_grad_(True)

    wall_s = time.perf_counter() - t0
    final_eval = evaluate_all_depths(model, val_blocks, device)
    final_samples = sample_completions(model, tokenizer, prompts, device)
    print(f"[Final Rescued Eval] {final_eval}")
    for p in prompts:
        print(f"  Prompt: {p!r}\n    Before: {init_samples[p]!r}\n    After:  {final_samples[p]!r}")

    assert final_eval["cortex_ppl"] < final_eval["limbic_ppl"] < final_eval["reflex_ppl"], (
        f"Expected Cortex < Limbic < Reflex PPL, got {final_eval}"
    )

    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    rescued_cfg = dict(raw_ckpt.get("config", {}) or {})
    rescued_cfg["depth_mode"] = "joint"
    rescued_cfg["exit_distill_coef"] = 0.25
    rescued_cfg["rescue_stages"] = {
        "stage1_steps": s1_executed,
        "stage1b_steps": s1b_executed,
        "stage15_steps": s15_executed,
        "stage2_steps": s2_executed,
    }

    prev_tokens = int(raw_ckpt.get("total_tokens_trained") or 0)
    total_step = int(meta["step"]) + s1_executed + s1b_executed + s15_executed + s2_executed
    model_bf16 = {
        k: (v.detach().to(device="cpu", dtype=torch.bfloat16) if v.is_floating_point() else v.detach().cpu())
        for k, v in model.state_dict().items()
    }
    payload = {
        "step": total_step,
        "model_state": model_bf16,
        "loss": final_eval["cortex_loss"],
        "total_tokens_trained": prev_tokens + new_tokens,
        "tokens_trained": prev_tokens + new_tokens,
        "config": rescued_cfg,
        "rescue_metrics": {
            "source_checkpoint": args.checkpoint,
            "initial": init_eval,
            "post_stage1": stage1_eval,
            "final": final_eval,
            "wall_time_s": round(wall_s, 2),
            "new_tokens_trained": new_tokens,
        },
    }
    out_ckpt = Path(args.output_checkpoint)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_ckpt))
    if stage1_tmp_path.exists() and not args.resume_stage1:
        stage1_tmp_path.unlink()
    print(f"[SAVED] Rescued checkpoint -> {out_ckpt}")

    report = {
        "source_checkpoint": args.checkpoint,
        "output_checkpoint": str(out_ckpt),
        "source_step": int(meta["step"]),
        "final_step": total_step,
        "new_tokens_trained": new_tokens,
        "wall_time_s": round(wall_s, 2),
        "tokens_per_sec": round(new_tokens / max(wall_s, 1e-6), 1),
        "initial_eval": init_eval,
        "stage1_eval": stage1_eval,
        "final_eval": final_eval,
        "initial_samples": init_samples,
        "final_samples": final_samples,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SAVED] Rescue telemetry JSON -> {out_json}")


if __name__ == "__main__":
    main()
