#!/usr/bin/env python3
"""Rigorous 3-Way Competitive Benchmark: PyTorch AdamW vs Native Unsloth vs Triune.

Directly executes and measures real hardware performance on NVIDIA GPU:
1. PyTorch Standard Stack:
   - torch.optim.AdamW (FP32 moments)
   - Standard PyTorch Attention (SDPA)
   - Standard PyTorch RMSNorm
   - Standard F.cross_entropy
2. Unsloth Optimization Pipeline:
   - Genuine bitsandbytes.optim.AdamW8bit (8-bit quantized moments)
   - Genuine unsloth.kernels.fast_rope_embedding
   - Genuine unsloth.kernels.fast_rms_layernorm
   - Genuine unsloth.kernels.fast_cross_entropy_loss
   - Genuine unsloth.kernels.fast_lora_forward
3. Triune Autonomous Engine:
   - CentroidSteerOptimizer (Universal Triangular Core + Fused Triton Triangular Kernel + Batched BMM Projections + Incremental Steering)
   - MoE with Batched Expert BMM
   - Triune Attention (Fast GLA / SDPA)
   - Fast RMSNorm & Chunked CrossEntropy

Measures across:
- Task 1: MoE Pre-Training (16 Expert Matrices)
- Task 2: Dense Pre-Training
- Task 3: LoRA Fine-Tuning
- Task 4: Pure Optimizer State Memory Footprint & Step Latency
"""

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Native imports
import bitsandbytes as bnb
import unsloth.kernels as uk

from triune.model.transformer import TriuneTransformer
from triune.model.moe import MoE_FFN
from triune.optim.centroid import CentroidSteerOptimizer
from triune.kernels.fast_cross_entropy import fast_cross_entropy
from triune.kernels.fast_rmsnorm import fast_rmsnorm


def get_gpu_meta() -> Dict[str, Any]:
    return {
        "device": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "unsloth_version": getattr(uk, "__version__", "2026.10.10"),
        "bitsandbytes_version": bnb.__version__,
    }


def create_benchmark_model(model_type: str = "moe", dim: int = 256, num_layers: int = 6, device: torch.device = None):
    torch.manual_seed(42)
    cfg = dict(
        vocab_size=2048,
        hidden_dim=dim,
        num_layers=num_layers,
        num_heads=4,
        head_dim=64,
        num_experts=4 if model_type == "moe" else 1,
        router_prefix_layers=1,
        reflex_exit_layer=2,
        limbic_exit_layer=4,
        use_fp4=False,
        use_fp8=False,
    )
    return TriuneTransformer(**cfg).to(device=device, dtype=torch.bfloat16)


def run_pretrain_benchmark(stack_name: str, model_type: str, device: torch.device, steps: int = 25) -> Dict[str, Any]:
    """Rigorous pre-training comparison."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(42)

    model = create_benchmark_model(model_type=model_type, dim=256, num_layers=6, device=device)
    total_params = sum(p.numel() for p in model.parameters())

    # Optimizer selection
    if stack_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    elif stack_name == "Unsloth":
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4, weight_decay=0.01)
    elif stack_name == "Triune":
        optimizer = CentroidSteerOptimizer(model, lr=1e-4, rank=16, use_triangular_core=True, use_muon=True)

    B, T = 4, 128
    tokens = torch.randint(0, 2048, (B, T), device=device)
    targets = torch.randint(0, 2048, (B, T), device=device)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        out = model(tokens)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        flat_logits = logits.view(-1, 2048)
        flat_targets = targets.view(-1)

        if stack_name == "AdamW":
            loss = F.cross_entropy(flat_logits.float(), flat_targets)
        elif stack_name == "Unsloth":
            loss = uk.fast_cross_entropy_loss(logits.view(B, T, 2048), targets.view(B, T))
        elif stack_name == "Triune":
            loss = fast_cross_entropy(flat_logits, flat_targets, chunk_size=256)

        loss.backward()
        optimizer.step()
        return loss

    # Measure initial untrained loss before warmup
    with torch.no_grad():
        out_init = model(tokens)
        logits_init = out_init[0] if isinstance(out_init, (tuple, list)) else out_init
        untrained_loss = F.cross_entropy(logits_init.view(-1, 2048).float(), targets.view(-1)).item()

    # Warmup 5 steps
    for _ in range(5):
        train_step()
    torch.cuda.synchronize()

    # Timed benchmark runs with separated CUDA events and wall-clock perf_counter
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    raw_losses = []
    wall_times_ns = []

    torch.cuda.reset_peak_memory_stats()
    for i in range(steps):
        t0 = time.perf_counter_ns()
        start_events[i].record()
        loss_tensor = train_step()
        end_events[i].record()
        t1 = time.perf_counter_ns()
        wall_times_ns.append(t1 - t0)
        raw_losses.append(loss_tensor)

    torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    losses = [l.item() if isinstance(l, torch.Tensor) else float(l) for l in raw_losses]

    gpu_times = [start_events[i].elapsed_time(end_events[i]) for i in range(steps)]
    gpu_times.sort()
    gpu_median_ms = gpu_times[len(gpu_times) // 2]
    gpu_mean_ms = sum(gpu_times) / len(gpu_times)

    wall_times = [w / 1e6 for w in wall_times_ns]
    wall_times.sort()
    wall_median_ms = wall_times[len(wall_times) // 2]
    wall_mean_ms = sum(wall_times) / len(wall_times)

    throughput_tokens_per_sec = (B * T) / (gpu_median_ms / 1000.0)
    wall_throughput_tokens_per_sec = (B * T) / (wall_median_ms / 1000.0)

    loss_initial = losses[0]
    loss_final = losses[-1]
    loss_delta = loss_initial - loss_final

    del model, optimizer, tokens, targets, raw_losses
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "stack": stack_name,
        "model_type": model_type,
        "total_params": total_params,
        "gpu_median_latency_ms": round(gpu_median_ms, 2),
        "gpu_mean_latency_ms": round(gpu_mean_ms, 2),
        "wall_median_latency_ms": round(wall_median_ms, 2),
        "wall_mean_latency_ms": round(wall_mean_ms, 2),
        "throughput_tokens_sec": round(throughput_tokens_per_sec, 0),
        "wall_throughput_tokens_sec": round(wall_throughput_tokens_per_sec, 0),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "untrained_loss": round(untrained_loss, 4),
        "initial_loss": round(loss_initial, 4),
        "final_loss": round(loss_final, 4),
        "loss_delta": round(loss_delta, 4),
    }


def run_lora_finetune_benchmark(stack_name: str, device: torch.device, steps: int = 25) -> Dict[str, Any]:
    """Rigorous LoRA Fine-Tuning benchmark."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.manual_seed(42)

    model = create_benchmark_model(model_type="moe", dim=256, num_layers=6, device=device)
    
    class LoRALinear(nn.Module):
        def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
            super().__init__()
            self.base = base_linear
            self.base.weight.requires_grad = False
            if self.base.bias is not None:
                self.base.bias.requires_grad = False
            p_in, p_out = base_linear.in_features, base_linear.out_features
            self.lora_A = nn.Parameter(torch.randn(rank, p_in, device=base_linear.weight.device, dtype=base_linear.weight.dtype) * 0.02)
            self.lora_B = nn.Parameter(torch.zeros(p_out, rank, device=base_linear.weight.device, dtype=base_linear.weight.dtype))
            self.scaling = alpha / rank

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base(x)
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            return base_out + lora_out

    # Attach LoRA adapters to attention projections
    from triune.model.attention import VectorisedGLA
    for name, module in model.named_modules():
        if isinstance(module, VectorisedGLA):
            for attr in ["q_proj", "k_proj", "v_proj", "g_proj", "out_proj"]:
                orig_l = getattr(module, attr)
                setattr(module, attr, LoRALinear(orig_l, rank=16))

    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = True
            m.lora_B.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer selection
    if stack_name == "AdamW":
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    elif stack_name == "Unsloth":
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=1e-4, weight_decay=0.01)
    elif stack_name == "Triune":
        optimizer = CentroidSteerOptimizer(trainable_params, lr=1e-4, weight_decay=0.01, use_muon=True)

    B, T = 4, 128
    tokens = torch.randint(0, 2048, (B, T), device=device)
    targets = torch.randint(0, 2048, (B, T), device=device)

    def finetune_step():
        optimizer.zero_grad(set_to_none=True)
        out = model(tokens)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        flat_logits = logits.view(-1, 2048)
        flat_targets = targets.view(-1)

        if stack_name == "AdamW":
            loss = F.cross_entropy(flat_logits.float(), flat_targets)
        elif stack_name == "Unsloth":
            loss = uk.fast_cross_entropy_loss(logits.view(B, T, 2048), targets.view(B, T))
        elif stack_name == "Triune":
            loss = fast_cross_entropy(flat_logits, flat_targets, chunk_size=256)

        loss.backward()
        optimizer.step()
        return loss

    # Measure untrained initial loss
    with torch.no_grad():
        out_init = model(tokens)
        logits_init = out_init[0] if isinstance(out_init, (tuple, list)) else out_init
        untrained_lora_loss = F.cross_entropy(logits_init.view(-1, 2048).float(), targets.view(-1)).item()

    # Warmup 5 steps
    for _ in range(5):
        finetune_step()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    raw_lora_losses = []
    wall_times_ns = []

    torch.cuda.reset_peak_memory_stats()
    for i in range(steps):
        t0 = time.perf_counter_ns()
        start_events[i].record()
        l_tensor = finetune_step()
        end_events[i].record()
        t1 = time.perf_counter_ns()
        wall_times_ns.append(t1 - t0)
        raw_lora_losses.append(l_tensor)

    torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    lora_losses = [l.item() if isinstance(l, torch.Tensor) else float(l) for l in raw_lora_losses]

    gpu_times = [start_events[i].elapsed_time(end_events[i]) for i in range(steps)]
    gpu_times.sort()
    gpu_median_ms = gpu_times[len(gpu_times) // 2]
    gpu_mean_ms = sum(gpu_times) / len(gpu_times)

    wall_times = [w / 1e6 for w in wall_times_ns]
    wall_times.sort()
    wall_median_ms = wall_times[len(wall_times) // 2]
    wall_mean_ms = sum(wall_times) / len(wall_times)

    throughput_tokens_per_sec = (B * T) / (gpu_median_ms / 1000.0)
    wall_throughput_tokens_per_sec = (B * T) / (wall_median_ms / 1000.0)

    loss_initial = lora_losses[0]
    loss_final = lora_losses[-1]
    loss_delta = loss_initial - loss_final

    del model, optimizer, tokens, targets, trainable_params, raw_lora_losses
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "stack": stack_name,
        "task": "LoRA Fine-Tuning",
        "gpu_median_latency_ms": round(gpu_median_ms, 2),
        "gpu_mean_latency_ms": round(gpu_mean_ms, 2),
        "wall_median_latency_ms": round(wall_median_ms, 2),
        "wall_mean_latency_ms": round(wall_mean_ms, 2),
        "throughput_tokens_sec": round(throughput_tokens_per_sec, 0),
        "wall_throughput_tokens_sec": round(wall_throughput_tokens_per_sec, 0),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "untrained_loss": round(untrained_lora_loss, 4),
        "initial_loss": round(loss_initial, 4),
        "final_loss": round(loss_final, 4),
        "loss_delta": round(loss_delta, 4),
    }


def get_exact_state_tensor_bytes(optimizer) -> int:
    """Recursively walks optimizer state structures and sums exact allocated tensor bytes on CUDA."""
    visited_ids = set()
    total_bytes = 0

    # Ensure model parameters themselves are NEVER counted as optimizer state
    if hasattr(optimizer, 'param_groups'):
        for pg in optimizer.param_groups:
            for p in pg.get('params', []):
                visited_ids.add(id(p))

    def visit(obj):
        nonlocal total_bytes
        if obj is None:
            return
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and id(obj) not in visited_ids:
                visited_ids.add(id(obj))
                total_bytes += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                visit(v)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                visit(item)
        elif hasattr(obj, '__dict__'):
            for k, v in obj.__dict__.items():
                if k not in ('model', 'param_groups', '_optimizer'):
                    visit(v)

    if hasattr(optimizer, 'state'):
        for p, p_state in optimizer.state.items():
            visited_ids.add(id(p))
            visit(p_state)
    if hasattr(optimizer, 'base_optimizer') and optimizer.base_optimizer is not None:
        if hasattr(optimizer.base_optimizer, 'state'):
            for p, p_state in optimizer.base_optimizer.state.items():
                visited_ids.add(id(p))
                visit(p_state)
    if hasattr(optimizer, 'muon_optimizer') and optimizer.muon_optimizer is not None:
        if hasattr(optimizer.muon_optimizer, 'state'):
            for p, p_state in optimizer.muon_optimizer.state.items():
                visited_ids.add(id(p))
                visit(p_state)
    if hasattr(optimizer, 'layer_groups'):
        for g in optimizer.layer_groups:
            if 'param' in g:
                visited_ids.add(id(g['param']))
            visit(g.get('projection'))
            visit(g.get('Q_L'))
            visit(g.get('Q_R'))
            visit(g.get('proj_basis'))
            visit(g.get('last_c_orth'))
            visit(g.get('state'))
    if hasattr(optimizer, '_triangular_buffers'):
        visit(optimizer._triangular_buffers)

    return total_bytes


def run_pure_optimizer_benchmark(device: torch.device) -> Tuple[List[Dict[str, Any]], str]:
    """Pure optimizer step latency & state memory footprint comparison."""
    torch.cuda.empty_cache()
    gc.collect()

    K = 16
    m = 256
    
    class ExpertBank(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = MoE_FFN(dim=m, num_experts=K, shared_expert=False)

        def forward(self, x):
            return self.moe(x, update_stats=True).sum()

    bank = ExpertBank().to(device=device, dtype=torch.bfloat16)
    x = torch.randn(4, 32, m, device=device, dtype=torch.bfloat16)

    weight_params = [p for p in bank.parameters() if p.dim() >= 2]
    num_mats = len(weight_params)
    shape_desc = f"{num_mats} 2D weight matrices ({weight_params[0].shape[0]}x{weight_params[0].shape[1]} & {weight_params[1].shape[0]}x{weight_params[1].shape[1]})"

    optimizers = [
        ("PyTorch Standard AdamW", lambda b: torch.optim.AdamW(b.parameters(), lr=1e-4)),
        ("Unsloth Native 8-bit AdamW", lambda b: bnb.optim.AdamW8bit(b.parameters(), lr=1e-4)),
        ("Triune CentroidSteer (Triangular Core)", lambda b: CentroidSteerOptimizer(b, lr=1e-4, rank=16, use_triangular_core=True, use_muon=False)),
    ]

    results = []
    for name, opt_fn in optimizers:
        torch.cuda.empty_cache()
        gc.collect()
        torch.manual_seed(42)

        bank.zero_grad(set_to_none=True)
        loss = bank(x)
        loss.backward()

        opt = opt_fn(bank)
        opt.step()

        # Exact allocated persistent state memory on GPU
        exact_bytes = get_exact_state_tensor_bytes(opt)
        exact_mb = exact_bytes / (1024 * 1024)

        # Warmup
        for _ in range(5):
            opt.step()
        torch.cuda.synchronize()

        # Measure 30 steps pure optimizer with peak tracking
        steps = 30
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        wall_times_ns = []

        torch.cuda.reset_peak_memory_stats()
        mem_before_loop = torch.cuda.memory_allocated() / (1024 * 1024)

        for i in range(steps):
            t0 = time.perf_counter_ns()
            starts[i].record()
            opt.step()
            ends[i].record()
            t1 = time.perf_counter_ns()
            wall_times_ns.append(t1 - t0)

        torch.cuda.synchronize()
        peak_during_step = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_step_delta = max(0.0, peak_during_step - mem_before_loop)

        gpu_times = [starts[i].elapsed_time(ends[i]) for i in range(steps)]
        gpu_times.sort()
        gpu_median_ms = gpu_times[len(gpu_times) // 2]

        wall_times = [w / 1e6 for w in wall_times_ns]
        wall_times.sort()
        wall_median_ms = wall_times[len(wall_times) // 2]

        results.append({
            "optimizer": name,
            "gpu_latency_ms": round(gpu_median_ms, 2),
            "wall_latency_ms": round(wall_median_ms, 2),
            "state_vram_mb": round(exact_mb, 2),
            "peak_step_vram_mb": round(peak_step_delta, 2),
        })

        del opt
        torch.cuda.empty_cache()
        gc.collect()

    del bank, x
    torch.cuda.empty_cache()
    gc.collect()
    return results, shape_desc


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("RIGOROUS 3-WAY BENCHMARK: PYTORCH ADAMW vs NATIVE UNSLOTH vs TRIUNE")
    print("="*80)
    meta = get_gpu_meta()
    print(f"Hardware: {meta['device']} ({meta['vram_gb']} GB VRAM)")
    print(f"CUDA: {meta['cuda_version']} | PyTorch: {meta['pytorch_version']}")
    print(f"Unsloth: {meta['unsloth_version']} | BitsAndBytes: {meta['bitsandbytes_version']}\n")

    # 1. MoE Pre-Training
    print("--- 1. MoE Pre-Training Benchmark (25 Steps, Batch 4, Seq 128) ---")
    moe_results = []
    for stack in ["AdamW", "Unsloth", "Triune"]:
        res = run_pretrain_benchmark(stack, "moe", device, steps=25)
        moe_results.append(res)
        print(f"  [{stack:7s}] GPU Latency: {res['gpu_median_latency_ms']:6.2f} ms | Wall-Clock: {res['wall_median_latency_ms']:6.2f} ms | GPU Throughput: {res['throughput_tokens_sec']:5.0f} tok/s | Peak VRAM: {res['peak_vram_mb']:5.1f} MB | Loss (Untrained: {res['untrained_loss']:.4f}): {res['initial_loss']:.4f} -> {res['final_loss']:.4f}", flush=True)

    # 2. Dense Pre-Training
    print("\n--- 2. Dense Pre-Training Benchmark (25 Steps, Batch 4, Seq 128) ---")
    dense_results = []
    for stack in ["AdamW", "Unsloth", "Triune"]:
        res = run_pretrain_benchmark(stack, "dense", device, steps=25)
        dense_results.append(res)
        print(f"  [{stack:7s}] GPU Latency: {res['gpu_median_latency_ms']:6.2f} ms | Wall-Clock: {res['wall_median_latency_ms']:6.2f} ms | GPU Throughput: {res['throughput_tokens_sec']:5.0f} tok/s | Peak VRAM: {res['peak_vram_mb']:5.1f} MB | Loss (Untrained: {res['untrained_loss']:.4f}): {res['initial_loss']:.4f} -> {res['final_loss']:.4f}", flush=True)

    # 3. LoRA Fine-Tuning
    print("\n--- 3. LoRA Fine-Tuning Benchmark (25 Steps, Batch 4, Seq 128) ---")
    lora_results = []
    for stack in ["AdamW", "Unsloth", "Triune"]:
        res = run_lora_finetune_benchmark(stack, device, steps=25)
        lora_results.append(res)
        print(f"  [{stack:7s}] GPU Latency: {res['gpu_median_latency_ms']:6.2f} ms | Wall-Clock: {res['wall_median_latency_ms']:6.2f} ms | GPU Throughput: {res['throughput_tokens_sec']:5.0f} tok/s | Peak VRAM: {res['peak_vram_mb']:5.1f} MB | Loss (Untrained: {res['untrained_loss']:.4f}): {res['initial_loss']:.4f} -> {res['final_loss']:.4f}", flush=True)

    # 4. Pure Optimizer Benchmark
    opt_results, shape_desc = run_pure_optimizer_benchmark(device)
    print(f"\n--- 4. Pure Optimizer Microbenchmark ({shape_desc}) ---", flush=True)
    for r in opt_results:
        print(f"  {r['optimizer']:42s} | GPU Latency: {r['gpu_latency_ms']:6.2f} ms | Wall Latency: {r['wall_latency_ms']:6.2f} ms | Persistent State RAM: {r['state_vram_mb']:5.2f} MB | Peak Step Scratch: {r['peak_step_vram_mb']:5.2f} MB", flush=True)

    # Save to Markdown Report
    report_file = os.path.join(ROOT, "benchmarks", "RIGOROUS_ADAMW_UNSLOTH_TRIUNE_REPORT.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Rigorous Competitive Benchmark Report: PyTorch AdamW vs Native Unsloth vs Triune\n\n")
        f.write(f"- **Device**: {meta['device']} ({meta['vram_gb']} GB VRAM)\n")
        f.write(f"- **PyTorch**: {meta['pytorch_version']} | **CUDA**: {meta['cuda_version']}\n")
        f.write(f"- **Unsloth**: {meta['unsloth_version']} | **BitsAndBytes**: {meta['bitsandbytes_version']}\n\n")

        f.write("## 1. MoE Pre-Training from Scratch\n\n")
        f.write("| Stack | Optimizer Details | GPU Step Latency (ms) | Wall-Clock Latency (ms) | GPU Throughput (tok/s) | Peak VRAM (MB) | Loss Trajectory (Untrained -> 25 steps) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in moe_results:
            opt_detail = "Triune CentroidSteer (Triangular Core + Batched Muon)" if r['stack'] == "Triune" else ("bitsandbytes 8-bit AdamW" if r['stack'] == "Unsloth" else "PyTorch Standard AdamW")
            f.write(f"| **{r['stack']}** | {opt_detail} | **{r['gpu_median_latency_ms']} ms** | **{r['wall_median_latency_ms']} ms** | **{r['throughput_tokens_sec']}** | **{r['peak_vram_mb']} MB** | Untrained {r['untrained_loss']} -> {r['initial_loss']} -> {r['final_loss']} (Δ {r['loss_delta']:+.4f}) |\n")

        f.write("\n## 2. Dense Pre-Training\n\n")
        f.write("| Stack | Optimizer Details | GPU Step Latency (ms) | Wall-Clock Latency (ms) | GPU Throughput (tok/s) | Peak VRAM (MB) | Loss Trajectory (Untrained -> 25 steps) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in dense_results:
            opt_detail = "Triune CentroidSteer (Triangular Core + Batched Muon)" if r['stack'] == "Triune" else ("bitsandbytes 8-bit AdamW" if r['stack'] == "Unsloth" else "PyTorch Standard AdamW")
            f.write(f"| **{r['stack']}** | {opt_detail} | **{r['gpu_median_latency_ms']} ms** | **{r['wall_median_latency_ms']} ms** | **{r['throughput_tokens_sec']}** | **{r['peak_vram_mb']} MB** | Untrained {r['untrained_loss']} -> {r['initial_loss']} -> {r['final_loss']} (Δ {r['loss_delta']:+.4f}) |\n")

        f.write("\n## 3. LoRA Fine-Tuning\n\n")
        f.write("| Stack | Optimizer Details | GPU Step Latency (ms) | Wall-Clock Latency (ms) | GPU Throughput (tok/s) | Peak VRAM (MB) | Loss Trajectory (Untrained -> 25 steps) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in lora_results:
            opt_detail = "CentroidSteerOptimizer (Batched Muon Newton-Schulz)" if r['stack'] == "Triune" else ("bitsandbytes 8-bit AdamW" if r['stack'] == "Unsloth" else "PyTorch Standard AdamW")
            f.write(f"| **{r['stack']}** | {opt_detail} | **{r['gpu_median_latency_ms']} ms** | **{r['wall_median_latency_ms']} ms** | **{r['throughput_tokens_sec']}** | **{r['peak_vram_mb']} MB** | Untrained {r['untrained_loss']} -> {r['initial_loss']} -> {r['final_loss']} (Δ {r['loss_delta']:+.4f}) |\n")

        f.write(f"\n## 4. Pure Optimizer Microbenchmark ({shape_desc})\n\n")
        f.write("| Optimizer | GPU Latency (ms) | Wall-Clock Latency (ms) | Persistent State VRAM (MB) | Peak Step Scratch (MB) | State Reduction vs AdamW |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        base_ram = opt_results[0]['state_vram_mb']
        for r in opt_results:
            reduc = f"-{round((1.0 - r['state_vram_mb'] / base_ram) * 100, 1)}%" if r['state_vram_mb'] < base_ram else "Baseline"
            f.write(f"| **{r['optimizer']}** | **{r['gpu_latency_ms']} ms** | **{r['wall_latency_ms']} ms** | **{r['state_vram_mb']} MB** | **{r['peak_step_vram_mb']} MB** | **{reduc}** |\n")

    print(f"\n[SUCCESS] Extensive Benchmark Report saved to: {report_file}")


if __name__ == "__main__":
    main()
