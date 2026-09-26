"""Dynamic Hardware Resource Manager, Automated Allotment Engine, and Live Monitor.

Monitors real physical GPU VRAM, enforces hard memory budgets, dynamically sizes
microbatches and gradient accumulation steps, and provides active defragmentation
to prevent CUDA driver mapping crashes on consumer GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class VRAMBudgetExceededError(RuntimeError):
    """Raised when a model or training configuration exceeds available VRAM and --force is not set."""
    pass


@dataclass
class HardwareProfile:
    device_name: str
    total_vram_gb: float
    free_vram_gb: float
    used_vram_gb: float
    compute_capability: Tuple[int, int]
    supports_bf16: bool
    supports_fp8: bool
    has_transformer_engine: bool = False


@dataclass
class FeasibilityAssessment:
    model_name: str
    num_layers: int
    num_experts: int
    hidden_dim: int
    total_params: int
    precision: str
    param_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    gradient_memory_gb: float
    cuda_overhead_gb: float
    estimated_total_gb: float
    available_vram_gb: float
    free_vram_gb: float
    safe_budget_gb: float
    headroom_gb: float
    status: str  # "FEASIBLE", "TIGHT", "HARD_LIMIT_EXCEEDED"
    is_feasible: bool
    is_hard_exceeded: bool
    suggested_batch_size: int
    suggested_grad_accum: int
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AllotmentPlan:
    config: Dict[str, Any]
    total_params: int
    param_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    gradient_memory_gb: float
    estimated_total_gb: float
    available_vram_gb: float
    batch_size: int
    grad_accum_steps: int
    precision: str
    headroom_gb: float
    auto_fitted: bool
    forced_override: bool = False
    adjustment_reason: Optional[str] = None


class LiveVRAMMonitor:
    """Real-time GPU memory monitor with active defragmentation and telemetry."""

    def __init__(self, device: torch.device, defrag_threshold: float = 0.85):
        self.device = device
        self.defrag_threshold = defrag_threshold
        self.total_gb = (
            torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            if device.type == "cuda" else 8.0
        )
        self.last_allocated_gb = 0.0
        self.last_reserved_gb = 0.0
        self.peak_allocated_gb = 0.0
        self.defrag_count = 0

    def tick(self) -> Dict[str, float]:
        """Sample memory stats and run proactive defragmentation if needed."""
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_gb": 0.0, "headroom_gb": 8.0}

        alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        res = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        self.last_allocated_gb = alloc
        self.last_reserved_gb = res
        self.peak_allocated_gb = peak

        # Proactive defragmentation: if reserved memory exceeds threshold, purge cached blocks
        if (res / self.total_gb) >= self.defrag_threshold:
            torch.cuda.empty_cache()
            self.defrag_count += 1
            res = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
            self.last_reserved_gb = res

        headroom = max(0.0, self.total_gb - res)
        frag_pct = max(0.0, (1.0 - (alloc / max(0.001, res))) * 100.0)

        return {
            "allocated_gb": round(alloc, 2),
            "reserved_gb": round(res, 2),
            "peak_gb": round(peak, 2),
            "headroom_gb": round(headroom, 2),
            "frag_pct": round(frag_pct, 1),
        }

    def format_telemetry(self) -> str:
        """Return a formatted one-line telemetry status string."""
        stats = self.tick()
        alloc_pct = (stats["allocated_gb"] / max(0.1, self.total_gb)) * 100.0
        return (
            f"VRAM: {stats['allocated_gb']:.2f}/{self.total_gb:.2f} GB ({alloc_pct:.1f}%) "
            f"| Res: {stats['reserved_gb']:.2f} GB "
            f"| Peak: {stats['peak_gb']:.2f} GB "
            f"| Headroom: {stats['headroom_gb']:.2f} GB"
        )


class DynamicResourceManager:
    """Probes hardware, validates budgets, and dynamically allots training resources."""

    @staticmethod
    def probe_hardware(device: Optional[torch.device] = None) -> HardwareProfile:
        """Probe the active hardware environment with live memory and library inspection."""
        has_te = False
        try:
            import transformer_engine.pytorch  # noqa: F401
            has_te = True
        except ImportError:
            has_te = False

        if not torch.cuda.is_available():
            return HardwareProfile(
                device_name="CPU",
                total_vram_gb=16.0,
                free_vram_gb=16.0,
                used_vram_gb=0.0,
                compute_capability=(0, 0),
                supports_bf16=False,
                supports_fp8=False,
                has_transformer_engine=has_te,
            )

        dev = device if device is not None else torch.device("cuda:0")
        props = torch.cuda.get_device_properties(dev)
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev)

        total_vram_gb = total_bytes / (1024 ** 3)
        free_vram_gb = free_bytes / (1024 ** 3)
        used_vram_gb = total_vram_gb - free_vram_gb

        major, minor = props.major, props.minor
        supports_bf16 = major >= 8
        # FP8 hardware tensor cores: Ada Lovelace (8.9), Hopper (9.0), Blackwell (10.0, 12.0)
        supports_fp8 = (major == 8 and minor == 9) or (major >= 9)

        return HardwareProfile(
            device_name=props.name,
            total_vram_gb=round(total_vram_gb, 2),
            free_vram_gb=round(free_vram_gb, 2),
            used_vram_gb=round(used_vram_gb, 2),
            compute_capability=(major, minor),
            supports_bf16=supports_bf16,
            supports_fp8=supports_fp8,
            has_transformer_engine=has_te,
        )

    @classmethod
    def calculate_model_size(cls, config: Dict[str, Any], model: Optional[nn.Module] = None) -> Tuple[int, float]:
        """Accurately compute total parameter count and raw parameter footprint in GB."""
        if model is not None:
            seen_ids = set()
            total_params = 0
            for p in model.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    total_params += p.numel()
        else:
            vocab_size = config.get("vocab_size", 32000)
            hidden_dim = config.get("hidden_dim", 1536)
            num_layers = config.get("num_layers", 24)
            num_experts = config.get("num_experts", 8)
            expert_multiplier = config.get("expert_hidden_multiplier", 6)
            reflex_exit_layer = config.get("reflex_exit_layer", 6)

            # 1. Embeddings & output norm
            embed_params = vocab_size * hidden_dim + hidden_dim

            # 2. Transformer layers
            total_layer_params = 0
            for i in range(num_layers):
                # Attention (Q, K, V, Gate, Out) + 2 RMSNorms
                attn_params = 5 * (hidden_dim ** 2) + 2 * hidden_dim

                # FFN: MoE vs dense prefix
                if i > reflex_exit_layer:
                    expert_dim = hidden_dim * expert_multiplier
                    routed_params = num_experts * (2 * hidden_dim * expert_dim)
                    shared_params = 2 * hidden_dim * expert_dim
                    router_params = hidden_dim * num_experts
                    ffn_params = routed_params + shared_params + router_params
                else:
                    ffn_params = 2 * hidden_dim * (hidden_dim * 4)

                total_layer_params += attn_params + ffn_params

            # 3. Exit heads (Reflex, Limbic, Cortex)
            exit_head_params = 3 * (hidden_dim * vocab_size) + 3 * vocab_size
            # 4. Depth router
            depth_router_params = hidden_dim * 3 + (num_layers * 2 * hidden_dim) + hidden_dim

            total_params = embed_params + total_layer_params + exit_head_params + depth_router_params

        # Precision bytes per param
        if config.get("use_fp4"):
            bytes_per_param = 0.5
        elif config.get("use_fp8"):
            bytes_per_param = 1.0
        else:
            bytes_per_param = 2.0  # BF16

        param_memory_gb = (total_params * bytes_per_param) / (1024 ** 3)
        return total_params, param_memory_gb

    @classmethod
    def assess_feasibility(
        cls,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        user_overrides: Optional[Dict[str, Any]] = None,
        safety_ceiling: float = 0.85,
        model: Optional[nn.Module] = None,
    ) -> FeasibilityAssessment:
        """Assesses hardware capacity and model memory demands, returning recommendations."""
        hw = cls.probe_hardware(device)
        total_vram_gb = hw.total_vram_gb
        free_vram_gb = hw.free_vram_gb
        safe_budget_gb = free_vram_gb * safety_ceiling

        cfg = dict(config)
        total_params, param_memory_gb = cls.calculate_model_size(cfg, model=model)

        use_muon = cfg.get("use_muon", True)
        opt_bytes_factor = 0.5 if use_muon else 0.8
        optimizer_memory_gb = param_memory_gb * opt_bytes_factor

        cuda_overhead_gb = 0.6  # PyTorch CUDA context + cuBLAS workspace
        gradient_memory_gb = param_memory_gb * 0.5  # Checkpointed/active gradients
        baseline_static_gb = param_memory_gb + optimizer_memory_gb + gradient_memory_gb + cuda_overhead_gb

        initial_batch = cfg.get("batch_size", 2)
        initial_accum = cfg.get("grad_accum_steps", 4)
        target_effective_batch = initial_batch * initial_accum
        seq_len = cfg.get("seq_len", 256)
        num_layers = cfg.get("num_layers", 24)
        hidden_dim = cfg.get("hidden_dim", 1536)
        num_experts = cfg.get("num_experts", 8)

        # Activation per sample (with gradient checkpointing)
        act_per_sample_bytes = num_layers * seq_len * hidden_dim * 2
        act_per_sample_gb = act_per_sample_bytes / (1024 ** 3)
        activation_memory_gb = initial_batch * act_per_sample_gb

        estimated_total_gb = baseline_static_gb + activation_memory_gb
        headroom_gb = safe_budget_gb - estimated_total_gb

        # Compute suggested microbatch and grad accum
        if total_vram_gb <= 8.5 and initial_batch > 1:
            suggested_batch = 1
        else:
            remaining_headroom_gb = max(0.1, safe_budget_gb - baseline_static_gb)
            suggested_batch = max(1, min(initial_batch, int(remaining_headroom_gb // max(0.001, act_per_sample_gb * 3.0))))

        suggested_accum = max(1, math.ceil(target_effective_batch / suggested_batch))

        # Precision description
        precision = "FP4" if cfg.get("use_fp4") else ("FP8" if cfg.get("use_fp8") else "BF16")
        model_name = cfg.get("model_name", f"{num_layers}L-{num_experts}E")

        # Determine Feasibility Status
        is_hard_exceeded = param_memory_gb >= free_vram_gb
        if is_hard_exceeded or baseline_static_gb >= total_vram_gb:
            status = "HARD_LIMIT_EXCEEDED"
            is_feasible = False
        elif estimated_total_gb >= safe_budget_gb:
            status = "TIGHT"
            is_feasible = False
        else:
            status = "FEASIBLE"
            is_feasible = True

        # Formulate actionable recommendations
        recommendations = []
        if status == "HARD_LIMIT_EXCEEDED":
            if is_hard_exceeded:
                recommendations.append(
                    f"Model weights alone (~{param_memory_gb:.2f} GB in {precision}) exceed available free VRAM ({free_vram_gb:.2f} GB)."
                )
            else:
                recommendations.append(
                    f"Total static footprint (~{baseline_static_gb:.2f} GB: weights + optimizer + gradients) exceeds total GPU VRAM ({total_vram_gb:.2f} GB)."
                )
            if hw.supports_fp8 and not cfg.get("use_fp8"):
                bf16_weights_gb = (total_params * 2.0) / (1024 ** 3)
                recommendations.append(
                    f"Enable Native FP8 (--use_fp8): Halves model weight footprint from ~{bf16_weights_gb:.2f} GB to "
                    f"~{bf16_weights_gb*0.5:.2f} GB on {hw.device_name} (native PyTorch scaled GEMM, zero external builds needed)."
                )
            if hw.has_transformer_engine and not cfg.get("use_fp4"):
                recommendations.append(
                    "Enable NVFP4 (--use_fp4): Quarters model weight footprint using Transformer Engine block scaling."
                )
            elif not hw.has_transformer_engine:
                recommendations.append(
                    "Note on FP4: Transformer Engine is not currently installed in this environment; compile it from source to unlock experimental NVFP4."
                )
            recommendations.append(
                "Enable Layer Streaming (--streaming): Virtualizes memory AirLLM-style by streaming one layer at a time from CPU RAM, keeping VRAM strictly <1.5 GB regardless of model size."
            )
            if num_layers > 18:
                recommendations.append(
                    "Consider '--model_name triune-small' (18 layers, 4 experts, ~3.91 GB BF16) which fits comfortably within your 8GB GPU."
                )
            recommendations.append(
                "Override Authority: Pass '--force' to bypass pre-flight budget checks and attempt running anyway."
            )
        elif status == "TIGHT":
            recommendations.append(
                f"Configuration requires ~{estimated_total_gb:.2f} GB, leaving only ~{max(0.0, headroom_gb):.2f} GB headroom under safe budget ({safe_budget_gb:.2f} GB)."
            )
            if initial_batch > suggested_batch:
                recommendations.append(
                    f"Scale micro-batch: Set '--batch_size {suggested_batch} --grad_accum_steps {suggested_accum}' to reduce activation memory by 50%+ while preserving effective batch size ({target_effective_batch})."
                )
            recommendations.append(
                "Override Authority: Pass '--force' to proceed without micro-batch adjustments."
            )
        else:
            recommendations.append(
                f"Configuration fits comfortably within hardware budget (Estimated Headroom: ~{headroom_gb:.2f} GB)."
            )

        return FeasibilityAssessment(
            model_name=model_name,
            num_layers=num_layers,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            total_params=total_params,
            precision=precision,
            param_memory_gb=round(param_memory_gb, 2),
            optimizer_memory_gb=round(optimizer_memory_gb, 2),
            activation_memory_gb=round(activation_memory_gb, 2),
            gradient_memory_gb=round(gradient_memory_gb, 2),
            cuda_overhead_gb=round(cuda_overhead_gb, 2),
            estimated_total_gb=round(estimated_total_gb, 2),
            available_vram_gb=round(total_vram_gb, 2),
            free_vram_gb=round(free_vram_gb, 2),
            safe_budget_gb=round(safe_budget_gb, 2),
            headroom_gb=round(headroom_gb, 2),
            status=status,
            is_feasible=is_feasible,
            is_hard_exceeded=is_hard_exceeded,
            suggested_batch_size=suggested_batch,
            suggested_grad_accum=suggested_accum,
            recommendations=recommendations,
        )

    @classmethod
    def format_assessment_report(cls, assessment: FeasibilityAssessment) -> str:
        """Formats an informative, beautiful terminal card of the assessment."""
        status_badges = {
            "FEASIBLE": "✅ FEASIBLE (Fits comfortably within safe budget)",
            "TIGHT": "⚠️ TIGHT (High memory pressure; adjustments recommended)",
            "HARD_LIMIT_EXCEEDED": "❌ HARD LIMIT EXCEEDED (Requires override or optimization)",
        }
        badge = status_badges.get(assessment.status, assessment.status)

        lines = [
            "=" * 78,
            "🔍 [Resource Manager] Hardware & Feasibility Assessment",
            "=" * 78,
            f"Architecture: {assessment.model_name} ({assessment.num_layers} layers, {assessment.num_experts} experts, {assessment.hidden_dim} dim)",
            f"Parameters:   {assessment.total_params:,} ({assessment.precision})",
            f"GPU Memory:   {assessment.free_vram_gb:.2f} GB free of {assessment.available_vram_gb:.2f} GB total physical VRAM",
            "-" * 78,
            "Estimated Footprint Breakdown:",
            f"  • Model Weights ({assessment.precision}):      ~{assessment.param_memory_gb:.2f} GB",
            f"  • Optimizer (Centroid+Muon):   ~{assessment.optimizer_memory_gb:.2f} GB",
            f"  • Activations & Gradients:     ~{assessment.activation_memory_gb + assessment.gradient_memory_gb:.2f} GB",
            f"  • CUDA Context & Overhead:     ~{assessment.cuda_overhead_gb:.2f} GB",
            f"  -------------------------------------------",
            f"  • Total Projected VRAM:        ~{assessment.estimated_total_gb:.2f} GB",
            f"  • Safe Budget Ceiling (85%):   ~{assessment.safe_budget_gb:.2f} GB",
            f"  • Headroom / Deficit:           {assessment.headroom_gb:+.2f} GB",
            "-" * 78,
            f"Feasibility Status: {badge}",
            "",
            "💡 Recommendations & Options:",
        ]
        for idx, rec in enumerate(assessment.recommendations, 1):
            lines.append(f"  {idx}. {rec}")
        lines.append("=" * 78)
        return "\n".join(lines)

    @classmethod
    def validate_and_allot(
        cls,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        force: bool = False,
        auto_fit: bool = False,
        user_overrides: Optional[Dict[str, Any]] = None,
        safety_ceiling: float = 0.85,
        model: Optional[nn.Module] = None,
    ) -> AllotmentPlan:
        """Validates configuration against physical VRAM and allots resources without altering architecture."""
        overrides = user_overrides or {}
        assessment = cls.assess_feasibility(
            config, device=device, user_overrides=overrides, safety_ceiling=safety_ceiling, model=model
        )

        cfg = dict(config)
        adjustment_reasons = []
        auto_fitted = False

        # If user explicitly asked for --force, override hard checks
        if force:
            adjustment_reasons.append("⚡ User override '--force' active: proceeding regardless of VRAM budget")
        elif not assessment.is_feasible:
            # Raise descriptive error with recommendations and override hint
            report = cls.format_assessment_report(assessment)
            raise VRAMBudgetExceededError(
                f"\n{report}\n\n"
                f"❌ Execution stopped: requested configuration exceeds safe VRAM ceiling.\n"
                f"💡 To proceed anyway with user override, append '--force' to your command.\n"
                f"💡 To automatically adopt safe micro-batch sizing, append '--auto_fit'."
            )

        # Micro-batch & grad-accum tuning:
        # ONLY apply if auto_fit is True AND the user did NOT explicitly provide batch_size via CLI
        if auto_fit and "batch_size" not in overrides:
            if assessment.suggested_batch_size != cfg.get("batch_size"):
                orig_bs = cfg.get("batch_size", 2)
                cfg["batch_size"] = assessment.suggested_batch_size
                cfg["grad_accum_steps"] = assessment.suggested_grad_accum
                auto_fitted = True
                adjustment_reasons.append(
                    f"Auto-fit adjusted micro-batch from {orig_bs} -> {assessment.suggested_batch_size}, "
                    f"grad_accum -> {assessment.suggested_grad_accum}"
                )

        chosen_batch = cfg.get("batch_size", assessment.suggested_batch_size)
        chosen_accum = cfg.get("grad_accum_steps", assessment.suggested_grad_accum)
        reason_str = " | ".join(adjustment_reasons) if adjustment_reasons else None

        return AllotmentPlan(
            config=cfg,
            total_params=assessment.total_params,
            param_memory_gb=assessment.param_memory_gb,
            optimizer_memory_gb=assessment.optimizer_memory_gb,
            activation_memory_gb=assessment.activation_memory_gb,
            gradient_memory_gb=assessment.gradient_memory_gb,
            estimated_total_gb=assessment.estimated_total_gb,
            available_vram_gb=assessment.available_vram_gb,
            batch_size=chosen_batch,
            grad_accum_steps=chosen_accum,
            precision=assessment.precision,
            headroom_gb=assessment.headroom_gb,
            auto_fitted=auto_fitted,
            forced_override=force,
            adjustment_reason=reason_str,
        )

    @staticmethod
    def safe_to_device(
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        force: bool = False,
        safety_ceiling_pct: float = 0.92,
    ) -> nn.Module:
        """Safely transfers model components to GPU layer-by-layer with memory checks.

        If force=True, allows CUDA to load layers without pre-emptive ceiling checks.
        """
        if device.type != "cuda" or not torch.cuda.is_available():
            return model.to(device=device, dtype=dtype)

        props = torch.cuda.get_device_properties(device)
        total_bytes = props.total_memory
        max_allowed_bytes = total_bytes * safety_ceiling_pct

        print(f"🛡️ [Resource Manager] Initiating safe staged GPU transfer (force={force})...", flush=True)

        try:
            # 1. Transfer non-layer parameters first (embeddings, router, heads)
            if hasattr(model, "token_embed"):
                model.token_embed.to(device=device, dtype=dtype)
            if hasattr(model, "router"):
                model.router.to(device=device, dtype=dtype)
            if hasattr(model, "final_norm"):
                model.final_norm.to(device=device, dtype=dtype)

            # 2. Transfer layers iteratively
            if hasattr(model, "layers"):
                for idx, layer in enumerate(model.layers):
                    layer.to(device=device, dtype=dtype)
                    allocated_bytes = torch.cuda.memory_allocated(device)
                    if not force and allocated_bytes >= max_allowed_bytes:
                        used_gb = allocated_bytes / (1024 ** 3)
                        total_gb = total_bytes / (1024 ** 3)
                        raise VRAMBudgetExceededError(
                            f"Aborting transfer at layer {idx + 1}/{len(model.layers)}: "
                            f"allocated {used_gb:.2f} GB of {total_gb:.2f} GB exceeds safe ceiling ({safety_ceiling_pct*100:.0f}%).\n"
                            f"Pass '--force' to override this check, or optimize precision/model size."
                        )

            # 3. Any remaining submodules
            # Removed self-defeating final model.to() call that bulk-transfers everything
            # model.to(device=device, dtype=dtype)
            allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"✅ [Resource Manager] Staged transfer complete. Allocated: {allocated_gb:.2f}/{total_bytes/(1024**3):.2f} GB", flush=True)
            return model

        except torch.cuda.OutOfMemoryError as oom_err:
            print(f"❌ [Resource Manager] Physical CUDA OutOfMemory during model transfer: {oom_err}", flush=True)
            model.to(device=torch.device("cpu"))
            gc.collect()
            torch.cuda.empty_cache()
            raise
        except Exception as err:
            print(f"❌ [Resource Manager] GPU transfer failed: {err}. Rolling back to CPU...", flush=True)
            model.to(device=torch.device("cpu"))
            gc.collect()
            torch.cuda.empty_cache()
            raise
