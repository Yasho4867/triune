"""Dynamic Plugin & Node Extension Registry with @register_node decorator."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type


class NodeRegistry:
    """Centralized Registry for Framework Nodes, Custom Layers, and Loss Functions."""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: Optional[str] = None,
        category: str = "Custom",
        description: str = "",
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
    ) -> Callable:
        """Decorator to register any Python function or class into the Triune ecosystem."""

        def decorator(target: Any) -> Any:
            node_name = name or (target.__name__ if hasattr(target, "__name__") else str(target))
            sig = inspect.signature(target) if inspect.isfunction(target) or inspect.isclass(target) else None

            params = {}
            if sig:
                for param_name, param in sig.parameters.items():
                    if param_name in ("self", "cls"):
                        continue
                    params[param_name] = {
                        "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": param.default if param.default != inspect.Parameter.empty else None
                    }

            schema = {
                "name": node_name,
                "category": category,
                "description": description or (target.__doc__ or ""),
                "inputs": inputs or ["in"],
                "outputs": outputs or ["out"],
                "parameters": params,
                "target": target,
            }

            self._nodes[node_name] = schema
            try:
                target._triune_node_schema = schema
            except Exception:
                pass
            return target

        return decorator

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        return self._nodes.get(name)

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        return self.get_node(name)

    def list_nodes(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": meta["name"],
                "category": meta["category"],
                "description": meta["description"],
                "inputs": meta["inputs"],
                "outputs": meta["outputs"],
                "parameters": meta["parameters"],
            }
            for meta in self._nodes.values()
        ]

    def register_builtin_nodes(self) -> None:
        """Register the complete library of built-in Triune architectural, optimizer, and runtime nodes."""
        builtins = [
            # Data & Tokenization
            {
                "name": "HuggingFaceStreamer",
                "category": "Data",
                "description": "Streams training batches directly from any Hugging Face Hub dataset (e.g. roneneldan/TinyStories, HuggingFaceFW/fineweb-edu, wikitext) with zero disk footprint.",
                "inputs": [],
                "outputs": ["dataset_stream", "text_sample"],
                "parameters": {"dataset_name": "roneneldan/TinyStories", "split": "train", "config": "", "text_column": "text", "buffer_size": 20},
            },
            {
                "name": "URLDatasetStreamer",
                "category": "Data",
                "description": "Streams remote .jsonl, .parquet, .csv, or .txt corpus directly over HTTP/HTTPS with live buffer prefetching.",
                "inputs": [],
                "outputs": ["dataset_stream", "text_sample"],
                "parameters": {"url": "https://example.com/corpus.jsonl", "text_column": "text", "buffer_size": 20},
            },
            {
                "name": "LocalFileReader",
                "category": "Data",
                "description": "Reads local text or JSONL training corpora from disk with sequence chunking.",
                "inputs": [],
                "outputs": ["dataset", "text_samples"],
                "parameters": {"file_path": "data/fineweb_sample.jsonl", "batch_size": 4, "seq_len": 64},
            },
            {
                "name": "BPETokenizer",
                "category": "Data",
                "description": "Byte-Pair Encoding tokenizer with 32,000 vocabulary, fast encode/decode, and vocabulary mapping.",
                "inputs": ["text"],
                "outputs": ["tokens", "token_ids"],
                "parameters": {"vocab_size": 32000, "pad_token": "[PAD]", "eos_token": "[EOS]"},
            },
            {
                "name": "CyclingDataLoader",
                "category": "Data",
                "description": "Continuous micro-batch generator with sequence padding and micro-batch accumulation.",
                "inputs": ["tokens"],
                "outputs": ["input_ids", "targets"],
                "parameters": {"batch_size": 4, "seq_len": 64, "shuffle": True},
            },

            # Model Architecture & Layers
            {
                "name": "TriuneTransformer",
                "category": "Model",
                "description": "Full Mixture-of-Experts core transformer with Vectorised GLA attention and dynamic hierarchical exit heads.",
                "inputs": ["input_ids"],
                "outputs": ["logits", "exit_logits", "cache"],
                "parameters": {"vocab_size": 32000, "hidden_dim": 1536, "num_layers": 24, "num_heads": 12, "num_experts": 8, "use_fp4": True},
            },
            {
                "name": "VectorisedGLA",
                "category": "Model",
                "description": "Gated Linear Attention layer with fast parallel training chunks and O(1) recurrent inference state cache.",
                "inputs": ["hidden_states"],
                "outputs": ["attention_out", "gla_cache"],
                "parameters": {"hidden_dim": 1536, "num_heads": 12, "head_dim": 128, "gate_low_rank_dim": 16},
            },
            {
                "name": "HybridAttention",
                "category": "Model",
                "description": "Vectorized GLA attention combined with Rotary Position Embeddings (RoPE) for long-context numerical stability.",
                "inputs": ["hidden_states", "rope_cos_sin"],
                "outputs": ["attention_out"],
                "parameters": {"hidden_dim": 1536, "num_heads": 12, "use_rope": True, "rope_max_seq_len": 4096},
            },
            {
                "name": "MoE_FFN",
                "category": "Model",
                "description": "Sparse Mixture-of-Experts FeedForward network with Gumbel-Softmax top-k gating, centroid tracking, and shared expert.",
                "inputs": ["hidden_states"],
                "outputs": ["moe_out", "routing_weights", "centroid_dist"],
                "parameters": {"num_experts": 8, "top_k": 2, "shared_expert": True, "capacity_multiplier": 1.25},
            },
            {
                "name": "DepthRouter",
                "category": "Model",
                "description": "Hierarchical Exit Head Router: Reflex (Layer 2/6), Limbic (Layer 4/16), and Cortex (Layer 6/24).",
                "inputs": ["hidden_states"],
                "outputs": ["exit_choice", "router_weights"],
                "parameters": {"reflex_exit_layer": 6, "limbic_exit_layer": 16, "target_depth_dist": [0.34, 0.33, 0.33]},
            },
            {
                "name": "FP8Linear",
                "category": "Model",
                "description": "Hardware-accelerated FP8 (E4M3) scaled matrix multiplication with dynamic forward/backward scaling.",
                "inputs": ["x"],
                "outputs": ["linear_out"],
                "parameters": {"in_features": 1536, "out_features": 1536, "bias": False, "dtype": "fp8_e4m3fn"},
            },
            {
                "name": "FP4Linear",
                "category": "Model",
                "description": "NVFP4 microscaling precision linear layer for Blackwell and Ada Lovelace architectures.",
                "inputs": ["x"],
                "outputs": ["linear_out"],
                "parameters": {"in_features": 1536, "out_features": 1536, "block_size": 16},
            },
            {
                "name": "RMSNorm",
                "category": "Model",
                "description": "Root Mean Square Normalization with custom Triton kernel and epsilon scaling.",
                "inputs": ["x"],
                "outputs": ["norm_out"],
                "parameters": {"dim": 1536, "eps": 1e-6},
            },
            {
                "name": "LoRAAdapter",
                "category": "Model",
                "description": "Parameter-Efficient Low-Rank Adaptation (LoRA) injection layer for fine-tuning.",
                "inputs": ["base_layer"],
                "outputs": ["adapted_layer"],
                "parameters": {"rank": 16, "alpha": 32.0, "dropout": 0.05, "target_modules": ["q_proj", "v_proj", "out_proj"]},
            },

            # Optimizers & Schedulers
            {
                "name": "CentroidSteerOptimizer",
                "category": "Optimizer",
                "description": "GaLore low-rank SVD projection combined with dynamic centroid steering and orthogonal complement updates.",
                "inputs": ["model_parameters", "loss"],
                "outputs": ["optimizer_state"],
                "parameters": {"lr": 1e-4, "betas": [0.9, 0.95], "weight_decay": 0.01, "steer_scale": 0.20, "galore_rank": 128},
            },
            {
                "name": "MuonOptimizer",
                "category": "Optimizer",
                "description": "Newton-Schulz iteration orthogonal matrix momentum optimizer for high-throughput parameter updates.",
                "inputs": ["model_parameters", "loss"],
                "outputs": ["optimizer_state"],
                "parameters": {"lr": 0.02, "momentum": 0.95, "ns_steps": 5},
            },
            {
                "name": "AdamWOptimizer",
                "category": "Optimizer",
                "description": "Standard AdamW optimizer with decoupled weight decay for 1D parameters, biases, and normalization weights.",
                "inputs": ["model_parameters", "loss"],
                "outputs": ["optimizer_state"],
                "parameters": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            },
            {
                "name": "CosineLRScheduler",
                "category": "Optimizer",
                "description": "Cosine annealing learning rate schedule with linear warmup.",
                "inputs": ["optimizer"],
                "outputs": ["scheduled_lr"],
                "parameters": {"warmup_steps": 200, "total_steps": 50000, "min_lr_ratio": 0.1},
            },

            # Loss Functions & Kernels
            {
                "name": "FastCrossEntropy",
                "category": "Loss",
                "description": "Fused chunked cross-entropy loss function with autograd gradient graph preservation.",
                "inputs": ["logits", "targets"],
                "outputs": ["loss"],
                "parameters": {"ignore_index": -100, "chunk_size": 2048, "label_smoothing": 0.0},
            },
            {
                "name": "JointExitLoss",
                "category": "Loss",
                "description": "Simultaneous multi-exit supervision combining Reflex (Exit 1), Limbic (Exit 2), and Cortex (Exit 3) cross-entropies with custom exit loss scaling.",
                "inputs": ["exit_logits", "targets"],
                "outputs": ["joint_loss", "per_exit_loss"],
                "parameters": {"lambda_reflex": 0.20, "lambda_limbic": 0.30, "lambda_cortex": 0.50, "label_smoothing": 0.0},
            },
            {
                "name": "RouterZLoss",
                "category": "Loss",
                "description": "Auxiliary router stability loss (logsumexp^2 penalty) to prevent router logit drift.",
                "inputs": ["router_logits"],
                "outputs": ["z_loss"],
                "parameters": {"coeff": 1e-3},
            },
            {
                "name": "FastRoPE",
                "category": "Kernel",
                "description": "Vectorized Rotary Position Embedding with Triton GPU acceleration and rotate_half fallback.",
                "inputs": ["q", "k", "cos", "sin"],
                "outputs": ["q_rot", "k_rot"],
                "parameters": {"dim": 64, "max_seq_len": 4096, "base": 10000.0},
            },
            {
                "name": "FastRMSNorm",
                "category": "Kernel",
                "description": "High-throughput fused RMSNorm Triton kernel with epsilon stabilization.",
                "inputs": ["x", "weight"],
                "outputs": ["norm_out"],
                "parameters": {"eps": 1e-6},
            },

            # Runtime & System
            {
                "name": "GradientAccumulator",
                "category": "Runtime",
                "description": "Multi-microbatch gradient accumulator with loss scaling (1/N), D2H asynchronous offloading, and step sync.",
                "inputs": ["loss", "model", "optimizer"],
                "outputs": ["accum_status", "effective_tokens"],
                "parameters": {"grad_accum_steps": 4, "max_grad_norm": 1.0, "sync_frequency": 1},
            },
            {
                "name": "CheckpointManager",
                "category": "Runtime",
                "description": "Schema-v2 canonical fingerprinting, atomic WSL2/disk checkpoint saving, and decoupled weights-only resume.",
                "inputs": ["model", "optimizer", "step"],
                "outputs": ["checkpoint_path", "fingerprint"],
                "parameters": {"save_dir": "checkpoints", "keep_last_n": 5, "weights_only_load": False},
            },
            {
                "name": "DynamicResourceManager",
                "category": "Runtime",
                "description": "Real-time hardware capability gating, device feasibility probing, and tied-weight deduplicated parameter tracking.",
                "inputs": ["model_config"],
                "outputs": ["feasibility_report", "vram_budget_gb"],
                "parameters": {"target_device": "cuda:0", "safety_margin_gb": 1.0},
            },
            {
                "name": "AutoregressiveGenerator",
                "category": "Model",
                "description": "O(1) GLA recurrent state-cached autoregressive generator with temperature, top-k/top-p, and repetition penalty.",
                "inputs": ["model", "tokenizer", "prompt"],
                "outputs": ["generated_text", "tokens_per_sec", "route_taken"],
                "parameters": {"max_tokens": 128, "temperature": 0.7, "top_k": 50, "repetition_penalty": 1.2, "force_depth": None},
            },
            {
                "name": "BYOKChatRouter",
                "category": "Agent",
                "description": "Multi-provider BYOK (Bring Your Own Key) chat router supporting local Triune weights, Ollama, Anthropic Claude, OpenAI, and DeepSeek.",
                "inputs": ["prompt", "system_prompt"],
                "outputs": ["response_text", "provider_telemetry"],
                "parameters": {"provider": "triune-local", "model": "triune-base", "temperature": 0.7, "max_tokens": 256},
            },
            {
                "name": "LayerStreamingEngine",
                "category": "Runtime",
                "description": "CPU-GPU ping-pong layer streaming with pinned memory buffers for training large models on 8GB VRAM.",
                "inputs": ["model"],
                "outputs": ["streaming_model"],
                "parameters": {"prefetch": True, "pin_memory": True, "d2h_async": True},
            },
            {
                "name": "VRAMProfiler",
                "category": "Runtime",
                "description": "Real-time GPU memory profiler: allocated, reserved, peak VRAM, and fragmentation leak detection.",
                "inputs": [],
                "outputs": ["vram_stats", "oom_risk"],
                "parameters": {"device": "cuda:0", "alert_threshold_gb": 7.2},
            },
            {
                "name": "PythonSandbox",
                "category": "Runtime",
                "description": "Subprocess-isolated Python execution sandbox with memory caps and execution timeout.",
                "inputs": ["code"],
                "outputs": ["result", "stdout", "stderr"],
                "parameters": {"timeout": 10, "max_memory_mb": 256},
            },
            {
                "name": "WandbLogger",
                "category": "Runtime",
                "description": "Weights & Biases cloud telemetry logger for loss curves, throughput, and exit head distribution.",
                "inputs": ["metrics"],
                "outputs": ["log_status"],
                "parameters": {"project": "triune-moe", "entity": "", "log_freq_steps": 10},
            },

            # Evaluation & Fine-Tuning
            {
                "name": "TrainingStep",
                "category": "Evaluation",
                "description": "Executes one autograd training step with gradient accumulation, clipping, and telemetry.",
                "inputs": ["model", "optimizer", "data_batch"],
                "outputs": ["step_metrics", "loss"],
                "parameters": {"grad_accum_steps": 4, "max_grad_norm": 1.0},
            },
            {
                "name": "PerplexityEvaluator",
                "category": "Evaluation",
                "description": "Computes validation loss, token prediction perplexity, and expert routing utilization.",
                "inputs": ["model", "val_dataset"],
                "outputs": ["perplexity", "val_loss"],
                "parameters": {"eval_steps": 50, "batch_size": 4},
            },
            {
                "name": "LoRAFineTuner",
                "category": "Evaluation",
                "description": "End-to-end parameter-efficient fine-tuning loop with periodic evaluation and checkpointing.",
                "inputs": ["model", "lora_config", "dataset"],
                "outputs": ["finetuned_adapter"],
                "parameters": {"epochs": 3, "lr": 2e-4, "save_adapter": True},
            },

            # Export & Deployment
            {
                "name": "SafeTensorsExport",
                "category": "Export",
                "description": "Exports model weights in SafeTensors format with memory-deduplicated tied embeddings.",
                "inputs": ["model"],
                "outputs": ["safetensors_file"],
                "parameters": {"filename": "model.safetensors", "include_optimizer": False},
            },
            {
                "name": "GGUFExport",
                "category": "Export",
                "description": "Quantizes and exports model weights to GGUF format for llama.cpp and Ollama inference.",
                "inputs": ["model"],
                "outputs": ["gguf_file"],
                "parameters": {"quantization": "Q4_K_M", "filename": "model.gguf"},
            },
            {
                "name": "ONNXExport",
                "category": "Export",
                "description": "Exports model computational graph to ONNX for TensorRT and ONNX Runtime acceleration.",
                "inputs": ["model"],
                "outputs": ["onnx_file"],
                "parameters": {"opset_version": 17, "dynamic_axes": True},
            },
        ]

        for item in builtins:
            self._nodes[item["name"]] = {
                "name": item["name"],
                "category": item["category"],
                "description": item["description"],
                "inputs": item["inputs"],
                "outputs": item["outputs"],
                "parameters": item["parameters"],
                "target": None,
            }


# Global Node Registry Instances & Aliases
global_registry = NodeRegistry()
global_registry.register_builtin_nodes()
node_registry = global_registry
register_node = global_registry.register
NODE_REGISTRY = global_registry._nodes


def list_registered_nodes() -> List[Dict[str, Any]]:
    return global_registry.list_nodes()


def get_node_executor(name: str) -> Optional[Any]:
    node = global_registry.get_node(name)
    return node["target"] if node else None
