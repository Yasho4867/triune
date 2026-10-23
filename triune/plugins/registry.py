"""Dynamic Plugin & Node Extension Registry with Strongly-Typed Ports and Execution Contracts."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type

from triune.plugins.node_schema import NodeExecutionType, PortType


def _normalize_ports(ports: Optional[List[Any]]) -> List[Dict[str, Any]]:
    if not ports:
        return []
    res = []
    for p in ports:
        if isinstance(p, dict):
            res.append({
                "name": str(p.get("name", "port")),
                "type": str(p.get("type", PortType.ANY.value)),
                "description": str(p.get("description", "")),
                "required": bool(p.get("required", True)),
            })
        elif isinstance(p, str):
            res.append({
                "name": p,
                "type": PortType.ANY.value,
                "description": "",
                "required": True,
            })
        elif hasattr(p, "name") and hasattr(p, "type"):
            res.append({
                "name": str(p.name),
                "type": str(p.type),
                "description": getattr(p, "description", ""),
                "required": bool(getattr(p, "required", True)),
            })
    return res


class NodeRegistry:
    """Centralized Registry for Framework Nodes, Custom Layers, and Loss Functions."""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._core_nodes: set[str] = set()

    def get_core_node_names(self) -> set[str]:
        """Return the set of core framework built-in node names."""
        return set(self._core_nodes)

    def register(
        self,
        name: Optional[str] = None,
        category: str = "Custom",
        description: str = "",
        inputs: Optional[List[Any]] = None,
        outputs: Optional[List[Any]] = None,
        execution_type: str = NodeExecutionType.EXECUTION.value,
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

            norm_inputs = _normalize_ports(inputs or ["in"])
            norm_outputs = _normalize_ports(outputs or ["out"])
            str_inputs = [p["name"] for p in norm_inputs]
            str_outputs = [p["name"] for p in norm_outputs]

            schema = {
                "name": node_name,
                "category": category,
                "execution_type": execution_type,
                "description": description or (target.__doc__ or ""),
                "inputs": str_inputs,
                "outputs": str_outputs,
                "typed_inputs": norm_inputs,
                "typed_outputs": norm_outputs,
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

    NODE_ALIASES = {
        "Data": "LocalFileReader",
        "DataLoader": "CyclingDataLoader",
        "Model": "TriuneTransformer",
        "Optimizer": "CentroidSteerOptimizer",
        "Loss": "FastCrossEntropy",
        "Evaluation": "PerplexityEvaluator",
        "Export": "SafeTensorsExport",
        "Tokenizer": "BPETokenizer",
    }

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        if name in self._nodes:
            return self._nodes[name]
        alias_target = self.NODE_ALIASES.get(name)
        if alias_target and alias_target in self._nodes:
            return self._nodes[alias_target]
        return None

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        return self.get_node(name)

    def list_nodes(self) -> List[Dict[str, Any]]:
        """Return backward-compatible node list with string input/output lists and typed ports."""
        res = []
        for meta in self._nodes.values():
            raw_inputs = meta.get("inputs", [])
            raw_outputs = meta.get("outputs", [])
            norm_in = _normalize_ports(raw_inputs)
            norm_out = _normalize_ports(raw_outputs)
            res.append({
                "name": meta["name"],
                "category": meta["category"],
                "execution_type": meta.get("execution_type", NodeExecutionType.EXECUTION.value),
                "description": meta["description"],
                "inputs": [p["name"] for p in norm_in],
                "outputs": [p["name"] for p in norm_out],
                "typed_inputs": norm_in,
                "typed_outputs": norm_out,
                "parameters": meta["parameters"],
            })
        return res

    def get_catalog(self) -> List[Dict[str, Any]]:
        """Return full typed catalog containing exact Port schemas and NodeExecutionType contracts."""
        res = []
        for meta in self._nodes.values():
            raw_inputs = meta.get("inputs", [])
            raw_outputs = meta.get("outputs", [])
            norm_in = _normalize_ports(raw_inputs)
            norm_out = _normalize_ports(raw_outputs)
            res.append({
                "id": meta["name"],
                "name": meta["name"],
                "category": meta["category"],
                "execution_type": meta.get("execution_type", NodeExecutionType.EXECUTION.value),
                "description": meta["description"],
                "inputs": norm_in,
                "outputs": norm_out,
                "parameters": meta.get("parameters", {}),
            })
        return res

    def register_builtin_nodes(self) -> None:
        """Register the complete library of built-in Triune architectural, optimizer, and runtime nodes."""
        builtins = [
            # Data & Tokenization
            {
                "name": "HuggingFaceStreamer",
                "category": "Data",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Streams training batches directly from any Hugging Face Hub dataset (e.g. roneneldan/TinyStories, HuggingFaceFW/fineweb-edu, wikitext) with zero disk footprint.",
                "inputs": [],
                "outputs": [
                    {"name": "dataset_stream", "type": PortType.DATASET_STREAM.value, "description": "Continuous stream iterator"},
                    {"name": "text_sample", "type": PortType.TEXT.value, "description": "Raw string text sample preview"}
                ],
                "parameters": {"dataset_name": "roneneldan/TinyStories", "split": "train", "config": "", "text_column": "text", "buffer_size": 20},
            },
            {
                "name": "URLDatasetStreamer",
                "category": "Data",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Streams remote .jsonl, .parquet, .csv, or .txt corpus directly over HTTP/HTTPS with live buffer prefetching.",
                "inputs": [],
                "outputs": [
                    {"name": "dataset_stream", "type": PortType.DATASET_STREAM.value, "description": "HTTP stream iterator"},
                    {"name": "text_sample", "type": PortType.TEXT.value, "description": "Raw string text sample"}
                ],
                "parameters": {"url": "https://example.com/corpus.jsonl", "text_column": "text", "buffer_size": 20},
            },
            {
                "name": "LocalFileReader",
                "category": "Data",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Reads local text or JSONL training corpora from disk with sequence chunking.",
                "inputs": [],
                "outputs": [
                    {"name": "dataset", "type": PortType.DATASET_STREAM.value, "description": "Local dataset stream"},
                    {"name": "dataset_stream", "type": PortType.DATASET_STREAM.value, "description": "Configured local dataset consumed by the shared training engine"},
                    {"name": "text_samples", "type": PortType.TEXT.value, "description": "Corpus text samples"},
                    {"name": "text_sample", "type": PortType.TEXT.value, "description": "Single corpus text sample preview"}
                ],
                "parameters": {"file_path": "data/fineweb_sample.jsonl", "batch_size": 4, "seq_len": 64, "configure_runtime": True},
            },
            {
                "name": "BPETokenizer",
                "category": "Data",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Byte-Pair Encoding tokenizer with 32,000 vocabulary, fast encode/decode, and vocabulary mapping.",
                "inputs": [
                    {"name": "text", "type": PortType.TEXT.value, "description": "Input raw string text"}
                ],
                "outputs": [
                    {"name": "tokens", "type": PortType.TENSOR.value, "description": "Token ID tensor of shape [batch, seq_len]"}
                ],
                "parameters": {"vocab_size": 32000, "pad_token": "[PAD]", "eos_token": "[EOS]"},
            },
            {
                "name": "CyclingDataLoader",
                "category": "Data",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Continuous micro-batch generator with sequence padding and micro-batch accumulation.",
                "inputs": [
                    {"name": "tokens", "type": PortType.TENSOR.value, "description": "Token stream tensor"}
                ],
                "outputs": [
                    {"name": "input_ids", "type": PortType.TENSOR.value, "description": "Input IDs tensor [batch, seq_len]"},
                    {"name": "targets", "type": PortType.TENSOR.value, "description": "Next-token target tensor [batch, seq_len]"}
                ],
                "parameters": {"batch_size": 4, "seq_len": 64, "shuffle": True},
            },

            # Model Architecture & Layers
            {
                "name": "TriuneTransformer",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Runs a forward pass on the active Studio model; set use_node_config=true to build a separate graph model.",
                "inputs": [
                    {"name": "input_ids", "type": PortType.TENSOR.value, "description": "Input sequence tokens [batch, seq_len]"},
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value, "description": "Optional upstream live Triune model"}
                ],
                "outputs": [
                    {"name": "logits", "type": PortType.TENSOR.value, "description": "Output logits tensor [batch, seq_len, vocab_size]"},
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value, "description": "Active PyTorch TriuneTransformer nn.Module"},
                    {"name": "exit_logits", "type": PortType.TENSOR_TUPLE.value, "description": "Tuple of Reflex, Limbic, and Cortex logits"},
                    {"name": "router_logits", "type": PortType.TENSOR.value, "description": "Router logits used by RouterZLoss"},
                    {"name": "balance_loss", "type": PortType.LOSS.value, "description": "MoE router load-balancing objective"},
                    {"name": "selected_depth", "type": PortType.CONFIG.value, "description": "Per-sequence dynamic depth choice"}
                ],
                "parameters": {"vocab_size": 32000, "hidden_dim": 256, "num_layers": 6, "num_heads": 4, "num_experts": 4, "use_fp4": False, "initialize_only": False},
            },
            {
                "name": "VectorisedGLA",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Gated Linear Attention layer with fast parallel training chunks and O(1) recurrent inference state cache.",
                "inputs": [{"name": "hidden_states", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "attention_out", "type": PortType.TENSOR.value}],
                "parameters": {"hidden_dim": 256, "num_heads": 4},
            },
            {
                "name": "HybridAttention",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Vectorized GLA attention combined with Rotary Position Embeddings (RoPE) for long-context numerical stability.",
                "inputs": [{"name": "hidden_states", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "attention_out", "type": PortType.TENSOR.value}],
                "parameters": {"hidden_dim": 256, "num_heads": 4},
            },
            {
                "name": "MoE_FFN",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Sparse Mixture-of-Experts FeedForward network with Gumbel-Softmax top-k gating, centroid tracking, and shared expert.",
                "inputs": [{"name": "hidden_states", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "moe_out", "type": PortType.TENSOR.value}],
                "parameters": {"num_experts": 4, "top_k": 2, "shared_expert": True},
            },
            {
                "name": "DepthRouter",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Hierarchical Exit Head Router: Reflex (Layer 2/6), Limbic (Layer 4/16), and Cortex (Layer 6/24).",
                "inputs": [{"name": "hidden_states", "type": PortType.TENSOR.value}],
                "outputs": [
                    {"name": "exit_choice", "type": PortType.CONFIG.value},
                    {"name": "router_logits", "type": PortType.TENSOR.value},
                    {"name": "balance_loss", "type": PortType.LOSS.value},
                ],
                "parameters": {"target_depth_dist": [0.34, 0.33, 0.33], "temperature": 1.0},
            },
            {
                "name": "FP8Linear",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Hardware-accelerated FP8 (E4M3) scaled matrix multiplication with dynamic forward/backward scaling.",
                "inputs": [{"name": "x", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "linear_out", "type": PortType.TENSOR.value}],
                "parameters": {"in_features": None, "out_features": None, "bias": False},
            },
            {
                "name": "FP4Linear",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Transformer Engine-backed linear operation; uses NVFP4 when the active precision recipe and hardware support it, otherwise the module's BF16 fallback.",
                "inputs": [{"name": "x", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "linear_out", "type": PortType.TENSOR.value}],
                "parameters": {"in_features": None, "out_features": None, "bias": True},
            },
            {
                "name": "TensorInspector",
                "category": "Analysis",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Reports real tensor shape, dtype, device, finite-value count, and numeric range/statistics.",
                "inputs": [{"name": "x", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "metrics", "type": PortType.METRICS.value}],
                "parameters": {},
            },
            {
                "name": "TokenBatchInspector",
                "category": "Analysis",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Validates and summarizes integer token IDs before training, including shape, range, unique count, and sequence length.",
                "inputs": [{"name": "tokens", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "metrics", "type": PortType.METRICS.value}],
                "parameters": {},
            },
            {
                "name": "RMSNorm",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Root Mean Square Normalization with custom Triton kernel and epsilon scaling.",
                "inputs": [{"name": "x", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "norm_out", "type": PortType.TENSOR.value}],
                "parameters": {"dim": None, "eps": 1e-6},
            },
            {
                "name": "LoRAAdapter",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Parameter-Efficient Low-Rank Adaptation (LoRA) injection layer for fine-tuning.",
                "inputs": [{"name": "base_layer", "type": PortType.MODEL_HANDLE.value}],
                "outputs": [{"name": "adapted_layer", "type": PortType.MODEL_HANDLE.value}],
                "parameters": {"rank": 16, "alpha": 32.0, "dropout": 0.05, "target_modules": ["q_proj", "v_proj", "out_proj"]},
            },

            # Optimizers & Schedulers
            {
                "name": "CentroidSteerOptimizer",
                "category": "Optimizer",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "GaLore low-rank SVD projection combined with dynamic centroid steering and orthogonal complement updates.",
                "inputs": [
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value},
                    {"name": "loss", "type": PortType.LOSS.value}
                ],
                "outputs": [
                    {"name": "optimizer_handle", "type": PortType.OPTIMIZER_HANDLE.value},
                    {"name": "metrics", "type": PortType.METRICS.value}
                ],
                "parameters": {"lr": 1e-4, "betas": [0.9, 0.95], "weight_decay": 0.01, "steer_scale": 0.20, "galore_rank": 128},
            },
            {
                "name": "MuonOptimizer",
                "category": "Optimizer",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Newton-Schulz iteration orthogonal matrix momentum optimizer for high-throughput parameter updates.",
                "inputs": [
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value},
                    {"name": "loss", "type": PortType.LOSS.value}
                ],
                "outputs": [
                    {"name": "optimizer_handle", "type": PortType.OPTIMIZER_HANDLE.value},
                    {"name": "metrics", "type": PortType.METRICS.value}
                ],
                "parameters": {"lr": 0.02, "momentum": 0.95, "ns_steps": 5},
            },
            {
                "name": "AdamWOptimizer",
                "category": "Optimizer",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Standard AdamW optimizer with decoupled weight decay for 1D parameters, biases, and normalization weights.",
                "inputs": [
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value},
                    {"name": "loss", "type": PortType.LOSS.value}
                ],
                "outputs": [
                    {"name": "optimizer_handle", "type": PortType.OPTIMIZER_HANDLE.value},
                    {"name": "metrics", "type": PortType.METRICS.value}
                ],
                "parameters": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            },
            {
                "name": "CosineLRScheduler",
                "category": "Optimizer",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Cosine annealing learning rate schedule with linear warmup.",
                "inputs": [{"name": "optimizer", "type": PortType.OPTIMIZER_HANDLE.value}],
                "outputs": [{"name": "scheduled_lr", "type": PortType.CONFIG.value}],
                "parameters": {"warmup_steps": 200, "total_steps": 50000, "min_lr_ratio": 0.1},
            },

            # Loss Functions & Kernels
            {
                "name": "FastCrossEntropy",
                "category": "Loss",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Fused chunked cross-entropy loss function with autograd gradient graph preservation.",
                "inputs": [
                    {"name": "logits", "type": PortType.TENSOR.value},
                    {"name": "targets", "type": PortType.TENSOR.value},
                    {"name": "balance_loss", "type": PortType.LOSS.value},
                    {"name": "router_logits", "type": PortType.TENSOR.value}
                ],
                "outputs": [
                    {"name": "loss", "type": PortType.LOSS.value},
                    {"name": "lm_loss", "type": PortType.LOSS.value}
                ],
                "parameters": {"ignore_index": -100, "chunk_size": 2048, "label_smoothing": 0.0, "balance_loss_weight": 0.3, "router_z_loss_weight": 1e-3},
            },
            {
                "name": "JointExitLoss",
                "category": "Loss",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Simultaneous multi-exit supervision combining Reflex (Exit 1), Limbic (Exit 2), and Cortex (Exit 3) cross-entropies.",
                "inputs": [
                    {"name": "exit_logits", "type": PortType.TENSOR_TUPLE.value},
                    {"name": "targets", "type": PortType.TENSOR.value},
                    {"name": "z_loss", "type": PortType.LOSS.value}
                ],
                "outputs": [
                    {"name": "joint_loss", "type": PortType.LOSS.value},
                    {"name": "per_exit_loss", "type": PortType.METRICS.value}
                ],
                "parameters": {"lambda_reflex": 0.20, "lambda_limbic": 0.30, "lambda_cortex": 0.50, "label_smoothing": 0.0},
            },
            {
                "name": "RouterZLoss",
                "category": "Loss",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Auxiliary router stability loss (logsumexp^2 penalty) to prevent router logit drift.",
                "inputs": [{"name": "router_logits", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "z_loss", "type": PortType.LOSS.value}],
                "parameters": {"coeff": 1e-3},
            },
            {
                "name": "FastRoPE",
                "category": "Kernel",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Vectorized Rotary Position Embedding with Triton GPU acceleration and rotate_half fallback.",
                "inputs": [{"name": "q", "type": PortType.TENSOR.value}, {"name": "k", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "q_rot", "type": PortType.TENSOR.value}, {"name": "k_rot", "type": PortType.TENSOR.value}],
                "parameters": {"dim": 64, "max_seq_len": 4096, "base": 10000.0},
            },
            {
                "name": "FastRMSNorm",
                "category": "Kernel",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "High-throughput fused RMSNorm Triton kernel with epsilon stabilization.",
                "inputs": [{"name": "x", "type": PortType.TENSOR.value}, {"name": "weight", "type": PortType.TENSOR.value}],
                "outputs": [{"name": "norm_out", "type": PortType.TENSOR.value}],
                "parameters": {"eps": 1e-6},
            },

            # Runtime & System
            {
                "name": "GradientAccumulator",
                "category": "Runtime",
                "execution_type": NodeExecutionType.SPECIFICATION.value,
                "description": "Declares gradient-accumulation settings for graph planning; use the shared TrainingStep for actual accumulation.",
                "inputs": [{"name": "loss", "type": PortType.LOSS.value}],
                "outputs": [{"name": "accum_status", "type": PortType.METRICS.value}],
                "parameters": {"grad_accum_steps": 4},
            },
            {
                "name": "CheckpointManager",
                "category": "Runtime",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Atomically saves the connected model and active optimizer state, with retention of recent checkpoints.",
                "inputs": [
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value},
                    {"name": "optimizer_handle", "type": PortType.OPTIMIZER_HANDLE.value}
                ],
                "outputs": [{"name": "checkpoint_path", "type": PortType.TEXT.value}],
                "parameters": {"save_dir": "checkpoints", "keep_last_n": 5},
            },
            {
                "name": "DynamicResourceManager",
                "category": "Runtime",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Probes the requested device and reports actual memory and hardware capabilities.",
                "inputs": [],
                "outputs": [{"name": "feasibility_report", "type": PortType.METRICS.value}],
                "parameters": {"target_device": "cuda:0", "safety_margin_gb": 1.0},
            },
            {
                "name": "AutoregressiveGenerator",
                "category": "Model",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Runs cached autoregressive generation with temperature, top-k sampling, and repetition penalty on the connected live model.",
                "inputs": [
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value},
                    {"name": "prompt", "type": PortType.TEXT.value}
                ],
                "outputs": [
                    {"name": "generated_text", "type": PortType.TEXT.value},
                    {"name": "tokens_per_sec", "type": PortType.METRICS.value}
                ],
                "parameters": {"max_tokens": 128, "temperature": 0.7, "top_k": 50, "repetition_penalty": 1.2, "force_depth": None},
            },
            {
                "name": "BYOKChatRouter",
                "category": "Agent",
                "execution_type": NodeExecutionType.SPECIFICATION.value,
                "description": "Provider configuration placeholder only; this node does not authenticate or dispatch remote requests.",
                "inputs": [{"name": "prompt", "type": PortType.TEXT.value}],
                "outputs": [{"name": "provider_config", "type": PortType.CONFIG.value}],
                "parameters": {"provider": "triune-local", "model": "triune-base", "temperature": 0.7, "max_tokens": 256},
            },
            {
                "name": "LayerStreamingEngine",
                "category": "Runtime",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Attaches Triune's layer-streaming engine to the connected live model and optimizer.",
                "inputs": [{"name": "model_handle", "type": PortType.MODEL_HANDLE.value}],
                "outputs": [{"name": "streaming_model", "type": PortType.MODEL_HANDLE.value}],
                "parameters": {"prefetch": True, "pin_memory": True, "chunk_size": 1},
            },
            {
                "name": "VRAMProfiler",
                "category": "Runtime",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Real-time GPU memory profiler: allocated, reserved, peak VRAM, and fragmentation leak detection.",
                "inputs": [],
                "outputs": [{"name": "vram_stats", "type": PortType.METRICS.value}],
                "parameters": {"device": "cuda:0", "alert_threshold_gb": 7.2},
            },
            {
                "name": "PythonSandbox",
                "category": "Runtime",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Subprocess-isolated Python execution sandbox with memory caps and execution timeout.",
                "inputs": [{"name": "code", "type": PortType.TEXT.value}],
                "outputs": [
                    {"name": "result", "type": PortType.TEXT.value},
                    {"name": "stdout", "type": PortType.TEXT.value}
                ],
                "parameters": {"timeout": 10, "max_memory_mb": 256},
            },
            {
                "name": "WandbLogger",
                "category": "Runtime",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Weights & Biases cloud telemetry logger for loss curves, throughput, and exit head distribution.",
                "inputs": [{"name": "metrics", "type": PortType.METRICS.value}],
                "outputs": [{"name": "log_status", "type": PortType.METRICS.value}],
                "parameters": {"step": None},
            },

            # Evaluation & Fine-Tuning
            {
                "name": "TrainingStep",
                "category": "Evaluation",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Executes one autograd training step with gradient accumulation, clipping, and telemetry.",
                "inputs": [
                    {"name": "dataset_stream", "type": PortType.DATASET_STREAM.value, "description": "Connected live dataset configured by the source node"},
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value},
                    {"name": "optimizer_handle", "type": PortType.OPTIMIZER_HANDLE.value},
                    {"name": "tokens", "type": PortType.TENSOR.value}
                ],
                "outputs": [
                    {"name": "step_metrics", "type": PortType.METRICS.value},
                    {"name": "loss", "type": PortType.LOSS.value},
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value, "description": "The model updated by this step"}
                ],
                "parameters": {"grad_accum_steps": 4, "max_grad_norm": 1.0},
            },
            {
                "name": "PerplexityEvaluator",
                "category": "Evaluation",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Computes validation loss and token prediction perplexity from a connected evaluation token batch.",
                "inputs": [
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value},
                    {"name": "tokens", "type": PortType.TENSOR.value},
                    {"name": "targets", "type": PortType.TENSOR.value},
                    {"name": "exit_logits", "type": PortType.TENSOR_TUPLE.value}
                ],
                "outputs": [
                    {"name": "perplexity", "type": PortType.METRICS.value},
                    {"name": "val_loss", "type": PortType.LOSS.value},
                    {"name": "per_exit_metrics", "type": PortType.METRICS.value}
                ],
                "parameters": {},
            },
            {
                "name": "LoRAFineTuner",
                "category": "Evaluation",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Attaches LoRA adapters and performs one real optimizer update on the connected token batch.",
                "inputs": [
                    {"name": "model_handle", "type": PortType.MODEL_HANDLE.value},
                    {"name": "tokens", "type": PortType.TENSOR.value}
                ],
                "outputs": [
                    {"name": "finetuned_adapter", "type": PortType.MODEL_HANDLE.value},
                    {"name": "final_loss", "type": PortType.LOSS.value}
                ],
                "parameters": {"rank": 16, "alpha": 32.0, "lr": 2e-4, "max_grad_norm": 1.0},
            },

            # Export & Deployment
            {
                "name": "SafeTensorsExport",
                "category": "Export",
                "execution_type": NodeExecutionType.EXECUTION.value,
                "description": "Exports model weights in SafeTensors format with memory-deduplicated tied embeddings.",
                "inputs": [{"name": "model_handle", "type": PortType.MODEL_HANDLE.value}],
                "outputs": [{"name": "safetensors_file", "type": PortType.TEXT.value}],
                "parameters": {"filename": "model.safetensors", "include_optimizer": False, "adapter_only": False},
            },
            {
                "name": "GGUFExport",
                "category": "Export",
                "execution_type": NodeExecutionType.SPECIFICATION.value,
                "description": "GGUF export configuration only; conversion is not performed by this node.",
                "inputs": [{"name": "model_handle", "type": PortType.MODEL_HANDLE.value}],
                "outputs": [{"name": "gguf_file", "type": PortType.CONFIG.value}],
                "parameters": {"quantization": "Q4_K_M", "filename": "model.gguf"},
            },
            {
                "name": "ONNXExport",
                "category": "Export",
                "execution_type": NodeExecutionType.SPECIFICATION.value,
                "description": "ONNX export configuration only; graph conversion is not performed by this node.",
                "inputs": [{"name": "model_handle", "type": PortType.MODEL_HANDLE.value}],
                "outputs": [{"name": "onnx_file", "type": PortType.CONFIG.value}],
                "parameters": {"opset_version": 17, "dynamic_axes": True},
            },
        ]

        for item in builtins:
            norm_in = _normalize_ports(item["inputs"])
            norm_out = _normalize_ports(item["outputs"])
            self._nodes[item["name"]] = {
                "name": item["name"],
                "category": item["category"],
                "execution_type": item.get("execution_type", NodeExecutionType.EXECUTION.value),
                "description": item["description"],
                "inputs": [p["name"] for p in norm_in],
                "outputs": [p["name"] for p in norm_out],
                "typed_inputs": norm_in,
                "typed_outputs": norm_out,
                "parameters": item["parameters"],
                "target": None,
            }
        self._core_nodes = set(self._nodes.keys())


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
