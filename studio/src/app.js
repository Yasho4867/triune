/**
 * Triune Studio v2.0 – Classic Papery & Beige Editorial AI Research IDE
 * Built with Pixel-Perfect Wire Alignment, Instant Wire Disconnecting & Live Hardware Telemetry Sync.
 */
(function () {
  const e = React.createElement;
  const { useState, useEffect, useRef } = React;

  // Blender Socket Taxonomy & Classification
  const SOCKET_TYPES = {
    TENSOR: { type: 'tensor', color: '#63c7b2', label: 'Tensor' },
    TOKENS: { type: 'tokens', color: '#a1e976', label: 'Tokens' },
    LOSS: { type: 'loss', color: '#ff4d4d', label: 'Loss' },
    OPTIMIZER: { type: 'optimizer', color: '#e8cb4f', label: 'Optimizer' },
    MODEL: { type: 'model', color: '#7a88cf', label: 'Model/Config' },
    STREAM: { type: 'stream', color: '#ffffff', label: 'Execution Flow' },
    SCALAR: { type: 'scalar', color: '#a0a0a0', label: 'Scalar' },
    REROUTE: { type: 'reroute', color: '#cccccc', label: 'Reroute' }
  };

  const PORT_TYPE_COLORS = {
    tensor: '#63c7b2',           // Blender Teal
    tokens: '#a1e976',           // Blender Lime Green
    loss: '#ff4d4d',             // Blender Red
    optimizer: '#e8cb4f',        // Blender Gold Yellow
    optimizer_handle: '#e8cb4f',
    model: '#7a88cf',            // Blender Slate Blue
    model_handle: '#7a88cf',
    stream: '#ffffff',           // Blender White Execution Flow
    dataset_stream: '#ffffff',
    scalar: '#a0a0a0',           // Blender Gray
    metrics: '#a0a0a0',
    config: '#7a88cf',
    text: '#a1e976',
    reroute: '#cccccc',
    any: '#a0a0a0'
  };

  const getPortColor = (type) => {
    if (!type) return PORT_TYPE_COLORS.tensor;
    const clean = String(type).toLowerCase().trim();
    if (SOCKET_TYPES[clean.toUpperCase()]) return SOCKET_TYPES[clean.toUpperCase()].color;
    return PORT_TYPE_COLORS[clean] || PORT_TYPE_COLORS.tensor;
  };

  const normalizePorts = (ports, defaultType = 'tensor') => {
    if (!ports || !Array.isArray(ports)) return [];
    return ports.map(p => {
      if (typeof p === 'string') {
        const lower = p.toLowerCase();
        let pType = defaultType;
        if (lower.includes('model') || lower.includes('layer') || lower.includes('adapter') || lower.includes('cfg') || lower.includes('config')) pType = 'model';
        else if (lower.includes('optimizer') || lower.includes('scheduler')) pType = 'optimizer';
        else if (lower.includes('loss')) pType = 'loss';
        else if (lower.includes('token') || lower.includes('text') || lower.includes('prompt') || lower.includes('code') || lower.includes('sample') || lower.includes('input_ids') || lower.includes('targets')) pType = 'tokens';
        else if (lower.includes('stream') || lower.includes('dataset') || lower.includes('flow')) pType = 'stream';
        else if (lower.includes('scalar') || lower.includes('metric') || lower.includes('vram') || lower.includes('stat') || lower.includes('lr') || lower.includes('step') || lower.includes('accum') || lower.includes('tokens_per_sec') || lower.includes('perplexity')) pType = 'scalar';
        else if (lower.includes('reroute')) pType = 'reroute';
        else if (lower.includes('tensor') || lower.includes('logits') || lower.includes('hidden') || lower.includes('cache') || lower.includes('attention') || lower.includes('weights')) pType = 'tensor';
        return { name: p, type: pType };
      }
      return { name: p.name || 'port', type: p.type || defaultType, description: p.description || '' };
    });
  };
  const isPortCompatible = (sourceType, targetType) => {
    const source = String(sourceType || 'any').toLowerCase();
    const target = String(targetType || 'any').toLowerCase();
    if (source === 'any' || target === 'any' || source === target) return true;
    return (source === 'dataset_stream' && target === 'text') ||
      (source === 'loss' && ['tensor', 'metrics'].includes(target)) ||
      (source === 'tensor' && ['loss', 'metrics'].includes(target));
  };

  // Frontend labels/default parameters for nodes executed by the backend registry.
  const BUILTIN_NODE_CATALOG = [
    // Data & Tokenization
    {
      name: 'HuggingFaceStreamer',
      title: 'Hugging Face Streamer',
      category: 'Data',
      description: 'Streams training batches directly from any Hugging Face Hub dataset (e.g. roneneldan/TinyStories, HuggingFaceFW/fineweb-edu, wikitext) with zero disk footprint.',
      inputs: [],
      outputs: ['dataset_stream', 'text_sample'],
      details: 'dataset_name=roneneldan/TinyStories\nsplit=train\nconfig=\ntext_column=text\nbuffer_size=20'
    },
    {
      name: 'URLDatasetStreamer',
      title: 'URL Dataset Streamer',
      category: 'Data',
      description: 'Streams remote .jsonl, .parquet, .csv, or .txt corpus directly over HTTP/HTTPS with live buffer prefetching.',
      inputs: [],
      outputs: ['dataset_stream', 'text_sample'],
      details: 'url=https://example.com/corpus.jsonl\ntext_column=text\nbuffer_size=20'
    },
    {
      name: 'LocalFileReader',
      title: 'Local File Reader',
      category: 'Data',
      description: 'Reads local text or JSONL training corpora from disk with sequence chunking.',
      inputs: [],
      outputs: ['dataset', 'dataset_stream', 'text_samples', 'text_sample'],
      details: 'file_path=data/fineweb_sample.jsonl\nbatch_size=4\nseq_len=64\nconfigure_runtime=false'
    },
    {
      name: 'BPETokenizer',
      title: 'BPE Tokenizer',
      category: 'Data',
      description: 'Byte-Pair Encoding tokenizer with 32,000 vocabulary, fast encode/decode, and vocabulary mapping.',
      inputs: ['text'],
      outputs: ['tokens'],
      details: 'vocab_size=32000\npad_token=[PAD]\neos_token=[EOS]'
    },
    {
      name: 'CyclingDataLoader',
      title: 'Cycling DataLoader',
      category: 'Data',
      description: 'Continuous micro-batch generator with sequence padding and micro-batch accumulation.',
      inputs: ['tokens'],
      outputs: ['input_ids', 'targets'],
      details: 'batch_size=4\nseq_len=64\nshuffle=True'
    },
    {
      name: 'TrainingStep',
      title: 'Autograd Training Step',
      category: 'Evaluation',
      description: 'Runs one real CLI-equivalent training step through the shared Triune TrainingEngine, including configured routing, balancing, gradient accumulation, clipping, optimizer, and telemetry.',
      inputs: ['dataset_stream'],
      outputs: ['step_metrics', 'loss', 'model_handle'],
      details: 'grad_accum_steps=4\nmax_grad_norm=1.0'
    },

    // Model Architecture & Layers
    {
      name: 'TriuneTransformer',
      title: 'Triune MoE Transformer',
      category: 'Model',
      description: 'Runs the active Studio model; set use_node_config=true to build a separate graph model.',
      inputs: ['input_ids', 'model_handle'],
      outputs: ['logits', 'exit_logits', 'router_logits', 'balance_loss', 'selected_depth', 'model_handle'],
      details: 'use_node_config=false\nreturn_all_exits=false\nreturn_balance_loss=false\nevaluation_mode=false'
    },
    {
      name: 'VectorisedGLA',
      title: 'Vectorised GLA Attention',
      category: 'Model',
      description: 'Executes Triune Gated Linear Attention on connected hidden states.',
      inputs: ['hidden_states'],
      outputs: ['attention_out'],
      details: 'num_heads=4'
    },
    {
      name: 'HybridAttention',
      title: 'Hybrid GLA + RoPE Attention',
      category: 'Model',
      description: 'Executes Triune GLA attention on connected hidden states.',
      inputs: ['hidden_states'],
      outputs: ['attention_out'],
      details: 'hidden_dim=256\nnum_heads=4'
    },
    {
      name: 'MoE_FFN',
      title: 'Sparse MoE FFN Layer',
      category: 'Model',
      description: 'Executes the configured Triune sparse MoE feed-forward layer on connected hidden states.',
      inputs: ['hidden_states'],
      outputs: ['moe_out'],
      details: 'num_experts=4\ntop_k=2\nshared_expert=True'
    },
    {
      name: 'DepthRouter',
      title: 'Hierarchical Depth Router',
      category: 'Model',
      description: 'Runs the Triune three-way depth router and returns selected exit choices and router logits.',
      inputs: ['hidden_states'],
      outputs: ['exit_choice', 'router_logits', 'balance_loss'],
      details: 'target_depth_dist=[0.34, 0.33, 0.33]\ntemperature=1.0'
    },
    {
      name: 'FP8Linear',
      title: 'FP8 Scaled Linear',
      category: 'Model',
      description: 'Hardware-accelerated FP8 (E4M3) scaled matrix multiplication with dynamic forward/backward scaling.',
      inputs: ['x'],
      outputs: ['linear_out'],
      details: 'bias=False'
    },
    {
      name: 'FP4Linear',
      title: 'NVFP4 Microscaling Linear',
      category: 'Model',
      description: 'Runs the Transformer Engine-backed linear layer under the active NVFP4 recipe when available, otherwise the BF16 fallback.',
      inputs: ['x'],
      outputs: ['linear_out'],
      details: 'in_features=1536\nout_features=1536\nbias=True'
    },
    {
      name: 'TensorInspector',
      title: 'Tensor Inspector',
      category: 'Analysis',
      description: 'Reports real tensor dimensions, device, type, finite values, and numeric statistics.',
      inputs: ['x'],
      outputs: ['metrics'],
      details: ''
    },
    {
      name: 'TokenBatchInspector',
      title: 'Token Batch Inspector',
      category: 'Analysis',
      description: 'Checks an integer token batch before training: dimensions, token count, range, unique IDs, and empty/invalid input.',
      inputs: ['tokens'],
      outputs: ['metrics'],
      details: ''
    },
    {
      name: 'RMSNorm',
      title: 'Fast RMSNorm',
      category: 'Model',
      description: 'Root Mean Square Normalization with custom Triton kernel and epsilon scaling.',
      inputs: ['x'],
      outputs: ['norm_out'],
      details: 'eps=1e-6'
    },
    {
      name: 'LoRAAdapter',
      title: 'LoRA Adapter Layer',
      category: 'Model',
      description: 'Parameter-Efficient Low-Rank Adaptation (LoRA) injection layer for fine-tuning.',
      inputs: ['base_layer'],
      outputs: ['adapted_layer'],
      details: 'rank=16\nalpha=32.0\ndropout=0.05\ntarget_modules=["q_proj", "v_proj", "out_proj"]'
    },

    // Optimizers & Schedulers
    {
      name: 'CentroidSteerOptimizer',
      title: 'CentroidSteer Optimizer',
      category: 'Optimizer',
      description: 'GaLore low-rank SVD projection combined with dynamic centroid steering and orthogonal complement updates.',
      inputs: ['model_handle', 'loss'],
      outputs: ['optimizer_handle', 'metrics'],
      details: 'lr=0.0004\nbetas=[0.9, 0.95]\nweight_decay=0.01\nsteer_scale=0.20\ngalore_rank=64'
    },
    {
      name: 'MuonOptimizer',
      title: 'Muon Matrix Optimizer',
      category: 'Optimizer',
      description: 'Newton-Schulz iteration orthogonal matrix momentum optimizer for high-throughput parameter updates.',
      inputs: ['model_handle', 'loss'],
      outputs: ['optimizer_handle', 'metrics'],
      details: 'lr=0.02\nmomentum=0.95\nns_steps=5'
    },
    {
      name: 'AdamWOptimizer',
      title: 'AdamW Optimizer',
      category: 'Optimizer',
      description: 'Standard AdamW optimizer with decoupled weight decay for 1D parameters, biases, and normalization weights.',
      inputs: ['model_handle', 'loss'],
      outputs: ['optimizer_handle', 'metrics'],
      details: 'lr=0.0004\nbetas=[0.9, 0.999]\neps=1e-8\nweight_decay=0.01'
    },
    {
      name: 'CosineLRScheduler',
      title: 'Cosine LR Scheduler',
      category: 'Optimizer',
      description: 'Cosine annealing learning rate schedule with linear warmup.',
      inputs: ['optimizer'],
      outputs: ['scheduled_lr'],
      details: 'warmup_steps=200\ntotal_steps=50000\nmin_lr_ratio=0.1'
    },

    // Loss Functions & Kernels
    {
      name: 'FastCrossEntropy',
      title: 'Fast Cross-Entropy Loss',
      category: 'Loss',
      description: 'Fused chunked cross-entropy loss function with autograd gradient graph preservation.',
      inputs: ['logits', 'targets', 'balance_loss', 'router_logits'],
      outputs: ['loss', 'lm_loss'],
      details: 'ignore_index=-100\nchunk_size=2048\nlabel_smoothing=0.0\nbalance_loss_weight=0.3\nrouter_z_loss_weight=0.001'
    },
    {
      name: 'JointExitLoss',
      title: 'Joint Exit Loss',
      category: 'Loss',
      description: 'Simultaneous multi-exit supervision combining Reflex (Exit 1), Limbic (Exit 2), and Cortex (Exit 3) cross-entropies with custom exit loss scaling.',
      inputs: ['exit_logits', 'targets', 'z_loss'],
      outputs: ['joint_loss', 'per_exit_loss'],
      details: 'lambda_reflex=0.20\nlambda_limbic=0.30\nlambda_cortex=0.50\nlabel_smoothing=0.0'
    },
    {
      name: 'RouterZLoss',
      title: 'Router Stability Z-Loss',
      category: 'Loss',
      description: 'Auxiliary router stability loss (logsumexp^2 penalty) to prevent router logit drift.',
      inputs: ['router_logits'],
      outputs: ['z_loss'],
      details: 'coeff=1e-3'
    },
    {
      name: 'FastRoPE',
      title: 'Fast RoPE Kernel',
      category: 'Model',
      description: 'Vectorized Rotary Position Embedding with Triton GPU acceleration and rotate_half fallback.',
      inputs: ['q', 'k'],
      outputs: ['q_rot', 'k_rot'],
      details: 'max_seq_len=4096\nbase=10000.0'
    },
    {
      name: 'FastRMSNorm',
      title: 'Fast RMSNorm Kernel',
      category: 'Model',
      description: 'High-throughput fused RMSNorm Triton kernel with epsilon stabilization.',
      inputs: ['x', 'weight'],
      outputs: ['norm_out'],
      details: 'eps=1e-6'
    },
    {
      name: 'AutoregressiveGenerator',
      title: 'Autoregressive Generator',
      category: 'Model',
      description: 'Generates from the connected live model using temperature, top-k sampling, and repetition penalty.',
      inputs: ['model_handle', 'prompt'],
      outputs: ['generated_text', 'tokens_per_sec'],
      details: 'max_tokens=128\ntemperature=0.7\ntop_k=50\nrepetition_penalty=1.2'
    },
    {
      name: 'BYOKChatRouter',
      title: 'BYOK Chat Router',
      category: 'Model',
      description: 'Provider configuration placeholder only; this node does not authenticate or dispatch remote requests.',
      inputs: ['prompt'],
      outputs: ['provider_config'],
      details: 'provider=triune-local\nmodel=triune-base\ntemperature=0.7\nmax_tokens=256'
    },

    // Runtime & System
    {
      name: 'GradientAccumulator',
      title: 'Gradient Accumulator',
      category: 'Runtime',
      description: 'Declares accumulation settings for graph planning; actual accumulation is performed by the shared TrainingStep.',
      inputs: ['loss'],
      outputs: ['accum_status'],
      details: 'grad_accum_steps=4'
    },
    {
      name: 'CheckpointManager',
      title: 'Checkpoint Manager',
      category: 'Runtime',
      description: 'Atomically saves the connected model and active optimizer state, retaining recent checkpoints.',
      inputs: ['model_handle', 'optimizer_handle'],
      outputs: ['checkpoint_path'],
      details: 'save_dir=checkpoints\nkeep_last_n=5'
    },
    {
      name: 'DynamicResourceManager',
      title: 'Dynamic Resource Manager',
      category: 'Runtime',
      description: 'Probes the requested device and reports actual memory and hardware capabilities.',
      inputs: [],
      outputs: ['feasibility_report'],
      details: 'target_device=cuda:0\nsafety_margin_gb=1.0'
    },
    {
      name: 'LayerStreamingEngine',
      title: 'Layer Streaming Engine',
      category: 'Runtime',
      description: 'Attaches Triune layer streaming to the connected live model and optimizer.',
      inputs: ['model_handle'],
      outputs: ['streaming_model'],
      details: 'prefetch=True\npin_memory=True\nchunk_size=1'
    },
    {
      name: 'VRAMProfiler',
      title: 'VRAM Profiler',
      category: 'Runtime',
      description: 'Real-time GPU memory profiler: allocated, reserved, peak VRAM, and fragmentation leak detection.',
      inputs: [],
      outputs: ['vram_stats'],
      details: 'device=cuda:0\nalert_threshold_gb=7.2'
    },
    {
      name: 'PythonSandbox',
      title: 'Restricted Python Runner',
      category: 'Runtime',
      description: 'Restricted Python runner with an execution timeout. Local access only by default; not a security boundary for untrusted code.',
      inputs: ['code'],
      outputs: ['result', 'stdout'],
      details: 'timeout=10\nmax_memory_mb=256'
    },
    {
      name: 'WandbLogger',
      title: 'WandB Telemetry Logger',
      category: 'Runtime',
      description: 'Logs connected metrics to an already active Weights & Biases run.',
      inputs: ['metrics'],
      outputs: ['log_status'],
      details: ''
    },

    // Evaluation & Fine-Tuning
    {
      name: 'TrainingStep',
      title: 'Autograd Training Step',
      category: 'Evaluation',
      description: 'Executes one autograd training step with gradient accumulation, clipping, and telemetry.',
      inputs: ['dataset_stream', 'model_handle', 'optimizer_handle', 'tokens'],
      outputs: ['step_metrics', 'loss'],
      details: 'grad_accum_steps=4\nmax_grad_norm=1.0'
    },
    {
      name: 'PerplexityEvaluator',
      title: 'Perplexity Evaluator',
      category: 'Evaluation',
      description: 'Computes validation loss and token prediction perplexity on the connected token batch.',
      inputs: ['model_handle', 'tokens', 'targets', 'exit_logits'],
      outputs: ['perplexity', 'val_loss', 'per_exit_metrics'],
      details: 'eval_steps=50\nbatch_size=4'
    },
    {
      name: 'LoRAFineTuner',
      title: 'LoRA Fine-Tuner Engine',
      category: 'Evaluation',
      description: 'Attaches LoRA adapters and performs one real optimizer update on the connected token batch.',
      inputs: ['model_handle', 'tokens'],
      outputs: ['finetuned_adapter', 'final_loss'],
      details: 'rank=16\nalpha=32.0\nlr=2e-4\nmax_grad_norm=1.0'
    },

    // Export & Deployment
    {
      name: 'SafeTensorsExport',
      title: 'SafeTensors Exporter',
      category: 'Export',
      description: 'Exports connected model weights in SafeTensors format.',
      inputs: ['model_handle'],
      outputs: ['safetensors_file'],
      details: 'filename=model.safetensors\ninclude_optimizer=False\nadapter_only=False'
    },
    {
      name: 'GGUFExport',
      title: 'GGUF Export (Configuration Only)',
      category: 'Export',
      description: 'Configuration-only placeholder. Triune GGUF conversion is not implemented; this node does not create a GGUF file.',
      inputs: ['model_handle'],
      outputs: ['gguf_file'],
      details: 'quantization=Q4_K_M\nfilename=model.gguf'
    },
    {
      name: 'ONNXExport',
      title: 'ONNX Graph Exporter',
      category: 'Export',
      description: 'ONNX export configuration only; graph conversion is not performed by this node.',
      inputs: ['model_handle'],
      outputs: ['onnx_file'],
      details: 'opset_version=17\ndynamic_axes=True'
    },
    {
      name: 'RerouteNode',
      title: 'Reroute Node',
      category: 'Reroute',
      description: 'Blender-style junction socket to organize and redirect wires cleanly across the canvas.',
      inputs: ['in'],
      outputs: ['out'],
      details: 'label=reroute'
    }
  ];

  // The default preset invokes the canonical TrainingEngine used by CLI and
  // Studio training. The engine owns routing, losses, optimizer, and accumulation.
  const DAG_PRESETS = {
    triune_canonical_hierarchical: {
      name: 'Run one shared-engine training step + checkpoint',
      description: 'A minimal orchestration graph that uses Studio’s active training engine instead of rebuilding a second model or optimizer.',
      requirements: 'connect a streamable dataset and initialize the active model; this graph runs one step, not a background training job.',
      nodes: [
        { id: 'node_stream', title: 'Hugging Face Streamer', type: 'Data', x: 40, y: 140, details: 'dataset_name=HuggingFaceFW/fineweb-edu\nsplit=train\nconfig=sample-10BT\ntext_column=text' },
        { id: 'node_training_step', title: 'Autograd Training Step', type: 'Evaluation', x: 430, y: 140, details: 'grad_accum_steps=4\nmax_grad_norm=1.0' },
        { id: 'node_checkpoint', title: 'Checkpoint Manager', type: 'Runtime', x: 850, y: 140, details: 'save_dir=checkpoints\nkeep_last_n=5' }
      ],
      edges: [
        { id: 'e1', source: 'node_stream', source_port: 'dataset_stream', target: 'node_training_step', target_port: 'dataset_stream' },
        { id: 'e2', source: 'node_training_step', source_port: 'model_handle', target: 'node_checkpoint', target_port: 'model_handle' }
      ]
    },
    moe_training: {
      name: 'Research: explicit MoE loss and optimizer update',
      description: 'Exposes tokenization, next-token batching, Triune forward pass, balancing and router losses, CentroidSteer update, and checkpointing as editable stages.',
      requirements: 'local corpus at data/fineweb_sample.jsonl and triune_tokenizer.json; model must fit the selected device.',
      nodes: [
        { id: 'node_data', title: 'Local File Reader', type: 'Data', x: 20, y: 150, details: 'file_path=data/fineweb_sample.jsonl\nconfigure_runtime=false' },
        { id: 'node_tokenizer', title: 'BPE Tokenizer', type: 'Data', x: 300, y: 150, details: 'tokenizer_path=triune_tokenizer.json' },
        { id: 'node_loader', title: 'Cycling DataLoader', type: 'Data', x: 560, y: 150, details: 'seq_len=64\nbatch_size=1' },
        { id: 'node_model', title: 'Triune MoE Transformer', type: 'Model', x: 840, y: 150, details: 'use_node_config=false\nreturn_balance_loss=true' },
        { id: 'node_loss', title: 'Fast Cross-Entropy Loss', type: 'Loss', x: 1150, y: 150, details: 'balance_loss_weight=0.3\nrouter_z_loss_weight=0.001' },
        { id: 'node_optimizer', title: 'CentroidSteer Optimizer', type: 'Optimizer', x: 1460, y: 150, details: 'lr=0.0004\nsteer_scale=0.2\ngalore_rank=64' },
        { id: 'node_checkpoint', title: 'Checkpoint Manager', type: 'Runtime', x: 1770, y: 150, details: 'save_dir=checkpoints\nkeep_last_n=5' }
      ],
      edges: [
        { id: 'e1', source: 'node_data', source_port: 'text_sample', target: 'node_tokenizer', target_port: 'text' },
        { id: 'e2', source: 'node_tokenizer', source_port: 'tokens', target: 'node_loader', target_port: 'tokens' },
        { id: 'e3', source: 'node_loader', source_port: 'input_ids', target: 'node_model', target_port: 'input_ids' },
        { id: 'e4', source: 'node_model', source_port: 'logits', target: 'node_loss', target_port: 'logits' },
        { id: 'e5', source: 'node_loader', source_port: 'targets', target: 'node_loss', target_port: 'targets' },
        { id: 'e6', source: 'node_model', source_port: 'balance_loss', target: 'node_loss', target_port: 'balance_loss' },
        { id: 'e7', source: 'node_model', source_port: 'router_logits', target: 'node_loss', target_port: 'router_logits' },
        { id: 'e8', source: 'node_model', source_port: 'model_handle', target: 'node_optimizer', target_port: 'model_handle' },
        { id: 'e9', source: 'node_loss', source_port: 'loss', target: 'node_optimizer', target_port: 'loss' },
        { id: 'e10', source: 'node_model', source_port: 'model_handle', target: 'node_checkpoint', target_port: 'model_handle' },
        { id: 'e11', source: 'node_optimizer', source_port: 'optimizer_handle', target: 'node_checkpoint', target_port: 'optimizer_handle' }
      ]
    },
    muon_layer_streaming: {
      name: 'Research: layer-streamed model with Muon',
      description: 'Makes layer staging and the Muon optimizer explicit so you can inspect or replace those parts of the training path.',
      requirements: 'local corpus and tokenizer; CUDA memory and model support for layer streaming.',
      nodes: [
        { id: 'node_data', title: 'Local File Reader', type: 'Data', x: 20, y: 160, details: 'file_path=data/fineweb_sample.jsonl\nconfigure_runtime=false' },
        { id: 'node_tokenizer', title: 'BPE Tokenizer', type: 'Data', x: 280, y: 160, details: 'tokenizer_path=triune_tokenizer.json' },
        { id: 'node_loader', title: 'Cycling DataLoader', type: 'Data', x: 530, y: 160, details: 'seq_len=64\nbatch_size=1' },
        { id: 'node_init', title: 'Triune MoE Transformer', type: 'Model', x: 790, y: 20, details: 'use_node_config=false\ninitialize_only=true' },
        { id: 'node_streaming', title: 'Layer Streaming Engine', type: 'Runtime', x: 1080, y: 20, details: 'prefetch=true\npin_memory=true\nchunk_size=1' },
        { id: 'node_forward', title: 'Triune MoE Transformer', type: 'Model', x: 1080, y: 280, details: 'return_balance_loss=true' },
        { id: 'node_loss', title: 'Fast Cross-Entropy Loss', type: 'Loss', x: 1380, y: 280, details: 'balance_loss_weight=0.3\nrouter_z_loss_weight=0.001' },
        { id: 'node_optimizer', title: 'Muon Matrix Optimizer', type: 'Optimizer', x: 1660, y: 280, details: 'lr=0.02\nmomentum=0.95\nns_steps=5' },
        { id: 'node_checkpoint', title: 'Checkpoint Manager', type: 'Runtime', x: 1950, y: 280, details: 'save_dir=checkpoints\nkeep_last_n=5' }
      ],
      edges: [
        { id: 'e1', source: 'node_data', source_port: 'text_sample', target: 'node_tokenizer', target_port: 'text' },
        { id: 'e2', source: 'node_tokenizer', source_port: 'tokens', target: 'node_loader', target_port: 'tokens' },
        { id: 'e3', source: 'node_init', source_port: 'model_handle', target: 'node_streaming', target_port: 'model_handle' },
        { id: 'e4', source: 'node_streaming', source_port: 'streaming_model', target: 'node_forward', target_port: 'model_handle' },
        { id: 'e5', source: 'node_loader', source_port: 'input_ids', target: 'node_forward', target_port: 'input_ids' },
        { id: 'e6', source: 'node_forward', source_port: 'logits', target: 'node_loss', target_port: 'logits' },
        { id: 'e7', source: 'node_loader', source_port: 'targets', target: 'node_loss', target_port: 'targets' },
        { id: 'e8', source: 'node_forward', source_port: 'balance_loss', target: 'node_loss', target_port: 'balance_loss' },
        { id: 'e9', source: 'node_forward', source_port: 'router_logits', target: 'node_loss', target_port: 'router_logits' },
        { id: 'e10', source: 'node_streaming', source_port: 'streaming_model', target: 'node_optimizer', target_port: 'model_handle' },
        { id: 'e11', source: 'node_loss', source_port: 'loss', target: 'node_optimizer', target_port: 'loss' },
        { id: 'e12', source: 'node_streaming', source_port: 'streaming_model', target: 'node_checkpoint', target_port: 'model_handle' },
        { id: 'e13', source: 'node_optimizer', source_port: 'optimizer_handle', target: 'node_checkpoint', target_port: 'optimizer_handle' }
      ]
    },
    lora_finetune: {
      name: 'Adapt a model with LoRA, then export the adapter',
      description: 'Loads a prompt sample, tokenizes it, performs one LoRA update on the active model, and writes adapter weights.',
      requirements: 'local corpus at data/fineweb_sample.jsonl, tokenizer, initialized model, and writable export location.',
      nodes: [
        { id: 'node_data', title: 'Local File Reader', type: 'Data', x: 40, y: 120, details: 'file_path=data/fineweb_sample.jsonl\nconfigure_runtime=false' },
        { id: 'node_tokenizer', title: 'BPE Tokenizer', type: 'Data', x: 330, y: 120, details: 'tokenizer_path=triune_tokenizer.json' },
        { id: 'node_model', title: 'Triune MoE Transformer', type: 'Model', x: 650, y: 120, details: 'use_node_config=false\ninitialize_only=true' },
        { id: 'node_lora', title: 'LoRA Fine-Tuner Engine', type: 'Evaluation', x: 990, y: 120, details: 'rank=16\nalpha=32.0\nlr=2e-4\nmax_grad_norm=1.0' },
        { id: 'node_export', title: 'SafeTensors Exporter', type: 'Export', x: 1360, y: 120, details: 'filename=lora_adapter.safetensors\nadapter_only=true' }
      ],
      edges: [
        { id: 'e1', source: 'node_data', source_port: 'text_sample', target: 'node_tokenizer', target_port: 'text' },
        { id: 'e2', source: 'node_model', source_port: 'model_handle', target: 'node_lora', target_port: 'model_handle' },
        { id: 'e3', source: 'node_tokenizer', source_port: 'tokens', target: 'node_lora', target_port: 'tokens' },
        { id: 'e4', source: 'node_lora', source_port: 'finetuned_adapter', target: 'node_export', target_port: 'model_handle' }
      ]
    },
    depth_routing_inference: {
      name: 'Evaluate Reflex, Limbic, and Cortex exits',
      description: 'Runs a held-out token batch through the active Triune model and reports perplexity and per-exit metrics.',
      requirements: 'local corpus and tokenizer; initialize the model before executing.',
      nodes: [
        { id: 'node_data', title: 'Local File Reader', type: 'Data', x: 30, y: 150, details: 'file_path=data/fineweb_sample.jsonl\nconfigure_runtime=false' },
        { id: 'node_tokenizer', title: 'BPE Tokenizer', type: 'Data', x: 330, y: 150, details: 'tokenizer_path=triune_tokenizer.json' },
        { id: 'node_loader', title: 'Cycling DataLoader', type: 'Data', x: 610, y: 150, details: 'seq_len=64\nbatch_size=1' },
        { id: 'node_model', title: 'Triune MoE Transformer', type: 'Model', x: 910, y: 150, details: 'use_node_config=false\nreturn_all_exits=true\nevaluation_mode=true' },
        { id: 'node_eval', title: 'Perplexity Evaluator', type: 'Evaluation', x: 1300, y: 150, details: '' }
      ],
      edges: [
        { id: 'e1', source: 'node_data', source_port: 'text_sample', target: 'node_tokenizer', target_port: 'text' },
        { id: 'e2', source: 'node_tokenizer', source_port: 'tokens', target: 'node_loader', target_port: 'tokens' },
        { id: 'e3', source: 'node_loader', source_port: 'input_ids', target: 'node_model', target_port: 'input_ids' },
        { id: 'e4', source: 'node_model', source_port: 'model_handle', target: 'node_eval', target_port: 'model_handle' },
        { id: 'e5', source: 'node_loader', source_port: 'targets', target: 'node_eval', target_port: 'targets' },
        { id: 'e6', source: 'node_model', source_port: 'exit_logits', target: 'node_eval', target_port: 'exit_logits' }
      ]
    },
    fp8_quantization_export: {
      name: 'Test the standalone FP8 linear operator',
      description: 'Routes the three-way router logits through a 3-input FP8 linear layer and inspects numeric output; it does not quantize/export the whole model.',
      requirements: 'local corpus and tokenizer; supported FP8 hardware/runtime, otherwise the node may report unavailable.',
      nodes: [
        { id: 'node_data', title: 'Local File Reader', type: 'Data', x: 25, y: 150, details: 'file_path=data/fineweb_sample.jsonl\nconfigure_runtime=false' },
        { id: 'node_tokenizer', title: 'BPE Tokenizer', type: 'Data', x: 300, y: 150, details: 'tokenizer_path=triune_tokenizer.json' },
        { id: 'node_loader', title: 'Cycling DataLoader', type: 'Data', x: 560, y: 150, details: 'seq_len=64\nbatch_size=1' },
        { id: 'node_model', title: 'Triune MoE Transformer', type: 'Model', x: 840, y: 150, details: 'use_node_config=false\nevaluation_mode=true' },
        { id: 'node_fp8', title: 'FP8 Scaled Linear', type: 'Model', x: 1170, y: 150, details: 'in_features=3\nout_features=3\nbias=false' },
        { id: 'node_inspect', title: 'Tensor Inspector', type: 'Analysis', x: 1500, y: 150, details: '' }
      ],
      edges: [
        { id: 'e1', source: 'node_data', source_port: 'text_sample', target: 'node_tokenizer', target_port: 'text' },
        { id: 'e2', source: 'node_tokenizer', source_port: 'tokens', target: 'node_loader', target_port: 'tokens' },
        { id: 'e3', source: 'node_loader', source_port: 'input_ids', target: 'node_model', target_port: 'input_ids' },
        { id: 'e4', source: 'node_model', source_port: 'router_logits', target: 'node_fp8', target_port: 'x' },
        { id: 'e5', source: 'node_fp8', source_port: 'linear_out', target: 'node_inspect', target_port: 'x' }
      ]
    },
    hybrid_gla_rope: {
      name: 'Generate from a local corpus prompt',
      description: 'Uses the first local text sample as a prompt and generates with the active Triune model’s cached autoregressive path.',
      requirements: 'local corpus, initialized model, and an available tokenizer.',
      nodes: [
        { id: 'node_prompt', title: 'Local File Reader', type: 'Data', x: 80, y: 150, details: 'file_path=data/fineweb_sample.jsonl\nconfigure_runtime=false' },
        { id: 'node_model', title: 'Triune MoE Transformer', type: 'Model', x: 460, y: 150, details: 'use_node_config=false\ninitialize_only=true' },
        { id: 'node_generate', title: 'Autoregressive Generator', type: 'Model', x: 850, y: 150, details: 'max_tokens=64\ntemperature=0.7\ntop_k=50\nrepetition_penalty=1.2' }
      ],
      edges: [
        { id: 'e1', source: 'node_prompt', source_port: 'text_sample', target: 'node_generate', target_port: 'prompt' },
        { id: 'e2', source: 'node_model', source_port: 'model_handle', target: 'node_generate', target_port: 'model_handle' }
      ]
    },
    joint_multi_exit_training: {
      name: 'Research: train all exits with explicit objectives',
      description: 'Combines Reflex/Limbic/Cortex supervision with router stability loss and a CentroidSteer update.',
      requirements: 'local corpus and tokenizer; the active model must return all exits and router logits.',
      nodes: [
        { id: 'node_data', title: 'Local File Reader', type: 'Data', x: 20, y: 150, details: 'file_path=data/fineweb_sample.jsonl\nconfigure_runtime=false' },
        { id: 'node_tokenizer', title: 'BPE Tokenizer', type: 'Data', x: 300, y: 150, details: 'tokenizer_path=triune_tokenizer.json' },
        { id: 'node_loader', title: 'Cycling DataLoader', type: 'Data', x: 570, y: 150, details: 'seq_len=64\nbatch_size=1' },
        { id: 'node_model', title: 'Triune MoE Transformer', type: 'Model', x: 860, y: 150, details: 'use_node_config=false\nreturn_all_exits=true' },
        { id: 'node_zloss', title: 'Router Stability Z-Loss', type: 'Loss', x: 1190, y: 30, details: 'coeff=0.001' },
        { id: 'node_joint', title: 'Joint Exit Loss', type: 'Loss', x: 1190, y: 260, details: 'lambda_reflex=0.2\nlambda_limbic=0.3\nlambda_cortex=0.5' },
        { id: 'node_optimizer', title: 'CentroidSteer Optimizer', type: 'Optimizer', x: 1520, y: 150, details: 'lr=0.0004\nsteer_scale=0.2\ngalore_rank=64' }
      ],
      edges: [
        { id: 'e1', source: 'node_data', source_port: 'text_sample', target: 'node_tokenizer', target_port: 'text' },
        { id: 'e2', source: 'node_tokenizer', source_port: 'tokens', target: 'node_loader', target_port: 'tokens' },
        { id: 'e3', source: 'node_loader', source_port: 'input_ids', target: 'node_model', target_port: 'input_ids' },
        { id: 'e4', source: 'node_model', source_port: 'exit_logits', target: 'node_joint', target_port: 'exit_logits' },
        { id: 'e5', source: 'node_loader', source_port: 'targets', target: 'node_joint', target_port: 'targets' },
        { id: 'e6', source: 'node_model', source_port: 'router_logits', target: 'node_zloss', target_port: 'router_logits' },
        { id: 'e7', source: 'node_zloss', source_port: 'z_loss', target: 'node_joint', target_port: 'z_loss' },
        { id: 'e8', source: 'node_model', source_port: 'model_handle', target: 'node_optimizer', target_port: 'model_handle' },
        { id: 'e9', source: 'node_joint', source_port: 'joint_loss', target: 'node_optimizer', target_port: 'loss' }
      ]
    },
    local_corpus_qa: {
      name: 'Check a local corpus and tokenizer (CPU-friendly)',
      description: 'Reads a real local sample, tokenizes it, then reports token tensor shape, range, and unique-ID count without replacing the active training dataset.',
      requirements: 'data/fineweb_sample.jsonl and triune_tokenizer.json; no model or GPU required.',
      nodes: [
        { id: 'node_data', title: 'Local File Reader', type: 'Data', x: 40, y: 160, details: 'file_path=data/fineweb_sample.jsonl\nconfigure_runtime=false' },
        { id: 'node_tokenizer', title: 'BPE Tokenizer', type: 'Data', x: 380, y: 160, details: 'tokenizer_path=triune_tokenizer.json\nconfigure_runtime=false' },
        { id: 'node_inspect', title: 'Token Batch Inspector', type: 'Analysis', x: 740, y: 160, details: '' }
      ],
      edges: [
        { id: 'e1', source: 'node_data', source_port: 'text_sample', target: 'node_tokenizer', target_port: 'text' },
        { id: 'e2', source: 'node_tokenizer', source_port: 'tokens', target: 'node_inspect', target_port: 'tokens' }
      ]
    }
  };

  const readSavedWorkflow = () => {
    try {
      const raw = window.localStorage.getItem('triune.studio.workflow.v1');
      const saved = raw ? JSON.parse(raw) : null;
      if (saved && saved.version === 1 && Array.isArray(saved.nodes) && Array.isArray(saved.edges)) {
        return { ...saved, frames: Array.isArray(saved.frames) ? saved.frames : [] };
      }
    } catch (_) {}
    return null;
  };

  const renderIcon = (name, size = 18, extraStyle = {}) => {
    const iconPaths = {
      chat: [
        e('path', { key: '1', d: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z' })
      ],
      nodes: [
        e('rect', { key: '1', x: '3', y: '3', width: '6', height: '6', rx: '1' }),
        e('rect', { key: '2', x: '15', y: '3', width: '6', height: '6', rx: '1' }),
        e('rect', { key: '3', x: '9', y: '15', width: '6', height: '6', rx: '1' }),
        e('path', { key: '4', d: 'M6 9v3a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V9' }),
        e('path', { key: '5', d: 'M12 13v2' })
      ],
      training: [
        e('polygon', { key: '1', points: '13 2 3 14 12 14 11 22 21 10 12 10 13 2' })
      ],
      checkpoints: [
        e('ellipse', { key: '1', cx: '12', cy: '5', rx: '9', ry: '3' }),
        e('path', { key: '2', d: 'M21 12c0 1.66-4 3-9 3s-9-1.34-9-3' }),
        e('path', { key: '3', d: 'M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5' })
      ],
      finetune: [
        e('circle', { key: '1', cx: '12', cy: '12', r: '10' }),
        e('circle', { key: '2', cx: '12', cy: '12', r: '6' }),
        e('circle', { key: '3', cx: '12', cy: '12', r: '2' })
      ],
      modules: [
        e('path', { key: '1', d: 'M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z' }),
        e('polyline', { key: '2', points: '3.27 6.96 12 12.01 20.73 6.96' }),
        e('line', { key: '3', x1: '12', y1: '22.08', x2: '12', y2: '12' })
      ],
      hardware: [
        e('rect', { key: '1', x: '4', y: '4', width: '16', height: '16', rx: '2' }),
        e('rect', { key: '2', x: '9', y: '9', width: '6', height: '6' }),
        e('path', { key: '3', d: 'M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3' })
      ],
      models: [
        e('polygon', { key: '1', points: '12 2 2 7 12 12 22 7 12 2' }),
        e('polyline', { key: '2', points: '2 17 12 22 22 17' }),
        e('polyline', { key: '3', points: '2 12 12 17 22 12' })
      ],
      datasets: [
        e('path', { key: '1', d: 'M3 3v18h18' }),
        e('rect', { key: '2', x: '7', y: '10', width: '3', height: '8', rx: '1' }),
        e('rect', { key: '3', x: '13', y: '6', width: '3', height: '12', rx: '1' }),
        e('rect', { key: '4', x: '19', y: '12', width: '3', height: '6', rx: '1' })
      ],
      notebook: [
        e('polyline', { key: '1', points: '16 18 22 12 16 6' }),
        e('polyline', { key: '2', points: '8 6 2 12 8 18' })
      ],
      byok: [
        e('circle', { key: '1', cx: '7.5', cy: '15.5', r: '5.5' }),
        e('path', { key: '2', d: 'M12 11l9-9M17 2l4 4M14 5l3 3' })
      ],
      copy: [
        e('rect', { key: '1', x: '9', y: '9', width: '13', height: '13', rx: '2' }),
        e('path', { key: '2', d: 'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1' })
      ],
      trash: [
        e('polyline', { key: '1', points: '3 6 5 6 21 6' }),
        e('path', { key: '2', d: 'M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2' })
      ],
      play: [
        e('polygon', { key: '1', points: '5 3 19 12 5 21 5 3', fill: 'currentColor' })
      ],
      pause: [
        e('rect', { key: '1', x: '6', y: '4', width: '4', height: '16', fill: 'currentColor' }),
        e('rect', { key: '2', x: '14', y: '4', width: '4', height: '16', fill: 'currentColor' })
      ],
      download: [
        e('path', { key: '1', d: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4' }),
        e('polyline', { key: '2', points: '7 10 12 15 17 10' }),
        e('line', { key: '3', x1: '12', y1: '15', x2: '12', y2: '3' })
      ],
      scissors: [
        e('circle', { key: '1', cx: '6', cy: '6', r: '3' }),
        e('circle', { key: '2', cx: '6', cy: '18', r: '3' }),
        e('line', { key: '3', x1: '20', y1: '4', x2: '8.12', y2: '15.88' }),
        e('line', { key: '4', x1: '14.47', y1: '14.48', x2: '20', y2: '20' }),
        e('line', { key: '5', x1: '8.12', y1: '8.12', x2: '12', y2: '12' })
      ],
      plus: [
        e('line', { key: '1', x1: '12', y1: '5', x2: '12', y2: '19' }),
        e('line', { key: '2', x1: '5', y1: '12', x2: '19', y2: '12' })
      ],
      library: [
        e('path', { key: '1', d: 'M4 19.5A2.5 2.5 0 0 1 6.5 17H20' }),
        e('path', { key: '2', d: 'M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z' })
      ],
      mute: [
        e('polygon', { key: '1', points: '11 5 6 9 2 9 2 15 6 15 11 19 11 5' }),
        e('line', { key: '2', x1: '23', y1: '9', x2: '17', y2: '15' }),
        e('line', { key: '3', x1: '17', y1: '9', x2: '23', y2: '15' })
      ],
      frame: [
        e('rect', { key: '1', x: '3', y: '3', width: '18', height: '18', rx: '2', strokeDasharray: '3 3' })
      ],
      bolt: [
        e('polygon', { key: '1', points: '13 2 3 14 12 14 11 22 21 10 12 10 13 2', fill: 'currentColor' })
      ],
      sparkle: [
        e('path', { key: '1', d: 'M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z', fill: 'currentColor' })
      ],
      minus: [
        e('line', { key: '1', x1: '5', y1: '12', x2: '19', y2: '12' })
      ],
      refresh: [
        e('path', { key: '1', d: 'M23 4v6h-6' }),
        e('path', { key: '2', d: 'M1 20v-6h6' }),
        e('path', { key: '3', d: 'M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15' })
      ],
      alert: [
        e('path', { key: '1', d: 'M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z' }),
        e('line', { key: '2', x1: '12', y1: '9', x2: '12', y2: '13' }),
        e('line', { key: '3', x1: '12', y1: '17', x2: '12.01', y2: '17' })
      ],
      check: [
        e('polyline', { key: '1', points: '20 6 9 17 4 12' })
      ],
      arrowRight: [
        e('line', { key: '1', x1: '5', y1: '12', x2: '19', y2: '12' }),
        e('polyline', { key: '2', points: '12 5 19 12 12 19' })
      ],
      search: [
        e('circle', { key: '1', cx: '11', cy: '11', r: '8' }),
        e('line', { key: '2', x1: '21', y1: '21', x2: '16.65', y2: '16.65' })
      ],
      heart: [
        e('path', { key: '1', d: 'M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z' })
      ],
      hub: [
        e('circle', { key: '1', cx: '12', cy: '12', r: '10' }),
        e('line', { key: '2', x1: '2', y1: '12', x2: '22', y2: '12' }),
        e('path', { key: '3', d: 'M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z' })
      ],
      git: [
        e('circle', { key: '1', cx: '18', cy: '18', r: '3' }),
        e('circle', { key: '2', cx: '6', cy: '6', r: '3' }),
        e('path', { key: '3', d: 'M6 21V9a9 9 0 0 0 9 9' })
      ],
      rocket: [
        e('path', { key: '1', d: 'M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z' }),
        e('path', { key: '2', d: 'M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z' })
      ],
      stream: [
        e('polyline', { key: '1', points: '22 12 18 12 15 21 9 3 6 12 2 12' })
      ],
      close: [
        e('line', { key: '1', x1: '18', y1: '6', x2: '6', y2: '18' }),
        e('line', { key: '2', x1: '6', y1: '6', x2: '18', y2: '18' })
      ],
      menu: [
        e('line', { key: '1', x1: '3', y1: '12', x2: '21', y2: '12' }),
        e('line', { key: '2', x1: '3', y1: '6', x2: '21', y2: '6' }),
        e('line', { key: '3', x1: '3', y1: '18', x2: '21', y2: '18' })
      ]
    };
    return e('svg', {
      xmlns: 'http://www.w3.org/2000/svg',
      width: size,
      height: size,
      viewBox: '0 0 24 24',
      fill: 'none',
      stroke: 'currentColor',
      strokeWidth: '1.75',
      strokeLinecap: 'round',
      strokeLinejoin: 'round',
      className: 'editorial-svg-icon',
      style: { display: 'inline-flex', verticalAlign: 'middle', flexShrink: 0, ...extraStyle }
    }, iconPaths[name] || null);
  };

  const MODEL_PRESETS = [
    {
      id: 'triune-nano',
      name: 'Triune-Nano (21M · 4L / 192D / 2E)',
      tier: 'Tier 1 · CPU & Rapid Debugging',
      layers: 4,
      hidden_dim: 192,
      heads: 3,
      head_dim: 64,
      experts_routed: 2,
      experts_shared: 1,
      exits: 'Prefix L0 → Reflex L2 → Limbic L3 → Cortex L4',
      params: '21.0M params (21.0M active)',
      desc: 'Minimal 4-layer Triune architecture (3 dense prefix/reflex blocks + 1 Sparse MoE block). Built for instant CPU debugging, pipeline verification, and zero-wait iteration.'
    },
    {
      id: 'triune-small',
      name: 'Triune-Small (36M · 6L / 256D / 4E)',
      tier: 'Tier 2 · Canonical Baseline (Default)',
      layers: 6,
      hidden_dim: 256,
      heads: 4,
      head_dim: 64,
      experts_routed: 4,
      experts_shared: 1,
      exits: 'Prefix L0 → Reflex L2 → Limbic L4 → Cortex L6',
      params: '36.0M params (32.9M active)',
      desc: 'Reference 6-layer Triune Transformer from config.py. Combines 3 dense early layers (L0–L2) and 3 Top-2 Sparse MoE layers (L3–L5) with Vectorised GLA + RoPE and 3-stage Gumbel exit routing.'
    },
    {
      id: 'triune-base',
      name: 'Triune-Base (76M · 8L / 384D / 4E)',
      tier: 'Tier 3 · Consumer GPU Pre-Training (6–8GB VRAM)',
      layers: 8,
      hidden_dim: 384,
      heads: 6,
      head_dim: 64,
      experts_routed: 4,
      experts_shared: 1,
      exits: 'Prefix L0 → Reflex L2 → Limbic L5 → Cortex L8',
      params: '75.8M params (64.0M active)',
      desc: 'Balanced 8-layer architecture expanding the Sparse MoE stack to 5 layers (L3–L7) at 384 hidden width. Ideal for training from scratch on TinyStories or FineWeb-Edu on a single 8GB GPU.'
    },
    {
      id: 'triune-medium',
      name: 'Triune-Medium (208M · 12L / 512D / 8E)',
      tier: 'Tier 4 · Sparse 8-Expert MoE (8–12GB VRAM)',
      layers: 12,
      hidden_dim: 512,
      heads: 8,
      head_dim: 64,
      experts_routed: 8,
      experts_shared: 1,
      exits: 'Prefix L0–L1 → Reflex L4 → Limbic L8 → Cortex L12',
      params: '207.5M params (119.4M active)',
      desc: '12-layer architecture scaling to 8 routed experts per MoE block across 7 MoE layers (L5–L11). Top-2 gating + 1 shared expert activates only 3 of 9 experts per token (57% active parameters).'
    },
    {
      id: 'triune-large',
      name: 'Triune-Large (574M · 16L / 768D / 8E)',
      tier: 'Tier 5 · High-Capacity MoE (12–16GB VRAM)',
      layers: 16,
      hidden_dim: 768,
      heads: 12,
      head_dim: 64,
      experts_routed: 8,
      experts_shared: 1,
      exits: 'Prefix L0–L1 → Reflex L5 → Limbic L10 → Cortex L16',
      params: '573.9M params (290.8M active)',
      desc: '16-layer half-billion parameter MoE (6 dense layers + 10 Sparse MoE layers). Designed for GaLore low-rank subspace projection and CentroidSteer + Muon training on 12GB–16GB GPUs.'
    },
    {
      id: 'triune-1b',
      name: 'Triune-1.5B (1.50B · 24L / 1024D / 8E)',
      tier: 'Billion-Scale · Entry Frontier MoE (16GB VRAM / 8GB Stream)',
      layers: 24,
      hidden_dim: 1024,
      heads: 16,
      head_dim: 64,
      experts_routed: 8,
      experts_shared: 1,
      exits: 'Prefix L0–L2 → Reflex L7 → Limbic L16 → Cortex L24',
      params: '1.50B params (696M active)',
      desc: 'Entry billion-parameter Sparse MoE (8 dense + 16 MoE layers). Activates only 696M parameters per token via Top-2 expert routing; trains on 16GB GPUs with GaLore + Muon or 8GB GPUs via Layer Streaming.'
    },
    {
      id: 'triune-3b',
      name: 'Triune-3B (3.07B · 30L / 1280D / 8E)',
      tier: 'Multi-Billion · 3B Workstation MoE (24GB VRAM / 12GB Stream)',
      layers: 30,
      hidden_dim: 1280,
      heads: 20,
      head_dim: 64,
      experts_routed: 8,
      experts_shared: 1,
      exits: 'Prefix L0–L2 → Reflex L7 → Limbic L19 → Cortex L30',
      params: '3.07B params (1.33B active)',
      desc: '3.07B parameter Sparse MoE across 30 layers (8 dense + 22 MoE blocks) with 1.33B active parameters per token. Built for 24GB single-GPU workstations or 8GB–12GB PCIe Layer Streaming.'
    },
    {
      id: 'triune-7b',
      name: 'Triune-7B (7.38B · 32L / 1920D / 8E)',
      tier: 'Multi-Billion · 7B Flagship MoE (24–48GB VRAM / Layer Stream)',
      layers: 32,
      hidden_dim: 1920,
      heads: 30,
      head_dim: 64,
      experts_routed: 8,
      experts_shared: 1,
      exits: 'Prefix L0–L3 → Reflex L7 → Limbic L20 → Cortex L32',
      params: '7.38B params (3.13B active)',
      desc: '7.38B parameter Flagship Sparse MoE (32 layers, 1920 hidden dim, 30 GLA heads) activating 3.13B parameters per token (42% active). Supports 4-bit NF4 QLoRA and asynchronous CPU-to-GPU Layer Streaming.'
    },
    {
      id: 'triune-14b',
      name: 'Triune-14B (14.5B · 40L / 2048D / 12E)',
      tier: 'Multi-Billion · 14B Deep Sparse MoE (48GB VRAM / Layer Stream)',
      layers: 40,
      hidden_dim: 2048,
      heads: 16,
      head_dim: 128,
      experts_routed: 12,
      experts_shared: 1,
      exits: 'Prefix L0–L3 → Reflex L9 → Limbic L25 → Cortex L40',
      params: '14.5B params (4.39B active)',
      desc: '14.5B parameter high-sparsity MoE (40 layers, 12 routed experts + 1 shared expert per MoE block). Top-2 routing keeps active compute to just 4.39B parameters per token (30% active ratio).'
    },
    {
      id: 'triune-32b',
      name: 'Triune-32B (34.5B · 48L / 2560D / 16E)',
      tier: 'Multi-Billion · 32B Frontier Sparse MoE (80GB A100/H100 / Stream)',
      layers: 48,
      hidden_dim: 2560,
      heads: 20,
      head_dim: 128,
      experts_routed: 16,
      experts_shared: 1,
      exits: 'Prefix L0–L4 → Reflex L11 → Limbic L30 → Cortex L48',
      params: '34.5B params (8.11B active)',
      desc: '34.5B parameter 16-expert Sparse MoE (48 layers, 2560 dim, 12 dense + 36 MoE blocks). Combines ultra-sparse Top-2 routing (only 8.11B active params/token, 23% active) with FP8/NVFP4 quantization.'
    },
    {
      id: 'triune-70b',
      name: 'Triune-70B (71.0B · 64L / 3072D / 16E)',
      tier: 'Multi-Billion · 70B Ultra-Scale Flagship MoE (Multi-GPU / H100)',
      layers: 64,
      hidden_dim: 3072,
      heads: 24,
      head_dim: 128,
      experts_routed: 16,
      experts_shared: 1,
      exits: 'Prefix L0–L5 → Reflex L11 → Limbic L40 → Cortex L64',
      params: '71.0B params (16.0B active)',
      desc: '71.0B parameter Datacenter Flagship MoE (64 layers, 3072 dim, 12 dense + 52 Sparse MoE blocks with 16 routed + 1 shared expert). Activates 16.0B parameters per token (22.5% active) with 3-stage early exits.'
    }
  ];

  const LORA_PRESETS = [
    { name: 'Low VRAM QLoRA (4-bit NF4)', rank: 8, alpha: 16, quantization: '4-bit NF4', lr: '0.0002', epochs: 3, dropout: 0.05, target_modules: "['q_proj', 'v_proj', 'out_proj']" },
    { name: 'High Rank LoRA (FP16, r=64)', rank: 64, alpha: 128, quantization: 'FP16 Half', lr: '0.0001', epochs: 5, dropout: 0.05, target_modules: "['q_proj', 'v_proj', 'out_proj']" },
    { name: 'Fast Adapter (8-bit, r=16)', rank: 16, alpha: 32, quantization: '8-bit', lr: '0.0003', epochs: 2, dropout: 0.05, target_modules: "['q_proj', 'v_proj', 'out_proj']" }
  ];

  const SYSTEM_PROMPT_PRESETS = [
    { name: 'MoE Core System', prompt: 'You are Triune Transformer, an advanced Mixture-of-Experts AI engine with dynamic exit-head routing.' },
    { name: 'Python Code Architect', prompt: 'You are an expert Python & PyTorch engineer specializing in deep learning performance, CUDA kernels, and clean code.' },
    { name: 'Research Assistant', prompt: 'You are a meticulous AI research assistant focusing on transformer architecture analysis and mathematical precision.' },
    { name: 'Uncensored / Objective Direct', prompt: 'You are an objective, uncensored AI assistant with refusal ablation. Answer technical and analytical inquiries with utmost candor, factual clarity, and zero boilerplate disclaimers.' },
    { name: 'Mathematical & Formal Logic', prompt: 'You are a formal logic and mathematical reasoning assistant. Provide rigorous, step-by-step mathematical reasoning with clear intermediate steps and proofs.' },
    { name: 'Creative Workshop', prompt: 'You are an imaginative creative writing partner skilled in literary tone, vivid world-building, and evocative prose.' }
  ];

  const POPULAR_DATASETS = [
    { name: 'FineWeb-Edu (10BT)', id: 'HuggingFaceFW/fineweb-edu', config: 'sample-10BT', split: 'train', col: 'text', desc: 'Highest quality educational web crawl', category: 'Reasoning' },
    { name: 'TinyStories', id: 'roneneldan/TinyStories', config: '', split: 'train', col: 'text', desc: 'Synthetic English stories, ultra-fast streaming', category: 'Pre-training' },
    { name: 'WikiText-103', id: 'Salesforce/wikitext', config: 'wikitext-103-raw-v1', split: 'train', col: 'text', desc: 'Verified Wikipedia articles', category: 'Pre-training' },
    { name: 'OpenWebText', id: 'Skylion007/openwebtext', config: '', split: 'train', col: 'text', desc: 'Reddit-curated open web text', category: 'Pre-training' },
    { name: 'Falcon RefinedWeb', id: 'tiiuae/falcon-refinedweb', config: '', split: 'train', col: 'content', desc: 'Strictly filtered multi-billion token corpus', category: 'Pre-training' },
    { name: 'The Stack (Python)', id: 'bigcode/the-stack-smol', config: 'data/python', split: 'train', col: 'content', desc: 'Permissively licensed Python source code', category: 'Code' },
    { name: 'GSM8K Math', id: 'openai/gsm8k', config: 'main', split: 'train', col: 'question', desc: 'Grade school math word problems with step-by-step solutions', category: 'Math' },
    { name: 'C4 Clean Crawl', id: 'allenai/c4', config: 'en', split: 'train', col: 'text', desc: 'Colossal Clean Crawled Corpus for large-scale pretraining', category: 'Pre-training' }
  ];

  const TAB_TITLES = {
    chat: 'Inference Console & Playground',
    nodegraph: 'Visual Node Canvas & Architecture Composer',
    training: 'Real-Time Training Monitor & Telemetry',
    checkpoints: 'Model Checkpoints & Weights Archive',
    finetune: 'LoRA & QLoRA Fine-Tuning Workshop',
    modules: 'Extensions Marketplace',
    customize: 'Appearance Marketplace',
    environment: 'System Diagnostics & Hardware Inspector',
    models: 'Model Zoo & Exporter Engine',
    datasets: 'Dataset Streaming & Corpus Management',
    notebook: 'Interactive Python Sandbox & Scratchpad',
    byok: 'API Keys & Bring-Your-Own-Key Subscriptions'
  };

  const TAB_DESCRIPTIONS = {
    chat: 'Send prompts to your selected local model or provider.',
    nodegraph: 'Compose, inspect, and apply a model execution graph.',
    training: 'Configure a run and monitor live training metrics.',
    checkpoints: 'Save, restore, and manage model checkpoints.',
    finetune: 'Configure and attach LoRA or QLoRA adapters.',
    modules: 'Discover curated integrations, local plugins, and community repositories.',
    customize: 'Choose a Studio edition, light or dark mode, and a curated color theme.',
    environment: 'Inspect hardware, runtime health, and performance settings.',
    models: 'Choose architectures and export trained models.',
    datasets: 'Connect streaming or local datasets and configure tokenization.',
    notebook: 'Run trusted local Python snippets with execution limits.',
    byok: 'Configure and test external model-provider credentials.'
  };

  const APPEARANCE_THEMES = [
    { id: 'modern-light', name: 'Studio Modern', mode: 'Light', edition: 'Modern', accent: '#635bff', description: 'Clean surfaces, compact navigation, and a focused technical workspace.' },
    { id: 'modern-dark', name: 'Studio Midnight', mode: 'Dark', edition: 'Modern', accent: '#8b83ff', description: 'A low-glare dark workspace with clear hierarchy and high-contrast controls.' },
    { id: 'modern-forest', name: 'Studio Forest', mode: 'Dark', edition: 'Modern', accent: '#55b987', description: 'A calm green-accented dark theme for long training and debugging sessions.' },
    { id: 'classic-light', name: 'Classic Editorial', mode: 'Light', edition: 'Classic', accent: '#9a3412', description: 'Keep the original warm paper, serif headings, and editorial character.' },
    { id: 'classic-dark', name: 'Classic Ink', mode: 'Dark', edition: 'Classic', accent: '#e08a63', description: 'The original editorial style adapted for a comfortable dark workspace.' }
  ];

  let activeApiBase = '';
  let apiAuthToken = '';
  let apiAuthPrompt = null;
  let apiDiscoveryPromise = null;

  function isLocalStudioHost(hostname) {
    const host = (hostname || '').toLowerCase().replace(/^\[|\]$/g, '');
    return host === 'localhost' || host === '::1' || host === '0:0:0:0:0:0:0:1' || /^127(?:\.\d{1,3}){3}$/.test(host);
  }

  function requestApiAuthToken() {
    if (apiAuthToken) return Promise.resolve(apiAuthToken);
    if (!apiAuthPrompt) {
      apiAuthPrompt = Promise.resolve(window.prompt('Enter the Triune remote API bearer token:'))
        .then(token => {
          apiAuthToken = (token || '').trim();
          return apiAuthToken;
        })
        .finally(() => { apiAuthPrompt = null; });
    }
    return apiAuthPrompt;
  }


  async function discoverApiBase() {
    // A remotely served Studio should use its own origin; local port discovery
    // could otherwise attach it to an unrelated API on the viewer's machine.
    if ((window.location.protocol === 'http:' || window.location.protocol === 'https:') && !isLocalStudioHost(window.location.hostname)) return '';
    if (activeApiBase) {
      const controller = new AbortController();
      const tid = setTimeout(() => controller.abort(), 800);
      try {
        const res = await fetch(activeApiBase + '/v1/system/diagnostics', { signal: controller.signal });
        if (res.ok) return activeApiBase;
      } catch (e) {
      } finally {
        clearTimeout(tid);
      }
    }
    if (apiDiscoveryPromise) return apiDiscoveryPromise;
    apiDiscoveryPromise = (async () => {
      const candidates = await Promise.all(Array.from({ length: 21 }, async (_, index) => {
        const candidate = `http://127.0.0.1:${8000 + index}`;
        const controller = new AbortController();
        const tid = setTimeout(() => controller.abort(), 700);
        try {
          const res = await fetch(candidate + '/v1/system/diagnostics', { signal: controller.signal });
          if (!res.ok) return null;
          const data = await res.json();
          return data.pytorch_version || data.cuda_available !== undefined || data.platform ? candidate : null;
        } catch (e) {
          return null;
        } finally {
          clearTimeout(tid);
        }
      }));
      const candidate = candidates.find(Boolean) || '';
      if (candidate) {
        console.log('[ApiFetch] Discovered PyTorch Engine on:', candidate);
        activeApiBase = candidate;
      }
      return candidate;
    })();
    try {
      return await apiDiscoveryPromise;
    } finally {
      apiDiscoveryPromise = null;
    }
  }

  async function apiFetch(path, options = {}) {
    const isRemoteOrigin = (window.location.protocol === 'http:' || window.location.protocol === 'https:') && !isLocalStudioHost(window.location.hostname);
    const apiBase = isRemoteOrigin ? '' : await discoverApiBase();
    if (!isRemoteOrigin && !apiBase) {
      throw new Error('Triune backend was not found on localhost ports 8000–8020. Check Studio startup and open the startup log.');
    }
    const url = apiBase ? (apiBase + path) : path;
    const send = (token) => {
      const headers = new Headers(options.headers || {});
      if (token) headers.set('Authorization', `Bearer ${token}`);
      return fetch(url, { ...options, headers });
    };
    let response;
    try {
      response = await send(apiAuthToken);
    } catch (err) {
      throw new Error(`Could not reach ${url}: ${err.message || 'connection failed'}. Check that the Studio backend is still running.`);
    }
    if (response.status !== 401) return response;
    const token = await requestApiAuthToken();
    if (!token) return response;
    try {
      response = await send(token);
    } catch (err) {
      throw new Error(`Could not reach ${url} after local API authentication: ${err.message || 'connection failed'}. Check the Studio startup log.`);
    }
    if (response.status === 401) apiAuthToken = '';
    return response;
  }

  // Persistent Top Global Status Bar Component Across All Tabs
  function GlobalStatusBar({
    hardware,
    vramAllocated,
    vramTotal,
    oomRisk,
    engineStatus,
    step,
    loss,
    throughput,
    exitDist,
    depthMode,
    onPurgeVRAM
  }) {
    const total = vramTotal || 8.0;
    const vramPct = Math.min(100, Math.max(0, (vramAllocated / total) * 100));
    const isRunning = engineStatus === 'Running';
    const isPaused = engineStatus === 'Paused';
    const statusColor = isRunning ? '#16a34a' : (isPaused ? '#d97706' : '#78716c');
    const statusBg = isRunning ? 'rgba(22, 163, 74, 0.12)' : (isPaused ? 'rgba(217, 119, 6, 0.12)' : 'rgba(120, 113, 108, 0.12)');

    return e('div', {
      className: 'global-status-bar',
      style: {
        height: '40px',
        minHeight: '40px',
        background: 'var(--bg-card, #fcfbfa)',
        borderBottom: '1px solid var(--border-color, #ded9cd)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 20px',
        fontSize: '12px',
        fontFamily: 'var(--font-family, sans-serif)',
        color: 'var(--text-main, #1c1917)',
        zIndex: 50,
        gap: '16px',
        flexWrap: 'nowrap',
        overflowX: 'auto',
        boxShadow: '0 1px 3px rgba(0,0,0,0.03)'
      }
    },
      // 1. Hardware & VRAM Telemetry
      e('div', { style: { display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0 } },
        e('div', { style: { display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, fontSize: '12px' } },
          renderIcon('hardware', 15),
          e('span', { style: { fontFamily: 'var(--font-heading, Newsreader)', fontSize: '13px' } }, hardware || 'NVIDIA RTX GPU')
        ),
        e('div', {
          style: {
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'var(--bg-surface, #ede9df)',
            padding: '4px 10px',
            borderRadius: '12px',
            border: '1px solid var(--border-color, #ded9cd)',
            fontSize: '11px',
            fontFamily: 'var(--font-mono, Consolas)'
          }
        },
          e('span', { style: { color: 'var(--text-muted)' } }, 'VRAM:'),
          e('span', { style: { fontWeight: 700, color: oomRisk ? 'var(--primary, #9a3412)' : 'var(--text-main)' } },
            `${vramAllocated} / ${total} GB`
          ),
          e('div', {
            style: {
              width: '60px',
              height: '7px',
              background: 'var(--border-dark, #c9c3b4)',
              borderRadius: '4px',
              overflow: 'hidden'
            }
          },
            e('div', {
              style: {
                width: `${vramPct}%`,
                height: '100%',
                background: oomRisk ? '#dc2626' : (vramPct > 80 ? '#d97706' : '#2b4c3f'),
                borderRadius: '4px',
                transition: 'width 0.3s ease'
              }
            })
          ),
          e('span', { style: { fontSize: '10px', color: 'var(--text-dim)' } }, `${vramPct.toFixed(0)}%`),
          onPurgeVRAM && e('button', {
            onClick: onPurgeVRAM,
            title: 'Purge GPU VRAM Caches',
            style: {
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '0 2px',
              fontSize: '11px',
              color: 'var(--accent-terracotta, #c2410c)',
              fontWeight: 700,
              display: 'inline-flex',
              alignItems: 'center'
            }
          }, renderIcon('trash', 12))
        )
      ),

      // 2. Engine Status & Training Metrics
      e('div', { style: { display: 'flex', alignItems: 'center', gap: '16px', flexShrink: 0 } },
        e('div', {
          style: {
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '3px 10px',
            borderRadius: '12px',
            background: statusBg,
            color: statusColor,
            fontWeight: 700,
            fontSize: '11px',
            letterSpacing: '0.5px',
            textTransform: 'uppercase'
          }
        },
          e('span', {
            style: {
              width: '7px',
              height: '7px',
              borderRadius: '50%',
              background: statusColor,
              display: 'inline-block',
              boxShadow: `0 0 6px ${statusColor}`
            }
          }),
          engineStatus
        ),
        e('div', { style: { display: 'flex', alignItems: 'center', gap: '5px', fontSize: '12px' } },
          e('span', { style: { color: 'var(--text-muted)' } }, 'Step:'),
          e('span', { style: { fontWeight: 700, fontFamily: 'var(--font-mono)' } }, (step || 0).toLocaleString())
        ),
        e('div', { style: { display: 'flex', alignItems: 'center', gap: '5px', fontSize: '12px' } },
          e('span', { style: { color: 'var(--text-muted)' } }, 'Loss:'),
          e('span', { style: { fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-terracotta, #9a3412)' } }, loss || '--')
        ),
        e('div', { style: { display: 'flex', alignItems: 'center', gap: '5px', fontSize: '12px' } },
          e('span', { style: { color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center' } }, [renderIcon('bolt', 11, { marginRight: '3px' }), 'Speed:']),
          e('span', { style: { fontWeight: 600, fontFamily: 'var(--font-mono)' } }, throughput || '--')
        )
      ),

      // 3. Exit Distribution (Reflex / Limbic / Cortex)
      e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 } },
        e('span', { style: { fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.4px', fontWeight: 600 } },
          `Exits [${depthMode || 'DYNAMIC'}]:`
        ),
        e('div', { style: { display: 'flex', gap: '5px', alignItems: 'center' } },
          e('span', {
            style: {
              padding: '2px 7px',
              borderRadius: '4px',
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              background: 'rgba(99, 199, 178, 0.15)',
              color: '#0d9488',
              border: '1px solid rgba(99, 199, 178, 0.4)'
            },
            title: 'Reflex Exit (L1)'
          }, `Reflex: ${(exitDist && exitDist.reflex !== undefined) ? exitDist.reflex : 34}%`),
          e('span', {
            style: {
              padding: '2px 7px',
              borderRadius: '4px',
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              background: 'rgba(232, 203, 79, 0.15)',
              color: '#b45309',
              border: '1px solid rgba(232, 203, 79, 0.4)'
            },
            title: 'Limbic Exit (L3)'
          }, `Limbic: ${(exitDist && exitDist.limbic !== undefined) ? exitDist.limbic : 33}%`),
          e('span', {
            style: {
              padding: '2px 7px',
              borderRadius: '4px',
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              fontWeight: 600,
              background: 'rgba(122, 136, 207, 0.15)',
              color: '#4338ca',
              border: '1px solid rgba(122, 136, 207, 0.4)'
            },
            title: 'Cortex Exit (L5)'
          }, `Cortex: ${(exitDist && exitDist.cortex !== undefined) ? exitDist.cortex : 33}%`)
        )
      )
    );
  }

  function TriuneStudio() {
    const cleanText = (str) => {
      if (!str) return '';
      return String(str)
        .replace(/â€™/g, "'")
        .replace(/â€˜/g, "'")
        .replace(/â€œ/g, '"')
        .replace(/â€/g, '"')
        .replace(/â€ /g, '"')
        .replace(/â€/g, '"')
        .replace(/â€“/g, '-')
        .replace(/â€”/g, '--')
        .replace(/â€¦/g, '...')
        .replace(/[\ufffd\uFFFD]/g, '')
        .replace(/Ġ/g, ' ')
        .replace(/Ċ/g, '\n')
        .replace(/\r\n?/g, '\n')
        .replace(/[ \t]+/g, ' ')
        .replace(/ *\n */g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
    };
    const [activeTab, setActiveTab] = useState(() => {
      try {
        const savedTab = window.localStorage.getItem('triune.studio.activeTab');
        return Object.prototype.hasOwnProperty.call(TAB_TITLES, savedTab) ? savedTab : 'chat';
      } catch (e) {
        return 'chat';
      }
    });
    const [backendStatus, setBackendStatus] = useState('checking');
    const [backendRetry, setBackendRetry] = useState(0);
    const [vramUsage, setVramUsage] = useState({ allocated: 0.0, reserved: 0.0, total: 0.0, oom_risk: false });
    const [systemDiagnostics, setSystemDiagnostics] = useState({ device_name: 'PyTorch Engine', cuda_available: false });
    const [activeModel, setActiveModel] = useState(() => {
    try {
      const saved = window.localStorage.getItem('triune.studio.activeModel');
      if (saved && saved !== 'gemini-3.8-flash') return saved;
      if (saved === 'gemini-3.8-flash') return 'gemini-2.5-flash';
    } catch (_) {}
    return 'triune-small';
  });
    const [weightsLoaded, setWeightsLoaded] = useState(false);
    const [precision, setPrecision] = useState('bf16');
    const [statusToast, setStatusToast] = useState(null);
    const [studioTheme, setStudioTheme] = useState(() => {
      try {
        const saved = window.localStorage.getItem('triune.studio.theme');
        return APPEARANCE_THEMES.some(theme => theme.id === saved) ? saved : 'modern-light';
      } catch (e) { return 'modern-light'; }
    });
    const [appearanceFilter, setAppearanceFilter] = useState('All');
    const [customAccent, setCustomAccent] = useState(() => {
      try { return window.localStorage.getItem('triune.studio.customAccent') || ''; }
      catch (e) { return ''; }
    });
    const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
      try { return window.localStorage.getItem('triune.studio.sidebarCollapsed') === 'true'; }
      catch (e) { return false; }
    });

    useEffect(() => {
      try {
        window.localStorage.setItem('triune.studio.activeTab', activeTab);
        window.localStorage.setItem('triune.studio.sidebarCollapsed', String(sidebarCollapsed));
        window.localStorage.setItem('triune.studio.theme', studioTheme);
        if (customAccent) window.localStorage.setItem('triune.studio.customAccent', customAccent);
        else window.localStorage.removeItem('triune.studio.customAccent');
      } catch (e) {}
    }, [activeTab, sidebarCollapsed, studioTheme, customAccent]);

    // Chat State
    const [messages, setMessages] = useState([
      {
        sender: 'Triune Studio',
        text: 'Welcome. Choose a model or provider, then send a prompt to begin. Backend connection status is shown above.',
        time: 'Just now',
        isAssistant: true
      }
    ]);
    const [chatInput, setChatInput] = useState('');
    const [isGenerating, setIsGenerating] = useState(false);
    const [route, setRoute] = useState('auto');
    const [systemPrompt, setSystemPrompt] = useState(SYSTEM_PROMPT_PRESETS[0].prompt);
    const [temperature, setTemperature] = useState(0.7);
    const [maxTokens, setMaxTokens] = useState(256);
    const chatBottomRef = useRef(null);
    const chatAbortControllerRef = useRef(null);

    // Real Training & Telemetry State
    const [isTraining, setIsTraining] = useState(false);
    const [metricsHistory, setMetricsHistory] = useState([]);
    const [metrics, setMetrics] = useState({ loss: null, lm_loss: null, router_loss: null, step: 0, throughput: 0 });
    const [exitUsage, setExitUsage] = useState({ reflex: 0, limbic: 0, cortex: 0 });
    const [telemetryLogs, setTelemetryLogs] = useState([]);
    const [lastSample, setLastSample] = useState('');
    const [trainBatchSize, setTrainBatchSize] = useState(4);
    const [trainSeqLen, setTrainSeqLen] = useState(64);
    const [trainGradAccum, setTrainGradAccum] = useState(4);
    const [trainLR, setTrainLR] = useState('0.0005');
    const [trainWarmup, setTrainWarmup] = useState(500);
    const [trainDepthMode, setTrainDepthMode] = useState('dynamic');
    const [trainPrecision, setTrainPrecision] = useState('bf16');
    const [isUpdatingTrainConfig, setIsUpdatingTrainConfig] = useState(false);
    const lastManualConfigEditRef = useRef(0);

    // WandB Experiment Tracking & Research Suite State
    const [chartMetrics, setChartMetrics] = useState({
      loss: true,
      lm_loss: true,
      router_loss: true,
      throughput: false,
      lr: false,
      ppl: false
    });
    const [chartSmoothing, setChartSmoothing] = useState(0.6);
    const [chartXAxis, setChartXAxis] = useState('steps'); // 'steps' | 'tokens'
    const [chartScale, setChartScale] = useState('linear'); // 'linear' | 'log'
    const [runName, setRunName] = useState('triune-canonical-run-1');
    const [chartHoverInfo, setChartHoverInfo] = useState(null);

    // Hardware Tuner & Auto-Offload State (LM Studio / Ollama Style)
    const [hardwareProfile, setHardwareProfile] = useState(null);
    const [selectedHwPreset, setSelectedHwPreset] = useState('turbo');
    const [gpuOffloadPct, setGpuOffloadPct] = useState(100);
    const [gpuLayers, setGpuLayers] = useState(6);
    const [isAutoTuning, setIsAutoTuning] = useState(false);
    const canvasRef = useRef(null);
    const [windowWidth, setWindowWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1200);
    const [nodeCatalog, setNodeCatalog] = useState(BUILTIN_NODE_CATALOG);

    // Visual Node Graph State (Unreal Engine / Blender Style Freedom)
    const [nodes, setNodes] = useState(() => (readSavedWorkflow() || DAG_PRESETS.triune_canonical_hierarchical).nodes);
    const [edges, setEdges] = useState(() => (readSavedWorkflow() || DAG_PRESETS.triune_canonical_hierarchical).edges);
    const [activePresetKey, setActivePresetKey] = useState(null);
    const [workflowName, setWorkflowName] = useState(() => (readSavedWorkflow() || {}).name || 'My Triune Workflow');
    const [workflowSaveState, setWorkflowSaveState] = useState('Saved locally');
    const [canvasPan, setCanvasPan] = useState({ x: 0, y: 0 });
    const [canvasZoom, setCanvasZoom] = useState(1.0);
    const [socketPositions, setSocketPositions] = useState({});
    const [isPanning, setIsPanning] = useState(false);
    const [panStart, setPanStart] = useState({ x: 0, y: 0 });
    const [connectingWire, setConnectingWire] = useState(null);
    const [canvasMousePos, setCanvasMousePos] = useState({ x: 0, y: 0, screenX: 0, screenY: 0 });
    const [omnibarQuery, setOmnibarQuery] = useState('');
    const [showOmnibarDropdown, setShowOmnibarDropdown] = useState(false);
    const [showWireList, setShowWireList] = useState(false);

    // Global Omnibar click-outside & window resize handler
    useEffect(() => {
      const handleResize = () => setWindowWidth(window.innerWidth);
      const handleClickOutside = (ev) => {
        if (!ev.target.closest('.topbar-omnibar-wrap')) {
          setShowOmnibarDropdown(false);
        }
      };
      window.addEventListener('resize', handleResize);
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        window.removeEventListener('resize', handleResize);
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }, []);

    const [draggingNodeId, setDraggingNodeId] = useState(null);
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
    const [connectingFromId, setConnectingFromId] = useState(null);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const [dagExecutionStatus, setDagExecutionStatus] = useState(null);
    const [nodeExecOutputs, setNodeExecOutputs] = useState({});
    const [activeRunningNodeId, setActiveRunningNodeId] = useState(null);
    const [showCustomNodeModal, setShowCustomNodeModal] = useState(false);
    const [showNodeLibraryModal, setShowNodeLibraryModal] = useState(false);
    const [showDAGPresetLibrary, setShowDAGPresetLibrary] = useState(false);
    const [nodeSearchQuery, setNodeSearchQuery] = useState('');
    const [selectedNodeCategory, setSelectedNodeCategory] = useState('All');
    const [customNodeTitle, setCustomNodeTitle] = useState('');
    const [customNodeType, setCustomNodeType] = useState('Model');
    const [customNodeDetails, setCustomNodeDetails] = useState('');
    const [selectedNodeId, setSelectedNodeId] = useState(null);
    const [selectedNodeIds, setSelectedNodeIds] = useState([]);
    const [selectedNodeParams, setSelectedNodeParams] = useState({});
    const [showNewNodeParam, setShowNewNodeParam] = useState(false);
    const [newNodeParamName, setNewNodeParamName] = useState('');
    const [isExecutingSingleNode, setIsExecutingSingleNode] = useState(false);
    const [showWorkspacePluginModal, setShowWorkspacePluginModal] = useState(false);
    const [newPluginName, setNewPluginName] = useState('CustomLossNode');
    const [newPluginCategory, setNewPluginCategory] = useState('Loss');
    const [frames, setFrames] = useState(() => (readSavedWorkflow() || {}).frames || []);
    const [quickAddMenu, setQuickAddMenu] = useState(null);
    const [isCompilingTraining, setIsCompilingTraining] = useState(false);
    const gridRef = useRef(null);

    // Derive wire endpoints from the rendered socket pins. Node headers,
    // variable port counts, fonts, and execution details all change card
    // geometry, so guessed pixel offsets inevitably drift.
    useEffect(() => {
      if (activeTab !== 'nodegraph' || !gridRef.current) return undefined;
      const frame = window.requestAnimationFrame(() => {
        if (!gridRef.current) return;
        const gridRect = gridRef.current.getBoundingClientRect();
        const measured = {};
        gridRef.current.querySelectorAll('[data-dag-input-node], [data-dag-output-node]').forEach(pin => {
          const nodeId = pin.dataset.dagInputNode || pin.dataset.dagOutputNode;
          const port = pin.dataset.dagInputPort || pin.dataset.dagOutputPort;
          const side = pin.dataset.dagInputNode ? 'in' : 'out';
          const pinRect = pin.getBoundingClientRect();
          measured[`${nodeId}|${side}|${port}`] = {
            x: (pinRect.left + pinRect.width / 2 - gridRect.left - canvasPan.x) / canvasZoom,
            y: (pinRect.top + pinRect.height / 2 - gridRect.top - canvasPan.y) / canvasZoom
          };
        });
        setSocketPositions(previous => {
          const oldKeys = Object.keys(previous);
          const newKeys = Object.keys(measured);
          if (oldKeys.length === newKeys.length && newKeys.every(key =>
            previous[key] && Math.abs(previous[key].x - measured[key].x) < 0.25 && Math.abs(previous[key].y - measured[key].y) < 0.25
          )) return previous;
          return measured;
        });
      });
      return () => window.cancelAnimationFrame(frame);
    }, [activeTab, nodes, edges, canvasPan, canvasZoom, nodeExecOutputs, studioTheme]);
    const workflowImportRef = useRef(null);

    // Global Key Listener for Blender / Unreal Controls: Shift+A (Quick Add), M (Mute), Ctrl+J (Frame), Escape (Close)
    useEffect(() => {
      const handleGlobalKeyDown = (ev) => {
        if (ev.key === 'Escape') {
          setShowOmnibarDropdown(false);
          setShowCustomNodeModal(false);
          setShowNodeLibraryModal(false);
          setShowDAGPresetLibrary(false);
          setShowWorkspacePluginModal(false);
          setQuickAddMenu(null);
          return;
        }

        const tag = (ev.target && ev.target.tagName) ? ev.target.tagName.toLowerCase() : '';
        const isTyping = tag === 'input' || tag === 'textarea' || tag === 'select';

        if (!isTyping && activeTab === 'nodegraph') {
          // Shift + A -> Quick Add Menu
          if ((ev.key === 'A' || ev.key === 'a') && ev.shiftKey) {
            ev.preventDefault();
            setQuickAddMenu({
              x: Math.min(window.innerWidth - 290, Math.max(20, (canvasMousePos && canvasMousePos.screenX) || 240)),
              y: Math.min(window.innerHeight - 320, Math.max(60, (canvasMousePos && canvasMousePos.screenY) || 180)),
              canvasX: (canvasMousePos && canvasMousePos.x) || 120,
              canvasY: (canvasMousePos && canvasMousePos.y) || 120,
              search: ''
            });
            return;
          }

          if ((ev.ctrlKey || ev.metaKey) && (ev.key === 'g' || ev.key === 'G')) {
            ev.preventDefault();
            handleGroupSelectedNodes();
            return;
          }

          // M -> Mute / Unmute selected node
          if (ev.key === 'm' || ev.key === 'M') {
            if (selectedNodeId) {
              ev.preventDefault();
              handleToggleMuteNode(selectedNodeId);
              return;
            }
          }

          // Ctrl + J -> Visual Group Frame
          if ((ev.ctrlKey || ev.metaKey) && (ev.key === 'j' || ev.key === 'J')) {
            ev.preventDefault();
            handleCreateVisualFrame();
            return;
          }
        }
      };
      window.addEventListener('keydown', handleGlobalKeyDown);
      return () => window.removeEventListener('keydown', handleGlobalKeyDown);
    }, [activeTab, selectedNodeId, selectedNodeIds, canvasMousePos, frames, nodes, edges]);

    // LoRA Fine-Tuner State
    const [loraConfig, setLoraConfig] = useState({
      rank: 16,
      alpha: 32,
      lr: '0.0002',
      epochs: 3,
      dataset: 'data/fineweb_sample.jsonl',
      quantization: '4-bit NF4'
    });
    const [fineTuningStatus, setFineTuningStatus] = useState(null);
    const [activeAdapter, setActiveAdapter] = useState(null);

    // Dataset Manager State
    const [sampleText, setSampleText] = useState('Triune Engine accelerates local transformer training with MoE exit heads.');
    const [tokens, setTokens] = useState([]);
    const [tokenizerStatus, setTokenizerStatus] = useState('Loading the configured tokenizer…');
    const [datasetList, setDatasetList] = useState([]);
    const [activeDataset, setActiveDataset] = useState('HuggingFaceFW/fineweb-edu');
    const [hfDatasetInput, setHfDatasetInput] = useState('roneneldan/TinyStories');
    const [hfConfigInput, setHfConfigInput] = useState('');
    const [hfSplitInput, setHfSplitInput] = useState('train');
    const [hfTextColInput, setHfTextColInput] = useState('');
    const [isConnectingHf, setIsConnectingHf] = useState(false);
    const [hfStreamResult, setHfStreamResult] = useState(null);
    const [localDatasetInput, setLocalDatasetInput] = useState('');

    // Notebook State
    const [notebookCode, setNotebookCode] = useState(
      `import torch\nimport triune\n\n# Query VRAM Profiler\nprint("VRAM Profile:", triune.VRAMProfiler.get_vram_stats())\n\n# Test PyTorch Engine\nprint("Engine Active:", torch.cuda.is_available())`
    );
    const [notebookOutput, setNotebookOutput] = useState('Click "Run Code in PythonSandbox" to execute on backend engine...');
    const [isExecutingNotebook, setIsExecutingNotebook] = useState(false);

    // BYOK State
    const [byokKeys, setByokKeys] = useState({ openai: '', anthropic: '', gemini: '', huggingface: '', wandb: '' });
    const [byokStatus, setByokStatus] = useState({});
    const [byokConfigured, setByokConfigured] = useState({});
    const [wandbUrl, setWandbUrl] = useState('');
    const [wandbActive, setWandbActive] = useState(false);

    // System Scanner & Module Marketplace State
    const [systemScan, setSystemScan] = useState(null);
    const [systemConfig, setSystemConfig] = useState({
      installation_path: 'C:\\TriuneStudio',
      models_path: 'C:\\TriuneStudio\\models',
      datasets_path: 'C:\\TriuneStudio\\datasets',
      checkpoints_path: 'C:\\TriuneStudio\\checkpoints',
      python_executable: '',
      hardware_mode: 'Auto Detect',
      auto_check_updates: true
    });
    const [moduleFilter, setModuleFilter] = useState('all');
    const [moduleSearchQuery, setModuleSearchQuery] = useState('');
    const [marketplaceData, setMarketplaceData] = useState({ curated: [], github: [], huggingface: [], installed_count: 0 });
    const [marketplaceError, setMarketplaceError] = useState('');
    const [installedModules, setInstalledModules] = useState([]);
    const [availableUpdates, setAvailableUpdates] = useState([]);
    const [isSearchingModules, setIsSearchingModules] = useState(false);
    const [marketplaceSubTab, setMarketplaceSubTab] = useState('installed');
    const [repoCloneUrl, setRepoCloneUrl] = useState('');
    const [repoCloneType, setRepoCloneType] = useState('auto');
    const [repoCloneBranch, setRepoCloneBranch] = useState('main');
    const [isCloningRepo, setIsCloningRepo] = useState(false);
    const [activePullingModId, setActivePullingModId] = useState(null);
    const [activeActivatingModId, setActiveActivatingModId] = useState(null);

    // Checkpoints & Dataset Telemetry State
    const [checkpoints, setCheckpoints] = useState([]);
    const [activeCheckpointName, setActiveCheckpointName] = useState(null);
    const [saveCkptName, setSaveCkptName] = useState('');
    const [isLoadingCkpt, setIsLoadingCkpt] = useState(false);
    const [datasetInfo, setDatasetInfo] = useState({
      name: 'HuggingFaceFW/fineweb-edu',
      type: 'streaming',
      total_tokens: 0,
      active_batch_preview: '',
      sequences_count: 0,
      batch_size: 4,
      seq_len: 64
    });
    const [activeBatchPreview, setActiveBatchPreview] = useState('');

    // Toast helper — uses ref to prevent rapid-fire timeout conflicts
    const toastTimerRef = useRef(null);
    const showToast = (msg) => {
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
      setStatusToast(msg);
      toastTimerRef.current = setTimeout(() => setStatusToast(null), 3000);
    };

    useEffect(() => {
      setWorkflowSaveState('Saving…');
      const timer = setTimeout(() => {
        try {
          window.localStorage.setItem('triune.studio.workflow.v1', JSON.stringify({
            format: 'triune-workflow', version: 1,
            name: workflowName || 'My Triune Workflow',
            nodes, edges, frames, saved_at: new Date().toISOString()
          }));
          setWorkflowSaveState('Saved locally');
        } catch (_) {
          setWorkflowSaveState('Local save unavailable');
        }
      }, 350);
      return () => clearTimeout(timer);
    }, [nodes, edges, frames, workflowName]);

    const handleExportWorkflow = () => {
      const payload = {
        format: 'triune-workflow', version: 1,
        name: workflowName || 'My Triune Workflow',
        nodes, edges, frames, exported_at: new Date().toISOString()
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      const safeName = (payload.name || 'triune-workflow').replace(/[^a-z0-9_-]+/gi, '-').replace(/^-|-$/g, '').toLowerCase() || 'triune-workflow';
      link.href = url;
      link.download = `${safeName}.triune.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      showToast('Workflow exported with its nested groups and connections.');
    };

    const handleImportWorkflowFile = async (file) => {
      if (!file) return;
      if (file.size > 10 * 1024 * 1024) {
        showToast('Workflow file is larger than the 10 MB safety limit.');
        return;
      }
      try {
        const payload = JSON.parse(await file.text());
        if (payload.format !== 'triune-workflow' || payload.version !== 1 || !Array.isArray(payload.nodes) || !Array.isArray(payload.edges)) {
          throw new Error('This is not a supported Triune workflow file.');
        }
        const ids = new Set();
        for (const node of payload.nodes) {
          if (!node || typeof node.id !== 'string' || !node.id.trim() || ids.has(node.id)) throw new Error('Workflow nodes need unique, non-empty IDs.');
          ids.add(node.id);
        }
        for (const edge of payload.edges) {
          if (!edge || !ids.has(edge.source) || !ids.has(edge.target)) throw new Error('A workflow wire references a missing node.');
        }
        const importedFrames = payload.frames === undefined ? [] : payload.frames;
        if (!Array.isArray(importedFrames) || importedFrames.some(frame =>
          !frame || typeof frame.id !== 'string' || typeof frame.title !== 'string' ||
          ![frame.x, frame.y, frame.width, frame.height].every(Number.isFinite) ||
          frame.width < 120 || frame.height < 80 || frame.width > 20000 || frame.height > 20000 ||
          (frame.color !== undefined && (typeof frame.color !== 'string' || !/^#[\da-f]{3,8}$/i.test(frame.color))) ||
          (frame.borderColor !== undefined && (typeof frame.borderColor !== 'string' || !/^#[\da-f]{3,8}$/i.test(frame.borderColor)))
        )) throw new Error('Workflow frames have invalid labels, sizes, or colors.');
        setNodes(payload.nodes);
        setEdges(payload.edges);
        setFrames(importedFrames);
        setActivePresetKey(null);
        setWorkflowName(typeof payload.name === 'string' && payload.name.trim() ? payload.name.trim() : file.name.replace(/\.triune\.json$|\.json$/i, ''));
        setNodeExecOutputs({});
        setSelectedNodeId(null);
        setSelectedNodeIds([]);
        setDagExecutionStatus(null);
        showToast(`Imported workflow “${payload.name || file.name}”.`);
      } catch (err) {
        showToast(`Could not import workflow: ${err.message || 'invalid file'}`);
      }
    };

    const refreshNodeCatalog = async () => {
      try {
        const res = await apiFetch('/v1/nodes/catalog');
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          setNodeCatalog(data.map(item => {
            const builtin = BUILTIN_NODE_CATALOG.find(candidate => candidate.name === item.name);
            return {
              ...item,
              title: item.title || (builtin && builtin.title) || item.name || 'Unnamed node',
              description: item.description || (builtin && builtin.description) || '',
              inputs: Array.isArray(item.inputs) ? item.inputs : [],
              outputs: Array.isArray(item.outputs) ? item.outputs : []
            };
          }));
        }
      } catch (e) {}
    };

    const updateDetailsString = (detailsStr, patchObj) => {
      const lines = (detailsStr || '').split('\n');
      const remaining = { ...patchObj };
      const nextLines = lines.map(line => {
        const t = line.trim();
        if (!t || t.startsWith('#')) return line;
        const eq = t.indexOf('=');
        if (eq !== -1) {
          const k = t.slice(0, eq).trim();
          if (Object.prototype.hasOwnProperty.call(remaining, k) && remaining[k] !== undefined) {
            const v = remaining[k];
            delete remaining[k];
            return `${k}=${v}`;
          }
        }
        return line;
      });
      Object.entries(remaining).forEach(([k, v]) => {
        if (v !== undefined) {
          nextLines.push(`${k}=${v}`);
        }
      });
      return nextLines.filter(Boolean).join('\n');
    };

    // Non-destructive bidirectional sync from UI/Engine state onto Visual Node Canvas
    const syncStateToDAGNodes = (patch) => {
      if (!patch || typeof patch !== 'object') return;
      setNodes(prev => prev.map(n => {
        const ntype = n.type || '';
        const title = n.title || '';
        const isStreamer = (
          n.id === 'node_1' ||
          ntype === 'Dataset' ||
          ntype === 'HuggingFaceStreamer' ||
          ntype === 'HFRestStreamReader' ||
          title.includes('Streamer') ||
          title.includes('Reader') ||
          (ntype === 'Data' && !title.includes('Token') && !title.includes('DataLoader'))
        );
        const isDataLoader = (
          n.id === 'node_3' ||
          ntype === 'DataLoader' ||
          ntype === 'CyclingDataLoader' ||
          title.includes('DataLoader')
        );
        const isModel = (
          n.id === 'node_5' ||
          ntype === 'Model' ||
          ntype === 'TriuneTransformer' ||
          title.includes('Transformer') ||
          title.includes('MoE') ||
          title.includes('Backbone')
        );
        const isRouter = (
          n.id === 'node_6' ||
          ntype === 'Router' ||
          ntype === 'ExitRouter' ||
          ntype === 'HierarchicalDepthRouter' ||
          title.includes('Router')
        );
        const isPrecision = (
          n.id === 'node_7' ||
          ntype === 'Precision' ||
          ntype === 'FP8Linear' ||
          ntype === 'NF4Quantizer' ||
          title.includes('FP8') ||
          title.includes('Quantiz')
        );
        const isOptimizer = (
          ntype === 'Optimizer' ||
          title.includes('AdamW')
        );
        const isRuntime = (
          ntype === 'Runtime' ||
          title.includes('Accumulat')
        );
        const isAdapter = (
          ntype === 'Adapter' ||
          ntype === 'LoRAAdapter' ||
          title.includes('LoRA')
        );

        let nodePatch = null;
        if (isStreamer && (patch.dataset_name !== undefined || patch.split !== undefined || patch.text_column !== undefined)) {
          nodePatch = {};
          if (patch.dataset_name !== undefined) nodePatch.dataset_name = patch.dataset_name;
          if (patch.split !== undefined) nodePatch.split = patch.split;
          if (patch.text_column !== undefined) nodePatch.text_column = patch.text_column;
        } else if (isDataLoader && (patch.batch_size !== undefined || patch.seq_len !== undefined)) {
          nodePatch = {};
          if (patch.batch_size !== undefined) nodePatch.batch_size = patch.batch_size;
          if (patch.seq_len !== undefined) nodePatch.seq_len = patch.seq_len;
        } else if (isModel && (patch.preset_id !== undefined || patch.num_layers !== undefined || patch.hidden_dim !== undefined || patch.num_heads !== undefined || patch.head_dim !== undefined || patch.num_experts !== undefined)) {
          nodePatch = {};
          if (patch.preset_id !== undefined) nodePatch.preset_id = patch.preset_id;
          if (patch.num_layers !== undefined) nodePatch.num_layers = patch.num_layers;
          if (patch.hidden_dim !== undefined) nodePatch.hidden_dim = patch.hidden_dim;
          if (patch.num_heads !== undefined) nodePatch.heads = patch.num_heads;
          if (patch.head_dim !== undefined) nodePatch.head_dim = patch.head_dim;
          if (patch.num_experts !== undefined) nodePatch.experts_routed = patch.num_experts;
        } else if (isRouter && (patch.depth_mode !== undefined || patch.num_layers !== undefined)) {
          nodePatch = {};
          if (patch.depth_mode !== undefined) nodePatch.depth_mode = patch.depth_mode;
          if (patch.num_layers !== undefined) {
            const nl = Math.max(2, parseInt(patch.num_layers) || 6);
            nodePatch.reflex_exit_layer = Math.max(1, Math.floor(nl / 3) - 1);
            nodePatch.limbic_exit_layer = Math.max(1, Math.floor((2 * nl) / 3) - 1);
            nodePatch.cortex_exit_layer = nl - 1;
          }
        } else if (isPrecision && patch.precision !== undefined) {
          const pLow = String(patch.precision).toLowerCase();
          nodePatch = {
            dtype: pLow === 'fp8' ? 'fp8_e4m3fn' : (pLow === 'fp4' ? 'fp4' : pLow)
          };
        } else if (isOptimizer && patch.lr !== undefined) {
          nodePatch = { lr: patch.lr };
        } else if (isRuntime && patch.grad_accum_steps !== undefined) {
          nodePatch = { grad_accum_steps: patch.grad_accum_steps };
        } else if (isAdapter && (patch.lora_rank !== undefined || patch.lora_alpha !== undefined)) {
          nodePatch = {};
          if (patch.lora_rank !== undefined) nodePatch.rank = patch.lora_rank;
          if (patch.lora_alpha !== undefined) nodePatch.alpha = patch.lora_alpha;
        }

        if (!nodePatch || Object.keys(nodePatch).length === 0) return n;
        return { ...n, details: updateDetailsString(n.details, nodePatch) };
      }));
    };

    // System Diagnostics, Catalog & Live Telemetry WebSocket
    useEffect(() => {
      refreshNodeCatalog();

      // Connect to WebSocket telemetry stream for live DAG and training events (with auto-reconnect)
      let ws = null;
      let wsReconnectTimer = null;
      let wsReconnectDelay = 1000;
      const connectWs = () => {
        try {
          const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
          const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
          ws = new WebSocket(wsUrl);
          ws.onopen = () => {
            wsReconnectDelay = 1000;
            // Cloudflare Access authenticates the browser at the edge and the
            // tunnel reaches this server over loopback. Only send a Triune
            // bearer token when a direct-remote API request has already needed it.
            if (!isLocalStudioHost(window.location.hostname) && apiAuthToken && ws.readyState === WebSocket.OPEN) {
              ws.send(apiAuthToken);
            }
          };
          ws.onmessage = (event) => {
            try {
              const msg = JSON.parse(event.data);
              if (msg.type === 'dag_event' && msg.data) {
                const d = msg.data;
                if (d.event === 'node_started') {
                  setActiveRunningNodeId(d.node_id);
                  setDagExecutionStatus(`Executing Node: ${d.node_name}...`);
                  setNodeExecOutputs(prev => ({
                    ...prev,
                    [d.node_id]: { status: 'running' }
                  }));
                } else if (d.event === 'node_finished') {
                  setNodeExecOutputs(prev => ({
                    ...prev,
                    [d.node_id]: {
                      status: 'completed',
                      output: d.output || {},
                      elapsed: d.elapsed_sec
                    }
                  }));
                } else if (d.event === 'node_failed') {
                  const nodeError = d.error || d.output?.error || d.output?.message || 'Node failed without a diagnostic from the execution engine.';
                  setNodeExecOutputs(prev => ({
                    ...prev,
                    [d.node_id]: {
                      status: 'failed',
                      error: nodeError,
                      output: d.output || {},
                      elapsed: d.elapsed_sec
                    }
                  }));
                  setDagExecutionStatus(`Node ${d.node_name || d.node_id} failed: ${nodeError}`);
                }
              } else if ((msg.type === 'training_step' || msg.type === 'step') && msg.data) {
                const d = msg.data;
                setMetrics(d);
                setMetricsHistory(prev => {
                  const next = [...prev, d];
                  return next.length > 100 ? next.slice(next.length - 100) : next;
                });
                if (d.exit_usage) setExitUsage(d.exit_usage);
                if (d.batch_preview) setActiveBatchPreview(d.batch_preview);
                if (d.active_checkpoint) setActiveCheckpointName(d.active_checkpoint);
                if (typeof d.vram_gb === 'number') {
                  setVramUsage(prev => ({ ...prev, allocated: d.vram_gb }));
                }
              } else if (msg.type === 'checkpoint_loaded' && msg.data) {
                const d = msg.data;
                if (d.active_checkpoint) setActiveCheckpointName(d.active_checkpoint);
                if (d.active_model) setActiveModel(d.active_model);
                if (d.architecture) {
                  setDatasetInfo(prev => ({
                    ...prev,
                    num_layers: d.architecture.num_layers ?? prev.num_layers,
                    hidden_dim: d.architecture.hidden_dim ?? prev.hidden_dim,
                    num_heads: d.architecture.num_heads ?? prev.num_heads,
                    num_experts: d.architecture.num_experts ?? prev.num_experts
                  }));
                }
                setWeightsLoaded(true);
                if (d.step !== undefined) {
                  setMetrics(prev => ({ ...prev, step: d.step, tokens_trained: d.tokens_trained || prev.tokens_trained }));
                }
                if (d.tokens_trained !== undefined) {
                  setDatasetInfo(prev => ({ ...prev, total_tokens: d.tokens_trained }));
                }
              }
            } catch (e) {}
          };
          ws.onclose = (event) => {
            if (event.code === 1008 && !isLocalStudioHost(window.location.hostname)) {
              apiAuthToken = '';
              // A direct remote server rejects the unauthenticated socket;
              // prompt only after that rejection. Access-protected tunnels
              // using loopback never hit this branch or show a token prompt.
              requestApiAuthToken().then(token => {
                if (token) connectWs();
              });
              return;
            }
            // Auto-reconnect with exponential backoff (max 15s)
            wsReconnectTimer = setTimeout(() => {
              wsReconnectDelay = Math.min(wsReconnectDelay * 1.5, 15000);
              connectWs();
            }, wsReconnectDelay);
          };
          ws.onerror = () => {};
        } catch (e) {}
      };
      connectWs();

      let fetchInFlight = false;
      let backendHealthFailures = 0;
      const fetchSystemInfo = async () => {
        if (fetchInFlight) return;
        fetchInFlight = true;
        let healthCheckSucceeded = false;
        try {
          const healthRes = await apiFetch('/healthz');
          if (!healthRes.ok) {
            backendHealthFailures += 1;
            if (backendHealthFailures >= 3) setBackendStatus('offline');
            return;
          }
          healthCheckSucceeded = true;
          backendHealthFailures = 0;
          setBackendStatus('online');

          const resDiag = await apiFetch('/v1/system/diagnostics');
          if (!resDiag.ok) return;
          const dataDiag = await resDiag.json();
          if (dataDiag.device_name) setSystemDiagnostics(dataDiag);

          const resVram = await apiFetch('/v1/vram/stats');
          const dataVram = await resVram.json();
          if (dataVram.total_gb !== undefined || dataVram.total !== undefined) {
            setVramUsage({
              allocated: (dataVram.allocated !== undefined && dataVram.allocated !== null) ? dataVram.allocated : ((dataVram.allocated_gb !== undefined && dataVram.allocated_gb !== null) ? dataVram.allocated_gb : 0.0),
              reserved: (dataVram.reserved !== undefined && dataVram.reserved !== null) ? dataVram.reserved : ((dataVram.reserved_gb !== undefined && dataVram.reserved_gb !== null) ? dataVram.reserved_gb : 0.0),
              total: (dataVram.total !== undefined && dataVram.total !== null) ? dataVram.total : ((dataVram.total_gb !== undefined && dataVram.total_gb !== null) ? dataVram.total_gb : 8.0),
              oom_risk: Boolean(dataVram.oom_risk)
            });
          }

          try {
            const resHw = await apiFetch('/v1/hardware/profile');
            const dataHw = await resHw.json();
            if (dataHw && dataHw.device_name) setHardwareProfile(dataHw);
          } catch (hwErr) {}

          const statusRes = await apiFetch('/v1/training/status');
          const statusData = await statusRes.json();
          setIsTraining(statusData.is_training);
          if (statusData.active_model) {
            setActiveModel(prev => {
              const isByok = prev && (prev.includes('gpt') || prev.includes('claude') || prev.includes('gemini'));
              return isByok ? prev : statusData.active_model;
            });
          }
          setWeightsLoaded(Boolean(statusData.weights_loaded));
          if (statusData.precision) {
            setTrainPrecision(statusData.precision);
            setPrecision(statusData.precision);
          }
          if (statusData.gpu_offload_pct !== undefined) setGpuOffloadPct(statusData.gpu_offload_pct);
          if (statusData.gpu_layers !== undefined) setGpuLayers(statusData.gpu_layers);
          if (statusData.warmup_steps !== undefined) setTrainWarmup(statusData.warmup_steps);
          if (statusData.active_adapter !== undefined) {
            setActiveAdapter(statusData.active_adapter);
          }
          if (statusData.wandb_url) setWandbUrl(statusData.wandb_url);
          if (statusData.wandb_active !== undefined) setWandbActive(statusData.wandb_active);
          if (statusData.dataset) {
            const arch = statusData.architecture || {};
            setDatasetInfo(prev => ({
              ...prev,
              ...statusData.dataset,
              num_layers: arch.num_layers || prev.num_layers || 6,
              hidden_dim: arch.hidden_dim || prev.hidden_dim || 256,
              num_heads: arch.num_heads || prev.num_heads || 4,
              num_experts: arch.num_experts || prev.num_experts || 4
            }));
            if (statusData.dataset.name) setActiveDataset(statusData.dataset.name);
            if (Date.now() - lastManualConfigEditRef.current > 4000) {
              if (statusData.dataset.batch_size) setTrainBatchSize(statusData.dataset.batch_size);
              if (statusData.dataset.seq_len) setTrainSeqLen(statusData.dataset.seq_len);
              if (statusData.dataset.grad_accum_steps) setTrainGradAccum(statusData.dataset.grad_accum_steps);
              if (statusData.dataset.depth_mode) setTrainDepthMode(statusData.dataset.depth_mode);
              if (statusData.dataset.lr_peak) setTrainLR(String(statusData.dataset.lr_peak));
              else if (statusData.lr_peak) setTrainLR(String(statusData.lr_peak));
            }
            if (statusData.dataset.active_batch_preview) {
              setActiveBatchPreview(statusData.dataset.active_batch_preview);
            }
          }
          setActiveCheckpointName(statusData.active_checkpoint || null);
          if (statusData.history && statusData.history.length > 0) {
            setMetricsHistory(statusData.history);
            const latest = statusData.history[statusData.history.length - 1];
            setMetrics(latest);
            if (latest.exit_usage) setExitUsage(latest.exit_usage);
            if (latest.batch_preview) setActiveBatchPreview(latest.batch_preview);
          }
          if (statusData.logs && statusData.logs.length > 0) setTelemetryLogs(statusData.logs);
        } catch (err) {
          if (!healthCheckSucceeded) {
            backendHealthFailures += 1;
            if (backendHealthFailures >= 3) setBackendStatus('offline');
          }
        } finally {
          fetchInFlight = false;
        }
      };
      fetchSystemInfo();
      const interval = setInterval(fetchSystemInfo, 3000);
      return () => {
        clearInterval(interval);
        if (wsReconnectTimer) clearTimeout(wsReconnectTimer);
        if (ws) {
          ws.onclose = null; // prevent reconnect on intentional close
          try { ws.close(); } catch (e) {}
        }
      };
    }, [backendRetry]);

    // Auto-scroll chat smoothly on new messages or generation updates
    useEffect(() => {
      if (activeTab === 'chat' && chatBottomRef.current) {
        try {
          chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
        } catch (e) {}
      }
    }, [messages, isGenerating, activeTab]);

    // Module Marketplace & System Scan Handlers
    const fetchSystemScan = async () => {
      try {
        const res = await apiFetch('/v1/system/scan');
        const data = await res.json();
        setSystemScan(data);
      } catch (err) {}
    };

    const fetchSystemConfig = async () => {
      try {
        const res = await apiFetch('/v1/system/config');
        const data = await res.json();
        setSystemConfig(data);
        if (data.byok_keys) {
          // Never place masked server values in the editable secret inputs.
          // They are display metadata only; blank means "keep the saved key".
          setByokConfigured(Object.fromEntries(
            Object.entries(data.byok_keys).map(([provider, value]) => [provider, Boolean(value)])
          ));
        }
      } catch (err) {}
    };

    const fetchDatasets = async () => {
      try {
        const res = await apiFetch('/v1/datasets');
        const data = await res.json();
        if (data.datasets && data.datasets.length > 0) {
          setDatasetList(data.datasets);
        }
        // Also query active dataset stream telemetry
        const sRes = await apiFetch('/v1/datasets/stream/status');
        const sData = await sRes.json();
        if (sData.is_streaming) {
          setHfStreamResult(sData);
          if (sData.dataset_name) {
            setActiveDataset(sData.dataset_name);
            setHfDatasetInput(sData.dataset_name);
          }
          if (sData.config) setHfConfigInput(sData.config);
          if (sData.split) setHfSplitInput(sData.split);
          if (sData.text_column) setHfTextColInput(sData.text_column);
          if (sData.active_batch_preview) setActiveBatchPreview(sData.active_batch_preview);
        }
      } catch (err) {}
    };

    const handleConnectHfStream = async (datasetName = hfDatasetInput, config = hfConfigInput, split = hfSplitInput, col = hfTextColInput) => {
      const dName = (datasetName || '').trim();
      if (!dName) {
        showToast('Please enter a valid Hugging Face dataset identifier or URL');
        return;
      }
      setHfDatasetInput(dName);
      setHfConfigInput(config || '');
      setHfSplitInput(split || 'train');
      setHfTextColInput(col || '');
      setIsConnectingHf(true);
      showToast(`Connecting to ${dName}...`);
      try {
        const res = await apiFetch('/v1/datasets/stream/connect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_name: dName,
            config: (config || '').trim() || undefined,
            split: (split || 'train').trim(),
            text_column: (col || '').trim() || undefined
          })
        });
        const data = await res.json();
        if (data.status === 'success') {
          setHfStreamResult(data);
          setActiveDataset(dName);
          setDatasetInfo(prev => ({
            ...prev,
            name: dName,
            type: dName.startsWith('http') ? 'url_streaming' : 'hf_streaming',
            is_streaming: true,
            active_batch_preview: data.sample_preview || prev.active_batch_preview
          }));
          if (data.sample_preview) {
            setActiveBatchPreview(data.sample_preview);
          }
          setLoraConfig(prev => ({ ...prev, dataset: dName }));
          syncStateToDAGNodes({
            dataset_name: dName,
            split: (split || 'train').trim(),
            text_column: data.text_column || (col || 'text').trim()
          });
          showToast(`Streaming live from ${dName}`);
          fetchDatasets();
        } else {
          showToast(`Connection failed: ${data.message || 'error'}`);
        }
      } catch (err) {
        showToast(`Streaming error: ${err.message}`);
      } finally {
        setIsConnectingHf(false);
      }
    };

    const handleSelectDataset = async (ds) => {
      const dPath = ds.path || ds.name;
      showToast(`Activating ${ds.name}...`);
      try {
        const res = await apiFetch('/v1/datasets/select', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dataset_path: dPath })
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
          setActiveDataset(ds.name);
          setDatasetInfo(prev => ({
            ...prev,
            name: ds.name,
            type: ds.type || (dPath.includes('/') && !dPath.startsWith('data/') ? 'streaming' : 'local'),
            active_batch_preview: data.sample_preview || prev.active_batch_preview
          }));
          if (data.sample_preview) {
            setActiveBatchPreview(data.sample_preview);
          }
          setLoraConfig(prev => ({ ...prev, dataset: dPath }));
          syncStateToDAGNodes({
            dataset_name: ds.name,
            split: 'train',
            text_column: data.text_column || 'text'
          });
          showToast(`Activated ${ds.name} across Training, LoRA & DAG!`);
          fetchDatasets();
        } else {
          showToast(`Could not activate dataset: ${data.message || 'error'}`);
        }
      } catch (e) {
        showToast(`Could not activate dataset ${ds.name}: ${e.message || 'backend request failed'}`);
      }
    };

    const handleRegisterLocalDataset = async () => {
      const p = (localDatasetInput || '').trim();
      if (!p) {
        showToast('Please enter a local file path (e.g. data/fineweb_sample.jsonl)');
        return;
      }
      showToast(`Registering local file ${p}...`);
      try {
        const res = await apiFetch('/v1/datasets/local/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: p })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(data.message);
          setLocalDatasetInput('');
          fetchDatasets();
        } else {
          showToast(data.message);
        }
      } catch (err) {
        showToast(`Error: ${err.message}`);
      }
    };

    const handleExportModel = async (modelId, format) => {
      showToast(`Exporting ${modelId} as ${format.toUpperCase()}...`);
      try {
        const res = await apiFetch('/v1/models/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_id: modelId, format: format.toLowerCase() })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(data.message || `Exported ${modelId} to ${data.path}`);
        } else {
          showToast(`Export failed: ${data.message || 'unknown error'}`);
        }
      } catch (err) {
        showToast(`Export error: ${err.message}`);
      }
    };

    const handleExportCheckpoint = async (checkpointPath) => {
      try {
        const res = await apiFetch('/v1/checkpoints/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ checkpoint_path: checkpointPath })
        });
        const data = await res.json();
        if (!res.ok || data.status !== 'success') {
          throw new Error(data.message || data.detail || `Export failed (${res.status})`);
        }
        showToast(data.message || `Exported selected checkpoint to ${data.path}`);
      } catch (err) {
        showToast(`Checkpoint export failed: ${err.message || 'backend request failed'}`);
      }
    };

    const saveSystemConfig = async (newCfg) => {
      try {
        const res = await apiFetch('/v1/system/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(newCfg)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message || `Save failed (${res.status})`);
        setSystemConfig(data);
        showToast('System configuration & custom paths saved!');
      } catch (err) {
        showToast(`Failed to save configuration: ${err.message || 'backend request failed'}`);
      }
    };

    const searchMarketplace = async (query = moduleSearchQuery, filter = moduleFilter, source = 'all') => {
      setIsSearchingModules(true);
      setMarketplaceError('');
      try {
        const res = await apiFetch(`/v1/modules/search?q=${encodeURIComponent(query)}&type=${encodeURIComponent(filter)}&source=${encodeURIComponent(source)}`);
        if (!res.ok) throw new Error(`Search failed (HTTP ${res.status})`);
        const data = await res.json();
        setMarketplaceData(data);
      } catch (err) {
        setMarketplaceError(err.message || 'Could not search the extension marketplace.');
      }
      setIsSearchingModules(false);
    };

    const fetchInstalledModules = async () => {
      try {
        const res = await apiFetch('/v1/modules/installed');
        const data = await res.json();
        setInstalledModules(data);
      } catch (err) {}
    };

    const checkModuleUpdates = async () => {
      try {
        const res = await apiFetch('/v1/modules/updates');
        if (!res.ok) throw new Error(`Update check failed (HTTP ${res.status})`);
        const data = await res.json();
        setAvailableUpdates(data);
        if (data.length > 0) {
          showToast(`${data.length} module update(s) available!`);
        } else {
          showToast('All modules are up to date.');
        }
      } catch (err) {
        showToast(`Could not check extension updates: ${err.message || 'backend unavailable'}`);
      }
    };

    const handleCloneCustomRepo = async (ev) => {
      if (ev) ev.preventDefault();
      if (!repoCloneUrl || !repoCloneUrl.trim()) {
        showToast('Please enter a Git repository or Hugging Face URL.');
        return;
      }
      setIsCloningRepo(true);
      showToast(`Cloning repo: ${repoCloneUrl.trim()}...`);
      try {
        const res = await apiFetch('/v1/modules/clone', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            url: repoCloneUrl.trim(),
            type: repoCloneType,
            branch: repoCloneBranch || 'main'
          })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(`Successfully cloned ${data.name || 'repository'}!`);
          setRepoCloneUrl('');
          await fetchInstalledModules();
          await refreshNodeCatalog();
          fetchDatasets();
          fetchCheckpoints();
          setMarketplaceSubTab('installed');
          if (data.registered_nodes && data.registered_nodes.length > 0) {
            showToast(`Registered DAG nodes: ${data.registered_nodes.join(', ')}`);
          }
        } else {
          showToast(`Clone failed: ${data.message || 'Unknown error'}`);
        }
      } catch (err) {
        showToast('Clone request failed. Check server connection.');
      }
      setIsCloningRepo(false);
    };

    const handleGitPullRepo = async (modId) => {
      setActivePullingModId(modId);
      showToast(`Git pulling latest changes for ${modId}...`);
      try {
        const res = await apiFetch('/v1/modules/pull', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: modId })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(data.message);
          await fetchInstalledModules();
          await refreshNodeCatalog();
        } else {
          showToast(`Notice: ${data.message || 'Git pull failed'}`);
        }
      } catch (err) {
        showToast('Git pull request failed.');
      }
      setActivePullingModId(null);
    };

    const installModule = async (modData) => {
      showToast(`Installing ${modData.name}...`);
      try {
        const res = await apiFetch('/v1/modules/install', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(modData)
        });
        const data = await res.json();
        if (!res.ok || data.success !== true) {
          throw new Error(data.message || `Install failed (${res.status})`);
        }
        showToast(data.message || `Installed ${modData.name}`);
        await fetchInstalledModules();
        await refreshNodeCatalog();
        await fetchDatasets();
        await fetchCheckpoints();
        await searchMarketplace(moduleSearchQuery, moduleFilter, 'all');
      } catch (err) {
        showToast(`Installation failed for ${modData.name}: ${err.message || 'backend request failed'}`);
      }
    };

    const uninstallModule = async (modId) => {
      if (!confirm(`Are you sure you want to uninstall module "${modId}"? This will remove all associated files.`)) return;
      try {
        const res = await apiFetch('/v1/modules/uninstall', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: modId })
        });
        const data = await res.json();
        if (!res.ok || data.success !== true) {
          throw new Error(data.message || `Uninstall failed (${res.status})`);
        }
        showToast(data.message || 'Module uninstalled');
        await fetchInstalledModules();
        await refreshNodeCatalog();
        await searchMarketplace(moduleSearchQuery, moduleFilter, 'all');
      } catch (err) {
        showToast(`Uninstall failed: ${err.message}`);
      }
    };

    const handleScanWorkspacePlugins = async () => {
      showToast('Scanning workspace for custom Python plugins...');
      try {
        const res = await apiFetch('/v1/modules/scan_local', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          showToast(data.message);
          fetchInstalledModules();
          refreshNodeCatalog();
        } else {
          showToast(`Notice: ${data.message || 'Scan completed'}`);
        }
      } catch (err) {
        showToast('Error scanning workspace plugins');
      }
    };

    const handleCreatePluginScript = async () => {
      if (!newPluginName.trim()) return;
      showToast(`Generating plugin template for ${newPluginName}...`);
      try {
        const res = await apiFetch('/v1/modules/create_plugin', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: newPluginName.trim(), category: newPluginCategory })
        });
        const data = await res.json();
        if (data.success) {
          showToast(data.message || `Created plugin template at ${data.file_path}. Review it, then scan the workspace to load it.`);
          setShowWorkspacePluginModal(false);
          fetchInstalledModules();
        } else {
          showToast(`Failed to create plugin: ${data.message}`);
        }
      } catch (err) {
        showToast(`Error: ${err.message}`);
      }
    };

    const handleAddModuleNodesToCanvas = (m) => {
      const nodeNames = m.registered_nodes || (m.artifacts && m.artifacts.nodes) || [];
      if (nodeNames.length === 0) {
        const newNodeId = `node_${Date.now()}`;
        const newModNode = {
          id: newNodeId,
          title: m.name || 'Module Node',
          type: m.type === 'dataset' ? 'Data' : (m.type === 'loss' ? 'Loss' : 'Model'),
          x: 200 + Math.floor(Math.random() * 200),
          y: 150 + Math.floor(Math.random() * 200),
          details: `module=${m.id}\ninstalled_at=${m.installed_at || ''}`
        };
        setNodes(prev => [...prev, newModNode]);
        setActiveTab('nodegraph');
        showToast(`Added [${m.name}] node onto Visual Canvas!`);
        return;
      }
      let added = 0;
      nodeNames.forEach(nName => {
        handleSpawnCatalogNode(nName);
        added++;
      });
      setActiveTab('nodegraph');
      showToast(`Added ${added} node(s) from ${m.name} to Visual Canvas!`);
    };

    const handleLoadModuleWeights = async (m) => {
      const weights = m.artifacts && m.artifacts.weights;
      const ckptTarget = (weights && weights.length > 0) ? weights[0] : (m.installed_at || m.id);
      showToast(`Loading weights for ${m.name}...`);
      try {
        const res = await apiFetch('/v1/models/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ checkpoint_path: ckptTarget })
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
          showToast(data.message || `Loaded model weights from ${m.name}.`);
          if (data.active_model) setActiveModel(data.active_model);
          setWeightsLoaded(Boolean(data.weights_loaded));
          if (data.checkpoint) setActiveCheckpointName(data.checkpoint);
          if (data.step !== undefined) {
            const restoredTok = data.tokens_trained !== undefined ? data.tokens_trained : (data.step * 1024);
            setMetrics(prev => ({ ...prev, step: data.step, tokens_trained: restoredTok }));
            setDatasetInfo(prev => ({ ...prev, total_tokens: restoredTok }));
          }
        } else {
          throw new Error(data.message || `Weight load failed (${res.status})`);
        }
      } catch (err) {
        showToast(`Could not load ${m.name}: ${err.message || 'backend request failed'}`);
        return;
      }
      setActiveTab('chat');
    };

    const handleStreamModuleDataset = (m) => {
      const dsName = m.repo_url && m.repo_url.includes('huggingface.co')
        ? m.repo_url.replace('https://huggingface.co/datasets/', '').replace('https://huggingface.co/', '')
        : (m.id || m.name);
      setHfDatasetInput(dsName);
      setDatasetInfo(prev => ({ ...prev, name: dsName }));
      setActiveTab('datasets');
      showToast(`Connecting live engine stream to ${dsName}...`);
      handleConnectHfStream(dsName, '', 'train', 'text');
    };

    const handleAttachAdapterToEngine = async (adapterSpec) => {
      const spec = adapterSpec || {
        id: `lora-r${loraConfig.rank}`,
        name: `LoRA-r${loraConfig.rank}`,
        rank: loraConfig.rank,
        alpha: loraConfig.alpha,
        quantization: loraConfig.quantization || '4-bit NF4'
      };
      try {
        const res = await apiFetch('/v1/finetune/adapter', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'attach',
            id: spec.id,
            name: spec.name,
            path: spec.path,
            rank: spec.rank || 16,
            alpha: spec.alpha || 32.0,
            quantization: spec.quantization || loraConfig.quantization || '4-bit NF4'
          })
        });
        const data = await res.json();
        if (!res.ok || data.status !== 'success' || !data.active_adapter) {
          throw new Error(data.message || `Adapter attach failed (${res.status})`);
        }
        setActiveAdapter(data.active_adapter);
        syncStateToDAGNodes({ lora_rank: data.active_adapter.rank, lora_alpha: data.active_adapter.alpha });
        showToast(data.message || `Attached ${data.active_adapter.name} to the engine.`);
      } catch (err) {
        showToast(`Could not attach ${spec.name}: ${err.message || 'backend request failed'}`);
      }
    };

    const handleDetachAdapterFromEngine = async () => {
      try {
        const res = await apiFetch('/v1/finetune/adapter', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'detach' })
        });
        const data = await res.json();
        if (!res.ok || data.status !== 'success' || data.action !== 'detached') {
          throw new Error(data.message || `Adapter detach failed (${res.status})`);
        }
        setActiveAdapter(null);
        showToast(data.message || 'Detached LoRA Adapter from Studio Engine');
      } catch (err) {
        showToast(`Could not detach adapter: ${err.message || 'backend request failed'}`);
      }
    };

    const handleAttachModuleAdapter = (m) => {
      handleAttachAdapterToEngine({
        id: m.id,
        name: m.name,
        path: m.installed_at,
        rank: 16,
        alpha: 32,
        quantization: loraConfig.quantization || '4-bit NF4'
      });
      setActiveTab('finetune');
    };

    // Checkpoint Management Handlers
    const fetchCheckpoints = async () => {
      try {
        const res = await apiFetch('/v1/checkpoints');
        const data = await res.json();
        if (data.checkpoints) {
          setCheckpoints(data.checkpoints);
        }
        if (data.active_checkpoint) {
          setActiveCheckpointName(data.active_checkpoint);
        }
      } catch (err) {}
    };

    const handleLoadCheckpoint = async (ckptPath) => {
      setIsLoadingCkpt(true);
      try {
        const res = await apiFetch('/v1/checkpoints/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ checkpoint_path: ckptPath })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(data.message);
          if (data.active_model) setActiveModel(data.active_model);
          setWeightsLoaded(Boolean(data.weights_loaded));
          setActiveCheckpointName(data.checkpoint);
          if (data.step !== undefined) {
            const restoredTok = data.tokens_trained !== undefined ? data.tokens_trained : (data.step * 1024);
            setMetrics(prev => ({ ...prev, step: data.step, tokens_trained: restoredTok }));
            setDatasetInfo(prev => ({ ...prev, total_tokens: restoredTok }));
          }
          fetchCheckpoints();
        } else {
          showToast(data.message);
        }
      } catch (err) {
        showToast(`Load error: ${err.message}`);
      } finally {
        setIsLoadingCkpt(false);
      }
    };

    const handleSaveCheckpointCustom = async () => {
      try {
        const res = await apiFetch('/v1/checkpoints/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: saveCkptName.trim() || undefined })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(`Saved checkpoint: ${data.filename} (${data.size_mb} MB)`);
          if (data.filename) setActiveCheckpointName(data.filename);
          setSaveCkptName('');
          fetchCheckpoints();
        } else {
          showToast(`Save error: ${data.message}`);
        }
      } catch (err) {
        showToast(`Save error: ${err.message}`);
      }
    };

    const handleDeleteCheckpoint = async (ckptPath) => {
      if (!confirm(`Are you sure you want to delete checkpoint: ${ckptPath}?`)) return;
      try {
        const res = await apiFetch('/v1/checkpoints/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ checkpoint_path: ckptPath })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(data.message);
          fetchCheckpoints();
        } else {
          showToast(`Delete failed: ${data.message}`);
        }
      } catch (err) {
        showToast(`Delete error: ${err.message}`);
      }
    };

    // On-Mount Initialization & Automatic Update Check on Startup
    useEffect(() => {
      fetchSystemScan();
      fetchSystemConfig();
      fetchHardwareProfile();
      fetchInstalledModules();
      fetchDatasets();
      fetchCheckpoints();
      searchMarketplace('', 'all');
      checkModuleUpdates();
    }, []);

    // Non-Blocking Active Hardware Telemetry Polling Loop
    useEffect(() => {
      let interval;
      if (isTraining) {
        interval = setInterval(async () => {
          try {
            const res = await apiFetch('/v1/training/status');
            const data = await res.json();
            if (data.active_model) {
              setActiveModel(data.active_model);
            }
            if (data.dataset) {
              setDatasetInfo(data.dataset);
              if (Date.now() - lastManualConfigEditRef.current > 4000) {
                if (data.dataset.batch_size) setTrainBatchSize(data.dataset.batch_size);
                if (data.dataset.seq_len) setTrainSeqLen(data.dataset.seq_len);
                if (data.dataset.grad_accum_steps) setTrainGradAccum(data.dataset.grad_accum_steps);
                if (data.dataset.depth_mode) setTrainDepthMode(data.dataset.depth_mode);
              }
              if (data.dataset.active_batch_preview) {
                setActiveBatchPreview(data.dataset.active_batch_preview);
              }
            }
            if (data.active_checkpoint) {
              setActiveCheckpointName(data.active_checkpoint);
            }
            if (data.history && data.history.length > 0) {
              setMetricsHistory(data.history);
              const latest = data.history[data.history.length - 1];
              setMetrics(latest);
              if (latest.exit_usage) setExitUsage(latest.exit_usage);
              if (latest.batch_preview) setActiveBatchPreview(latest.batch_preview);
            }
            if (data.logs && data.logs.length > 0) setTelemetryLogs(data.logs);
            if (data.last_sample) setLastSample(data.last_sample.replace(/Ġ/g, ' ').replace(/\s+/g, ' ').trim());
          } catch (err) {}
        }, 300);
      }
      return () => clearInterval(interval);
    }, [isTraining]);

    // Tokenizer Sandbox
    useEffect(() => {
      if (!sampleText) {
        setTokens([]);
        setTokenizerStatus('Enter text to preview tokens.');
        return;
      }
      let cancelled = false;
      const tid = setTimeout(async () => {
        try {
          const res = await apiFetch('/v1/tokenizer/tokenize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: sampleText })
          });
          const data = await res.json();
          if (cancelled) return;
          if (data.tokens && data.tokens.length > 0) {
            setTokens(data.tokens);
            setTokenizerStatus(`${data.tokenizer || 'Configured tokenizer'} · ${data.tokens.length} tokens`);
            return;
          }
          setTokens([]);
          setTokenizerStatus(data.message || 'No configured tokenizer is available. Load a tokenizer to see real token IDs.');
        } catch (err) {
          if (cancelled) return;
          setTokens([]);
          setTokenizerStatus(`Tokenizer service unavailable: ${err.message || 'could not reach the engine'}`);
        }
      }, 200);
      return () => { cancelled = true; clearTimeout(tid); };
    }, [sampleText]);

    // WandB-Style Multi-Metric Loss & Telemetry Canvas Chart Renderer
    useEffect(() => {
      if (activeTab === 'training' && canvasRef.current) {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const displayWidth = Math.max(300, Math.floor(rect.width || 850));
        const displayHeight = Math.max(180, Math.floor(rect.height || 260));
        const dpr = window.devicePixelRatio || 1;
        canvas.width = displayWidth * dpr;
        canvas.height = displayHeight * dpr;
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        const chartStyle = window.getComputedStyle(canvas);
        const chartColor = (name, fallback) => chartStyle.getPropertyValue(name).trim() || fallback;
        const chartColors = {
          grid: chartColor('--chart-grid', '#e8e4d8'),
          axis: chartColor('--chart-axis', '#78716c'),
          loss: chartColor('--chart-loss', '#b45309'),
          lmLoss: chartColor('--chart-lm-loss', '#2b4c3f'),
          routerLoss: chartColor('--chart-router-loss', '#d97706'),
          throughput: chartColor('--chart-throughput', '#0d9488'),
          ppl: chartColor('--chart-ppl', '#6366f1')
        };
        const width = displayWidth;
        const height = displayHeight;
        ctx.clearRect(0, 0, width, height);

        // Grid lines with classic editorial warm tint
        ctx.strokeStyle = chartColors.grid;
        ctx.lineWidth = 1;
        for (let x = 60; x < width - 15; x += 60) {
          ctx.beginPath(); ctx.moveTo(x, 15); ctx.lineTo(x, height - 30); ctx.stroke();
        }
        for (let y = 15; y < height - 30; y += 35) {
          ctx.beginPath(); ctx.moveTo(60, y); ctx.lineTo(width - 15, y); ctx.stroke();
        }

        if (metricsHistory && metricsHistory.length >= 1) {
          const metricConfigs = [
            { key: 'loss', name: 'Total Loss', color: chartColors.loss, active: chartMetrics.loss },
            { key: 'lm_loss', name: 'LM Loss', color: chartColors.lmLoss, active: chartMetrics.lm_loss },
            { key: 'router_loss', name: 'Router Loss', color: chartColors.routerLoss, active: chartMetrics.router_loss },
            { key: 'throughput', name: 'tok/s', color: chartColors.throughput, active: chartMetrics.throughput },
            { key: 'ppl', name: 'Perplexity', color: chartColors.ppl, active: chartMetrics.ppl }
          ];

          const allLossValues = [];
          metricsHistory.forEach(m => {
            if (chartMetrics.loss && typeof m.loss === 'number') allLossValues.push(m.loss);
            if (chartMetrics.lm_loss && typeof m.lm_loss === 'number') allLossValues.push(m.lm_loss);
            if (chartMetrics.router_loss && typeof m.router_loss === 'number') allLossValues.push(m.router_loss);
            if (chartMetrics.throughput && typeof m.throughput === 'number') allLossValues.push(m.throughput);
            if (chartMetrics.ppl) {
              const lm = Number(m.lm_loss || m.loss || 0);
              if (lm > 0) allLossValues.push(Math.min(100, Math.exp(Math.min(lm, 10))));
            }
          });

          let minY = allLossValues.length ? Math.min(...allLossValues) : 0;
          let maxY = allLossValues.length ? Math.max(...allLossValues) : 4.0;
          if (minY === maxY) {
            minY = Math.max(0, minY - 0.5);
            maxY = maxY + 0.5;
          } else {
            const pad = (maxY - minY) * 0.12;
            minY = Math.max(0, minY - pad);
            maxY = maxY + pad;
          }
          if (chartScale === 'log') {
            minY = Math.max(0.001, minY);
            maxY = Math.max(minY * 1.5, maxY);
          }
          const rangeY = Math.max(maxY - minY, 0.0001);

          // Render Y-Axis numeric ticks
          ctx.fillStyle = chartColors.axis;
          ctx.font = '10px monospace';
          ctx.textAlign = 'right';
          const yTicks = 4;
          for (let i = 0; i <= yTicks; i++) {
            const frac = i / yTicks;
            const val = chartScale === 'log'
              ? Math.exp(Math.log(minY) + frac * (Math.log(maxY) - Math.log(minY)))
              : minY + frac * rangeY;
            const yPos = (height - 30) - frac * (height - 50);
            ctx.fillText(val.toFixed(3), 54, yPos + 3);
          }

          // Render X-Axis ticks
          ctx.textAlign = 'center';
          const n = metricsHistory.length;
          const firstStep = metricsHistory[0].step || 0;
          const lastStep = metricsHistory[n - 1].step || (firstStep + n);
          ctx.fillText(`Step ${firstStep}`, 65, height - 12);
          ctx.fillText(`Step ${Math.round((firstStep + lastStep) / 2)}`, width / 2, height - 12);
          ctx.fillText(`Step ${lastStep}`, width - 35, height - 12);

          // Plot each active series
          const plotWidth = width - 85;
          const plotHeight = height - 50;
          const startX = 65;
          const startY = height - 30;

          metricConfigs.forEach(cfg => {
            if (!cfg.active) return;
            const rawVals = metricsHistory.map(m => {
              if (cfg.key === 'ppl') {
                const lm = Number(m.lm_loss || m.loss || 0);
                return lm > 0 ? Math.min(100, Math.exp(Math.min(lm, 10))) : 1;
              }
              if (cfg.key === 'throughput') {
                return Number(m.throughput || 0);
              }
              return Number(m[cfg.key] || 0);
            });

            // Bias-corrected EMA smoothing
            const smoothedVals = [];
            const alpha = chartSmoothing;
            let s = rawVals[0];
            rawVals.forEach((val, idx) => {
              if (alpha <= 0) {
                smoothedVals.push(val);
              } else {
                s = alpha * s + (1 - alpha) * val;
                const biasCorr = 1 - Math.pow(alpha, idx + 1);
                smoothedVals.push(biasCorr > 0 ? s / biasCorr : s);
              }
            });

            const getY = (val) => {
              let norm = (val - minY) / rangeY;
              if (chartScale === 'log') {
                norm = (Math.log(Math.max(0.001, val)) - Math.log(minY)) / (Math.log(maxY) - Math.log(minY));
              }
              norm = Math.max(0, Math.min(1, norm));
              return startY - norm * plotHeight;
            };

            if (n === 1) {
              const y0 = getY(smoothedVals[0]);
              ctx.beginPath();
              ctx.strokeStyle = cfg.color;
              ctx.lineWidth = 2.4;
              ctx.moveTo(startX, y0);
              ctx.lineTo(startX + plotWidth, y0);
              ctx.stroke();
              ctx.beginPath();
              ctx.fillStyle = cfg.color;
              ctx.arc(startX + plotWidth / 2, y0, 4, 0, Math.PI * 2);
              ctx.fill();
              return;
            }

            // 1. Raw Ghost Signal Line (light translucent)
            if (alpha > 0.05) {
              ctx.beginPath();
              ctx.strokeStyle = cfg.color;
              ctx.globalAlpha = 0.22;
              ctx.lineWidth = 1.2;
              rawVals.forEach((val, idx) => {
                const x = startX + (idx / (n - 1)) * plotWidth;
                const y = getY(val);
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
              });
              ctx.stroke();
              ctx.globalAlpha = 1.0;
            }

            // 2. Smoothed Solid Trend Line
            ctx.beginPath();
            ctx.strokeStyle = cfg.color;
            ctx.lineWidth = 2.4;
            smoothedVals.forEach((val, idx) => {
              const x = startX + (idx / (n - 1)) * plotWidth;
              const y = getY(val);
              if (idx === 0) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
            });
            ctx.stroke();
          });
        } else {
          ctx.fillStyle = chartColors.axis;
          ctx.font = 'italic 13px Newsreader, Georgia, serif';
          ctx.textAlign = 'center';
          ctx.fillText('No training metrics yet. Start Training or choose Train 1 Step to begin.', width / 2, height / 2);
          ctx.textAlign = 'start';
        }
      }
    }, [activeTab, metricsHistory, chartMetrics, chartSmoothing, chartScale, chartXAxis, windowWidth]);

    // WandB Experiment Research Export Handlers
    const handleExportRunCSV = () => {
      if (!metricsHistory || metricsHistory.length === 0) {
        showToast('No metrics to export yet. Run training steps first.');
        return;
      }
      const headers = ['step', 'loss', 'lm_loss', 'router_loss', 'perplexity', 'throughput_tok_s', 'vram_gb', 'reflex_pct', 'limbic_pct', 'cortex_pct', 'lr', 'tokens_trained'];
      const rows = metricsHistory.map(m => {
        const lm = Number(m.lm_loss || m.loss || 0);
        const ppl = lm > 0 ? Math.exp(Math.min(lm, 20)).toFixed(2) : '1.00';
        const exit = m.exit_usage || {};
        return [
          m.step || 0,
          m.loss !== undefined ? Number(m.loss).toFixed(4) : '',
          m.lm_loss !== undefined ? Number(m.lm_loss).toFixed(4) : '',
          m.router_loss !== undefined ? Number(m.router_loss).toFixed(4) : '',
          ppl,
          m.throughput || 0,
          m.vram_gb || 0,
          exit.reflex || 0,
          exit.limbic || 0,
          exit.cortex || 0,
          m.lr || trainLR,
          m.tokens_trained || 0
        ].join(',');
      });
      const csvStr = [headers.join(','), ...rows].join('\n');
      const blob = new Blob([csvStr], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${runName || 'triune-run'}_${Date.now()}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      showToast(`Exported ${metricsHistory.length} steps to CSV`);
    };

    const handleExportRunJSON = () => {
      const losses = metricsHistory.map(m => Number(m.loss)).filter(l => !isNaN(l));
      const payload = {
        format: 'wandb_experiment_run_v1',
        run_name: runName,
        project: 'TriuneTransformer',
        created_at: new Date().toISOString(),
        architecture: {
          model_preset: activeModel,
          batch_size: trainBatchSize,
          seq_len: trainSeqLen,
          grad_accum: trainGradAccum,
          depth_mode: trainDepthMode,
          precision: trainPrecision,
          optimizer: 'CentroidSteer + Muon (GaLore)',
          dataset: datasetInfo.name || 'HuggingFaceFW/fineweb-edu'
        },
        summary: {
          total_steps: metrics.step || 0,
          total_tokens: metrics.tokens_trained || datasetInfo.total_tokens || 0,
          min_loss: losses.length ? Math.min(...losses) : null,
          final_loss: losses.length ? losses[losses.length - 1] : null,
          peak_throughput: Math.max(...metricsHistory.map(m => m.throughput || 0), 0)
        },
        history: metricsHistory
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${runName || 'triune-run'}_wandb.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      showToast('Exported WandB experiment JSON');
    };

    const handleCopyLatexTable = () => {
      const losses = metricsHistory.map(m => Number(m.loss)).filter(l => !isNaN(l));
      const minL = losses.length ? Math.min(...losses).toFixed(4) : '--';
      const finL = losses.length ? losses[losses.length - 1].toFixed(4) : '--';
      const maxTok = Math.max(...metricsHistory.map(m => m.throughput || 0), 0);
      const latex = `\\begin{table}[h]\n\\centering\n\\caption{Triune MoE Training Run Evaluation: ${runName}}\n\\begin{tabular}{lcccc}\n\\hline\n\\textbf{Model} & \\textbf{Steps} & \\textbf{Min Loss} & \\textbf{Final Loss} & \\textbf{Peak Throughput} \\\\\n\\hline\n${activeModel} & ${metrics.step || 0} & ${minL} & ${finL} & ${maxTok.toLocaleString()} tok/s \\\\\n\\hline\n\\end{tabular}\n\\end{table}`;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(latex);
        showToast('Copied LaTeX table to clipboard');
      }
    };

    const handleLoadDAGPreset = (presetKey) => {
      const preset = DAG_PRESETS[presetKey];
      if (preset) {
        setActivePresetKey(presetKey);
        setShowDAGPresetLibrary(false);
        setWorkflowName(preset.name);
        setNodes(preset.nodes.map(node => ({ ...node })));
        setEdges(preset.edges.map(edge => ({ ...edge })));
        setFrames(Array.isArray(preset.frames) ? preset.frames.map(frame => ({ ...frame })) : []);
        setNodeExecOutputs({});
        setSelectedNodeId(null);
        setSelectedNodeIds([]);
        setTimeout(() => handleFitCanvasView(preset.nodes), 0);
        showToast(`Loaded ${preset.name} onto the canvas. Use “Apply to Live Engine” when you want to change training settings.`);
      }
    };

    const handleSelectModelPreset = async (presetId) => {
      const preset = MODEL_PRESETS.find(p => p.id === presetId) || MODEL_PRESETS[0];
      const newBatchSize = preset.recommended_batch || (preset.layers > 8 ? 2 : 4);
      const newSeqLen = preset.recommended_seq || (preset.layers > 8 ? 64 : 128);
      const currentDepthMode = trainDepthMode || preset.depth_mode || 'dynamic';
      try {
        const res = await apiFetch('/v1/models/preset', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            preset_id: preset.id,
            num_layers: preset.layers,
            hidden_dim: preset.hidden_dim,
            num_heads: preset.heads,
            head_dim: preset.head_dim,
            num_experts: preset.experts_routed,
            batch_size: newBatchSize,
            seq_len: newSeqLen,
            depth_mode: currentDepthMode,
            precision: trainPrecision
          })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data.status !== 'success') {
          throw new Error(data.message || `Model configuration failed (${res.status})`);
        }

        lastManualConfigEditRef.current = Date.now();
        setActiveModel(data.active_model || preset.id);
        setWeightsLoaded(false);
        setActiveCheckpointName(null);
        setTrainBatchSize(newBatchSize);
        setTrainSeqLen(newSeqLen);
        setDatasetInfo(prev => ({
          ...prev,
          num_layers: preset.layers,
          hidden_dim: preset.hidden_dim,
          num_heads: preset.heads,
          num_experts: preset.experts_routed,
          batch_size: newBatchSize,
          seq_len: newSeqLen,
          depth_mode: currentDepthMode
        }));

        // Update the graph only after the engine accepts the same configuration.
        syncStateToDAGNodes({
          preset_id: preset.id,
          num_layers: preset.layers,
          hidden_dim: preset.hidden_dim,
          num_heads: preset.heads,
          head_dim: preset.head_dim,
          num_experts: preset.experts_routed,
          batch_size: newBatchSize,
          seq_len: newSeqLen,
          depth_mode: currentDepthMode,
          precision: trainPrecision
        });
        if (data && data.profile) {
          setHardwareProfile(data.profile);
          if (data.profile.gpu_offload_pct !== undefined) setGpuOffloadPct(data.profile.gpu_offload_pct);
          if (data.profile.gpu_layers !== undefined) setGpuLayers(data.profile.gpu_layers);
        }
        if (data && data.architecture) {
          setDatasetInfo(prev => ({
            ...prev,
            num_layers: data.architecture.num_layers || preset.layers,
            hidden_dim: data.architecture.hidden_dim || preset.hidden_dim,
            num_heads: data.architecture.num_heads || preset.heads,
            num_experts: data.architecture.num_experts || preset.experts_routed
          }));
        }
        showToast(data.message || `Loaded ${preset.name} and synchronized training settings and graph.`);
      } catch (err) {
        showToast(`Model selection failed: ${err.message || 'could not reach the backend'}`);
      }
    };


    // Toggle Real PyTorch Hardware Training Loop On/Off
    const handleToggleTraining = async () => {
      const targetState = !isTraining;
      const endpoint = targetState ? '/v1/training/start' : '/v1/training/pause';
      try {
        const res = await apiFetch(endpoint, { method: 'POST' });
        if (res.ok) {
          setIsTraining(targetState);
          showToast(targetState ? 'PyTorch hardware training loop started!' : 'Paused PyTorch Training Loop!');
        } else {
          showToast(`API Error: ${res.status} – training ${targetState ? 'start' : 'pause'} failed`);
        }
      } catch (e) {
        showToast('Network error – could not reach backend engine');
      }
    };

    const handleStepTrainingOnce = async () => {
      try {
        const res = await apiFetch('/v1/training/step', { method: 'POST' });
        const data = await res.json();
        if (data && data.step !== undefined) {
          setMetrics(prev => ({ ...prev, ...data }));
          setMetricsHistory(prev => {
            const next = [...prev, data];
            return next.length > 200 ? next.slice(next.length - 200) : next;
          });
          if (data.exit_usage) setExitUsage(data.exit_usage);
          if (data.batch_preview) setActiveBatchPreview(data.batch_preview);
          if (data.tokens_trained !== undefined) {
            setDatasetInfo(prev => ({ ...prev, total_tokens: data.tokens_trained }));
          }
          showToast(`Step ${data.step} completed (Loss: ${Number(data.loss || 0).toFixed(4)})`);
        }
      } catch (e) {
        showToast('Could not execute single training step');
      }
    };

    // Hardware Tuner, Auto-Offload & Dynamic Allotment Handlers (LM Studio / Ollama Style)
    const fetchHardwareProfile = async () => {
      try {
        const res = await apiFetch('/v1/hardware/profile');
        const data = await res.json();
        if (data && data.device_name) {
          setHardwareProfile(data);
          if (data.current_config) {
            setTrainBatchSize(data.current_config.batch_size);
            setTrainSeqLen(data.current_config.seq_len);
            setTrainGradAccum(data.current_config.grad_accum_steps);
            if (data.current_config.depth_mode) setTrainDepthMode(data.current_config.depth_mode);
            if (data.current_config.precision) {
              setTrainPrecision(data.current_config.precision);
              setPrecision(data.current_config.precision);
            }
            if (data.current_config.lr !== undefined) setTrainLR(String(data.current_config.lr));
            if (data.current_config.warmup_steps !== undefined) setTrainWarmup(data.current_config.warmup_steps);
            setGpuOffloadPct(data.gpu_offload_pct !== undefined ? data.gpu_offload_pct : 100);
            setGpuLayers(data.gpu_layers !== undefined ? data.gpu_layers : (data.num_layers || 6));
          }
          if (data.recommended_preset) {
            setSelectedHwPreset(data.recommended_preset);
          }
        }
      } catch (err) {}
    };

    const handleApplyHardwareTune = async (overrides = {}) => {
      setIsUpdatingTrainConfig(true);
      try {
        const pKey = overrides.preset !== undefined ? overrides.preset : selectedHwPreset;
        const bSize = overrides.batch_size !== undefined ? overrides.batch_size : (parseInt(trainBatchSize) || 4);
        const sLen = overrides.seq_len !== undefined ? overrides.seq_len : (parseInt(trainSeqLen) || 64);
        const gAccum = overrides.grad_accum_steps !== undefined ? overrides.grad_accum_steps : (parseInt(trainGradAccum) || 4);
        const dMode = overrides.depth_mode !== undefined ? overrides.depth_mode : trainDepthMode;
        const prec = overrides.precision !== undefined ? overrides.precision : trainPrecision;
        const offPct = overrides.gpu_offload_pct !== undefined ? overrides.gpu_offload_pct : gpuOffloadPct;
        const offLayers = overrides.gpu_layers !== undefined ? overrides.gpu_layers : gpuLayers;
        const lrVal = overrides.lr !== undefined ? overrides.lr : (parseFloat(trainLR) || 0.0005);
        const warmupVal = overrides.warmup_steps !== undefined ? overrides.warmup_steps : (parseInt(trainWarmup) || 0);

        const res = await apiFetch('/v1/hardware/tune', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            preset: pKey,
            batch_size: bSize,
            seq_len: sLen,
            grad_accum_steps: gAccum,
            depth_mode: dMode,
            precision: prec,
            gpu_offload_pct: offPct,
            gpu_layers: offLayers,
            lr: lrVal,
            warmup_steps: warmupVal
          })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(data.message || 'Hardware tuning applied');
          if (data.profile) {
            setHardwareProfile(data.profile);
            const cc = data.profile.current_config;
            setTrainBatchSize(cc.batch_size);
            setTrainSeqLen(cc.seq_len);
            setTrainGradAccum(cc.grad_accum_steps);
            setTrainDepthMode(cc.depth_mode);
            if (cc.precision) {
              setTrainPrecision(cc.precision);
              setPrecision(cc.precision);
            }
            if (cc.lr !== undefined) setTrainLR(String(cc.lr));
            if (cc.warmup_steps !== undefined) setTrainWarmup(cc.warmup_steps);
            setGpuOffloadPct(data.profile.gpu_offload_pct);
            setGpuLayers(data.profile.gpu_layers);
            setDatasetInfo(prev => ({
              ...prev,
              batch_size: cc.batch_size,
              seq_len: cc.seq_len,
              grad_accum_steps: cc.grad_accum_steps,
              depth_mode: cc.depth_mode
            }));
            syncStateToDAGNodes({
              batch_size: cc.batch_size,
              seq_len: cc.seq_len,
              grad_accum_steps: cc.grad_accum_steps,
              depth_mode: cc.depth_mode,
              precision: cc.precision || prec,
              lr: cc.lr !== undefined ? cc.lr : lrVal
            });
          }
        } else {
          showToast(`Notice: ${data.detail || data.message || `Config update failed (${res.status})`}`);
        }
      } catch (err) {
        showToast('Network error: Could not reach hardware tune endpoint');
      }
      setIsUpdatingTrainConfig(false);
    };

    const handleAutoTuneHardware = async () => {
      setIsAutoTuning(true);
      try {
        const res = await apiFetch('/v1/hardware/autotune', { method: 'POST' });
        const responseText = await res.text();
        let data;
        try {
          data = responseText ? JSON.parse(responseText) : {};
        } catch (_) {
          throw new Error(`Auto-tune endpoint returned a non-JSON response (HTTP ${res.status}). The backend may be outdated; restart Studio and retry.`);
        }
        if (!res.ok) {
          throw new Error(data.detail || data.message || `Auto-tune request failed (HTTP ${res.status}).`);
        }
        if (data.status === 'success') {
          showToast(data.gpu_probe?.status === 'unavailable'
            ? 'CUDA was not detected by the backend; CPU-safe training settings were applied.'
            : (data.message || 'GPU probe passed; recommended training profile applied.'));
          if (data.profile) {
            setHardwareProfile(data.profile);
            setSelectedHwPreset(data.recommended_preset || 'turbo');
            const cc = data.profile.current_config;
            setTrainBatchSize(cc.batch_size);
            setTrainSeqLen(cc.seq_len);
            setTrainGradAccum(cc.grad_accum_steps);
            setTrainDepthMode(cc.depth_mode);
            if (cc.precision) {
              setTrainPrecision(cc.precision);
              setPrecision(cc.precision);
            }
            if (cc.lr !== undefined) setTrainLR(String(cc.lr));
            if (cc.warmup_steps !== undefined) setTrainWarmup(cc.warmup_steps);
            setGpuOffloadPct(data.profile.gpu_offload_pct);
            setGpuLayers(data.profile.gpu_layers);
            setDatasetInfo(prev => ({
              ...prev,
              batch_size: cc.batch_size,
              seq_len: cc.seq_len,
              grad_accum_steps: cc.grad_accum_steps,
              depth_mode: cc.depth_mode
            }));
            syncStateToDAGNodes({
              batch_size: cc.batch_size,
              seq_len: cc.seq_len,
              grad_accum_steps: cc.grad_accum_steps,
              depth_mode: cc.depth_mode,
              precision: cc.precision || trainPrecision
            });
          }
        } else {
          showToast(`Auto-tune did not apply: ${data.detail || data.message || 'The backend returned no successful result.'}`);
        }
      } catch (err) {
        showToast(`GPU auto-tune failed: ${err.message || String(err)}`);
      } finally {
        setIsAutoTuning(false);
      }
    };

    const handleSelectPreset = (presetKey) => {
      setSelectedHwPreset(presetKey);
      if (hardwareProfile && hardwareProfile.presets && hardwareProfile.presets[presetKey]) {
        const p = hardwareProfile.presets[presetKey];
        setTrainBatchSize(p.batch_size);
        setTrainSeqLen(p.seq_len);
        setTrainGradAccum(p.grad_accum_steps);
        setTrainDepthMode(p.depth_mode);
        if (p.precision) {
          setTrainPrecision(p.precision);
          setPrecision(p.precision);
        }
        setGpuOffloadPct(p.gpu_offload_pct);
        setGpuLayers(p.gpu_layers);
        handleApplyHardwareTune({
          preset: presetKey,
          batch_size: p.batch_size,
          seq_len: p.seq_len,
          grad_accum_steps: p.grad_accum_steps,
          depth_mode: p.depth_mode,
          precision: p.precision,
          gpu_offload_pct: p.gpu_offload_pct,
          gpu_layers: p.gpu_layers
        });
      }
    };

    // Render LM Studio / Ollama Style Hardware Acceleration & GPU Offload Deck
    const renderHardwareTunerDeck = () => {
      const numLayers = (hardwareProfile && hardwareProfile.num_layers) ? hardwareProfile.num_layers : (datasetInfo.num_layers || 6);
      const hiddenDim = (hardwareProfile && hardwareProfile.hidden_dim) ? hardwareProfile.hidden_dim : (datasetInfo.hidden_dim || 256);
      const deviceTitle = (hardwareProfile && hardwareProfile.device_name) ? hardwareProfile.device_name : systemDiagnostics.device_name;
      const totalVram = Number((hardwareProfile && hardwareProfile.total_vram_gb) || vramUsage.total || 0);
      const effTokens = (parseInt(trainBatchSize) || 4) * (parseInt(trainSeqLen) || 64) * (parseInt(trainGradAccum) || 4);

      // Memory breakdown estimates from backend profile or dynamic fallback
      const mb = (hardwareProfile && hardwareProfile.current_config && hardwareProfile.current_config.memory_breakdown) || {};
      const weightsMb = mb.weights_mb !== undefined ? mb.weights_mb : 84.0;
      const optMb = mb.optimizer_mb !== undefined ? mb.optimizer_mb : 336.0;
      const actMb = mb.activations_mb !== undefined
        ? mb.activations_mb
        : Math.round(((parseInt(trainBatchSize) || 4) * (parseInt(trainSeqLen) || 64) * hiddenDim * numLayers * 20) / (1024 * 1024) * 10) / 10;
      const cudaMb = mb.cuda_context_mb !== undefined ? mb.cuda_context_mb : ((hardwareProfile && hardwareProfile.is_cuda) ? 600.0 : 50.0);
      const totalEstMb = mb.total_estimated_mb !== undefined ? mb.total_estimated_mb : (weightsMb + optMb + actMb + cudaMb);
      const totalEstGb = mb.total_estimated_gb !== undefined ? Number(mb.total_estimated_gb).toFixed(2) : (totalEstMb / 1024.0).toFixed(2);
      const speedEstimate = Number(hardwareProfile?.presets?.[selectedHwPreset]?.estimated_tok_s);

      const totalCapMb = totalVram > 0 ? totalVram * 1024.0 : 0;
      const pctWeights = totalCapMb ? Math.min(100, (weightsMb / totalCapMb) * 100) : 0;
      const pctOpt = totalCapMb ? Math.min(100, (optMb / totalCapMb) * 100) : 0;
      const pctAct = totalCapMb ? Math.min(100, (actMb / totalCapMb) * 100) : 0;
      const pctCuda = totalCapMb ? Math.min(100, (cudaMb / totalCapMb) * 100) : 0;
      const pctUsed = totalCapMb ? Math.min(100, (totalEstMb / totalCapMb) * 100) : 0;

      return e('div', { className: 'hardware-tuner-card' },
        e('div', { className: 'hardware-tuner-header' },
          e('div', { style: { display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' } },
            e('span', { style: { fontFamily: 'Newsreader', fontSize: '18px', fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center' } },
              renderIcon('hardware', 16, { marginRight: '6px' }),
              'Hardware Acceleration & GPU Offload (Ollama / LM Studio Tuner)'
            ),
            e('span', { className: 'hw-device-badge' },
              `● ${deviceTitle} (${totalVram > 0 ? `${totalVram} GB VRAM` : 'VRAM capacity unavailable'})`
            )
          ),
          e('button', {
            className: 'btn-autotune',
            disabled: isAutoTuning || isUpdatingTrainConfig,
            onClick: handleAutoTuneHardware,
            title: 'Choose the largest conservative training preset that fits available GPU memory and apply its settings'
          },
            e('span', { style: { display: 'inline-flex', alignItems: 'center' } },
              isAutoTuning
                ? [renderIcon('refresh', 12, { animation: 'spin 1.5s linear infinite', marginRight: '5px' }), 'Probing GPU...']
                : [renderIcon('bolt', 12, { marginRight: '5px' }), 'Auto-Optimize Training']
            )
          )
        ),
        e('div', { className: 'hardware-preset-chips' },
          e('span', { style: { fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', marginRight: '4px' } }, 'Tuning Presets:'),
          e('button', {
            className: `hw-preset-chip ${selectedHwPreset === 'turbo' ? 'active turbo' : ''}`,
            onClick: () => handleSelectPreset('turbo')
          }, [renderIcon('rocket', 13, { marginRight: '5px' }), 'Turbo (Max Throughput)']),
          e('button', {
            className: `hw-preset-chip ${selectedHwPreset === 'balanced' ? 'active' : ''}`,
            onClick: () => handleSelectPreset('balanced')
          }, [renderIcon('models', 13, { marginRight: '5px' }), 'Balanced (Recommended)']),
          e('button', {
            className: `hw-preset-chip ${selectedHwPreset === 'eco' ? 'active' : ''}`,
            onClick: () => handleSelectPreset('eco')
          }, [renderIcon('sparkle', 13, { marginRight: '5px' }), 'Eco (<1GB VRAM)']),
          e('button', {
            className: `hw-preset-chip ${selectedHwPreset === 'streaming' ? 'active streaming' : ''}`,
            onClick: () => handleSelectPreset('streaming')
          }, [renderIcon('stream', 13, { marginRight: '5px' }), 'AirLLM Streaming (<600MB)']),
          selectedHwPreset === 'custom' && e('span', {
            className: 'hw-preset-chip active',
            style: { background: 'var(--border-dark)', borderColor: 'var(--border-dark)', color: 'var(--text-main)', display: 'inline-flex', alignItems: 'center' }
          }, [renderIcon('nodes', 13, { marginRight: '5px' }), 'Custom Tuning'])
        ),
        e('div', { className: 'hw-sliders-grid' },
          // 1. GPU Offload Layers
          e('div', { className: 'hw-slider-box' },
            e('div', { className: 'hw-slider-header' },
              e('span', { className: 'hw-slider-label' }, 'GPU Layer Offload'),
              e('span', { className: 'hw-slider-val' }, `${gpuLayers} / ${numLayers} layers (${gpuOffloadPct}%)`)
            ),
            e('input', {
              type: 'range',
              className: 'hw-slider-input',
              min: 0,
              max: numLayers,
              step: 1,
              value: gpuLayers,
              onChange: ev => {
                const l = parseInt(ev.target.value);
                const pct = Math.round((l / numLayers) * 100);
                setGpuLayers(l);
                setGpuOffloadPct(pct);
                setSelectedHwPreset('custom');
              }
            }),
            e('div', { className: 'hw-slider-sub' },
              gpuLayers === numLayers
                ? '100% VRAM Native: All layers in VRAM (Zero PCIe copies, max tok/s)'
                : (gpuLayers === 0 ? 'CPU Host Execution' : `Hybrid: ${gpuLayers} in VRAM, ${numLayers - gpuLayers} in RAM`)
            )
          ),
          // 2. Context Window / Sequence Length
          e('div', { className: 'hw-slider-box' },
            e('div', { className: 'hw-slider-header' },
              e('span', { className: 'hw-slider-label' }, 'Context / Sequence Length'),
              e('span', { className: 'hw-slider-val' }, `${trainSeqLen} tokens`)
            ),
            e('input', {
              type: 'range',
              className: 'hw-slider-input',
              min: 64,
              max: 512,
              step: 64,
              value: trainSeqLen,
              onChange: ev => {
                setTrainSeqLen(parseInt(ev.target.value));
                setSelectedHwPreset('custom');
              }
            }),
            e('div', { className: 'hw-slider-sub' },
              trainSeqLen >= 256
                ? 'High Context: Saturates Tensor Cores with dense matrix kernels'
                : 'Fast Micro-Context: Quick single-turn steps'
            )
          ),
          // 3. Micro-Batch Size
          e('div', { className: 'hw-slider-box' },
            e('div', { className: 'hw-slider-header' },
              e('span', { className: 'hw-slider-label' }, 'Micro-Batch Size'),
              e('span', { className: 'hw-slider-val' }, `${trainBatchSize} seqs`)
            ),
            e('input', {
              type: 'range',
              className: 'hw-slider-input',
              min: 1,
              max: 16,
              step: 1,
              value: trainBatchSize,
              onChange: ev => {
                setTrainBatchSize(parseInt(ev.target.value));
                setSelectedHwPreset('custom');
              }
            }),
            e('div', { className: 'hw-slider-sub' },
              trainBatchSize >= 8
                ? 'High Parallel Batch: Squeezes maximum compute from Tensor Cores'
                : 'Lightweight Batch: Low activation memory footprint'
            )
          ),
          // 4. Gradient Accumulation
          e('div', { className: 'hw-slider-box' },
            e('div', { className: 'hw-slider-header' },
              e('span', { className: 'hw-slider-label' }, 'Gradient Accumulation'),
              e('span', { className: 'hw-slider-val' }, `${trainGradAccum}x accum`)
            ),
            e('input', {
              type: 'range',
              className: 'hw-slider-input',
              min: 1,
              max: 16,
              step: 1,
              value: trainGradAccum,
              onChange: ev => {
                setTrainGradAccum(parseInt(ev.target.value));
                setSelectedHwPreset('custom');
              }
            }),
            e('div', { className: 'hw-slider-sub' },
              `Effective Batch: ${effTokens.toLocaleString()} tokens per optimizer step`
            )
          )
        ),
        // Supervision Mode, Precision, LR, and Warmup
        e('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px', marginBottom: '14px' } },
          e('div', { className: 'hyperparam-col' },
            e('label', null, 'Depth Mode (Supervision):'),
            e('select', {
              value: trainDepthMode,
              onChange: ev => {
                const newMode = ev.target.value;
                lastManualConfigEditRef.current = Date.now();
                setTrainDepthMode(newMode);
                setSelectedHwPreset('custom');
                handleApplyHardwareTune({ preset: 'custom', depth_mode: newMode });
              }
            },
              e('option', { value: 'cortex' }, 'Full Cortex (100% Backbone Grad - Fastest)'),
              e('option', { value: 'joint' }, 'Joint Multi-Exit (Simultaneous 3 Exits)'),
              e('option', { value: 'dynamic' }, 'Dynamic Router (Adaptive Depth)')
            )
          ),
          e('div', { className: 'hyperparam-col' },
            e('label', null, 'Precision & Tensor Cores:'),
            e('select', {
              value: trainPrecision,
              onChange: ev => {
                const newPrec = ev.target.value;
                lastManualConfigEditRef.current = Date.now();
                setTrainPrecision(newPrec);
                setPrecision(newPrec);
                setSelectedHwPreset('custom');
                handleApplyHardwareTune({ preset: 'custom', precision: newPrec });
              }
            },
              e('option', { value: 'bf16' }, 'BF16 (Native Tensor Cores)'),
              e('option', { value: 'fp8' }, 'FP8 (Scaled GEMM - Half VRAM)'),
              e('option', { value: 'fp4' }, 'FP4 (NVFP4 / NF4 Quantized)'),
              e('option', { value: 'fp32' }, 'FP32 (Standard Float32)')
            )
          ),
          e('div', { className: 'hyperparam-col' },
            e('label', null, 'Peak Learning Rate:'),
            e('select', {
              value: trainLR,
              onChange: ev => {
                const newLR = ev.target.value;
                lastManualConfigEditRef.current = Date.now();
                setTrainLR(newLR);
                setSelectedHwPreset('custom');
                handleApplyHardwareTune({ preset: 'custom', lr: parseFloat(newLR) });
              }
            },
              e('option', { value: '0.0001' }, '1e-4 (Gentle)'),
              e('option', { value: '0.0003' }, '3e-4 (Moderate)'),
              e('option', { value: '0.0005' }, '5e-4 (Recommended)'),
              e('option', { value: '0.0008' }, '8e-4 (Fast Converge)'),
              e('option', { value: '0.001' }, '1e-3 (Aggressive)')
            )
          ),
          e('div', { className: 'hyperparam-col' },
            e('label', null, 'Warmup Steps:'),
            e('input', {
              type: 'number',
              min: 0,
              max: 5000,
              step: 50,
              value: trainWarmup,
              onChange: ev => {
                const w = parseInt(ev.target.value) || 0;
                lastManualConfigEditRef.current = Date.now();
                setTrainWarmup(w);
                setSelectedHwPreset('custom');
              }
            })
          )
        ),
        // Footer: Effective tokens, estimated allocation, & apply button
        e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' } },
          e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' } },
            e('span', { className: 'badge-eff-tokens', style: { display: 'inline-flex', alignItems: 'center' } },
              renderIcon('bolt', 11, { marginRight: '4px' }),
              `Effective: ${effTokens.toLocaleString()} tokens/step`
            ),
            e('span', { className: 'badge-eff-tokens', style: { background: 'rgba(154, 52, 18, 0.08)', color: 'var(--accent-terracotta)', borderColor: 'rgba(154, 52, 18, 0.25)' } },
              speedEstimate > 0
                ? `Recent training rate: ~${Math.round(speedEstimate).toLocaleString()} tok/s`
                : 'Throughput: not benchmarked'
            ),
            e('span', { className: 'badge-eff-tokens', style: { background: 'rgba(43, 76, 63, 0.08)', color: 'var(--accent-olive, #2b4c3f)', borderColor: 'rgba(43, 76, 63, 0.25)' } },
              totalVram > 0
                ? `Est. memory: ${totalEstGb} GB / ${totalVram} GB`
                : `Est. memory: ${totalEstGb} GB · device capacity unknown`
            )
          ),
          e('button', {
            className: 'btn-action-apply',
            disabled: isUpdatingTrainConfig,
            onClick: () => handleApplyHardwareTune(),
            style: { display: 'inline-flex', alignItems: 'center', gap: '6px' }
          }, isUpdatingTrainConfig ? 'Applying...' : [renderIcon('bolt', 12), 'Apply Hardware Settings'])
        )
      );
    };

    // Node Canvas Dragging, Interactive Wire Connection & Node Inspector (Unreal/Blender Style)
    const NODE_WIDTH = 290;
    const NODE_PORT_CENTER_Y = 19;

    const handleToggleMuteNode = (nodeId) => {
      const targetId = nodeId || selectedNodeId;
      if (!targetId) return;
      setNodes(prev => prev.map(n => {
        if (n.id === targetId) {
          const nextMuted = !n.muted;
          showToast(`${nextMuted ? 'Muted' : 'Unmuted'} [${n.title}]`);
          return { ...n, muted: nextMuted };
        }
        return n;
      }));
    };

    const handleCreateVisualFrame = (titleText) => {
      const selNode = nodes.find(n => n.id === selectedNodeId);
      const fx = selNode ? (selNode.x - 20) : ((canvasMousePos && canvasMousePos.x) || 100);
      const fy = selNode ? (selNode.y - 36) : ((canvasMousePos && canvasMousePos.y) || 100);
      const fw = selNode ? (NODE_WIDTH + 40) : 340;
      const fh = selNode ? 340 : 260;
      const defaultTitle = selNode ? `Frame: ${selNode.title}` : 'Visual Node Frame';
      // Use a predictable default instead of a blocking browser prompt (which
      // is unavailable in some embedded WebView configurations).
      const frameTitle = titleText || defaultTitle;
      const frameColors = ['#63c7b2', '#7a88cf', '#e8cb4f', '#a1e976', '#ff4d4d'];
      const chosenColor = frameColors[frames.length % frameColors.length];
      const newFrame = {
        id: `frame_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
        title: frameTitle,
        x: Math.round(fx),
        y: Math.round(fy),
        width: Math.round(fw),
        height: Math.round(fh),
        color: chosenColor + '18',
        borderColor: chosenColor
      };
      setFrames(prev => [...prev, newFrame]);
      showToast(`Backdrop added: "${frameTitle}"`);
    };

    const handleCompileDAGToTraining = async () => {
      setIsCompilingTraining(true);
      showToast('Compiling DAG into live PyTorch training pipeline...');
      const activeNodes = nodes.filter(n => !n.muted);
      const activeEdges = edges.filter(e => {
        const s = nodes.find(n => n.id === e.source);
        const t = nodes.find(n => n.id === e.target);
        return s && !s.muted && t && !t.muted;
      });

      try {
        const res = await apiFetch('/v1/dag/compile_to_training', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ nodes: activeNodes, edges: activeEdges })
        });
        if (res.ok) {
          const data = await res.json();
          if (data.status !== 'success') {
            throw new Error(data.message || data.error || 'Backend rejected this graph.');
          }
          showToast(data.message || `Applied ${activeNodes.length} active graph nodes to the training engine.`);
          if (data.active_model) {
            setActiveModel(data.active_model);
          }
          if (data.profile) {
            setHardwareProfile(data.profile);
            if (data.profile.gpu_offload_pct !== undefined) setGpuOffloadPct(data.profile.gpu_offload_pct);
            if (data.profile.gpu_layers !== undefined) setGpuLayers(data.profile.gpu_layers);
          }
          if (data.applied_config) {
            const ac = data.applied_config;
            if (ac.batch_size) setTrainBatchSize(ac.batch_size);
            if (ac.seq_len) setTrainSeqLen(ac.seq_len);
            if (ac.grad_accum_steps) setTrainGradAccum(ac.grad_accum_steps);
            if (ac.depth_mode) setTrainDepthMode(ac.depth_mode);
            if (ac.precision) {
              setTrainPrecision(ac.precision);
              setPrecision(ac.precision);
            }
            if (ac.lr) setTrainLR(String(ac.lr));
            if (ac.dataset) setActiveDataset(ac.dataset);
            if (ac.active_adapter !== undefined) setActiveAdapter(ac.active_adapter);
            setDatasetInfo(prev => ({
              ...prev,
              name: ac.dataset || prev.name,
              batch_size: ac.batch_size || prev.batch_size,
              seq_len: ac.seq_len || prev.seq_len,
              grad_accum_steps: ac.grad_accum_steps || prev.grad_accum_steps,
              depth_mode: ac.depth_mode || prev.depth_mode,
              num_layers: (ac.model_architecture && ac.model_architecture.num_layers) || prev.num_layers,
              hidden_dim: (ac.model_architecture && ac.model_architecture.hidden_dim) || prev.hidden_dim,
              num_heads: (ac.model_architecture && ac.model_architecture.num_heads) || prev.num_heads,
              num_experts: (ac.model_architecture && ac.model_architecture.num_experts) || prev.num_experts
            }));
          }
          if (data.metrics) {
            setMetrics(prev => ({ ...prev, ...data.metrics }));
          }
        } else {
          throw new Error(`Backend returned HTTP ${res.status}.`);
        }
      } catch (err) {
        showToast(`Graph remains on the canvas, but was not applied to training: ${err.message || 'backend request failed'}`);
      } finally {
        setIsCompilingTraining(false);
      }
    };

    const getNodeTensorBadge = (node, execOutput) => {
      if (execOutput && execOutput.shape) {
        return `${execOutput.shape} ${execOutput.dtype || 'fp32'}`;
      }
      if (execOutput && execOutput.tokens_loaded) {
        return `${execOutput.tokens_loaded} tok`;
      }
      if (execOutput && execOutput.loss !== undefined) {
        return `Loss ${Number(execOutput.loss).toFixed(3)}`;
      }
      const b = trainBatchSize || 4;
      const s = trainSeqLen || 64;
      const d = datasetInfo.hidden_dim || 256;
      const p = trainPrecision || 'bf16';
      const lower = (node.title + ' ' + (node.type || '')).toLowerCase();
      if (lower.includes('streamer') || lower.includes('reader')) return 'Stream [txt]';
      if (lower.includes('tokenizer')) return `[${b}, ${s}] int64`;
      if (lower.includes('dataloader')) return `[${b}, ${s}] int64`;
      if (lower.includes('moe') || lower.includes('transformer')) return `[${b}, ${s}, ${d}] ${p}`;
      if (lower.includes('gla') || lower.includes('attention')) return `[${b}, ${s}, ${d}] ${p}`;
      if (lower.includes('linear')) return `[${b}, ${s}, ${d}] fp8`;
      if (lower.includes('router')) return `[${b}, 3] exit_dist`;
      if (lower.includes('loss')) return 'Loss scalar';
      if (lower.includes('optimizer')) return 'OptState';
      if (lower.includes('export')) return 'Safetensors';
      if (lower.includes('reroute')) return 'Reroute';
      return '[Tensor]';
    };

    const handleUpdateNodeParam = (nodeId, key, value) => {
      setNodes(prev => prev.map(n => {
        if (n.id !== nodeId) return n;
        const curDetails = n.details || '';
        const lines = curDetails.split('\n');
        let found = false;
        const newLines = lines.map(line => {
          const t = line.trim();
          if (t && !t.startsWith('#')) {
            const eq = t.indexOf('=');
            if (eq !== -1 && t.slice(0, eq).trim() === key) {
              found = true;
              return `${key}=${value}`;
            }
          }
          return line;
        });
        if (!found) {
          newLines.push(`${key}=${value}`);
        }
        const updatedDetails = newLines.join('\n');
        return { ...n, details: updatedDetails };
      }));

      if (selectedNodeId === nodeId) {
        setSelectedNodeParams(prev => ({ ...prev, [key]: String(value) }));
      }

      // Immediately sync common node hyperparameters to Training / Dataset / Hardware UI state
      const lk = String(key).toLowerCase();
      if (lk === 'batch_size') {
        const b = parseInt(value) || 4;
        setTrainBatchSize(b);
        setDatasetInfo(prev => ({ ...prev, batch_size: b }));
      } else if (lk === 'seq_len') {
        const s = parseInt(value) || 64;
        setTrainSeqLen(s);
        setDatasetInfo(prev => ({ ...prev, seq_len: s }));
      } else if (lk === 'grad_accum_steps') {
        const g = parseInt(value) || 4;
        setTrainGradAccum(g);
        setDatasetInfo(prev => ({ ...prev, grad_accum_steps: g }));
      } else if (lk === 'depth_mode') {
        const dm = String(value) === 'static' ? 'cortex' : String(value);
        setTrainDepthMode(dm);
        setDatasetInfo(prev => ({ ...prev, depth_mode: dm }));
      } else if (lk === 'lr') {
        setTrainLR(String(value));
      } else if (lk === 'dataset_name') {
        setActiveDataset(String(value));
        setDatasetInfo(prev => ({ ...prev, name: String(value) }));
      } else if (lk === 'num_layers') {
        setDatasetInfo(prev => ({ ...prev, num_layers: parseInt(value) || prev.num_layers }));
      } else if (lk === 'hidden_dim') {
        setDatasetInfo(prev => ({ ...prev, hidden_dim: parseInt(value) || prev.hidden_dim }));
      }
    };

    const renderNodeInlineControls = (node) => {
      const detailsStr = node.details || '';
      const lines = detailsStr.split('\n');
      const paramEntries = [];
      lines.forEach(line => {
        const t = line.trim();
        if (!t || t.startsWith('#')) return;
        const eq = t.indexOf('=');
        if (eq !== -1) {
          paramEntries.push([t.slice(0, eq).trim(), t.slice(eq + 1).trim()]);
        }
      });

      if (paramEntries.length === 0) {
        return e('div', { style: { fontSize: '11px', color: 'var(--text-dim)', fontStyle: 'italic', padding: '2px 0' } }, 'No inline controls');
      }

      const visibleEntries = paramEntries.slice(0, 3);
      const remainingCount = paramEntries.length - visibleEntries.length;

      return e('div', {
        className: 'node-inline-controls',
        style: {
          display: 'flex',
          flexDirection: 'column',
          gap: '5px',
          maxHeight: '135px',
          overflowY: 'hidden',
          paddingRight: '2px',
          pointerEvents: 'auto'
        },
        onMouseDown: (ev) => ev.stopPropagation(),
        onClick: (ev) => ev.stopPropagation()
      },
        visibleEntries.map(([key, val]) => {
          const lowerKey = key.toLowerCase();
          const cleanLabel = key.replace(/_/g, ' ');

          // 1. Numeric Steppers
          const isIntStepper = [
            'num_layers', 'hidden_dim', 'heads', 'num_heads', 'head_dim',
            'batch_size', 'seq_len', 'experts_routed', 'experts_shared', 'num_experts',
            'grad_accum_steps', 'buffer_size', 'reflex_exit_layer', 'limbic_exit_layer',
            'cortex_exit_layer', 'layer', 'rank', 'epochs', 'top_k'
          ].includes(lowerKey);

          if (isIntStepper) {
            const numVal = parseInt(val) || 0;
            const stepDelta = ['hidden_dim', 'seq_len'].includes(lowerKey) ? 64 : (['head_dim'].includes(lowerKey) ? 16 : 1);
            return e('div', {
              key: key,
              style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }
            },
              e('span', { style: { color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '110px' }, title: key }, cleanLabel),
              e('div', { style: { display: 'flex', alignItems: 'center', gap: '2px' } },
                e('button', {
                  type: 'button',
                  style: {
                    width: '18px',
                    height: '18px',
                    padding: 0,
                    lineHeight: '16px',
                    textAlign: 'center',
                    background: 'var(--bg-surface, #ede9df)',
                    border: '1px solid var(--border-color, #ded9cd)',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '11px',
                    fontWeight: 700
                  },
                  onMouseDown: (ev) => ev.stopPropagation(),
                  onClick: (ev) => {
                    ev.stopPropagation();
                    const nextVal = Math.max(1, numVal - stepDelta);
                    handleUpdateNodeParam(node.id, key, nextVal);
                  }
                }, '-'),
                e('input', {
                  type: 'number',
                  value: numVal,
                  style: {
                    width: '48px',
                    height: '18px',
                    textAlign: 'center',
                    background: 'var(--bg-input, #f2eee5)',
                    border: '1px solid var(--border-color, #ded9cd)',
                    borderRadius: '3px',
                    fontSize: '11px',
                    fontFamily: 'var(--font-mono)'
                  },
                  onMouseDown: (ev) => ev.stopPropagation(),
                  onChange: (ev) => {
                    ev.stopPropagation();
                    const n = parseInt(ev.target.value) || 0;
                    handleUpdateNodeParam(node.id, key, n);
                  }
                }),
                e('button', {
                  type: 'button',
                  style: {
                    width: '18px',
                    height: '18px',
                    padding: 0,
                    lineHeight: '16px',
                    textAlign: 'center',
                    background: 'var(--bg-surface, #ede9df)',
                    border: '1px solid var(--border-color, #ded9cd)',
                    borderRadius: '3px',
                    cursor: 'pointer',
                    fontSize: '11px',
                    fontWeight: 700
                  },
                  onMouseDown: (ev) => ev.stopPropagation(),
                  onClick: (ev) => {
                    ev.stopPropagation();
                    const nextVal = numVal + stepDelta;
                    handleUpdateNodeParam(node.id, key, nextVal);
                  }
                }, '+')
              )
            );
          }

          // 2. Sliders
          const isSlider = [
            'lr', 'steer_scale', 'balance_loss_weight', 'lambda_weight',
            'lambda_reflex', 'lambda_limbic', 'lambda_cortex', 'temperature',
            'dropout', 'weight_decay', 'momentum'
          ].includes(lowerKey);

          if (isSlider) {
            const floatVal = parseFloat(val) || 0;
            const isLr = lowerKey === 'lr';
            const minV = isLr ? 0.0001 : 0.0;
            const maxV = isLr ? 0.05 : (['temperature'].includes(lowerKey) ? 2.0 : 1.0);
            const stepV = isLr ? 0.0001 : (['dropout', 'weight_decay'].includes(lowerKey) ? 0.01 : 0.05);

            return e('div', {
              key: key,
              style: { display: 'flex', flexDirection: 'column', gap: '2px', fontSize: '11px' }
            },
              e('div', { style: { display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' } },
                e('span', { style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, cleanLabel),
                e('span', { style: { fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-main)' } }, isLr ? floatVal.toExponential(1) : floatVal.toFixed(2))
              ),
              e('input', {
                type: 'range',
                min: minV,
                max: maxV,
                step: stepV,
                value: floatVal,
                style: { width: '100%', height: '4px', cursor: 'pointer' },
                onMouseDown: (ev) => ev.stopPropagation(),
                onChange: (ev) => {
                  ev.stopPropagation();
                  handleUpdateNodeParam(node.id, key, parseFloat(ev.target.value));
                }
              })
            );
          }

          // 3. Dropdown Selects
          if (lowerKey === 'depth_mode') {
            return e('div', {
              key: key,
              style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }
            },
              e('span', { style: { color: 'var(--text-muted)' } }, cleanLabel),
              e('select', {
                value: val,
                style: { height: '20px', fontSize: '10.5px', background: 'var(--bg-input, #f2eee5)', border: '1px solid var(--border-color, #ded9cd)', borderRadius: '3px' },
                onMouseDown: (ev) => ev.stopPropagation(),
                onChange: (ev) => {
                  ev.stopPropagation();
                  handleUpdateNodeParam(node.id, key, ev.target.value);
                }
              },
                e('option', { value: 'dynamic' }, 'dynamic'),
                e('option', { value: 'joint' }, 'joint'),
                e('option', { value: 'early_exit' }, 'early_exit'),
                e('option', { value: 'static' }, 'static')
              )
            );
          }

          if (lowerKey === 'optimizer_type') {
            return e('div', {
              key: key,
              style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }
            },
              e('span', { style: { color: 'var(--text-muted)' } }, cleanLabel),
              e('select', {
                value: val,
                style: { height: '20px', fontSize: '10.5px', background: 'var(--bg-input, #f2eee5)', border: '1px solid var(--border-color, #ded9cd)', borderRadius: '3px' },
                onMouseDown: (ev) => ev.stopPropagation(),
                onChange: (ev) => {
                  ev.stopPropagation();
                  handleUpdateNodeParam(node.id, key, ev.target.value);
                }
              },
                e('option', { value: 'CentroidSteer' }, 'CentroidSteer'),
                e('option', { value: 'Muon' }, 'Muon'),
                e('option', { value: 'AdamW' }, 'AdamW')
              )
            );
          }

          if (lowerKey === 'split') {
            return e('div', {
              key: key,
              style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }
            },
              e('span', { style: { color: 'var(--text-muted)' } }, cleanLabel),
              e('select', {
                value: val,
                style: { height: '20px', fontSize: '10.5px', background: 'var(--bg-input, #f2eee5)', border: '1px solid var(--border-color, #ded9cd)', borderRadius: '3px' },
                onMouseDown: (ev) => ev.stopPropagation(),
                onChange: (ev) => {
                  ev.stopPropagation();
                  handleUpdateNodeParam(node.id, key, ev.target.value);
                }
              },
                e('option', { value: 'train' }, 'train'),
                e('option', { value: 'validation' }, 'validation'),
                e('option', { value: 'test' }, 'test')
              )
            );
          }

          if (lowerKey === 'dtype') {
            return e('div', {
              key: key,
              style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }
            },
              e('span', { style: { color: 'var(--text-muted)' } }, cleanLabel),
              e('select', {
                value: val,
                style: { height: '20px', fontSize: '10.5px', background: 'var(--bg-input, #f2eee5)', border: '1px solid var(--border-color, #ded9cd)', borderRadius: '3px' },
                onMouseDown: (ev) => ev.stopPropagation(),
                onChange: (ev) => {
                  ev.stopPropagation();
                  handleUpdateNodeParam(node.id, key, ev.target.value);
                }
              },
                e('option', { value: 'fp8_e4m3fn' }, 'fp8_e4m3fn'),
                e('option', { value: 'fp4' }, 'fp4'),
                e('option', { value: 'bf16' }, 'bf16'),
                e('option', { value: 'fp16' }, 'fp16'),
                e('option', { value: 'fp32' }, 'fp32')
              )
            );
          }

          // 4. Boolean Toggles
          const isBool = val === 'True' || val === 'False' || val === 'true' || val === 'false' || [
            'use_fp4', 'shuffle', 'prefetch', 'pin_memory', 'shared_expert',
            'use_rope', 'full_capacity', 'save_adapter', 'd2h_async', 'bias'
          ].includes(lowerKey);

          if (isBool) {
            const isChecked = val === 'True' || val === 'true';
            return e('div', {
              key: key,
              style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }
            },
              e('span', { style: { color: 'var(--text-muted)' } }, cleanLabel),
              e('input', {
                type: 'checkbox',
                checked: isChecked,
                style: { cursor: 'pointer' },
                onMouseDown: (ev) => ev.stopPropagation(),
                onChange: (ev) => {
                  ev.stopPropagation();
                  handleUpdateNodeParam(node.id, key, ev.target.checked ? 'True' : 'False');
                }
              })
            );
          }

          // 5. Default Compact Text Field
          return e('div', {
            key: key,
            style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px', gap: '6px' }
          },
            e('span', { style: { color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '80px' }, title: key }, cleanLabel),
            e('input', {
              type: 'text',
              value: val,
              style: {
                flex: 1,
                height: '18px',
                padding: '0 4px',
                background: 'var(--bg-input, #f2eee5)',
                border: '1px solid var(--border-color, #ded9cd)',
                borderRadius: '3px',
                fontSize: '10.5px',
                fontFamily: 'var(--font-mono)'
              },
              onMouseDown: (ev) => ev.stopPropagation(),
              onChange: (ev) => {
                ev.stopPropagation();
                handleUpdateNodeParam(node.id, key, ev.target.value);
              }
            })
          );
        }),
        remainingCount > 0 && e('div', {
          style: { fontSize: '10px', color: 'var(--text-dim)', textAlign: 'right', fontStyle: 'italic', marginTop: '2px' }
        }, `+${remainingCount} more in Inspector`)
      );
    };

    const getNodePorts = (node) => {
      const cat = nodeCatalog || BUILTIN_NODE_CATALOG;
      const meta = cat.find(c => c.name === node.type || c.title === node.title || c.name === node.title);
      const rawIns = (meta && (meta.typed_inputs || meta.inputs)) ? (meta.typed_inputs || meta.inputs) : (node.inputs || ['in']);
      const rawOuts = (meta && (meta.typed_outputs || meta.outputs)) ? (meta.typed_outputs || meta.outputs) : (node.outputs || ['out']);
      return {
        inputs: normalizePorts(rawIns, 'tensor'),
        outputs: normalizePorts(rawOuts, 'tensor')
      };
    };

    const handleSelectNode = (nodeId) => {
      setSelectedNodeId(nodeId);
      setSelectedNodeIds([nodeId]);
      const node = nodes.find(n => n.id === nodeId);
      if (node) {
        const params = {};
        if (node.details) {
          node.details.split('\n').forEach(line => {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) return;
            const eqIdx = trimmed.indexOf('=');
            if (eqIdx !== -1) {
              const k = trimmed.slice(0, eqIdx).trim();
              const v = trimmed.slice(eqIdx + 1).trim();
              params[k] = v;
            }
          });
        }
        setSelectedNodeParams(params);
      }
    };

    const handleToggleMultiSelectNode = (nodeId) => {
      const next = selectedNodeIds.includes(nodeId)
        ? selectedNodeIds.filter(id => id !== nodeId)
        : [...selectedNodeIds, nodeId];
      setSelectedNodeIds(next);
      setSelectedNodeId(next.includes(nodeId) ? nodeId : (next[next.length - 1] || null));
    };

    const handleGroupSelectedNodes = () => {
      const selected = nodes.filter(node => selectedNodeIds.includes(node.id));
      if (selected.length < 2) {
        showToast('Ctrl-click at least two nodes to create a reusable group.');
        return;
      }
      const selectedIds = new Set(selected.map(node => node.id));
      const incoming = edges.filter(edge => selectedIds.has(edge.target) && !selectedIds.has(edge.source));
      const outgoing = edges.filter(edge => selectedIds.has(edge.source) && !selectedIds.has(edge.target));
      if ([...incoming, ...outgoing].some(edge => !edge.source_port || !edge.target_port)) {
        showToast('This group has an untyped boundary wire. Add explicit ports before grouping.');
        return;
      }

      const groupId = `group_${Date.now()}`;
      const origin = {
        x: Math.min(...selected.map(node => Number(node.x) || 0)),
        y: Math.min(...selected.map(node => Number(node.y) || 0))
      };
      const nestedNodes = selected.map(node => ({ ...node }));
      const nestedEdges = edges.filter(edge => selectedIds.has(edge.source) && selectedIds.has(edge.target)).map(edge => ({ ...edge }));
      const groupInputs = [];
      const groupOutputs = [];
      const outerEdges = edges.filter(edge => !selectedIds.has(edge.source) && !selectedIds.has(edge.target));

      incoming.forEach((edge, index) => {
        const name = `input_${index + 1}`;
        const sourceNode = nodes.find(node => node.id === edge.source);
        const sourcePort = sourceNode ? getNodePorts(sourceNode).outputs.find(port => port.name === edge.source_port) : null;
        const type = sourcePort ? sourcePort.type : (edge.port_type || 'any');
        const boundaryId = `${groupId}_${name}`;
        groupInputs.push({ name, type, required: true, boundary_node: boundaryId, boundary_port: 'value' });
        nestedNodes.push({
          id: boundaryId, title: `Input: ${name}`, type: 'SubgraphInput',
          x: origin.x - 260, y: origin.y + index * 90,
          inputs: [], outputs: [{ name: 'value', type }],
          params: { public_name: name }
        });
        nestedEdges.push({
          id: `${groupId}_inner_${name}`, source: boundaryId, source_port: 'value',
          target: edge.target, target_port: edge.target_port, port_type: type
        });
        outerEdges.push({ ...edge, id: `${groupId}_outer_${name}`, target: groupId, target_port: name });
      });

      const outputMap = new Map();
      outgoing.forEach((edge, index) => {
        const key = `${edge.source}:${edge.source_port}`;
        let output = outputMap.get(key);
        if (!output) {
          const name = `output_${outputMap.size + 1}`;
          const sourceNode = selected.find(node => node.id === edge.source);
          const sourcePort = sourceNode ? getNodePorts(sourceNode).outputs.find(port => port.name === edge.source_port) : null;
          const type = sourcePort ? sourcePort.type : (edge.port_type || 'any');
          output = { name, type, required: true, source_node: edge.source, source_port: edge.source_port };
          outputMap.set(key, output);
          groupOutputs.push(output);
        }
        outerEdges.push({ ...edge, id: `${groupId}_outer_out_${index + 1}`, source: groupId, source_port: output.name });
      });

      const groupedNode = {
        id: groupId,
        title: `Group (${selected.length} nodes)`,
        type: 'Subgraph',
        x: origin.x,
        y: origin.y,
        inputs: groupInputs.map(({ name, type, required }) => ({ name, type, required })),
        outputs: groupOutputs.map(({ name, type, required }) => ({ name, type, required })),
        subgraph: { nodes: nestedNodes, edges: nestedEdges, inputs: groupInputs, outputs: groupOutputs, origin }
      };
      setNodes(prev => [...prev.filter(node => !selectedIds.has(node.id)), groupedNode]);
      setEdges(outerEdges);
      setSelectedNodeId(groupId);
      setSelectedNodeIds([groupId]);
      setNodeExecOutputs({});
      showToast(`Grouped ${selected.length} nodes. Select the group and choose Ungroup to edit its contents.`);
    };

    const handleUngroupSelectedNode = () => {
      const group = nodes.find(node => node.id === selectedNodeId && node.type === 'Subgraph');
      const definition = group && group.subgraph;
      if (!group || !definition || !Array.isArray(definition.nodes) || !Array.isArray(definition.edges)) {
        showToast('Select a grouped graph node first.');
        return;
      }
      const boundaryIds = new Set((definition.inputs || []).map(item => item.boundary_node));
      const groupIncoming = edges.filter(edge => edge.target === group.id);
      const groupOutgoing = edges.filter(edge => edge.source === group.id);
      const incomingByPort = new Map(groupIncoming.map(edge => [edge.target_port, edge]));
      const outgoingByPort = new Map((definition.outputs || []).map(item => [item.name, item]));
      const restoredEdges = edges.filter(edge => edge.source !== group.id && edge.target !== group.id);

      (definition.inputs || []).forEach(input => {
        const external = incomingByPort.get(input.name);
        if (!external) return;
        definition.edges.filter(edge => edge.source === input.boundary_node && edge.source_port === input.boundary_port).forEach((inner, index) => {
          restoredEdges.push({
            ...external,
            id: `${group.id}_restore_in_${input.name}_${index}`,
            target: inner.target,
            target_port: inner.target_port,
            port_type: input.type
          });
        });
      });
      groupOutgoing.forEach((external, index) => {
        const output = outgoingByPort.get(external.source_port);
        if (!output) return;
        restoredEdges.push({
          ...external,
          id: `${group.id}_restore_out_${index}`,
          source: output.source_node,
          source_port: output.source_port,
          port_type: output.type
        });
      });

      const dx = (Number(group.x) || 0) - (Number(definition.origin && definition.origin.x) || 0);
      const dy = (Number(group.y) || 0) - (Number(definition.origin && definition.origin.y) || 0);
      const restoredNodes = definition.nodes.filter(node => !boundaryIds.has(node.id)).map(node => ({
        ...node,
        x: (Number(node.x) || 0) + dx,
        y: (Number(node.y) || 0) + dy
      }));
      setNodes(prev => [...prev.filter(node => node.id !== group.id), ...restoredNodes]);
      setEdges(restoredEdges);
      setSelectedNodeId(null);
      setSelectedNodeIds([]);
      setNodeExecOutputs({});
      showToast(`Ungrouped ${restoredNodes.length} nodes; connections have been restored.`);
    };

    const handleSaveNodeParams = async () => {
      if (!selectedNodeId) return;
      const detailsStr = Object.entries(selectedNodeParams)
        .map(([k, v]) => `${k}=${v}`)
        .join('\n');
      const nextNodes = nodes.map(n => n.id === selectedNodeId ? { ...n, details: detailsStr } : n);
      setNodes(nextNodes);
      const foundNode = nodes.find(n => n.id === selectedNodeId);
      const title = (foundNode && foundNode.title) ? foundNode.title : selectedNodeId;

      try {
        const activeNodes = nextNodes.filter(n => !n.muted);
        const activeEdges = edges.filter(e => {
          const s = nextNodes.find(n => n.id === e.source);
          const t = nextNodes.find(n => n.id === e.target);
          return s && !s.muted && t && !t.muted;
        });
        const res = await apiFetch('/v1/dag/compile_to_training', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ nodes: activeNodes, edges: activeEdges })
        });
        const data = await res.json();
        if (!res.ok || data.status !== 'success') {
          throw new Error(data.message || data.error || `Compile failed (${res.status})`);
        }
        Object.entries(selectedNodeParams).forEach(([k, v]) => {
          const lk = String(k).toLowerCase();
          if (lk === 'batch_size') setTrainBatchSize(parseInt(v) || 4);
          if (lk === 'seq_len') setTrainSeqLen(parseInt(v) || 64);
          if (lk === 'grad_accum_steps') setTrainGradAccum(parseInt(v) || 4);
          if (lk === 'depth_mode') setTrainDepthMode(String(v) === 'static' ? 'cortex' : String(v));
          if (lk === 'lr') setTrainLR(String(v));
          if (lk === 'dataset_name') setActiveDataset(String(v));
        });
        if (data.applied_config && data.applied_config.active_adapter !== undefined) {
          setActiveAdapter(data.applied_config.active_adapter);
        }
        if (data.active_model) setActiveModel(data.active_model);
        if (data.profile) {
          setHardwareProfile(data.profile);
          if (data.profile.gpu_offload_pct !== undefined) setGpuOffloadPct(data.profile.gpu_offload_pct);
          if (data.profile.gpu_layers !== undefined) setGpuLayers(data.profile.gpu_layers);
        }
        showToast(data.message || `Saved and applied parameters for ${title}.`);
      } catch (err) {
        showToast(`Canvas node updated locally, but engine sync failed: ${err.message || 'backend request failed'}`);
      }
    };

    const handleExecuteSingleNode = async (nodeId) => {
      const targetId = nodeId || selectedNodeId;
      if (!targetId) return;
      const node = nodes.find(n => n.id === targetId);
      if (!node) return;

      setIsExecutingSingleNode(true);
      showToast(`Executing [${node.title}] in isolation...`);
      try {
        const res = await apiFetch('/v1/dag/execute_node', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            node: {
              ...node,
              params: targetId === selectedNodeId ? selectedNodeParams : (node.params || {})
            }
          })
        });
        const data = await res.json();
        if (data.status === 'completed') {
          setNodeExecOutputs(prev => ({
            ...prev,
            [targetId]: {
              status: 'completed',
              output: data.output,
              elapsed: data.elapsed_sec
            }
          }));
          showToast(`Node [${node.title}] executed in ${(data.elapsed_sec * 1000).toFixed(0)}ms`);
        } else {
          showToast(`Execution failed: ${data.error || 'Unknown'}`);
        }
      } catch (err) {
        showToast(`Error: ${err.message}`);
      } finally {
        setIsExecutingSingleNode(false);
      }
    };

    const handleNodeMouseDown = (ev, nodeId) => {
      if (ev.target.closest('.socket-pin') || ev.target.closest('.node-delete-btn') || ev.target.closest('.node-mute-btn') || ev.target.closest('input') || ev.target.closest('select') || ev.target.closest('button')) return;
      ev.stopPropagation();
      if (ev.ctrlKey || ev.metaKey) {
        handleToggleMultiSelectNode(nodeId);
        return;
      }
      setDraggingNodeId(nodeId);
      handleSelectNode(nodeId);
      const node = nodes.find(n => n.id === nodeId);
      if (node && gridRef.current) {
        const rect = gridRef.current.getBoundingClientRect();
        const screenX = ev.clientX - rect.left;
        const screenY = ev.clientY - rect.top;
        const canvasX = (screenX - canvasPan.x) / canvasZoom;
        const canvasY = (screenY - canvasPan.y) / canvasZoom;
        setDragOffset({ x: canvasX - node.x, y: canvasY - node.y });
      }
    };

    const handleCanvasMouseDown = (ev) => {
      if (ev.target === gridRef.current || (ev.target.tagName && ev.target.tagName.toLowerCase() === 'svg') || ev.target.classList.contains('node-canvas-viewport')) {
        setIsPanning(true);
        setPanStart({ x: ev.clientX - canvasPan.x, y: ev.clientY - canvasPan.y });
        setSelectedNodeId(null);
        setSelectedNodeIds([]);
      }
    };

    const handleCanvasMouseMove = (ev) => {
      if (!gridRef.current) return;
      const rect = gridRef.current.getBoundingClientRect();
      const screenX = ev.clientX - rect.left;
      const screenY = ev.clientY - rect.top;
      const canvasX = (screenX - canvasPan.x) / canvasZoom;
      const canvasY = (screenY - canvasPan.y) / canvasZoom;
      setCanvasMousePos({ x: canvasX, y: canvasY, screenX, screenY });
      setMousePos({ x: canvasX, y: canvasY });

      if (isPanning) {
        setCanvasPan({
          x: ev.clientX - panStart.x,
          y: ev.clientY - panStart.y
        });
        return;
      }

      if (draggingNodeId) {
        setNodes(prev => prev.map(n => {
          if (n.id === draggingNodeId) {
            return {
              ...n,
              x: Math.round(canvasX - dragOffset.x),
              y: Math.round(canvasY - dragOffset.y)
            };
          }
          return n;
        }));
      }
    };

    const handleCanvasMouseUp = () => {
      setIsPanning(false);
      setDraggingNodeId(null);
      setConnectingWire(null);
      setConnectingFromId(null);
    };

    // Non-passive wheel handler attached directly to canvas DOM element to strictly
    // prevent browser window zoom while zooming canvas nodes smoothly
    useEffect(() => {
      const grid = gridRef.current;
      if (!grid || activeTab !== 'nodegraph') return;
      const onGridWheel = (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        const zoomFactor = ev.deltaY < 0 ? 1.08 : 0.92;
        setCanvasZoom(prevZoom => {
          const newZoom = Math.min(Math.max(prevZoom * zoomFactor, 0.25), 2.5);
          if (gridRef.current) {
            const rect = gridRef.current.getBoundingClientRect();
            const mouseScreenX = ev.clientX - rect.left;
            const mouseScreenY = ev.clientY - rect.top;
            setCanvasPan(prevPan => ({
              x: mouseScreenX - (mouseScreenX - prevPan.x) * (newZoom / prevZoom),
              y: mouseScreenY - (mouseScreenY - prevPan.y) * (newZoom / prevZoom)
            }));
          }
          return newZoom;
        });
      };
      grid.addEventListener('wheel', onGridWheel, { passive: false });
      return () => grid.removeEventListener('wheel', onGridWheel);
    }, [activeTab]);

    // Keep drags and wires alive when the pointer leaves the canvas, and make
    // the same gestures work with touch and pen input.
    useEffect(() => {
      const isCanvasInteraction = () => draggingNodeId || isPanning || connectingWire;
      const onPointerMove = (ev) => {
        if (!gridRef.current || activeTab !== 'nodegraph') return;
        if (gridRef.current.contains(ev.target) || isCanvasInteraction()) handleCanvasMouseMove(ev);
      };
      const onPointerUp = (ev) => {
        if (connectingWire) {
          // Touch browsers may keep pointer capture on the source socket; hit
          // test the release point so dropping still connects the target.
          const hit = document.elementFromPoint(ev.clientX, ev.clientY);
          const input = hit && hit.closest('[data-dag-input-node]');
          if (input) {
            handleEndWire(ev, input.dataset.dagInputNode, input.dataset.dagInputPort, input.dataset.dagInputType);
          } else {
            setConnectingWire(null);
            setConnectingFromId(null);
          }
        }
        if (draggingNodeId || isPanning) handleCanvasMouseUp();
      };
      window.addEventListener('pointermove', onPointerMove);
      window.addEventListener('pointerup', onPointerUp);
      window.addEventListener('pointercancel', onPointerUp);
      return () => {
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
        window.removeEventListener('pointercancel', onPointerUp);
      };
    }, [activeTab, draggingNodeId, isPanning, connectingWire, canvasPan, panStart, dragOffset]);

    const handleStartWire = (ev, nodeId, portName, portType, portIndex) => {
      ev.stopPropagation();
      ev.preventDefault();
      const pinRect = ev.currentTarget.getBoundingClientRect();
      const gridRect = gridRef.current ? gridRef.current.getBoundingClientRect() : { left: 0, top: 0 };
      const startX = (pinRect.left + pinRect.width / 2 - gridRect.left - canvasPan.x) / canvasZoom;
      const startY = (pinRect.top + pinRect.height / 2 - gridRect.top - canvasPan.y) / canvasZoom;
      setConnectingWire({
        source: nodeId,
        source_port: portName,
        port_type: portType,
        startX: startX,
        startY: startY
      });
      setConnectingFromId(nodeId);
    };

    const handleEndWire = (ev, targetNodeId, targetPortName, targetPortType) => {
      ev.stopPropagation();
      ev.preventDefault();
      if (connectingWire && connectingWire.source !== targetNodeId) {
      if (!isPortCompatible(connectingWire.port_type, targetPortType)) {
        showToast(`Cannot connect ${connectingWire.port_type} to ${targetPortType}.`);
        setConnectingWire(null);
        setConnectingFromId(null);
        return;
      }
      const pending = [targetNodeId];
      const visited = new Set();
      while (pending.length) {
        const current = pending.pop();
        if (current === connectingWire.source) {
          showToast('That connection would create a cycle. DAGs must stay acyclic.');
          setConnectingWire(null);
          setConnectingFromId(null);
          return;
        }
        if (visited.has(current)) continue;
        visited.add(current);
        edges.forEach(edge => {
          if (edge.source === current && edge.target !== targetNodeId) pending.push(edge.target);
        });
      }
      const newEdge = {
          id: `e_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
          source: connectingWire.source,
          source_port: connectingWire.source_port,
          target: targetNodeId,
          target_port: targetPortName,
          port_type: connectingWire.port_type
        };
        setEdges(prev => [...prev.filter(e => !(e.target === targetNodeId && e.target_port === targetPortName)), newEdge]);
        showToast(`Connected Wire: [${connectingWire.source}:${connectingWire.source_port}] → [${targetNodeId}:${targetPortName}]`);
      }
      setConnectingWire(null);
      setConnectingFromId(null);
    };

    const handleDisconnectWire = (edgeId) => {
      setEdges(prev => prev.filter(e => e.id !== edgeId));
      showToast('Disconnected Wire');
    };

    const handleFitCanvasView = (graphNodes = nodes) => {
      if (!graphNodes.length || !gridRef.current) return;
      const minX = Math.min(...graphNodes.map(n => n.x));
      const maxX = Math.max(...graphNodes.map(n => n.x + NODE_WIDTH));
      const minY = Math.min(...graphNodes.map(n => n.y));
      const maxY = Math.max(...graphNodes.map(n => n.y + 260));
      const rect = gridRef.current.getBoundingClientRect();
      const graphW = (maxX - minX) + 160;
      const graphH = (maxY - minY) + 160;
      const scale = Math.min(Math.max(Math.min((rect.width - 60) / graphW, (rect.height - 60) / graphH), 0.35), 1.15);
      setCanvasZoom(scale);
      setCanvasPan({
        x: (rect.width - graphW * scale) / 2 - minX * scale + 80 * scale,
        y: (rect.height - graphH * scale) / 2 - minY * scale + 80 * scale
      });
      showToast('Centered & Fitted DAG View');
    };

    const handleDuplicateSelectedNode = () => {
      if (!selectedNodeId) return;
      const src = nodes.find(n => n.id === selectedNodeId);
      if (!src) return;
      const newId = `node_${Date.now().toString().slice(-4)}_${nodes.length + 1}`;
      const clone = {
        ...src,
        id: newId,
        title: `${src.title} (Copy)`,
        x: src.x + 40,
        y: src.y + 40
      };
      setNodes(prev => [...prev, clone]);
      setSelectedNodeId(newId);
      showToast(`Duplicated Node [${clone.title}]`);
    };

    const handleAddDatasetNodeToDAG = (ds) => {
      const dsId = typeof ds === 'string' ? ds : (ds.id || ds.name || 'roneneldan/TinyStories');
      const dsName = typeof ds === 'string' ? ds : (ds.name || dsId);
      const newId = `node_stream_${Date.now().toString().slice(-4)}`;
      const newNode = {
        id: newId,
        title: `Stream: ${dsName.split('/').pop()}`,
        type: 'Data',
        x: 60 + (nodes.length % 5) * 40,
        y: 80 + (nodes.length % 5) * 40,
        details: `dataset_name=${dsId}\nsplit=${ds.split || 'train'}\nbuffer_size=20\ntext_column=${ds.col || 'text'}`
      };
      setNodes(prev => [...prev, newNode]);
      setActiveTab('nodegraph');
      showToast(`Added Dataset Node [${newNode.title}] to DAG Canvas`);
    };

    const handleAddModuleNodeToDAG = (mod) => {
      const nodeNames = mod.registered_nodes || (mod.artifacts && mod.artifacts.nodes) || [];
      if (nodeNames.length > 0) {
        handleAddModuleNodesToCanvas(mod);
        return;
      }
      const newId = `node_mod_${Date.now().toString().slice(-4)}`;
      const category = (mod.type === 'plugin' ? 'Custom' : (mod.type === 'adapter' ? 'Model' : (mod.type === 'dataset' ? 'Data' : 'Runtime')));
      const newNode = {
        id: newId,
        title: mod.name || 'Module Node',
        type: category,
        x: 100 + (nodes.length % 5) * 40,
        y: 100 + (nodes.length % 5) * 40,
        details: `module_id=${mod.id}\nversion=${mod.version || '1.0'}\nauthor=${mod.author || 'User'}`
      };
      setNodes(prev => [...prev, newNode]);
      setActiveTab('nodegraph');
      showToast(`Added Module Node [${newNode.title}] to DAG Canvas`);
    };

    const handleMouseDown = handleNodeMouseDown;
    const handleMouseMove = handleCanvasMouseMove;
    const handleMouseUp = handleCanvasMouseUp;

    // Node Deletion Handler
    const handleDeleteNode = (nodeId) => {
      setNodes(prev => prev.filter(n => n.id !== nodeId));
      setEdges(prev => prev.filter(e => e.source !== nodeId && e.target !== nodeId));
      setSelectedNodeIds(prev => prev.filter(id => id !== nodeId));
      if (selectedNodeId === nodeId) setSelectedNodeId(null);
      showToast(`Deleted Node [${nodeId}]`);
    };

    const handleSpawnCatalogNode = (nodeIdentifier, posX, posY) => {
      const cat = nodeCatalog || BUILTIN_NODE_CATALOG;
      const item = cat.find(n => n.name === nodeIdentifier || n.title === nodeIdentifier);
      const newId = `node_${Date.now().toString().slice(-4)}_${nodes.length + 1}`;
      const defaultTitle = item ? item.title : (typeof nodeIdentifier === 'string' ? nodeIdentifier : 'Custom Node');
      const category = item ? item.category : 'Custom';
      const details = item
        ? (item.details || Object.entries(item.parameters || {}).map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`).join('\n'))
        : `Type: ${category}\nState: Ready`;

      // Calculate neat grid placement or use provided coordinates
      const col = nodes.length % 5;
      const row = Math.floor(nodes.length / 5);
      const targetX = (typeof posX === 'number') ? posX : (60 + col * 300);
      const targetY = (typeof posY === 'number') ? posY : (80 + row * 180);
      const newNode = {
        id: newId,
        title: defaultTitle,
        type: category,
        x: Math.round(targetX),
        y: Math.round(targetY),
        details: details
      };
      setNodes(prev => [...prev, newNode]);
      setShowNodeLibraryModal(false);
      showToast(`Spawned [${defaultTitle}] on canvas`);
    };

    const handleAddNode = (type) => {
      const cat = nodeCatalog || BUILTIN_NODE_CATALOG;
      const firstOfType = cat.find(n => n.category === type);
      if (firstOfType) {
        handleSpawnCatalogNode(firstOfType.name);
      } else {
        const newId = `node_${nodes.length + 1}`;
        const titles = { Data: 'Custom Data Node', Model: 'LoRA Adapter Node', Optimizer: 'AdamW Optimizer', Export: 'Exporter Node' };
        const newNode = {
          id: newId,
          title: titles[type] || 'Custom Node',
          type: type,
          x: 60 + (nodes.length % 5) * 280,
          y: 80 + Math.floor(nodes.length / 5) * 160,
          details: `Type: ${type}\nState: Ready`
        };
        setNodes(prev => [...prev, newNode]);
        showToast(`Added ${type} Node`);
      }
    };

    const handleCreateCustomNode = () => {
      if (!customNodeTitle.trim()) return;
      const newId = `custom_node_${Date.now()}`;
      const newNode = {
        id: newId,
        title: customNodeTitle.trim(),
        type: customNodeType,
        x: 80 + nodes.length * 30,
        y: 120 + (nodes.length % 3) * 30,
        details: customNodeDetails.trim() || `Custom Node (${customNodeType})\nRegistered in Plugin Registry`
      };
      setNodes([...nodes, newNode]);
      setShowCustomNodeModal(false);
      setCustomNodeTitle('');
      setCustomNodeDetails('');
      showToast(`Registered Custom Node: ${newNode.title}`);
    };


    const handleSaveCheckpoint = async () => {
      showToast('Saving model checkpoint to disk...');
      try {
        const res = await apiFetch('/v1/training/save_checkpoint', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(`Checkpoint saved (${data.size_mb} MB) -> ${data.filename}`);
          if (data.filename) setActiveCheckpointName(data.filename);
          fetchCheckpoints();
        } else {
          showToast(`Save failed: ${data.message || 'unknown error'}`);
        }
      } catch (err) {
        showToast('Error saving checkpoint to disk');
      }
    };

    const handleLoadModel = async (modelId) => {
      const matchedPreset = MODEL_PRESETS.find(p => p.id === modelId);
      if (matchedPreset && !modelId.endsWith('.pt') && !modelId.endsWith('.safetensors') && !modelId.includes('/')) {
        await handleSelectModelPreset(modelId);
        return;
      }
      showToast(`Loading ${modelId} into engine...`);
      try {
        const isCkpt = typeof modelId === 'string' && (modelId.endsWith('.pt') || modelId.endsWith('.bin') || modelId.endsWith('.safetensors') || modelId.includes('/'));
        const candidatePaths = isCkpt ? [modelId] : [
          `checkpoints/${modelId}.pt`,
          `checkpoints_full/${modelId}.pt`,
          `checkpoints_full/best.pt`,
          `checkpoints_full/latest.pt`,
          `checkpoints/triune_studio_step_0.pt`
        ];
        let loaded = false;
        for (const p of candidatePaths) {
          const res = await apiFetch('/v1/models/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ checkpoint_path: p })
          });
          const data = await res.json();
          if (res.ok && data.status === 'success') {
            showToast(data.message || `Loaded ${p} into live engine`);
            if (data.active_model) setActiveModel(data.active_model);
            setWeightsLoaded(Boolean(data.weights_loaded));
            if (data.checkpoint) setActiveCheckpointName(data.checkpoint);
            if (data.step !== undefined) {
              const restoredTok = data.tokens_trained !== undefined ? data.tokens_trained : (data.step * 1024);
              setMetrics(prev => ({ ...prev, step: data.step, tokens_trained: restoredTok }));
              setDatasetInfo(prev => ({ ...prev, total_tokens: restoredTok }));
            }
            if (data.architecture) {
              setDatasetInfo(prev => ({
                ...prev,
                num_layers: data.architecture.num_layers || prev.num_layers,
                hidden_dim: data.architecture.hidden_dim || prev.hidden_dim,
                num_heads: data.architecture.num_heads || prev.num_heads,
                num_experts: data.architecture.num_experts || prev.num_experts
              }));
              syncStateToDAGNodes({
                num_layers: data.architecture.num_layers,
                hidden_dim: data.architecture.hidden_dim,
                num_heads: data.architecture.num_heads,
                head_dim: data.architecture.head_dim,
                num_experts: data.architecture.num_experts
              });
            }
            loaded = true;
            break;
          }
        }
        if (!loaded) {
          showToast(`Could not load ${modelId}: no matching checkpoint could be loaded.`);
          return;
        }
      } catch (err) {
        showToast(`Could not load ${modelId}: ${err.message || 'backend request failed'}`);
        return;
      }
    };

    const handleExecuteDAG = async () => {
      setDagExecutionStatus('Initiating ExecutionEngine on PyTorch backend...');
      setNodeExecOutputs({});
      setActiveRunningNodeId(null);

      try {
        const activeNodes = nodes.filter(n => !n.muted);
        const activeEdges = edges.filter(e => {
          const s = nodes.find(n => n.id === e.source);
          const t = nodes.find(n => n.id === e.target);
          return s && !s.muted && t && !t.muted;
        });
        const res = await apiFetch('/v1/dag/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ nodes: activeNodes, edges: activeEdges })
        });
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`Server returned HTTP ${res.status}: ${errText.slice(0, 100)}`);
        }
        const data = await res.json();
        if ((data.status === 'error' || data.status === 'failed') && !data.results) {
          setDagExecutionStatus(`Engine ${data.status}: ${data.error || data.message || 'one or more nodes failed'}`);
          return;
        }
        const results = data.results || {};
        const nodeIds = Object.keys(results);

        if (nodeIds.length === 0) {
          setDagExecutionStatus('ExecutionEngine returned no executed nodes.');
          return;
        }

        // Update node execution outputs directly from genuine backend execution results
        const updatedOutputs = {};
        for (const [nid, nres] of Object.entries(results)) {
          updatedOutputs[nid] = {
            status: nres.status || 'completed',
            output: nres.output || {},
            error: nres.error || nres.output?.error || nres.output?.message || null,
            elapsed: nres.elapsed_sec
          };

          // If optimizer or loss node produced real step loss, update metrics
          if (nres.status === 'completed' && nres.output?.real_computation === true &&
              Number(nres.output?.step_completed) > 0 &&
              (nres.output.loss !== undefined || nres.output.loss_val !== undefined)) {
            const dagLoss = parseFloat(nres.output.loss !== undefined ? nres.output.loss : nres.output.loss_val);
            setMetrics(prev => ({
              ...prev,
              loss: dagLoss,
              step: (prev.step || 0) + 1
            }));
            setMetricsHistory(prev => [
              ...prev,
              {
                step: (prev.length > 0 ? (prev[prev.length - 1].step || 0) + 1 : 1),
                loss: dagLoss,
                lm_loss: nres.output.lm_loss !== undefined ? parseFloat(nres.output.lm_loss) : dagLoss,
                router_loss: nres.output.router_loss !== undefined ? parseFloat(nres.output.router_loss) : null,
                throughput: prev.throughput || 0,
                device: systemDiagnostics.device_name
              }
            ]);
          }
        }

        setNodeExecOutputs(prev => ({ ...prev, ...updatedOutputs }));
        setActiveRunningNodeId(null);

        let details = [];
        for (const [k, v] of Object.entries(results)) {
          const out = v.output || {};
          if (out.tokens_loaded) details.push(`Data: ${out.tokens_loaded} tok (${out.dataset_file || ''})`);
          if (out.total_params) details.push(`Model: ${out.total_params}`);
          if (out.loss !== undefined) details.push(`Loss: ${out.loss}`);
          if (out.loss_val !== undefined) details.push(`Loss: ${out.loss_val}`);
          if (out.perplexity !== undefined) details.push(`PPL: ${out.perplexity}`);
          if (out.file_size_mb) details.push(`Export: ${out.file_size_mb} MB`);
        }
        const specificationCount = Object.values(results).filter(
          result => result.output?.execution_type === 'specification'
        ).length;
        const failedCount = Object.values(results).filter(result => result.status === 'failed').length;
        const executedCount = nodeIds.length - specificationCount - failedCount;
        const summary = details.length > 0 ? ` (${details.join(' | ')})` : '';
        const specNote = specificationCount > 0
          ? ` ${specificationCount} node(s) only produced configuration/specification output; no computation was performed for those nodes.`
          : '';
        const failure = Object.entries(results).find(([, result]) => result.status === 'failed');
        const failureName = failure ? (nodes.find(node => node.id === failure[0])?.title || failure[0]) : '';
        const failureError = failure && (failure[1].error || failure[1].output?.error || failure[1].output?.message);
        const failedNote = failedCount > 0
          ? ` First failure: ${failureName}${failureError ? ` — ${failureError}` : ''}.`
          : '';
        setDagExecutionStatus(`ExecutionEngine finished ${executedCount} node(s); ${specificationCount} specification-only node(s); ${failedCount} failed.${summary}${specNote}${failedNote}`);
        showToast(failedCount > 0 ? `DAG stopped after ${failedCount} node failure(s)` : (specificationCount > 0
          ? `DAG finished with ${specificationCount} specification-only node(s)`
          : 'DAG Engine Pipeline Completed!'));
      } catch (err) {
        setActiveRunningNodeId(null);
        setDagExecutionStatus('Execution error: ' + err.message);
        showToast('DAG Execution Failed');
      }
    };

    const handleSendMessage = async (overridePrompt = null) => {
      const userText = (overridePrompt !== null ? overridePrompt : chatInput).trim();
      if (!userText || isGenerating) return;

      if (overridePrompt === null) {
        setMessages(prev => [
          ...prev,
          { sender: 'User', text: userText, time: new Date().toLocaleTimeString(), isAssistant: false }
        ]);
        setChatInput('');
      }
      setIsGenerating(true);

      const controller = new AbortController();
      chatAbortControllerRef.current = controller;
      const requestStartedAt = performance.now();
      let streamedMessageId = null;
      let streamedText = '';
      let streamFirstTokenMs = null;

      try {
        const historyTurns = messages
          .filter(m => m.sender !== 'System')
          .slice(-8)
          .map(m => ({
            role: m.isAssistant ? 'assistant' : 'user',
            content: m.text
          }));

        const requestMessages = [
          { role: 'system', content: systemPrompt },
          ...historyTurns,
          { role: 'user', content: userText }
        ];

        const response = await apiFetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            model: activeModel,
            messages: requestMessages,
            temperature: temperature,
            max_tokens: maxTokens,
            route: route,
            stream: true
          })
        });
        let data = {};
        let streamTelemetry = null;
        if (response.headers.get('content-type')?.includes('text/event-stream')) {
          if (!response.ok || !response.body) {
            const rawText = await response.text().catch(() => '');
            throw new Error(rawText.trim() || `Request failed (${response.status})`);
          }
          streamedMessageId = `triune-stream-${Date.now()}-${Math.random().toString(36).slice(2)}`;
          const streamingSender = activeModel.includes('gpt') || activeModel.includes('claude') || activeModel.includes('gemini')
            ? activeModel.toUpperCase() : 'Triune MoE Engine';
          setMessages(prev => [...prev, {
            id: streamedMessageId,
            sender: streamingSender,
            text: '',
            time: new Date().toLocaleTimeString(),
            isAssistant: true
          }]);

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let pending = '';
          let streamCompleted = false;
          const consumeSseLine = line => {
            const trimmed = line.trim();
            if (!trimmed.startsWith('data:')) return;
            const payload = trimmed.slice(5).trim();
            if (!payload) return;
            if (payload === '[DONE]') {
              streamCompleted = true;
              return;
            }
            let chunk;
            try { chunk = JSON.parse(payload); } catch (_) { return; }
            if (chunk.error?.message) throw new Error(chunk.error.message);
            const choice = chunk.choices?.[0];
            if (choice?.delta?.content !== undefined) {
              if (streamFirstTokenMs === null && String(choice.delta.content).length > 0) {
                streamFirstTokenMs = performance.now() - requestStartedAt;
              }
              streamedText = chunk.triune_snapshot
                ? String(choice.delta.content)
                : streamedText + String(choice.delta.content);
              setMessages(prev => prev.map(message => message.id === streamedMessageId
                ? { ...message, text: streamedText }
                : message));
            }
            if (chunk.telemetry) streamTelemetry = chunk.telemetry;
          };

          while (true) {
            const { value, done } = await reader.read();
            pending += decoder.decode(value || new Uint8Array(), { stream: !done });
            const lines = pending.split(/\r?\n/);
            pending = lines.pop() || '';
            for (const line of lines) consumeSseLine(line);
            if (done) break;
          }
          if (pending) consumeSseLine(pending);
          if (!streamCompleted && !controller.signal.aborted)
            throw new Error('The inference stream ended before the backend completed the response.');
          data = {
            choices: [{ message: { content: streamedText } }],
            telemetry: {
              ...(streamTelemetry || { route: route.toUpperCase(), latency_ms: Math.round(performance.now() - requestStartedAt) }),
              client_ttft_ms: streamFirstTokenMs
            }
          };
        } else {
          try {
            data = await response.json();
          } catch (_jsonErr) {
            const rawText = await response.text().catch(() => '');
            data = { detail: rawText.trim() || `Request failed (${response.status})` };
          }
        }
        if (!response.ok) {
          const detail = typeof data.detail === 'string' ? data.detail : (data.message || `Request failed (${response.status})`);
          throw new Error(detail);
        }
        const isExternal = activeModel.includes('gpt') || activeModel.includes('claude') || activeModel.includes('gemini');
        const rawReply = data.choices
          ? data.choices[0].message.content
          : (data.error ? `[API Key Notice]: ${data.error}` : 'Engine processing completed.');
        const reply = cleanText(rawReply);
        const senderName = isExternal ? activeModel.toUpperCase() : 'Triune MoE Engine';
        const tele = data.telemetry || { route: route.toUpperCase(), vram: `${vramUsage.allocated} GB`, latency: '18ms' };
        if (tele.weights_loaded !== undefined && !isExternal) setWeightsLoaded(Boolean(tele.weights_loaded));
        if (tele.checkpoint !== undefined && !isExternal) setActiveCheckpointName(tele.checkpoint || null);
        if (tele.active_model && !isExternal) setActiveModel(tele.active_model);

        const assistantMessage = {
            sender: senderName,
            text: reply,
            time: new Date().toLocaleTimeString(),
            isAssistant: true,
            telemetry: {
              route: isExternal ? 'CLOUD API' : (tele.route || route.toUpperCase()),
              vram: isExternal ? '0.0 GB (Cloud)' : (tele.vram_gb !== undefined ? `${tele.vram_gb} GB` : `${vramUsage.allocated} GB`),
              latency: tele.latency_ms !== undefined ? `${tele.latency_ms}ms` : '18ms',
              speed: tele.tokens_per_sec ? `${tele.tokens_per_sec} t/s` : undefined,
              tokens_count: tele.tokens_count,
              ttft: tele.client_ttft_ms !== undefined && tele.client_ttft_ms !== null
                ? `${Math.round(tele.client_ttft_ms)}ms first token`
                : (tele.ttft_ms !== undefined && tele.ttft_ms !== null ? `${tele.ttft_ms}ms first token` : undefined),
              decodeSpeed: tele.decode_tokens_per_sec ? `${tele.decode_tokens_per_sec} decode t/s` : undefined,
              device: tele.device || (isExternal ? 'cloud' : undefined)
            }
          };
        if (streamedMessageId) {
          setMessages(prev => prev.map(message => message.id === streamedMessageId
            ? { ...message, ...assistantMessage }
            : message));
        } else {
          setMessages(prev => [...prev, assistantMessage]);
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          if (streamedMessageId && streamedText) {
            setMessages(prev => prev.map(message => message.id === streamedMessageId
              ? { ...message, text: streamedText, time: new Date().toLocaleTimeString(), telemetry: { route: 'CANCELLED', latency: `${Math.round(performance.now() - requestStartedAt)}ms` } }
              : message));
          }
          showToast('Inference cancelled by user');
          return;
        }
        setMessages(prev => [
          ...prev.map(message => message.id === streamedMessageId && streamedText
            ? { ...message, text: streamedText, telemetry: { route: 'INTERRUPTED', latency: `${Math.round(performance.now() - requestStartedAt)}ms` } }
            : message).filter(message => message.id !== streamedMessageId || streamedText),
          {
            sender: 'System',
            text: `Chat request failed: ${err.message || 'The backend or selected provider did not return a response.'}`,
            time: new Date().toLocaleTimeString(),
            isAssistant: true,
            telemetry: {
              route: 'ERROR',
              vram: '0.0 GB',
              latency: `${Math.max(0, Math.round(performance.now() - requestStartedAt))}ms`
            }
          }
        ]);
      } finally {
        setIsGenerating(false);
        chatAbortControllerRef.current = null;
      }
    };

    const handleStopGeneration = async () => {
      const isLocalModel = activeModel.startsWith('triune');
      if (isLocalModel) {
        try {
          const response = await apiFetch('/v1/chat/cancel', { method: 'POST' });
          const result = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(result.detail || `Cancel failed (${response.status})`);
          showToast(result.status === 'idle' ? result.message : 'Stopping local inference after the current model step…');
        } catch (error) {
          showToast(`Could not stop local inference: ${error.message}`);
        }
        return;
      }
      if (chatAbortControllerRef.current) {
        chatAbortControllerRef.current.abort();
        chatAbortControllerRef.current = null;
      }
      setIsGenerating(false);
      showToast('Provider request cancelled');
    };

    const handleClearChat = () => {
      setMessages([
        {
          sender: 'Triune Engine',
          text: 'Conversation cleared. Ready for fresh research questions or prompt evaluations.',
          time: new Date().toLocaleTimeString(),
          isAssistant: true,
          telemetry: { route: 'AUTO', vram: 'Live', latency: 'Direct' }
        }
      ]);
      showToast('Chat session reset');
    };

    const handleCopyMessage = (text) => {
      if (typeof navigator !== 'undefined' && navigator.clipboard) {
        navigator.clipboard.writeText(text);
        showToast('Copied to clipboard');
      }
    };

    const handleRegenerateLast = () => {
      if (isGenerating) return;
      for (let i = messages.length - 1; i >= 0; i--) {
        if (!messages[i].isAssistant && messages[i].sender !== 'System') {
          const lastUserText = messages[i].text;
          setMessages(prev => {
            const nextList = [...prev];
            if (nextList.length > 0 && nextList[nextList.length - 1].isAssistant) {
              nextList.pop();
            }
            return nextList;
          });
          handleSendMessage(lastUserText);
          break;
        }
      }
    };


    const handleStartFineTuning = async () => {
      setFineTuningStatus('Attaching LoRA adapters & initiating fine-tuning run on PyTorch backend...');
      try {
        const res = await apiFetch('/v1/finetune/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_path: loraConfig.dataset,
            lora_rank: loraConfig.rank,
            lora_alpha: loraConfig.alpha,
            epochs: loraConfig.epochs,
            lr: parseFloat(loraConfig.lr) || 2e-4,
            quantization: loraConfig.quantization || '4-bit NF4'
          })
        });
        const data = await res.json();
        if (data.status === 'started') {
          if (data.active_adapter) {
            setActiveAdapter(data.active_adapter);
          }
          syncStateToDAGNodes({ adapter: { rank: loraConfig.rank, alpha: loraConfig.alpha } });
          showToast('LoRA fine-tuning started on PyTorch engine!');
          const poller = setInterval(async () => {
            try {
              const sRes = await apiFetch('/v1/finetune/status');
              const sData = await sRes.json();
              if (sData.status === 'completed') {
                clearInterval(poller);
                setFineTuningStatus(`Fine-tuning completed! Final Loss: ${sData.final_loss}. Saved to: ${sData.output_dir}`);
                const attached = sData.active_adapter || { name: `LoRA-r${loraConfig.rank}`, rank: loraConfig.rank, alpha: loraConfig.alpha, path: sData.output_dir };
                setActiveAdapter(attached);
                fetchCheckpoints();
                showToast('LoRA Fine-Tuning Completed & Attached to Engine!');
              } else if (sData.status === 'failed') {
                clearInterval(poller);
                setFineTuningStatus(`Fine-tuning failed: ${sData.error}`);
                showToast('Fine-Tuning Failed');
              } else if (sData.status === 'running') {
                setFineTuningStatus(`Training in progress... Step ${sData.step} (Trainable params: ${sData.trainable_params})`);
              }
            } catch (err) {
              clearInterval(poller);
            }
          }, 1000);
        } else {
          setFineTuningStatus(data.message || 'Could not start fine-tuning');
        }
      } catch (err) {
        setFineTuningStatus('Network error: Could not reach backend fine-tuning endpoint');
      }
    };

    const handleActivateModule = async (m) => {
      const needsCodeTrust = m.type === 'plugin' && !String(m.download_url || '').startsWith('builtin://');
      if (needsCodeTrust && !window.confirm(
        `Activate third-party plugin “${m.name}”?\n\nThis runs Python code inside the Triune process with the same permissions as the current user. The static safety checks are not a sandbox. Only continue if you have reviewed and trust this source.`
      )) return;
      setActiveActivatingModId(m.id);
      showToast(`Activating ${m.name}...`);
      try {
        const res = await apiFetch('/v1/modules/activate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: m.id, confirm_untrusted: needsCodeTrust })
        });
        const data = await res.json();
        if (data.status === 'success') {
          showToast(data.message);
          await fetchInstalledModules();
          refreshNodeCatalog();
          if (data.registered_nodes && data.registered_nodes.length > 0) {
            showToast(`Added ${data.registered_nodes.join(', ')} to DAG Canvas!`);
          }
          if (data.type === 'dataset') {
            handleSelectDataset(m);
          } else if (data.type === 'adapter') {
            const adp = data.active_adapter || { name: m.name, rank: 16, path: m.installed_at };
            setActiveAdapter(adp);
            syncStateToDAGNodes({ adapter: adp });
          } else if (data.type === 'checkpoint') {
            fetchCheckpoints();
          }
        } else {
          showToast(data.message || 'Activation failed');
        }
      } catch (err) {
        showToast(`Could not activate ${m.name}: ${err.message || 'backend request failed'}`);
      } finally {
        setActiveActivatingModId(null);
      }
    };

    const handlePurgeVRAM = async () => {
      try {
        const res = await apiFetch('/v1/vram/offload', { method: 'POST' });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data.status !== 'purged') {
          throw new Error(data.message || `Cache clear failed (${res.status})`);
        }
        showToast('Cleared unused GPU cache. Memory held by active models is unchanged.');
      } catch (err) {
        showToast(`Could not clear GPU cache: ${err.message || 'backend request failed'}`);
      }
    };

    const handleExecuteNotebook = async () => {
      setIsExecutingNotebook(true);
      setNotebookOutput('Sending code to backend PythonSandbox...');
      try {
        const res = await apiFetch('/v1/sandbox/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: notebookCode })
        });
        const data = await res.json();
        if (data.success) {
          setNotebookOutput(`[PythonSandbox Result - ${data.exec_time_sec}s]\n${data.output}`);
        } else {
          setNotebookOutput(`[PythonSandbox Error]\n${data.error}`);
        }
      } catch (err) {
        setNotebookOutput(`[PythonSandbox Error]\n${err.message || 'Could not reach the local Python runner.'}`);
      }
      setIsExecutingNotebook(false);
    };

    const testBYOKKey = async (provider) => {
      setByokStatus(prev => ({ ...prev, [provider]: 'Testing key...' }));
      try {
        const res = await apiFetch('/v1/byok/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ provider, key: byokKeys[provider] || '' })
        });
        const data = await res.json();
        if (data.status === 'valid') {
          setByokStatus(prev => ({ ...prev, [provider]: 'Connection verified' }));
          showToast(data.message || `${provider.toUpperCase()} connection verified.`);
        } else {
          setByokStatus(prev => ({ ...prev, [provider]: data.message || 'Verification failed' }));
        }
      } catch (err) {
        setByokStatus(prev => ({ ...prev, [provider]: `Test failed: ${err.message || 'backend unavailable'}` }));
      }
    };

    const handleSaveBYOK = async () => {
      try {
        const res = await apiFetch('/v1/byok/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ keys: byokKeys })
        });
        const data = await res.json();
        if (!res.ok || data.status !== 'saved') {
          throw new Error(data.detail || data.message || `Save failed (${res.status})`);
        }
        setByokConfigured(prev => ({
          ...prev,
          ...Object.fromEntries(Object.entries(byokKeys).filter(([, value]) => value.trim()).map(([provider]) => [provider, true]))
        }));
        setByokKeys({ openai: '', anthropic: '', gemini: '', huggingface: '', wandb: '' });
        showToast(data.message || 'Provider credentials saved.');
      } catch (e) {
        showToast(`Could not save provider credentials: ${e.message || 'backend unavailable'}`);
      }
    };

    return e('div', {
      className: `react-studio-app theme-${studioTheme}`,
      style: customAccent ? { '--user-accent': customAccent } : undefined,
      onMouseMove: handleMouseMove,
      onMouseUp: handleMouseUp
    },
      // Toast Banner
      statusToast && e('div', { className: `pill ${/\b(error|failed|failure|unable|could not|invalid|unavailable|denied|rejected|refused)\b/i.test(String(statusToast)) ? 'warning' : 'online'}`, role: 'status', 'aria-live': 'polite', style: { position: 'fixed', top: '16px', right: '20px', zIndex: 100, background: 'var(--bg-card)', border: '1px solid var(--border-dark)' } }, statusToast),

      // Custom Node Creation Modal
      showCustomNodeModal && e('div', { className: 'modal-overlay', onClick: () => setShowCustomNodeModal(false) },
        e('div', { className: 'modal-box', onClick: ev => ev.stopPropagation() },
          e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', marginBottom: '12px' } }, 'Create & Register Custom Node'),
          e('div', { className: 'field-group' },
            e('label', null, 'Node Title:'),
            e('input', { value: customNodeTitle, placeholder: 'e.g. Custom Loss Layer Node', onChange: ev => setCustomNodeTitle(ev.target.value) })
          ),
          e('div', { className: 'field-group', style: { marginTop: '10px' } },
            e('label', null, 'Node Type Category:'),
            e('select', { value: customNodeType, onChange: ev => setCustomNodeType(ev.target.value) },
              e('option', { value: 'Data' }, 'Data Loader Node'),
              e('option', { value: 'Model' }, 'Model / Layer Node'),
              e('option', { value: 'Optimizer' }, 'Optimizer Node'),
              e('option', { value: 'Export' }, 'Exporter Node')
            )
          ),
          e('div', { className: 'field-group', style: { marginTop: '10px' } },
            e('label', null, 'Parameter Configuration / Description:'),
            e('textarea', {
              style: { height: '80px', width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-color)', color: 'var(--text-main)', padding: '8px', borderRadius: '4px' },
              value: customNodeDetails,
              placeholder: 'e.g. Custom PyTorch Module\nParam: lr=1e-4',
              onChange: ev => setCustomNodeDetails(ev.target.value)
            })
          ),
          e('div', { style: { display: 'flex', gap: '10px', marginTop: '16px', justifyContent: 'flex-end' } },
            e('button', { className: 'btn-sec', onClick: () => setShowCustomNodeModal(false) }, 'Cancel'),
            e('button', { className: 'btn-send', onClick: handleCreateCustomNode }, 'Register Custom Node')
          )
        )
      ),

      // Node Library Catalog Modal
      showNodeLibraryModal && e('div', { className: 'node-library-modal-overlay', onClick: () => setShowNodeLibraryModal(false) },
        e('div', { className: 'node-library-modal', onClick: ev => ev.stopPropagation() },
          e('div', { className: 'node-library-header' },
            e('div', null,
              e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', margin: 0 } }, 'Triune Architectural Node Library'),
              e('div', { style: { fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' } }, 'Backend-registered operators, specifications, and extensions')
            ),
            e('button', {
              className: 'node-delete-btn',
              style: { padding: '4px 8px', background: 'transparent', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center' },
              onClick: () => setShowNodeLibraryModal(false)
            }, renderIcon('close', 14))
          ),
          e('div', { className: 'node-library-search-bar' },
            e('input', {
              type: 'text',
              placeholder: 'Search nodes by name, description, or component (e.g. Muon, FP8, GLA, MoE, LoRA, Streaming)...',
              value: nodeSearchQuery,
              onChange: ev => setNodeSearchQuery(ev.target.value),
              style: { flex: 1, minWidth: '240px', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)', fontSize: '13px' }
            }),
            e('div', { style: { display: 'flex', gap: '6px', flexWrap: 'wrap' } },
              ['All', ...new Set((nodeCatalog || BUILTIN_NODE_CATALOG).map(item => item.category).filter(Boolean))].map(cat =>
                e('button', {
                  key: cat,
                  className: `preset-chip ${selectedNodeCategory === cat ? 'active-preset' : ''}`,
                  onClick: () => setSelectedNodeCategory(cat)
                }, cat)
              )
            )
          ),
          e('div', { className: 'node-library-grid' },
            (nodeCatalog || BUILTIN_NODE_CATALOG)
              .filter(item => {
                const matchesCat = selectedNodeCategory === 'All' || item.category === selectedNodeCategory;
                const q = nodeSearchQuery.toLowerCase().trim();
                const matchesQuery = !q || item.name.toLowerCase().includes(q) || item.title.toLowerCase().includes(q) || item.description.toLowerCase().includes(q);
                return matchesCat && matchesQuery;
              })
              .map(item =>
                e('div', { key: item.name, className: 'node-catalog-card', 'data-category': item.category },
                  e('div', null,
                    e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' } },
                      e('div', { style: { fontWeight: 700, fontSize: '14px', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '6px' } },
                        item.title,
                        (item.execution_type === 'execution' || item.real_computation)
                          ? e('span', { style: { fontSize: '9px', background: '#e3ebd8', color: '#2b4c3f', padding: '1px 5px', borderRadius: '4px', fontWeight: 600 } }, '● EXECUTION')
                          : e('span', { style: { fontSize: '9px', background: '#faeed7', color: '#78350f', padding: '1px 5px', borderRadius: '4px', fontWeight: 600 } }, '○ SPEC')
                      ),
                      e('span', { className: 'node-type-tag', style: { fontSize: '10px' } }, item.category)
                    ),
                    e('p', { style: { fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.45', margin: '4px 0 8px 0' } }, item.description),
                    e('div', { className: 'node-catalog-ports' },
                      item.inputs && item.inputs.length > 0 && e('span', null, `In: [${item.inputs.map(port => typeof port === 'string' ? port : port.name).join(', ')}]`),
                      item.outputs && item.outputs.length > 0 && e('span', null, `Out: [${item.outputs.map(port => typeof port === 'string' ? port : port.name).join(', ')}]`)
                    )
                  ),
                  e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', paddingTop: '10px', borderTop: '1px solid #f0ede6' } },
                    e('span', { style: { fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-dim)' } }, item.name),
                    e('button', {
                      className: 'btn-action',
                      style: { fontSize: '11.5px', padding: '4px 10px', background: 'var(--accent-olive, #2b4c3f)', color: '#fff' },
                      onClick: () => handleSpawnCatalogNode(item.name)
                    }, '+ Add to Canvas')
                  )
                )
              )
          )
        )
      ),

      showDAGPresetLibrary && e('div', { className: 'node-library-modal-overlay', onClick: () => setShowDAGPresetLibrary(false) },
        e('div', { className: 'node-library-modal dag-preset-modal', onClick: ev => ev.stopPropagation() },
          e('div', { className: 'node-library-header' },
            e('div', null,
              e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', margin: 0 } }, 'Workflow starters by goal'),
              e('div', { style: { fontSize: '12px', color: 'var(--text-muted)', marginTop: '3px' } }, 'These are editable graphs of real Triune operations. Check each workflow’s prerequisites before running it.')
            ),
            e('button', {
              className: 'node-delete-btn',
              style: { padding: '4px 8px', background: 'transparent', border: 'none', cursor: 'pointer' },
              onClick: () => setShowDAGPresetLibrary(false),
              title: 'Close workflow presets'
            }, renderIcon('close', 14))
          ),
          e('div', { className: 'dag-preset-grid' },
            Object.entries(DAG_PRESETS).map(([key, preset]) => e('article', { key, className: 'dag-preset-card' },
              e('div', { className: 'dag-preset-card-meta' },
                e('span', null, `${preset.nodes.length} nodes`),
                e('span', null, `${preset.edges.length} connections`),
                key === 'local_corpus_qa' && e('span', null, 'No model / GPU needed')
              ),
              e('h4', null, preset.name),
              e('p', null, preset.description),
              e('small', null, `Before running: ${preset.requirements}`),
              e('button', {
                className: 'btn-action',
                onClick: () => handleLoadDAGPreset(key),
                title: `Load ${preset.name}`
              }, 'Load workflow')
            ))
          )
        )
      ),

      // Workspace Plugin Scaffolding Modal
      showWorkspacePluginModal && e('div', { className: 'node-library-modal-overlay', onClick: () => setShowWorkspacePluginModal(false) },
        e('div', { className: 'node-library-modal', style: { maxWidth: '520px' }, onClick: ev => ev.stopPropagation() },
          e('div', { className: 'node-library-header' },
            e('div', null,
              e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', margin: 0 } }, 'Scaffold Custom DAG Plugin Script'),
              e('div', { style: { fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' } }, 'Generates a ready-to-run Python plugin script in triune/plugins/ with @register_node schema')
            ),
            e('button', {
              className: 'node-delete-btn',
              style: { padding: '4px 8px', background: 'transparent', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center' },
              onClick: () => setShowWorkspacePluginModal(false)
            }, renderIcon('close', 14))
          ),
          e('div', { style: { padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: '14px' } },
            e('div', { className: 'field-group' },
              e('label', null, 'Plugin Class Name:'),
              e('input', {
                type: 'text',
                placeholder: 'e.g. AdaptiveCurvatureLoss or QuantumAttention',
                value: newPluginName,
                onChange: ev => setNewPluginName(ev.target.value),
                style: { width: '100%', height: '38px', padding: '0 10px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px' }
              })
            ),
            e('div', { className: 'field-group' },
              e('label', null, 'Component Category:'),
              e('select', {
                value: newPluginCategory,
                onChange: ev => setNewPluginCategory(ev.target.value),
                style: { width: '100%', height: '38px', padding: '0 10px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px' }
              },
                e('option', { value: 'Loss' }, 'Loss Function (e.g. Joint loss, Curvature loss)'),
                e('option', { value: 'Model' }, 'Model Architecture (e.g. Attention, FFN, MoE)'),
                e('option', { value: 'Optimizer' }, 'Optimizer / Scheduler (e.g. Muon, GaLore variant)'),
                e('option', { value: 'Data' }, 'Data Loader / Streamer (e.g. Custom reader)'),
                e('option', { value: 'Runtime' }, 'Runtime & Memory Management')
              )
            ),
            e('div', { style: { background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: '6px', border: '1px solid var(--border-color)', fontSize: '12px', color: 'var(--text-muted)' } },
              'Once generated, your new plugin will be saved to ',
              e('code', { style: { fontFamily: 'monospace', color: 'var(--primary)' } }, 'triune/plugins/'),
              '. Review the generated code, then scan the workspace to register it.'
            ),
            e('div', { style: { display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '10px' } },
              e('button', { className: 'btn-sec', onClick: () => setShowWorkspacePluginModal(false) }, 'Cancel'),
              e('button', {
                className: 'btn-send',
                style: { background: 'var(--accent-olive, #2b4c3f)', display: 'inline-flex', alignItems: 'center' },
                onClick: handleCreatePluginScript
              }, [renderIcon('plus', 12, { marginRight: '5px' }), 'Create Plugin'])
            )
          )
        )
      ),

      // Sidebar Navigation
      e('aside', { className: `react-sidebar ${sidebarCollapsed ? 'collapsed' : ''}` },
        e('div', { className: 'react-brand' },
          e('div', { className: 'brand-logo-icon' }, 'T'),
          !sidebarCollapsed && e('div', { className: 'brand-title-wrap' },
            e('span', { className: 'brand-title' }, 'Triune Studio'),
            e('span', { className: 'brand-sub' }, 'Model workspace')
          )
        ),
        e('nav', { className: 'react-nav', 'aria-label': 'Studio sections' },
          e('button', { className: `nav-item ${activeTab === 'chat' ? 'active' : ''}`, title: 'Chat & Playground', onClick: () => setActiveTab('chat') },
            e('span', { className: 'nav-icon' }, renderIcon('chat', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'Chat & Playground')
          ),
          e('button', { className: `nav-item ${activeTab === 'nodegraph' ? 'active' : ''}`, title: 'Visual Node Canvas', onClick: () => setActiveTab('nodegraph') },
            e('span', { className: 'nav-icon' }, renderIcon('nodes', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'Visual Node Canvas')
          ),
          e('button', { className: `nav-item ${activeTab === 'training' ? 'active' : ''}`, title: 'Training & Telemetry', onClick: () => setActiveTab('training') },
            e('span', { className: 'nav-icon' }, renderIcon('training', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'Training & Telemetry')
          ),
          e('button', { className: `nav-item ${activeTab === 'checkpoints' ? 'active' : ''}`, title: 'Checkpoints & Weights', onClick: () => { setActiveTab('checkpoints'); fetchCheckpoints(); } },
            e('span', { className: 'nav-icon' }, renderIcon('checkpoints', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'Checkpoints & Weights')
          ),
          e('button', { className: `nav-item ${activeTab === 'finetune' ? 'active' : ''}`, title: 'LoRA / QLoRA Tuner', onClick: () => setActiveTab('finetune') },
            e('span', { className: 'nav-icon' }, renderIcon('finetune', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'LoRA / QLoRA Tuner')
          ),
          e('button', { className: `nav-item ${activeTab === 'modules' ? 'active' : ''}`, title: 'Modules & Repos', onClick: () => { setActiveTab('modules'); fetchInstalledModules(); searchMarketplace(); } },
            e('span', { className: 'nav-icon' }, renderIcon('modules', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'Extensions Marketplace')
          ),
          e('button', { className: `nav-item ${activeTab === 'customize' ? 'active' : ''}`, title: 'Appearance Marketplace', onClick: () => setActiveTab('customize') },
            e('span', { className: 'nav-icon' }, renderIcon('sparkle', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'Appearance & Themes')
          ),
          e('button', { className: `nav-item ${activeTab === 'environment' ? 'active' : ''}`, title: 'System & Hardware', onClick: () => setActiveTab('environment') },
            e('span', { className: 'nav-icon' }, renderIcon('hardware', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'System & Hardware')
          ),
          e('button', { className: `nav-item ${activeTab === 'models' ? 'active' : ''}`, title: 'Model Zoo & Exporters', onClick: () => setActiveTab('models') },
            e('span', { className: 'nav-icon' }, renderIcon('models', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'Model Zoo & Exporters')
          ),
          e('button', { className: `nav-item ${activeTab === 'datasets' ? 'active' : ''}`, title: 'Dataset & Tokenizer', onClick: () => setActiveTab('datasets') },
            e('span', { className: 'nav-icon' }, renderIcon('datasets', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'Dataset & Tokenizer')
          ),
          e('button', { className: `nav-item ${activeTab === 'notebook' ? 'active' : ''}`, title: 'Python Sandbox', onClick: () => setActiveTab('notebook') },
            e('span', { className: 'nav-icon' }, renderIcon('notebook', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'Python Sandbox')
          ),
          e('button', { className: `nav-item ${activeTab === 'byok' ? 'active' : ''}`, title: 'BYOK Subscriptions', onClick: () => setActiveTab('byok') },
            e('span', { className: 'nav-icon' }, renderIcon('byok', 17)),
            !sidebarCollapsed && e('span', { className: 'nav-label' }, 'BYOK Subscriptions')
          )
        ),
      ),

      // Main Content Area
      e('main', { className: 'react-main' },
        e('header', { className: 'react-topbar' },
          e('div', { className: 'topbar-left' },
            e('button', {
              className: 'sidebar-toggle-btn',
              onClick: () => setSidebarCollapsed(!sidebarCollapsed),
              title: sidebarCollapsed ? 'Expand Navigation Sidebar' : 'Collapse Navigation Sidebar',
              'aria-label': sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation',
              'aria-expanded': !sidebarCollapsed,
              style: { display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }
            }, sidebarCollapsed ? renderIcon('menu', 16) : renderIcon('arrowRight', 16, { transform: 'rotate(180deg)' })),
            e('div', { className: 'topbar-title-group' },
              e('h2', { className: 'topbar-title' }, TAB_TITLES[activeTab] || (activeTab.charAt(0).toUpperCase() + activeTab.slice(1).replace(/_/g, ' '))),
              e('span', { className: 'topbar-subtitle' }, TAB_DESCRIPTIONS[activeTab] || 'Triune model workspace')
            )
          ),


          // Universal Topbar Omnibar (Dataset Explorer, DAG Nodes & Modules)
          e('div', { className: 'topbar-omnibar-wrap' },
            e('div', { className: 'topbar-search-input-box' },
              e('span', { className: 'topbar-search-icon' }, renderIcon('search', 14)),
              e('input', {
                type: 'text',
                className: 'topbar-search-input',
                placeholder: 'Browse & search datasets (TinyStories, fineweb), DAG nodes, modules...',
                value: omnibarQuery,
                onChange: (ev) => {
                  setOmnibarQuery(ev.target.value);
                  setShowOmnibarDropdown(true);
                },
                onFocus: () => setShowOmnibarDropdown(true),
                onKeyDown: (ev) => {
                  if (ev.key === 'Escape') setShowOmnibarDropdown(false);
                }
              }),
              omnibarQuery && e('button', {
                className: 'topbar-search-clear',
                onClick: () => { setOmnibarQuery(''); setShowOmnibarDropdown(false); }
              }, renderIcon('close', 11))
            ),
            showOmnibarDropdown && (() => {
              const q = (omnibarQuery || '').toLowerCase().trim();
              const matchDatasets = POPULAR_DATASETS.filter(d => !q || d.name.toLowerCase().includes(q) || d.id.toLowerCase().includes(q) || (d.desc && d.desc.toLowerCase().includes(q))).slice(0, 4);
              const cat = nodeCatalog || BUILTIN_NODE_CATALOG;
              const matchNodes = cat.filter(n => !q || n.title.toLowerCase().includes(q) || n.name.toLowerCase().includes(q) || (n.description && n.description.toLowerCase().includes(q))).slice(0, 5);
              const matchPresets = Object.keys(DAG_PRESETS).filter(k => !q || DAG_PRESETS[k].name.toLowerCase().includes(q) || k.toLowerCase().includes(q)).slice(0, 3);
              const curMods = (marketplaceData && marketplaceData.curated) || [];
              const matchMods = curMods.filter(m => !q || m.name.toLowerCase().includes(q) || (m.description && m.description.toLowerCase().includes(q))).slice(0, 3);
              const hasResults = matchDatasets.length > 0 || matchNodes.length > 0 || matchPresets.length > 0 || matchMods.length > 0;

              return e('div', { className: 'topbar-omnibar-dropdown' },
                e('div', { style: { display: 'flex', justifyContent: 'space-between', padding: '4px 14px 6px 14px', borderBottom: '1px solid var(--border-color)', fontSize: '11px', color: 'var(--text-muted)' } },
                  e('span', null, q ? `Search results for "${omnibarQuery}"` : [renderIcon('bolt', 12, { marginRight: '6px' }), 'Quick Jump & Stream Hub']),
                  e('button', { style: { background: 'none', border: 'none', cursor: 'pointer', fontSize: '11px', color: 'var(--text-dim)', display: 'inline-flex', alignItems: 'center', gap: '3px' }, onClick: () => setShowOmnibarDropdown(false) }, ['Close ', renderIcon('close', 10)])
                ),
                matchDatasets.length > 0 && e('div', null,
                  e('div', { className: 'omnibar-section-title' }, [renderIcon('datasets', 13, { marginRight: '6px' }), 'STREAMING DATASETS (Hugging Face / Web)']),
                  matchDatasets.map(ds =>
                    e('div', { key: ds.id, className: 'omnibar-item' },
                      e('div', { className: 'omnibar-item-left' },
                        e('span', { className: 'omnibar-item-icon' }, renderIcon('datasets', 16)),
                        e('div', null,
                          e('div', { className: 'omnibar-item-name' }, ds.name),
                          e('div', { className: 'omnibar-item-desc' }, `${ds.id} • ${ds.desc || 'Zero-disk stream'}`)
                        )
                      ),
                      e('div', { className: 'omnibar-item-actions' },
                        e('button', {
                          className: 'omnibar-btn primary',
                          title: 'Stream into PyTorch CUDA backward graph immediately',
                          onClick: () => {
                            setHfDatasetInput(ds.id);
                            setHfConfigInput(ds.config || '');
                            setHfSplitInput(ds.split || 'train');
                            setHfTextColInput(ds.col || 'text');
                            handleConnectHfStream(ds.id, ds.config || '', ds.split || 'train', ds.col || 'text');
                            setShowOmnibarDropdown(false);
                          }
                        }, [renderIcon('play', 11, { marginRight: '4px' }), 'Stream in Engine']),
                        e('button', {
                          className: 'omnibar-btn',
                          title: 'Drop streaming node onto DAG visual canvas',
                          onClick: () => {
                            handleAddDatasetNodeToDAG(ds);
                            setShowOmnibarDropdown(false);
                          }
                        }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Add to DAG'])
                      )
                    )
                  )
                ),
                matchNodes.length > 0 && e('div', null,
                  e('div', { className: 'omnibar-section-title' }, [renderIcon('nodes', 13, { marginRight: '6px' }), 'DAG ARCHITECTURE NODES']),
                  matchNodes.map(node =>
                    e('div', { key: node.name, className: 'omnibar-item' },
                      e('div', { className: 'omnibar-item-left' },
                        e('span', { className: 'omnibar-item-icon' }, renderIcon('nodes', 16)),
                        e('div', null,
                          e('div', { className: 'omnibar-item-name' }, node.title),
                          e('div', { className: 'omnibar-item-desc' }, node.description || '')
                        )
                      ),
                      e('div', { className: 'omnibar-item-actions' },
                        e('button', {
                          className: 'omnibar-btn primary',
                          onClick: () => {
                            handleSpawnCatalogNode(node.name);
                            setActiveTab('nodegraph');
                            setShowOmnibarDropdown(false);
                          }
                        }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Add to DAG'])
                      )
                    )
                  )
                ),
                matchPresets.length > 0 && e('div', null,
                  e('div', { className: 'omnibar-section-title' }, [renderIcon('training', 13, { marginRight: '6px' }), 'DAG ARCHITECTURE PRESETS']),
                  matchPresets.map(pkey =>
                    e('div', { key: pkey, className: 'omnibar-item' },
                      e('div', { className: 'omnibar-item-left' },
                        e('span', { className: 'omnibar-item-icon' }, renderIcon('models', 16)),
                        e('div', null,
                          e('div', { className: 'omnibar-item-name' }, DAG_PRESETS[pkey].name),
                          e('div', { className: 'omnibar-item-desc' }, `${DAG_PRESETS[pkey].nodes.length} nodes • ${DAG_PRESETS[pkey].edges.length} wires`)
                        )
                      ),
                      e('div', { className: 'omnibar-item-actions' },
                        e('button', {
                          className: 'omnibar-btn primary',
                          onClick: () => {
                            handleLoadDAGPreset(pkey);
                            setActiveTab('nodegraph');
                            setShowOmnibarDropdown(false);
                          }
                        }, 'Load Preset')
                      )
                    )
                  )
                ),
                matchMods.length > 0 && e('div', null,
                  e('div', { className: 'omnibar-section-title' }, [renderIcon('modules', 13, { marginRight: '6px' }), 'MODULES & EXTENSIONS']),
                  matchMods.map(mod =>
                    e('div', { key: mod.id, className: 'omnibar-item' },
                      e('div', { className: 'omnibar-item-left' },
                        e('span', { className: 'omnibar-item-icon' }, renderIcon('modules', 16)),
                        e('div', null,
                          e('div', { className: 'omnibar-item-name' }, mod.name),
                          e('div', { className: 'omnibar-item-desc' }, mod.description || '')
                        )
                      ),
                      e('div', { className: 'omnibar-item-actions' },
                        e('button', {
                          className: 'omnibar-btn primary',
                          onClick: () => {
                            handleAddModuleNodeToDAG(mod);
                            setShowOmnibarDropdown(false);
                          }
                        }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Add to DAG'])
                      )
                    )
                  )
                ),
                !hasResults && e('div', { style: { padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' } },
                  `No matching items found for "${omnibarQuery}". Try searching "TinyStories", "fineweb", "MoE", "Muon", or "Centroid".`
                )
              );
            })()
          ),

          e('div', { className: 'topbar-pills' },
            e('button', {
              className: 'appearance-quick-toggle',
              type: 'button',
              title: /dark|forest/.test(studioTheme) ? 'Switch to light mode' : 'Switch to dark mode',
              'aria-label': /dark|forest/.test(studioTheme) ? 'Switch to light mode' : 'Switch to dark mode',
              onClick: () => {
                const isDark = /dark|forest/.test(studioTheme);
                const edition = studioTheme.startsWith('classic') ? 'classic' : 'modern';
                setStudioTheme(`${edition}-${isDark ? 'light' : 'dark'}`);
              }
            }, [/dark|forest/.test(studioTheme) ? '☀' : '☾', /dark|forest/.test(studioTheme) ? 'Light' : 'Dark']),
            e('span', {
              className: `pill ${backendStatus === 'online' ? 'online' : (backendStatus === 'offline' ? 'warning' : '')}`,
              role: 'status',
              'aria-live': 'polite',
              title: backendStatus === 'online' ? 'Studio API is responding' : 'Studio API connection state'
            }, backendStatus === 'online' ? '● API online' : (backendStatus === 'offline' ? '● API offline' : '◌ Connecting…')),

            // Engine Status Chip
            e('span', {
              className: `pill ${backendStatus !== 'online' ? 'warning' : (isTraining ? 'online' : ((metrics && metrics.step > 0) ? 'warning' : ''))}`,
              title: backendStatus !== 'online' ? 'Engine status unavailable until the API reconnects' : `Engine: ${isTraining ? 'Running' : ((metrics && metrics.step > 0) ? 'Paused' : 'Idle')}`
            }, `● ${backendStatus !== 'online' ? 'Unavailable' : (isTraining ? 'Running' : ((metrics && metrics.step > 0) ? 'Paused' : 'Idle'))}`),

            // Step & Loss Telemetry
            (metrics && metrics.step > 0) && e('span', { className: 'pill', title: 'Global Step' }, `Step: ${metrics.step.toLocaleString()}`),
            (metrics && metrics.step > 0 && metrics.loss !== null && metrics.loss !== undefined) && e('span', {
              className: 'pill',
              style: { color: 'var(--accent-terracotta, #9a3412)', fontWeight: 600 },
              title: 'Total Loss'
            }, `Loss: ${Number(metrics.loss).toFixed(4)}`),
            (metrics && metrics.step > 0 && metrics.throughput > 0) && e('span', { className: 'pill', title: 'Training Throughput', style: { display: 'inline-flex', alignItems: 'center' } }, [renderIcon('bolt', 11, { marginRight: '4px' }), `${metrics.throughput} tok/s`]),

            // VRAM Meter Pill
            e('div', {
              className: `pill ${vramUsage.oom_risk ? 'warning' : ''}`,
              style: { display: 'flex', alignItems: 'center', gap: '6px' },
              title: `${systemDiagnostics.device_name || 'GPU'} VRAM used: ${vramUsage.allocated} GB${vramUsage.total > 0 ? ` / ${vramUsage.total} GB` : ' (capacity unavailable)'}`
            },
              e('span', { style: { fontSize: '11px', color: 'var(--text-muted)' } }, 'VRAM:'),
              e('span', { style: { fontWeight: 600 } }, `${vramUsage.allocated}G`),
              e('div', {
                style: {
                  width: '38px',
                  height: '6px',
                  background: 'var(--border-dark, #c9c3b4)',
                  borderRadius: '3px',
                  overflow: 'hidden'
                }
              },
                e('div', {
                  style: {
                    width: `${vramUsage.total > 0 ? Math.min(100, Math.max(0, (vramUsage.allocated / vramUsage.total) * 100)) : 0}%`,
                    height: '100%',
                    background: vramUsage.oom_risk ? '#dc2626' : '#2b4c3f',
                    borderRadius: '3px',
                    transition: 'width 0.3s ease'
                  }
                })
              ),
              e('button', {
                onClick: handlePurgeVRAM,
            title: 'Clear unused GPU cache; active model memory is not released',
                style: {
                  background: 'rgba(154, 52, 18, 0.08)',
                  border: '1px solid rgba(154, 52, 18, 0.25)',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  padding: '1px 5px',
                  fontSize: '10px',
                  color: 'var(--accent-terracotta, #9a3412)',
                  fontWeight: 600,
                  marginLeft: '2px'
                }
              }, 'Purge')
            ),

            // Model & Config Pills
            e('span', {
              className: 'pill',
              style: { cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: '5px', borderColor: 'var(--accent-olive, #2b4c3f)', color: 'var(--accent-olive, #2b4c3f)', fontWeight: 600 },
              title: 'Click to open Architecture Zoo & Model Presets',
              onClick: () => setActiveTab('models')
            }, [renderIcon('models', 12), `Model: ${activeModel}`]),
            e('span', { className: 'pill' }, `Precision: ${trainPrecision || precision}`)
          )
        ),

        backendStatus !== 'online' && e('div', {
          className: `backend-connect-banner ${backendStatus}`,
          role: backendStatus === 'offline' ? 'alert' : 'status',
          'aria-live': 'polite'
        },
          e('div', null,
            e('strong', null, backendStatus === 'offline' ? 'Cannot reach the Triune backend.' : 'Connecting to the Triune backend…'),
            backendStatus === 'offline' && e('span', null, ' Start or restart Studio’s backend; API features will be unavailable until it reconnects.')
          ),
          backendStatus === 'offline' && e('button', {
            type: 'button',
            onClick: () => {
              setBackendStatus('checking');
              setBackendRetry(retry => retry + 1);
            }
          }, 'Retry connection')
        ),

        e('div', { className: 'react-view-container' },
          // Tab 1: Chat & Playground
          activeTab === 'chat' && e('div', { className: 'view-chat' },
            e('div', { className: 'preset-bar', style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' } },
              e('div', { style: { display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '6px' } },
                e('span', { className: 'preset-title' }, 'System Persona:'),
                SYSTEM_PROMPT_PRESETS.map((p, idx) =>
                  e('button', {
                    key: idx,
                    className: `preset-chip ${systemPrompt === p.prompt ? 'active' : ''}`,
                    onClick: () => { setSystemPrompt(p.prompt); showToast(`Applied Persona: ${p.name}`); }
                  }, p.name)
                )
              ),
              e('button', { className: 'btn-clear-chat', onClick: handleClearChat, title: 'Clear conversation history' }, [renderIcon('trash', 13, { marginRight: '6px' }), 'Clear Chat'])
            ),
            e('div', { className: 'chat-scroll-area' },
              messages.map((m, idx) =>
                e('div', { key: idx, className: `chat-bubble ${m.isAssistant ? 'assistant' : 'user'}` },
                  e('div', { className: 'bubble-header' },
                    e('span', { className: 'bubble-sender' }, m.sender),
                    e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                      e('span', { className: 'bubble-time' }, m.time),
                      e('button', {
                        className: 'btn-copy-bubble',
                        title: 'Copy message text',
                        onClick: () => handleCopyMessage(m.text)
                      }, [renderIcon('copy', 12, { marginRight: '4px' }), 'Copy'])
                    )
                  ),
                  e('div', { className: 'bubble-text', style: { whiteSpace: 'pre-wrap', lineHeight: '1.6' } }, m.text),
                  m.telemetry && e('div', { className: 'bubble-telemetry-badge' },
                    e('span', null, renderIcon('bolt', 11, { marginRight: '3px' }), m.telemetry.speed || 'Direct'),
                    e('span', null, `Route: ${m.telemetry.route}`),
                    e('span', null, `VRAM: ${m.telemetry.vram}`),
                    e('span', null, `Latency: ${m.telemetry.latency}`),
                    m.telemetry.ttft && e('span', null, `TTFT: ${m.telemetry.ttft}`),
                    m.telemetry.decodeSpeed && e('span', null, m.telemetry.decodeSpeed),
                    m.telemetry.device && e('span', null, `Device: ${m.telemetry.device}`),
                    m.telemetry.tokens_count !== undefined && e('span', null, `${m.telemetry.tokens_count} toks`)
                  )
                )
              ),
              isGenerating && e('div', { className: 'chat-bubble assistant generating' },
                e('div', { className: 'bubble-header' },
                  e('span', { className: 'bubble-sender' }, activeModel.includes('gpt') || activeModel.includes('claude') || activeModel.includes('gemini') ? activeModel.toUpperCase() : 'Triune MoE Engine'),
                  e('span', { className: 'bubble-time' }, 'Synthesizing...')
                ),
                e('div', { className: 'bubble-text', style: { fontStyle: 'italic', color: 'var(--text-muted)' } },
                  activeModel.startsWith('triune')
                    ? (!weightsLoaded
                      ? 'Loading and validating trained weights, then generating locally. First load can take a while.'
                      : (!systemDiagnostics.cuda_available
                        ? 'Generating locally on CPU. This is functional but much slower than GPU inference.'
                      : 'Generating locally on GPU. Tokens will appear as they are produced.'))
                    : 'Waiting for the selected provider. The response appears when this request finishes.'
                )
              ),
              e('div', { ref: chatBottomRef })
            ),
            e('div', { className: 'chat-controls-box' },
              e('div', { className: 'route-select-row' },
                e('div', null,
                  e('label', { style: { marginRight: '8px', fontWeight: 600 } }, 'Model:'),
                  e('select', {
                    value: activeModel,
                    onChange: ev => {
                      const val = ev.target.value;
                      if (val.startsWith('triune-')) {
                        handleSelectModelPreset(val);
                      } else {
                        setActiveModel(val);
                        showToast(`Active BYOK Model: ${val}`);
                      }
                    }
                  },
                    MODEL_PRESETS.map(p =>
                      e('option', { key: p.id, value: p.id }, p.name)
                    ),
                    e('option', { value: 'gpt-4o-mini' }, 'GPT-4o Mini (OpenAI BYOK)'),
                    e('option', { value: 'claude-3-5-sonnet-latest' }, 'Claude 3.5 Sonnet (Anthropic BYOK)'),
                    e('option', { value: 'gemini-2.5-flash' }, 'Gemini 2.5 Flash (Google BYOK)'),
                    e('option', { value: 'gemini-1.5-flash' }, 'Gemini 1.5 Flash (Google BYOK)'),
                    e('option', { value: 'gemini-1.5-pro' }, 'Gemini 1.5 Pro (Google BYOK)')
                  )
                ),
                activeModel.startsWith('triune') ? e('div', null,
                  e('span', {
                    className: `pill ${weightsLoaded ? 'online' : 'warning'}`,
                    title: weightsLoaded
                      ? `Trained weights loaded${activeCheckpointName ? `: ${activeCheckpointName}` : ''}`
                      : 'No trained weights are currently loaded; local inference requires a checkpoint or a completed training step.'
                  }, weightsLoaded ? `Weights: ${activeCheckpointName || 'in memory'}` : 'No trained weights loaded'),
                  !systemDiagnostics.cuda_available && e('span', {
                    className: 'pill warning',
                    title: 'This Studio backend is using CPU-only PyTorch. API online means the server is reachable; it does not mean local GPU inference is active.'
                  }, 'CPU inference · slow'),
                  e('label', { style: { marginRight: '8px', fontWeight: 600 } }, 'Exit Route:'),
                  e('select', { value: route, onChange: ev => setRoute(ev.target.value) },
                    e('option', { value: 'auto' }, 'Auto (Adaptive Layer Exit)'),
                    e('option', { value: 'reflex' }, 'Reflex (Shallow Exit - Fast)'),
                    e('option', { value: 'limbic' }, 'Limbic (Mid Depth Exit)'),
                    e('option', { value: 'cortex' }, 'Cortex (Full Layer Depth)')
                  )
                ) : e('div', null,
                  e('span', { className: 'pill', style: { background: '#faeed7', color: '#78350f', borderColor: '#edd5a6' } }, 'BYOK API Mode')
                ),
                activeAdapter && e('div', null,
                  e('span', { className: 'pill online', title: `Trained LoRA Adapter: ${activeAdapter.name}` }, renderIcon('finetune', 12, { marginRight: '4px' }), activeAdapter.name)
                ),
                e('div', null,
                  e('label', { style: { marginRight: '8px' } }, `Temp: ${temperature}`),
                  e('input', { type: 'range', min: 0.1, max: 1.5, step: 0.1, value: temperature, onChange: ev => setTemperature(parseFloat(ev.target.value)) })
                ),
                e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                  e('label', { style: { marginRight: '4px' } }, `Max Tokens:`),
                  e('input', { type: 'range', min: 64, max: 8192, step: 64, value: maxTokens, onChange: ev => setMaxTokens(parseInt(ev.target.value, 10)) }),
                  e('input', {
                    type: 'number',
                    min: 1,
                    max: 8192,
                    value: maxTokens,
                    style: { width: '60px', height: '22px', textAlign: 'center', fontFamily: 'var(--font-mono)', fontSize: '11px', background: 'var(--bg-input, #f2eee5)', border: '1px solid var(--border-color, #ded9cd)', borderRadius: '3px' },
                    onChange: ev => setMaxTokens(Math.min(8192, Math.max(1, parseInt(ev.target.value, 10) || 1)))
                  })
                )
              ),
              e('div', { className: 'chat-input-row' },
                e('textarea', {
                  placeholder: isGenerating ? 'Generating response…' : 'Ask a research question or input prompt...',
                  value: chatInput,
                  disabled: isGenerating,
                  onChange: ev => setChatInput(ev.target.value),
                  onKeyDown: ev => { if (ev.key === 'Enter' && !ev.shiftKey && !isGenerating) { ev.preventDefault(); handleSendMessage(); } }
                }),
                !isGenerating && messages.some(m => !m.isAssistant && m.sender !== 'System') && e('button', {
                  className: 'btn-regen',
                  title: 'Regenerate last response',
                  onClick: handleRegenerateLast
                }, [renderIcon('refresh', 13, { marginRight: '5px' }), 'Retry']),
                isGenerating ? e('button', {
                  className: 'btn-stop',
                  onClick: handleStopGeneration
                }, [renderIcon('pause', 13, { marginRight: '5px' }), 'Stop']) : e('button', {
                  className: 'btn-send',
                  disabled: !chatInput.trim(),
                  style: !chatInput.trim() ? { opacity: 0.5, cursor: 'not-allowed' } : {},
                  onClick: () => handleSendMessage()
                }, 'Send Prompt')
              )
            )
          ),

          // Tab 2: Training Dashboard & Live Telemetry
          activeTab === 'training' && e('div', { className: 'view-training' },
            // Active Architecture & Model Preset Switcher Banner
            e('div', {
              style: {
                padding: '12px 18px',
                marginBottom: '12px',
                background: 'var(--bg-card, #fcfbfa)',
                borderRadius: '8px',
                border: '1px solid var(--border-color, #ded9cd)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '12px',
                boxShadow: '0 1px 3px rgba(0,0,0,0.03)'
              }
            },
              e('div', { style: { display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' } },
                e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                  e('span', { style: { display: 'flex', alignItems: 'center' } }, renderIcon('models', 18, { color: 'var(--accent-olive, #2b4c3f)' })),
                  e('span', { style: { fontFamily: 'Newsreader, Georgia, serif', fontSize: '17px', fontWeight: 600 } }, 'Active Architecture:'),
                  e('select', {
                    value: activeModel,
                    onChange: ev => handleSelectModelPreset(ev.target.value),
                    style: {
                      fontFamily: 'var(--font-mono, Consolas)',
                      fontSize: '13px',
                      fontWeight: 600,
                      padding: '4px 10px',
                      borderRadius: '6px',
                      background: 'var(--bg-surface, #ede9df)',
                      border: '1px solid var(--border-color, #ded9cd)',
                      color: 'var(--text-main)',
                      cursor: 'pointer'
                    }
                  },
                    MODEL_PRESETS.map(p => e('option', { key: p.id, value: p.id }, p.name))
                  )
                ),
                e('div', { style: { display: 'flex', gap: '6px', flexWrap: 'wrap', alignItems: 'center' } },
                  e('span', { className: 'model-spec-chip' }, `${datasetInfo.num_layers || 6} Layers`),
                  e('span', { className: 'model-spec-chip' }, `${datasetInfo.hidden_dim || 256} Dim`),
                  e('span', { className: 'model-spec-chip' }, `${datasetInfo.num_heads || 4} Heads`),
                  e('span', { className: 'model-spec-chip' }, `${datasetInfo.num_experts || 4} Experts`),
                  e('span', { className: 'model-spec-chip' }, `Depth: ${(datasetInfo.depth_mode || trainDepthMode).toUpperCase()}`)
                )
              ),
              e('div', { style: { display: 'flex', gap: '8px', alignItems: 'center' } },
                (wandbUrl || byokConfigured.wandb) && e('a', {
                  href: wandbUrl || 'https://wandb.ai',
                  target: '_blank',
                  rel: 'noreferrer',
                  className: 'btn-sec',
                  style: {
                    fontSize: '11.5px',
                    padding: '5px 12px',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '5px',
                    textDecoration: 'none',
                    borderColor: '#ffbe00',
                    color: '#d97706',
                    fontWeight: 600
                  }
                }, ['📊 WandB Dashboard ↗']),
                e('button', {
                  className: 'btn-sec',
                  style: { fontSize: '11.5px', padding: '5px 12px' },
                  onClick: () => setActiveTab('models')
                }, [renderIcon('models', 12, { marginRight: '5px' }), 'Open Model Zoo', renderIcon('arrowRight', 10, { marginLeft: '4px' })])
              )
            ),

            // Active Checkpoint Quick Status Bar
            e('div', { className: 'ckpt-card-header', style: { padding: '10px 16px', marginBottom: '0px' } },
              e('div', { style: { display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' } },
                e('span', { className: 'badge-ckpt-active' }, '● Active Model Weights'),
                e('span', { style: { fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 600 } },
                  activeCheckpointName || (metrics.step ? `triune_studio_step_${metrics.step}.pt` : 'No Weights Loaded')
                ),
                e('span', { style: { fontSize: '12px', color: 'var(--text-muted)' } },
                  `(Global Step ${metrics.step || 0})`
                )
              ),
              e('div', { style: { display: 'flex', gap: '8px' } },
                e('button', {
                  className: 'btn-sec',
                  style: { fontSize: '11.5px', padding: '5px 12px' },
                  onClick: () => { setActiveTab('checkpoints'); fetchCheckpoints(); }
                }, [renderIcon('checkpoints', 13, { marginRight: '6px' }), 'Checkpoint Browser', renderIcon('arrowRight', 11, { marginLeft: '5px' })])
              )
            ),
            // Hardware Tuner & GPU Offload Deck (LM Studio / Ollama Style)
            renderHardwareTunerDeck(),
            e('div', { className: 'metrics-cards-grid' },
              e('div', { className: 'card-stat' },
                e('div', { className: 'stat-label' }, 'Total Loss'),
                e('div', { className: 'stat-value' }, (metrics.loss !== undefined && metrics.loss !== null) ? Number(metrics.loss).toFixed(4) : '--'),
                e('div', { className: 'stat-sub' }, `Batch: ${trainBatchSize || datasetInfo.batch_size || 4} | Accum: ${trainGradAccum || datasetInfo.grad_accum_steps || 4}x | LR: ${Number(trainLR || 0.0005).toExponential(1)}`)
              ),
              e('div', { className: 'card-stat' },
                e('div', { className: 'stat-label' }, 'Global Steps'),
                e('div', { className: 'stat-value' }, metrics.step || 0),
                e('div', { className: 'stat-sub' }, `Tokens ingested: ${(datasetInfo.total_tokens || 0).toLocaleString()}`)
              ),
              e('div', { className: 'card-stat' },
                e('div', { className: 'stat-label' }, 'Exit Head Ratio'),
                e('div', { className: 'stat-value' }, (metrics.step && metrics.step > 0) ? `R:${exitUsage.reflex}% L:${exitUsage.limbic}% C:${exitUsage.cortex}%` : '--'),
                e('div', { className: 'stat-sub' }, `Mode: ${(trainDepthMode || datasetInfo.depth_mode || 'dynamic').toUpperCase()}`)
              ),
              e('div', { className: 'card-stat' },
                e('div', { className: 'stat-label' }, 'Throughput'),
                e('div', { className: 'stat-value' }, metrics.throughput ? `${metrics.throughput} tok/s` : '--'),
                e('div', { className: 'stat-sub' }, `${((trainBatchSize || datasetInfo.batch_size || 4) * (trainSeqLen || datasetInfo.seq_len || 64) * (trainGradAccum || datasetInfo.grad_accum_steps || 4)).toLocaleString()} tok/step`)
              )
            ),
            // Live Dataset Streaming Telemetry & Batch Ingestion Box
            e('div', { className: 'stream-ingestion-box' },
              e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' } },
                e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                  e('span', { className: 'stream-badge-pulse' }, isTraining ? 'Training from stream' : datasetInfo.name ? 'Stream ready · training idle' : 'No dataset connected'),
                  e('span', { style: { fontWeight: 600, fontSize: '13px' } }, `Source: ${datasetInfo.name || 'Select a dataset in Dataset & Tokenizer'}${datasetInfo.type ? ` (${datasetInfo.type})` : ''}`)
                ),
                e('span', { style: { fontSize: '12px', color: 'var(--text-muted)' } },
                  `Cumulative Tokens Ingested: ${(datasetInfo.total_tokens || metrics.tokens_trained || 0).toLocaleString()} tokens`
                )
              ),
              e('div', { style: { fontSize: '11px', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' } },
                e('span', null, 'Active GPU Micro-Batch Sequence (Decoded x[0] entering Autograd Graph):'),
                e('span', null, `Batch Size: ${datasetInfo.batch_size || 4} | Seq Len: ${datasetInfo.seq_len || 64}`)
              ),
              e('div', { className: 'stream-preview-text' },
                activeBatchPreview
                  ? `"${cleanText(activeBatchPreview)}"`
                  : isTraining
                    ? 'Waiting for the next decoded training batch…'
                    : 'Training is idle. Start Training or choose Train 1 Step to read from this source.'
              )
            ),
            e('div', { className: 'wandb-research-card' },
              // 1. WandB Header Bar
              e('div', { className: 'wandb-header-bar' },
                e('div', { className: 'wandb-run-info' },
                  e('span', { style: { fontWeight: 700, fontFamily: 'var(--font-mono)', fontSize: '13px' } }, [renderIcon('datasets', 13, { marginRight: '6px' }), 'Experiment Run:']),
                  e('input', {
                    className: 'wandb-run-input',
                    value: runName,
                    onChange: (ev) => setRunName(ev.target.value),
                    placeholder: 'run-name-e.g-triune-baseline'
                  }),
                  e('span', { className: `pill ${isTraining ? 'online' : ''}` }, isTraining ? '● Tracking Live' : '● Engine Idle')
                ),
                e('div', { className: 'wandb-actions-group' },
                  e('button', { className: `btn-action ${isTraining ? 'pause' : 'start'}`, onClick: handleToggleTraining },
                    isTraining ? [renderIcon('pause', 13, { marginRight: '6px' }), 'Pause Training'] : [renderIcon('play', 13, { marginRight: '6px' }), 'Start Training']
                  ),
                  !isTraining && e('button', {
                    className: 'btn-sec',
                    style: { padding: '6px 12px', fontSize: '12px', fontWeight: 600 },
                    onClick: handleStepTrainingOnce,
                    title: 'Execute a single unified TrainingEngine step'
                  }, [renderIcon('bolt', 12, { marginRight: '5px' }), 'Train 1 Step']),
                  e('button', { className: 'btn-action', style: { background: 'var(--accent-olive, #2b4c3f)', color: '#fff' }, onClick: handleSaveCheckpoint },
                    [renderIcon('checkpoints', 13, { marginRight: '6px' }), 'Save Checkpoint']
                  ),
                  e('details', { className: 'training-export-menu' },
                    e('summary', { className: 'btn-wandb-export', title: 'Export run metrics or copy a research table' }, [renderIcon('download', 12, { marginRight: '5px' }), 'Export']),
                    e('div', { className: 'training-export-options' },
                      e('button', { className: 'btn-wandb-export', onClick: ev => { handleExportRunCSV(); ev.currentTarget.closest('details').open = false; }, title: 'Export all training metrics to CSV file' }, 'CSV metrics'),
                      e('button', { className: 'btn-wandb-export', onClick: ev => { handleExportRunJSON(); ev.currentTarget.closest('details').open = false; }, title: 'Export run history in WandB JSON schema' }, 'WandB JSON'),
                      e('button', { className: 'btn-wandb-export', onClick: ev => { handleCopyLatexTable(); ev.currentTarget.closest('details').open = false; }, title: 'Copy research results table to clipboard as LaTeX' }, 'Copy LaTeX table')
                    )
                  )
                )
              ),

              // 2. WandB Metric Toggles & Controls Row
              e('div', { className: 'wandb-toolbar-row' },
                e('div', { className: 'metric-toggle-group' },
                  e('span', { style: { fontSize: '11px', color: 'var(--text-muted)', marginRight: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' } }, 'Metrics:'),
                  [
                    { key: 'loss', label: 'Loss', color: 'var(--chart-loss)' },
                    { key: 'lm_loss', label: 'LM Loss', color: 'var(--chart-lm-loss)' },
                    { key: 'router_loss', label: 'Router Loss', color: 'var(--chart-router-loss)' },
                    { key: 'throughput', label: 'Speed', color: 'var(--chart-throughput)' },
                    { key: 'ppl', label: 'PPL', color: 'var(--chart-ppl)' }
                  ].map(m => {
                    const isActive = chartMetrics && (Array.isArray(chartMetrics) ? chartMetrics.includes(m.key) : Boolean(chartMetrics[m.key]));
                    return e('div', {
                      key: m.key,
                      className: `metric-toggle-pill ${isActive ? 'active' : ''}`,
                      style: { color: m.color, borderColor: isActive ? m.color : 'var(--border-color)', cursor: 'pointer' },
                      onClick: () => {
                        setChartMetrics(prev => {
                          if (Array.isArray(prev)) {
                            return prev.includes(m.key) ? (prev.length > 1 ? prev.filter(k => k !== m.key) : prev) : [...prev, m.key];
                          }
                          const nextVal = !prev[m.key];
                          const activeCount = Object.values(prev).filter(Boolean).length;
                          if (!nextVal && activeCount <= 1) return prev;
                          return { ...prev, [m.key]: nextVal };
                        });
                      }
                    },
                      e('span', { className: 'metric-dot', style: { background: m.color } }),
                      m.label
                    );
                  })
                ),
                e('div', { className: 'wandb-chart-controls' },
                  e('div', { className: 'smoothing-control' },
                    e('span', null, `Smoothing: ${Math.round(chartSmoothing * 100)}%`),
                    e('input', {
                      type: 'range',
                      className: 'smoothing-slider',
                      min: 0,
                      max: 0.99,
                      step: 0.05,
                      value: chartSmoothing,
                      onChange: (ev) => setChartSmoothing(parseFloat(ev.target.value))
                    })
                  ),
                  e('button', {
                    className: `axis-mode-btn ${chartScale === 'log' ? 'active' : ''}`,
                    onClick: () => setChartScale(chartScale === 'linear' ? 'log' : 'linear'),
                    title: 'Toggle Linear / Logarithmic Y-axis'
                  }, `Scale: ${chartScale.toUpperCase()}`)
                )
              ),

              // 3. Canvas Multi-Series Graph
              e('div', { style: { padding: '12px 18px' } },
                e('canvas', { ref: canvasRef, width: 850, height: 260, className: 'loss-canvas' })
              ),

              // 4. Latest Model Generation Preview
              lastSample && e('div', { className: 'card-stat', style: { margin: '0 18px 12px 18px', textAlign: 'left', background: 'var(--bg-card, #f4efe6)', border: '1px solid var(--border-color, #d4ccb8)', borderRadius: '8px', padding: '12px 16px' } },
                e('div', { className: 'stat-label', style: { color: 'var(--accent-terracotta, #9a3412)', fontWeight: 600, marginBottom: '4px', display: 'flex', alignItems: 'center' } }, [renderIcon('sparkle', 12, { marginRight: '6px' }), 'Latest Live Model Generation (32k Vocab BPE):']),
                e('div', { style: { fontFamily: 'var(--font-mono)', fontSize: '13px', color: 'var(--text-main, #24211e)', lineHeight: '1.5' } }, `"${cleanText(lastSample)}"`)
              ),

              // 5. Telemetry Logs Terminal Stream
              e('div', { style: { padding: '0 18px 16px 18px' } },
                e('div', { className: 'telemetry-box' },
                  telemetryLogs.length > 0
                    ? telemetryLogs.map((log, idx) => e('div', { key: idx }, log.replace(/Ġ/g, ' ')))
                    : e('div', null, '[TELEMETRY] Training events will appear here when a run starts.')
                )
              )
            )
          ),

          // Tab 6: Interactive SVG Visual Node Canvas IDE with Precise Relative Mouse Tracking
          activeTab === 'nodegraph' && e('div', { className: 'view-nodegraph' },
            e('div', { className: 'node-canvas-react' },
              e('div', { className: 'node-canvas-header', style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', paddingBottom: '10px', borderBottom: '1px solid var(--border-color)', flexWrap: 'wrap', gap: '8px' } },
                e('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' } },
                  e('span', { style: { fontFamily: 'Newsreader, Georgia, serif', fontSize: '17px', fontWeight: 700, color: 'var(--text-main)' } }, 'Visual DAG Architecture'),
                  e('input', {
                    value: workflowName,
                    onChange: ev => setWorkflowName(ev.target.value),
                    'aria-label': 'Workflow name',
                    placeholder: 'Workflow name',
                    style: { width: '150px', padding: '5px 7px', fontSize: '11px', borderRadius: '5px', border: '1px solid var(--border-color)', background: 'var(--bg-input, #fff)', color: 'var(--text-main)' }
                  }),
                  e('button', {
                    className: 'btn-icon-tool',
                    onClick: () => setShowDAGPresetLibrary(true),
                    title: 'Browse workflows by goal, see prerequisites, and load one onto the canvas'
                  }, [renderIcon('library', 13, { marginRight: '6px' }), 'Workflow Presets'])
                ),
                e('div', { className: 'canvas-toolbar', style: { display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' } },
                  e('button', {
                    className: 'btn-action-compile-live',
                    style: {
                      background: 'linear-gradient(135deg, #9a3412 0%, #c2410c 100%)',
                      color: '#ffffff',
                      fontWeight: 700,
                      boxShadow: '0 2px 8px rgba(154, 52, 18, 0.3)',
                      border: 'none',
                      padding: '5px 12px',
                      borderRadius: '5px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '5px',
                      fontSize: '12px'
                    },
                    onClick: handleCompileDAGToTraining,
                    disabled: isCompilingTraining,
                    title: 'Compile visual DAG and apply to live PyTorch training loop (/v1/dag/compile_to_training)'
                  }, isCompilingTraining ? [renderIcon('refresh', 13, { marginRight: '6px' }), 'Compiling...'] : [renderIcon('bolt', 13, { marginRight: '6px' }), 'Apply to Live Engine']),
                  e('button', {
                    className: 'btn-icon-tool',
                    style: { fontWeight: 600, borderColor: 'var(--primary, #9a3412)' },
                    onClick: () => setQuickAddMenu({
                      x: Math.min(window.innerWidth - 290, 260),
                      y: 180,
                      canvasX: -canvasPan.x / canvasZoom + 100,
                      canvasY: -canvasPan.y / canvasZoom + 100,
                      search: ''
                    }),
                    title: 'Quick-add node near cursor (Shift+A)'
                  }, [renderIcon('plus', 13, { marginRight: '6px' }), 'Add Node (Shift+A)']),
                  e('button', {
                    className: 'btn-icon-tool',
                    style: { background: 'var(--accent-terracotta, #9a3412)', color: '#fff', border: 'none' },
                    onClick: () => setShowNodeLibraryModal(true),
                    title: 'Browse categorized nodes, executable operators, specifications, and plugins'
                  }, [renderIcon('library', 13, { marginRight: '6px' }), 'Library (' + nodeCatalog.length + ')']),
                  e('button', {
                    className: 'btn-icon-tool',
                    onClick: handleGroupSelectedNodes,
                    disabled: selectedNodeIds.length < 2,
                    title: 'Collapse selected nodes into a reusable typed subgraph (Ctrl+G)'
                  }, [renderIcon('nodes', 13, { marginRight: '6px' }), `Group${selectedNodeIds.length > 1 ? ` (${selectedNodeIds.length})` : ''}`]),
                  e('button', {
                    className: 'btn-icon-tool',
                    onClick: handleUngroupSelectedNode,
                    disabled: !nodes.some(node => node.id === selectedNodeId && node.type === 'Subgraph'),
                    title: 'Expand selected reusable subgraph and restore its wires'
                  }, [renderIcon('frame', 13, { marginRight: '6px' }), 'Ungroup']),
                  e('input', {
                    ref: workflowImportRef,
                    type: 'file',
                    accept: '.json,.triune.json,application/json',
                    style: { display: 'none' },
                    onChange: ev => {
                      const file = ev.target.files && ev.target.files[0];
                      handleImportWorkflowFile(file);
                      ev.target.value = '';
                    }
                  }),
                  e('button', {
                    className: 'btn-icon-tool',
                    onClick: () => workflowImportRef.current && workflowImportRef.current.click(),
                    title: 'Import a saved Triune workflow JSON file'
                  }, [renderIcon('folder', 13, { marginRight: '6px' }), 'Open']),
                  e('button', {
                    className: 'btn-icon-tool',
                    onClick: handleExportWorkflow,
                    title: 'Export this workflow and all nested groups as a portable JSON file'
                  }, [renderIcon('checkpoints', 13, { marginRight: '6px' }), 'Save As']),
                  e('span', { style: { fontSize: '10px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }, title: 'Workflow is autosaved in this browser profile' }, workflowSaveState),
                  e('button', {
                    className: 'btn-icon-tool',
                    onClick: () => { if (selectedNodeId) handleToggleMuteNode(selectedNodeId); else showToast('Select a node first to mute (M key)'); },
                    title: 'Mute / Bypass selected node in execution graph (M key)'
                  }, [renderIcon('mute', 13, { marginRight: '6px' }), 'Mute (M)']),
                  e('button', {
                    className: 'btn-icon-tool',
                    onClick: () => handleCreateVisualFrame(),
                    title: 'Add a labeled visual backdrop (Ctrl+J)'
                  }, [renderIcon('frame', 13, { marginRight: '6px' }), 'Add Backdrop']),
                  e('button', {
                    className: 'btn-action start',
                    style: { padding: '5px 12px', fontSize: '12px' },
                    onClick: handleExecuteDAG,
                    title: 'Run one forward-and-optimizer pass for this graph (not a long-running training job)'
                  }, [renderIcon('play', 13, { marginRight: '6px' }), 'Run One Pass']),
                  e('button', {
                    className: 'btn-icon-tool',
                    onClick: () => setShowWireList(prev => !prev),
                    style: { display: 'flex', alignItems: 'center', gap: '4px' },
                    title: 'Toggle active wire connections list'
                  }, [renderIcon('scissors', 13, { marginRight: '5px' }), `Wires (${edges.length}) ${showWireList ? '▲' : '▼'}`])
                )
              ),
              activePresetKey && DAG_PRESETS[activePresetKey] && e('div', { className: 'dag-preset-guide' },
                e('strong', null, DAG_PRESETS[activePresetKey].name),
                e('span', null, DAG_PRESETS[activePresetKey].description),
                e('small', null, `Before running: ${DAG_PRESETS[activePresetKey].requirements}`)
              ),
              showWireList && edges.length > 0 && e('div', {
                className: 'canvas-wires-panel',
                style: {
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: '6px',
                  padding: '8px 12px',
                  marginBottom: '8px',
                  background: 'var(--bg-surface, #f9f8f5)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  maxHeight: '110px',
                  overflowY: 'auto'
                }
              },
                e('span', { style: { fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)', alignSelf: 'center' } }, 'Active Wires (Click to disconnect):'),
                edges.map(edge => {
                  const srcNode = nodes.find(n => n.id === edge.source);
                  const tgtNode = nodes.find(n => n.id === edge.target);
                  return e('button', {
                    key: edge.id,
                    className: 'preset-chip',
                    style: { border: '1px dashed var(--primary)', color: 'var(--primary)', cursor: 'pointer', fontSize: '11px', padding: '2px 8px' },
                    onClick: () => handleDisconnectWire(edge.id),
                    title: 'Click to disconnect wire'
                  }, [srcNode ? srcNode.title : edge.source, ' → ', tgtNode ? tgtNode.title : edge.target, ' ', renderIcon('scissors', 11, { marginLeft: '4px' })]);
                })
              ),
              dagExecutionStatus && e('div', { className: 'telemetry-box', style: { marginBottom: '8px' } }, dagExecutionStatus),
              e('div', {
                className: 'node-grid-area',
                ref: gridRef,
                style: {
                  backgroundPosition: `${canvasPan.x}px ${canvasPan.y}px, ${canvasPan.x}px ${canvasPan.y}px, ${canvasPan.x}px ${canvasPan.y}px, ${canvasPan.x}px ${canvasPan.y}px, ${canvasPan.x}px ${canvasPan.y}px`,
                  backgroundSize: `${20 * canvasZoom}px ${20 * canvasZoom}px, ${20 * canvasZoom}px ${20 * canvasZoom}px, ${20 * canvasZoom}px ${20 * canvasZoom}px, ${100 * canvasZoom}px ${100 * canvasZoom}px, ${100 * canvasZoom}px ${100 * canvasZoom}px`
                },
                onPointerDown: handleCanvasMouseDown
              },
                // Viewport Container with 2D Pan and Zoom Transform
                e('div', {
                  className: 'node-canvas-viewport',
                  style: {
                    transform: `translate(${canvasPan.x}px, ${canvasPan.y}px) scale(${canvasZoom})`,
                    transformOrigin: '0 0'
                  }
                },
                  // Blender / Unreal Visual Group Backdrop Frames (Ctrl+J)
                  frames.map(f => e('div', {
                    key: f.id,
                    className: 'canvas-node-frame',
                    style: {
                      position: 'absolute',
                      left: `${f.x}px`,
                      top: `${f.y}px`,
                      width: `${f.width}px`,
                      height: `${f.height}px`,
                      background: f.color || 'rgba(99, 199, 178, 0.08)',
                      border: `2px dashed ${f.borderColor || '#63c7b2'}`,
                      borderRadius: '12px',
                      pointerEvents: 'none',
                      zIndex: 1
                    }
                  },
                    e('div', {
                      style: {
                        padding: '4px 10px',
                        background: f.borderColor || '#63c7b2',
                        color: '#fff',
                        fontSize: '11px',
                        fontWeight: 700,
                        fontFamily: 'var(--font-family)',
                        borderTopLeftRadius: '10px',
                        borderTopRightRadius: '10px',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '8px',
                        pointerEvents: 'auto'
                      }
                    },
                      e('span', null, f.title),
                      e('button', {
                        style: { background: 'none', border: 'none', color: '#fff', cursor: 'pointer', padding: '0 2px', display: 'flex', alignItems: 'center' },
                        title: 'Delete Frame',
                        onClick: (ev) => { ev.stopPropagation(); setFrames(prev => prev.filter(item => item.id !== f.id)); }
                      }, renderIcon('close', 10, { color: '#fff' }))
                    )
                  )),

                  // SVG Wires Layer with Exact Port Pin Bezier Alignment & Click-to-Disconnect
                  (() => {
                    // The SVG wire plane must cover the whole canvas coordinate
                    // space, including negative coordinates from panning/layout.
                    // Previously it started at (0, 0) and used a partial,
                    // positive-only size, silently clipping wires near bounds.
                    const wireOriginX = Math.min(0, ...nodes.map(n => n.x - 800));
                    const wireOriginY = Math.min(0, ...nodes.map(n => n.y - 800));
                    const wireRight = Math.max(12000, ...nodes.map(n => n.x + NODE_WIDTH + 800));
                    const wireBottom = Math.max(12000, ...nodes.map(n => n.y + 800));
                    const wireWidth = wireRight - wireOriginX;
                    const wireHeight = wireBottom - wireOriginY;
                    return e('svg', {
                      className: 'svg-wire-layer',
                      viewBox: `${wireOriginX} ${wireOriginY} ${wireWidth} ${wireHeight}`,
                      preserveAspectRatio: 'none',
                      style: { left: `${wireOriginX}px`, top: `${wireOriginY}px`, width: `${wireWidth}px`, height: `${wireHeight}px`, pointerEvents: 'none' }
                    },
                      edges.map(edge => {
                        const srcNode = nodes.find(n => n.id === edge.source);
                        const tgtNode = nodes.find(n => n.id === edge.target);
                        if (!srcNode || !tgtNode) return null;
                        const srcPorts = getNodePorts(srcNode);
                        const tgtPorts = getNodePorts(tgtNode);
                        const outIdx = Math.max(0, srcPorts.outputs.findIndex(p => p.name === edge.source_port));
                        const inIdx = Math.max(0, tgtPorts.inputs.findIndex(p => p.name === edge.target_port));
                        const sourcePoint = socketPositions[`${edge.source}|out|${edge.source_port}`];
                        const targetPoint = socketPositions[`${edge.target}|in|${edge.target_port}`];
                        const x1 = sourcePoint ? sourcePoint.x : srcNode.x + NODE_WIDTH;
                        const y1 = sourcePoint ? sourcePoint.y : srcNode.y + 73 + outIdx * 24;
                        const x2 = targetPoint ? targetPoint.x : tgtNode.x;
                        const y2 = targetPoint ? targetPoint.y : tgtNode.y + 73 + inIdx * 24;
                        const dx = Math.max(Math.abs(x2 - x1) * 0.45, 40);
                        const pathD = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
                        const wireColor = getPortColor(edge.port_type || (srcPorts.outputs[outIdx] && srcPorts.outputs[outIdx].type));

                        return e('g', { key: edge.id },
                          // Invisible wider hit-target for effortless click-to-disconnect
                          e('path', {
                            d: pathD,
                            stroke: 'transparent',
                            strokeWidth: 16,
                            fill: 'none',
                            style: { pointerEvents: 'stroke', cursor: 'pointer' },
                            title: `Connection: [${srcNode.title}:${edge.source_port}] → [${tgtNode.title}:${edge.target_port}] (Click to disconnect)`,
                            onClick: (ev) => { ev.stopPropagation(); handleDisconnectWire(edge.id); }
                          }),
                          // High-contrast themed wire
                          e('path', {
                            d: pathD,
                            className: 'svg-wire',
                            stroke: wireColor,
                            strokeWidth: 2.6,
                            fill: 'none',
                            style: { pointerEvents: 'none', filter: 'drop-shadow(0 1px 2px rgba(60, 45, 30, 0.25))' }
                          })
                        );
                      }),
                      // Live Wire Dragging Preview with Animated Dash & Matching Socket Color
                      connectingWire && (() => {
                        const x1 = connectingWire.startX;
                        const y1 = connectingWire.startY;
                        const x2 = canvasMousePos.x;
                        const y2 = canvasMousePos.y;
                        const dx = Math.max(Math.abs(x2 - x1) * 0.45, 40);
                        const wireColor = getPortColor(connectingWire.port_type);
                        return e('path', {
                          d: `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`,
                          className: 'svg-wire dragging',
                          stroke: wireColor,
                          strokeWidth: 2.6,
                          strokeDasharray: '6 4',
                          fill: 'none',
                          style: { filter: 'drop-shadow(0 1px 3px rgba(60, 45, 30, 0.3))' }
                        });
                      })()
                    );
                  })(),

                  // Blender & Unreal Blueprints Draggable Nodes with Individual Sockets
                  nodes.map(n => {
                    const exec = nodeExecOutputs[n.id];
                    const isRunning = activeRunningNodeId === n.id || (exec && exec.status === 'running');
                    const isCompleted = exec && exec.status === 'completed';
                    const isSpecification = exec && exec.status === 'specification';
                    const isFailed = exec && exec.status === 'failed';
                    const isSelected = selectedNodeIds.includes(n.id);
                    const isMuted = !!n.muted;
                    const cardClass = `node-card-react ${isSelected ? 'node-selected' : ''} ${isRunning ? 'running' : (isCompleted ? 'completed' : (isFailed ? 'failed' : (isSpecification ? 'specification' : '')))} ${isMuted ? 'node-muted' : ''}`;
                    const { inputs, outputs } = getNodePorts(n);
                    const tensorBadge = getNodeTensorBadge(n, exec && exec.output);

                    return e('div', {
                      key: n.id,
                      className: cardClass,
                      'data-category': n.type,
                      style: {
                        left: `${n.x}px`,
                        top: `${n.y}px`,
                        width: `${NODE_WIDTH}px`,
                        opacity: isMuted ? 0.45 : 1.0,
                        filter: isMuted ? 'grayscale(0.85)' : 'none',
                        borderStyle: isMuted ? 'dashed' : 'solid'
                      },
                      onClick: (ev) => { ev.stopPropagation(); },
                      onPointerDown: (ev) => handleNodeMouseDown(ev, n.id)
                    },
                      // Node Header
                      e('div', { className: 'node-head' },
                        e('span', { style: { fontFamily: 'var(--font-serif)', fontWeight: 700, fontSize: '13.5px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } },
                          n.title,
                          isMuted && e('span', { style: { fontSize: '10px', color: 'var(--accent-terracotta, #9a3412)', marginLeft: '4px' } }, '[MUTED]')
                        ),
                        e('div', { style: { display: 'flex', alignItems: 'center', gap: '5px' } },
                          exec && e('span', { className: `node-status-badge ${exec.status}`, style: { display: 'inline-flex', alignItems: 'center' } },
                            exec.status === 'running' ? 'RUNNING' : (exec.status === 'completed' ? [renderIcon('check', 10, { marginRight: '2px' }), 'READY'] : (isSpecification ? 'SPEC' : 'FAIL'))
                          ),
                          e('button', {
                            className: 'node-mute-btn',
                            style: {
                              background: isMuted ? '#9a3412' : 'var(--bg-surface, #ede9df)',
                              color: isMuted ? '#fff' : 'var(--text-muted)',
                              border: '1px solid var(--border-color)',
                              borderRadius: '3px',
                              padding: '1px 5px',
                              fontSize: '10px',
                              cursor: 'pointer',
                              fontWeight: 700
                            },
                            title: 'Mute / Bypass Node (M key)',
                            onClick: (ev) => { ev.stopPropagation(); handleToggleMuteNode(n.id); }
                          }, isMuted ? renderIcon('mute', 10) : 'M'),
                          e('button', {
                            className: 'node-delete-btn',
                            title: 'Delete Node',
                            style: { display: 'inline-flex', alignItems: 'center', justifyContent: 'center' },
                            onClick: (ev) => { ev.stopPropagation(); handleDeleteNode(n.id); }
                          }, renderIcon('close', 10))
                        )
                      ),
                      // Node Subbar: Category, Execution Time & Live Pin Output Badge
                      e('div', { className: 'node-subbar', style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '3px 10px', background: 'var(--bg-surface)', borderBottom: '1px solid var(--border-color)', height: '22px' } },
                        e('div', { style: { display: 'flex', alignItems: 'center', gap: '6px' } },
                          e('span', { className: 'node-type-tag' }, `${n.type} NODE`),
                          e('span', {
                            className: 'node-live-pin-badge',
                            style: {
                              fontSize: '9px',
                              fontFamily: 'var(--font-mono)',
                              padding: '1px 5px',
                              borderRadius: '3px',
                              background: 'var(--status-completed-bg, rgba(99, 199, 178, 0.18))',
                              color: 'var(--status-completed-text, #0f766e)',
                              border: '1px solid var(--status-completed-border, rgba(99, 199, 178, 0.4))',
                              fontWeight: 600
                            },
                            title: 'Live Executed Tensor Shape & Dtype'
                          }, tensorBadge)
                        ),
                        exec && exec.elapsed !== undefined && e('span', { style: { fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' } }, `${(exec.elapsed * 1000).toFixed(0)}ms`)
                      ),
                      // Individual Typed Sockets Grid (Editorial Workshop blueprint socket rows)
                      e('div', { className: 'node-sockets-grid', style: { display: 'flex', justifyContent: 'space-between', padding: '4px 0', minHeight: '44px', background: 'transparent' } },
                        // Left Column: Inputs
                        e('div', { className: 'socket-column inputs', style: { display: 'flex', flexDirection: 'column', gap: '2px', flex: '1 1 50%' } },
                          inputs.map((inp, idx) => {
                            const pColor = getPortColor(inp.type);
                            const isTargeting = connectingWire && connectingWire.source !== n.id;
                            return e('div', {
                              key: inp.name || idx,
                              className: 'socket-row input',
                              style: { display: 'flex', alignItems: 'center', height: '22px', position: 'relative', paddingLeft: '8px' }
                            },
                              e('div', {
                                className: `socket-pin input ${isTargeting ? 'snap-target' : ''}`,
                                style: {
                                  position: 'absolute',
                                  left: '-7px',
                                  width: '13px',
                                  height: '13px',
                                  borderRadius: '50%',
                                  background: pColor,
                                  border: '2px solid var(--bg-card)',
                                  boxShadow: '0 1px 3px rgba(60, 45, 30, 0.25)',
                                  cursor: 'crosshair',
                                  zIndex: 20
                                },
                                title: `Input: ${inp.name} (${inp.type}) – Drop wire here`,
                                'data-dag-input-node': n.id,
                                'data-dag-input-port': inp.name,
                                'data-dag-input-type': inp.type,
                              }),
                              e('span', { className: 'socket-label', style: { fontSize: '11px', color: 'var(--text-main)', marginLeft: '10px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, inp.name),
                              e('span', { className: 'socket-type-tag', style: { fontSize: '8.5px', color: pColor, marginLeft: '4px', opacity: 0.85, fontFamily: 'var(--font-mono)' } }, `[${inp.type}]`)
                            );
                          })
                        ),
                        // Right Column: Outputs
                        e('div', { className: 'socket-column outputs', style: { display: 'flex', flexDirection: 'column', gap: '2px', flex: '1 1 50%', alignItems: 'flex-end' } },
                          outputs.map((out, idx) => {
                            const pColor = getPortColor(out.type);
                            return e('div', {
                              key: out.name || idx,
                              className: 'socket-row output',
                              style: { display: 'flex', alignItems: 'center', justifyContent: 'flex-end', height: '22px', position: 'relative', paddingRight: '8px' }
                            },
                              e('span', { className: 'socket-type-tag', style: { fontSize: '8.5px', color: pColor, marginRight: '4px', opacity: 0.85, fontFamily: 'var(--font-mono)' } }, `[${out.type}]`),
                              e('span', { className: 'socket-label', style: { fontSize: '11px', color: 'var(--text-main)', marginRight: '10px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, out.name),
                              e('div', {
                                className: 'socket-pin output',
                                style: {
                                  position: 'absolute',
                                  right: '-7px',
                                  width: '13px',
                                  height: '13px',
                                  borderRadius: '50%',
                                  background: pColor,
                                  border: '2px solid var(--bg-card)',
                                  boxShadow: '0 1px 3px rgba(60, 45, 30, 0.25)',
                                  cursor: 'crosshair',
                                  zIndex: 20
                                },
                                title: `Output: ${out.name} (${out.type}) – Drag wire to connect`,
                                'data-dag-output-node': n.id,
                                'data-dag-output-port': out.name,
                                onPointerDown: (ev) => handleStartWire(ev, n.id, out.name, out.type, idx)
                              })
                            );
                          })
                        )
                      ),
                      // Node Body: In-Node Inline Editable Controls & Live Output
                      e('div', { className: 'node-body', style: { padding: '8px 10px 10px 10px', borderTop: '1px solid var(--border-color, #ded9cd)' } },
                        renderNodeInlineControls(n),
                        // Live Output Inspector Box
                        exec && (exec.output || exec.error) && (() => {
                          const out = exec.output || {};
                          let outText = '';
                          if (exec.status === 'failed') {
                            outText = exec.error || out.error || out.message || 'Node execution failed without a diagnostic.';
                          } else if (out.tokens_loaded) {
                            outText = `Loaded ${out.tokens_loaded} tokens (${out.dataset_file || 'stream'})`;
                          } else if (out.total_params) {
                            outText = `${out.total_params} params (${out.architecture || 'MoE'})\n${out.exit_heads || '3 Exits'}`;
                          } else if (out.loss !== undefined) {
                            outText = `Step 1 | Loss: ${out.loss}\nGrad Norm: ${out.grad_norm || 0.45}\nOptimizer: Active`;
                          } else if (out.file_size_mb) {
                            outText = `Exported ${out.file_size_mb} MB (${out.format || 'safetensors'})`;
                          } else if (out.text !== undefined) {
                            outText = `Generated: "${out.text.slice(0, 70)}..."\nLatency: ${((out.elapsed_sec || 0) * 1000).toFixed(1)}ms`;
                          } else if (out.status_note) {
                            outText = `${out.status_note}`;
                          } else if (out.note) {
                            outText = `${out.note}`;
                          } else {
                            outText = `Processed by ExecutionEngine`;
                          }
                          return e('div', { className: 'node-live-output', style: { marginTop: '6px' } }, outText);
                        })()
                      )
                    );
                  })
                ),
                // Empty Canvas Prompt
                nodes.length === 0 && e('div', {
                  style: {
                    position: 'absolute',
                    top: '45%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    textAlign: 'center',
                    background: 'var(--bg-card)',
                    padding: '24px 32px',
                    borderRadius: '10px',
                    border: '1px dashed var(--border-dark)',
                    boxShadow: '0 4px 14px rgba(60, 45, 30, 0.08)',
                    pointerEvents: 'auto',
                    zIndex: 15
                  }
                },
                  e('h4', { style: { fontFamily: 'var(--font-serif)', fontSize: '18px', color: 'var(--text-main)', marginBottom: '8px' } }, 'Node Canvas is Empty'),
                  e('p', { style: { fontSize: '13px', color: 'var(--text-muted)', marginBottom: '14px' } }, 'Select an architecture preset above, click "+ Node" on the toolbar, or spawn from the Node Library.'),
                  e('button', { className: 'btn-action', style: { background: 'var(--accent-olive, #2b4c3f)', color: '#fff', display: 'inline-flex', alignItems: 'center' }, onClick: () => handleLoadDAGPreset('triune_canonical_hierarchical') }, [renderIcon('bolt', 12, { marginRight: '6px' }), 'Load CLI Training Graph'])
                ),
                // Quick-Add Popup Menu (Shift+A)
                quickAddMenu && e('div', {
                  className: 'quick-add-popup',
                  style: {
                    position: 'absolute',
                    left: `${quickAddMenu.x}px`,
                    top: `${quickAddMenu.y}px`,
                    width: '300px',
                    maxHeight: '380px',
                    background: 'var(--bg-card, #fcfbfa)',
                    border: '1px solid var(--border-color, #ded9cd)',
                    borderRadius: '8px',
                    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.22)',
                    zIndex: 9999,
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'hidden',
                    fontSize: '12px',
                    fontFamily: 'var(--font-family, sans-serif)'
                  },
                  onClick: (ev) => ev.stopPropagation(),
                  onMouseDown: (ev) => ev.stopPropagation()
                },
                  e('div', {
                    style: {
                      padding: '8px 10px',
                      borderBottom: '1px solid var(--border-color, #ded9cd)',
                      background: 'var(--bg-surface, #f7f4ed)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px'
                    }
                  },
                    renderIcon('bolt', 13),
                    e('input', {
                      type: 'text',
                      autoFocus: true,
                      placeholder: 'Search nodes (Shift+A)...',
                      value: quickAddMenu.search || '',
                      onChange: (ev) => {
                        const val = ev.target.value;
                        setQuickAddMenu(prev => prev ? { ...prev, search: val } : null);
                      },
                      onKeyDown: (ev) => {
                        if (ev.key === 'Escape') setQuickAddMenu(null);
                      },
                      style: {
                        flex: 1,
                        background: 'var(--bg-input, #fff)',
                        border: '1px solid var(--border-color, #ded9cd)',
                        borderRadius: '4px',
                        padding: '4px 8px',
                        fontSize: '12px',
                        outline: 'none'
                      }
                    }),
                    e('button', {
                      onClick: () => setQuickAddMenu(null),
                      style: {
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: '2px',
                        display: 'flex',
                        alignItems: 'center',
                        color: 'var(--text-muted)'
                      }
                    }, renderIcon('close', 11))
                  ),
                  e('div', {
                    style: {
                      overflowY: 'auto',
                      maxHeight: '320px',
                      padding: '4px 0'
                    }
                  },
                    (() => {
                      const q = (quickAddMenu.search || '').toLowerCase().trim();
                      const cat = nodeCatalog || BUILTIN_NODE_CATALOG;
                      const filtered = cat.filter(item =>
                        !q || item.title.toLowerCase().includes(q) || item.name.toLowerCase().includes(q) || (item.category && item.category.toLowerCase().includes(q))
                      );
                      if (filtered.length === 0) {
                        return e('div', { style: { padding: '14px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '11px' } }, 'No matching nodes');
                      }
                      return filtered.map(item => {
                        const catColor = getPortColor(item.category ? item.category.toLowerCase() : 'tensor');
                        return e('div', {
                          key: item.name,
                          style: {
                            padding: '6px 12px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            borderBottom: '1px solid rgba(0,0,0,0.03)',
                            transition: 'background 0.15s ease'
                          },
                          className: 'quick-add-item',
                          onMouseEnter: (ev) => { ev.currentTarget.style.background = 'var(--bg-surface, #ede9df)'; },
                          onMouseLeave: (ev) => { ev.currentTarget.style.background = 'transparent'; },
                          onClick: () => {
                            handleSpawnCatalogNode(item.name, quickAddMenu.canvasX, quickAddMenu.canvasY);
                            setQuickAddMenu(null);
                          }
                        },
                          e('div', { style: { display: 'flex', flexDirection: 'column' } },
                            e('span', { style: { fontWeight: 600, color: 'var(--text-main)' } }, item.title),
                            e('span', { style: { fontSize: '10px', color: 'var(--text-muted)' } }, `${item.name} • ${item.category}`)
                          ),
                          e('span', {
                            style: {
                              fontSize: '10px',
                              padding: '2px 5px',
                              borderRadius: '3px',
                              background: `${catColor}22`,
                              color: catColor,
                              border: `1px solid ${catColor}`
                            }
                          }, item.category)
                        );
                      });
                    })()
                  )
                ),
                // Blender / Unreal Engine Navigation Overlay (Zoom In, Zoom Out, 100%, Fit View, Duplicate, Delete)
                e('div', { className: 'canvas-nav-overlay' },
                  e('div', { className: 'canvas-nav-controls' },
                    e('button', { className: 'canvas-nav-btn', title: 'Zoom In (+)', onClick: () => setCanvasZoom(z => Math.min(2.5, z * 1.15)) }, renderIcon('plus', 13)),
                    e('button', { className: 'canvas-nav-btn', title: 'Zoom Out (-)', onClick: () => setCanvasZoom(z => Math.max(0.25, z / 1.15)) }, renderIcon('minus', 13)),
                    e('button', { className: 'canvas-nav-btn', title: 'Reset to 100%', onClick: () => { setCanvasZoom(1.0); setCanvasPan({ x: 0, y: 0 }); } }, '100%'),
                    e('button', { className: 'canvas-nav-btn', title: 'Fit DAG to Viewport', onClick: handleFitCanvasView }, renderIcon('frame', 13)),
                    e('button', { className: 'canvas-nav-btn', title: 'Duplicate Selected Node', onClick: handleDuplicateSelectedNode, disabled: !selectedNodeId }, renderIcon('copy', 13)),
                    e('button', { className: 'canvas-nav-btn', title: 'Delete Selected Node', onClick: () => { if (selectedNodeId) handleDeleteNode(selectedNodeId); }, disabled: !selectedNodeId }, renderIcon('trash', 13))
                  ),
                  e('div', { className: 'canvas-status-pill' },
                    `Zoom: ${(canvasZoom * 100).toFixed(0)}% | Pan: ${Math.round(canvasPan.x)}, ${Math.round(canvasPan.y)} | Nodes: ${nodes.length} | Wires: ${edges.length}`
                  )
                )
              ),
              // Interactive Node Inspector Drawer
              selectedNodeId && (() => {
                const selNode = nodes.find(n => n.id === selectedNodeId);
                if (!selNode) return null;
                const cat = nodeCatalog || BUILTIN_NODE_CATALOG;
                const catItem = cat.find(c => c.name === selNode.name || c.title === selNode.title || c.name === selNode.type);
                const { inputs, outputs } = getNodePorts(selNode);
                const exec = nodeExecOutputs[selNode.id];

                return e('div', { className: 'node-inspector-drawer' },
                  e('div', { className: 'inspector-header' },
                    e('div', null,
                      e('span', { className: 'inspector-type-badge' }, selNode.type || 'NODE'),
                      e('h4', { className: 'inspector-title' }, selNode.title),
                      e('div', { className: 'inspector-id' }, `ID: ${selNode.id}`)
                    ),
                    e('button', {
                      className: 'inspector-close-btn',
                      style: { display: 'flex', alignItems: 'center', justifyContent: 'center' },
                      title: 'Close Inspector',
                      onClick: () => setSelectedNodeId(null)
                    }, renderIcon('close', 12))
                  ),
                  e('div', { className: 'inspector-body' },
                    // Description section
                    e('div', { className: 'inspector-section' },
                      e('span', { className: 'inspector-section-label' }, 'Description'),
                      e('div', { className: 'inspector-desc' },
                        (catItem && catItem.description) || `Custom ${selNode.type} component in Triune execution graph.`
                      )
                    ),
                    // I/O Ports
                    e('div', { className: 'inspector-section' },
                      e('span', { className: 'inspector-section-label' }, 'I/O Ports'),
                      e('div', { style: { display: 'flex', flexDirection: 'column', gap: '4px' } },
                        e('div', null,
                          e('span', { style: { fontSize: '11px', color: 'var(--text-muted)', marginRight: '6px' } }, 'Inputs:'),
                          inputs.length > 0
                            ? inputs.map((pin, idx) => e('span', { key: idx, className: 'pin-pill in' }, `→ ${pin.name} (${pin.type})`))
                            : e('span', { style: { fontSize: '11px', color: 'var(--text-dim)' } }, 'None (Source Node)')
                        ),
                        e('div', null,
                          e('span', { style: { fontSize: '11px', color: 'var(--text-muted)', marginRight: '6px' } }, 'Outputs:'),
                          outputs.length > 0
                            ? outputs.map((pin, idx) => e('span', { key: idx, className: 'pin-pill out' }, `${pin.name} (${pin.type}) →`))
                            : e('span', { style: { fontSize: '11px', color: 'var(--text-dim)' } }, 'None (Terminal Node)')
                        )
                      )
                    ),
                    // Editable Hyperparameters Section
                    e('div', { className: 'inspector-section' },
                      e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' } },
                        e('span', { className: 'inspector-section-label' }, 'Hyperparameters'),
                        e('button', {
                          className: 'btn-icon-tool',
                          style: { padding: '2px 8px', fontSize: '11px' },
                          onClick: () => { setNewNodeParamName(''); setShowNewNodeParam(true); }
                        }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Param'])
                      ),
                      showNewNodeParam && e('div', { className: 'inspector-field-row' },
                        e('input', {
                          className: 'inspector-field-input',
                          value: newNodeParamName,
                          placeholder: 'Parameter name',
                          'aria-label': 'New parameter name',
                          autoFocus: true,
                          onChange: ev => setNewNodeParamName(ev.target.value),
                          onKeyDown: ev => {
                            if (ev.key === 'Escape') setShowNewNodeParam(false);
                            if (ev.key === 'Enter') {
                              ev.preventDefault();
                              const key = newNodeParamName.trim();
                              if (!key) return;
                              if (Object.prototype.hasOwnProperty.call(selectedNodeParams, key)) {
                                showToast(`Parameter "${key}" already exists.`);
                                return;
                              }
                              setSelectedNodeParams(prev => ({ ...prev, [key]: '' }));
                              setShowNewNodeParam(false);
                            }
                          }
                        }),
                        e('button', {
                          className: 'btn-icon-tool',
                          title: 'Add parameter',
                          onClick: () => {
                            const key = newNodeParamName.trim();
                            if (!key) return;
                            if (Object.prototype.hasOwnProperty.call(selectedNodeParams, key)) {
                              showToast(`Parameter "${key}" already exists.`);
                              return;
                            }
                            setSelectedNodeParams(prev => ({ ...prev, [key]: '' }));
                            setShowNewNodeParam(false);
                          }
                        }, renderIcon('check', 12))
                      ),
                      Object.keys(selectedNodeParams).length === 0 ?
                        e('div', { style: { fontSize: '11.5px', color: 'var(--text-dim)', fontStyle: 'italic' } }, 'No parameters configured.') :
                        Object.entries(selectedNodeParams).map(([k, v]) =>
                          e('div', { key: k, className: 'inspector-field-row' },
                            e('span', { className: 'inspector-field-key', title: k }, k),
                            e('input', {
                              className: 'inspector-field-input',
                              value: v,
                              onChange: (ev) => {
                                const val = ev.target.value;
                                setSelectedNodeParams(prev => ({ ...prev, [k]: val }));
                              }
                            })
                          )
                        ),
                      e('div', { style: { display: 'flex', gap: '8px', marginTop: '6px' } },
                        e('button', {
                          className: 'btn-sec',
                          style: { flex: 1, padding: '6px 0', fontSize: '12px' },
                          onClick: handleSaveNodeParams
                        }, [renderIcon('checkpoints', 12, { marginRight: '5px' }), 'Save Params']),
                        e('button', {
                          className: 'btn-send',
                          style: { flex: 1.2, padding: '6px 0', fontSize: '12px', background: 'var(--accent-terracotta, #9a3412)' },
                          disabled: isExecutingSingleNode,
                          onClick: () => handleExecuteSingleNode(selNode.id)
                        }, isExecutingSingleNode ? [renderIcon('refresh', 12, { marginRight: '5px' }), 'Running...'] : [renderIcon('play', 12, { marginRight: '5px' }), 'Run Node'])
                      )
                    ),
                    // Live Execution Output
                    e('div', { className: 'inspector-section' },
                      e('span', { className: 'inspector-section-label' }, 'Execution Output'),
                      exec && exec.output ?
                        e('pre', { className: 'inspector-output-json' }, JSON.stringify(exec.output, null, 2)) :
                        e('div', { style: { fontSize: '11.5px', color: 'var(--text-dim)', fontStyle: 'italic' } }, 'Node not executed yet. Click "Run Node" above to test in isolation.')
                    )
                  )
                );
              })()
            )
          ),

          // Tab 3: LoRA Fine-Tuner
          activeTab === 'finetune' && e('div', { className: 'view-finetune' },
            e('div', { className: 'card-finetune' },
              e('div', { className: 'preset-bar', style: { marginBottom: '20px' } },
                e('span', { className: 'preset-title' }, 'LoRA Presets:'),
                LORA_PRESETS.map((p, idx) =>
                  e('button', {
                    key: idx,
                    className: 'preset-chip',
                    onClick: () => {
                      setLoraConfig({ ...loraConfig, rank: p.rank, alpha: p.alpha, quantization: p.quantization, lr: p.lr, epochs: p.epochs });
                      showToast(`Applied LoRA Preset: ${p.name}`);
                    }
                  }, p.name)
                )
              ),
              e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', marginBottom: '8px' } }, 'Unified LoRA & QLoRA Fine-Tuner Abstraction'),
              e('p', { style: { color: 'var(--text-muted)', fontSize: '13px' } },
                'Attach adapter weights to native PyTorch models with dynamic gradient accumulation.'
              ),
              e('div', { className: 'finetune-grid' },
                e('div', { className: 'field-group' },
                  e('label', null, 'Dataset Path (JSONL / CSV):'),
                  e('input', { value: loraConfig.dataset, onChange: ev => setLoraConfig({ ...loraConfig, dataset: ev.target.value }) })
                ),
                e('div', { className: 'field-group' },
                  e('label', null, 'Quantization Precision:'),
                  e('select', { value: loraConfig.quantization, onChange: ev => setLoraConfig({ ...loraConfig, quantization: ev.target.value }) },
                    e('option', { value: '4-bit NF4' }, '4-bit NF4 (QLoRA - Ultra Low VRAM)'),
                    e('option', { value: 'FP8 E4M3' }, 'FP8 E4M3 (Ada / Blackwell Tensor Cores)'),
                    e('option', { value: '8-bit' }, '8-bit Int8 Quantization'),
                    e('option', { value: 'BF16' }, '16-bit BF16 Brain Float Precision'),
                    e('option', { value: 'FP16 Half' }, '16-bit FP16 Half Precision')
                  )
                ),
                e('div', { className: 'field-group' },
                  e('label', null, `LoRA Rank r (${loraConfig.rank}):`),
                  e('input', { type: 'range', min: 4, max: 128, step: 4, value: loraConfig.rank, onChange: ev => setLoraConfig({ ...loraConfig, rank: parseInt(ev.target.value) }) })
                ),
                e('div', { className: 'field-group' },
                  e('label', null, `LoRA Alpha Scaling (${loraConfig.alpha}):`),
                  e('input', { type: 'range', min: 8, max: 256, step: 8, value: loraConfig.alpha, onChange: ev => setLoraConfig({ ...loraConfig, alpha: parseInt(ev.target.value) }) })
                )
              ),
              e('div', { style: { display: 'flex', gap: '10px', marginTop: '10px' } },
                e('button', {
                  className: 'btn-send',
                  style: { flex: 2, height: '44px' },
                  onClick: handleStartFineTuning
                }, 'Start LoRA Fine-Tuning Run'),
                e('button', {
                  className: 'btn-sec',
                  style: { flex: 1, height: '44px', fontSize: '12px' },
                  onClick: () => handleAttachAdapterToEngine(loraConfig)
                }, activeAdapter ? 'Re-attach Adapter' : 'Attach to Engine')
              ),
              activeAdapter && e('div', { style: { marginTop: '14px', padding: '12px 16px', background: '#e3ebd8', border: '1px solid #c5d6b4', borderRadius: '6px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' } },
                e('div', null,
                  e('div', { style: { fontWeight: 700, color: '#2b4c3f', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '5px' } }, [
                    renderIcon('check', 13, { color: '#2b4c3f' }),
                    `${activeAdapter.name} Active in Studio Engine`
                  ]),
                  e('div', { style: { fontSize: '11.5px', color: '#3f6212' } }, `Rank: ${activeAdapter.rank} | Model: ${activeModel} | Precision: ${activeAdapter.quantization || loraConfig.quantization}`)
                ),
                e('button', { className: 'btn-sec', style: { fontSize: '11px', padding: '4px 10px' }, onClick: handleDetachAdapterFromEngine }, 'Detach')
              ),
              fineTuningStatus && e('div', { className: 'notebook-terminal', style: { marginTop: '14px' } }, fineTuningStatus)
            )
          ),

          // Tab: Checkpoints & Model Weights Browser
          activeTab === 'checkpoints' && e('div', { className: 'view-checkpoints' },
            e('div', { className: 'ckpt-card-header' },
              e('div', null,
                e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', marginBottom: '4px' } }, [renderIcon('checkpoints', 18, { marginRight: '8px' }), 'Checkpoint Browser & Model Weights Manager']),
                e('div', { className: 'ckpt-run-status' },
                  e('p', { style: { color: 'var(--text-main)', fontSize: '13px', margin: '0 0 4px' } },
                    [e('strong', { key: 'label' }, 'Loaded weights: '), activeCheckpointName || 'No checkpoint loaded']
                  ),
                  e('p', { style: { color: 'var(--text-muted)', fontSize: '12px', margin: 0 } },
                    `Current run: Step ${(metrics.step || 0).toLocaleString()} · ${(metrics.tokens_trained || 0).toLocaleString()} tokens trained · ${(datasetInfo.total_tokens || 0).toLocaleString()} tokens ingested`
                  )
                )
              ),
              e('div', { className: 'ckpt-save-bar' },
                e('input', {
                  className: 'ckpt-input',
                  placeholder: `triune_studio_step_${metrics.step || 0}.pt (or custom name)`,
                  value: saveCkptName,
                  onChange: ev => setSaveCkptName(ev.target.value)
                }),
                e('button', {
                  className: 'btn-action',
                  style: { background: 'var(--accent-olive, #2b4c3f)', color: '#fff' },
                  onClick: handleSaveCheckpointCustom
                }, [renderIcon('checkpoints', 13, { marginRight: '6px' }), 'Save Checkpoint']),
                e('button', {
                  className: 'btn-sec',
                  onClick: fetchCheckpoints
                }, [renderIcon('refresh', 13, { marginRight: '6px' }), 'Refresh'])
              )
            ),
            e('div', { className: 'ckpt-table-wrap', role: 'region', 'aria-label': 'Checkpoint files', tabIndex: 0 },
              e('table', { className: 'ckpt-table' },
              e('thead', null,
                e('tr', null,
                  e('th', null, 'Checkpoint File'),
                  e('th', null, 'Location'),
                  e('th', null, 'File Size'),
                  e('th', null, 'Step'),
                  e('th', null, 'Cumulative Tokens'),
                  e('th', null, 'Modified'),
                  e('th', null, 'Engine Status'),
                  e('th', { style: { textAlign: 'right' } }, 'Actions')
                )
              ),
              e('tbody', null,
                checkpoints.length > 0
                  ? checkpoints.map((c, idx) => {
                      const isActive = c.is_active || (activeCheckpointName && (activeCheckpointName === c.filename || activeCheckpointName.includes(c.filename)));
                      return e('tr', { key: c.id || idx, style: isActive ? { background: 'rgba(43, 76, 63, 0.06)' } : {} },
                        e('td', { style: { fontWeight: isActive ? 700 : 500, fontFamily: 'var(--font-mono)' } },
                          [renderIcon('checkpoints', 13, { marginRight: '6px', color: 'var(--accent-terracotta)' }), c.filename]
                        ),
                        e('td', null, e('span', { className: 'badge-folder' }, c.folder)),
                        e('td', null, c.size_formatted || `${c.size_mb} MB`),
                        e('td', null, c.step !== null && c.step !== undefined ? `Step ${c.step.toLocaleString()}` : '—'),
                        e('td', { style: { fontFamily: 'var(--font-mono)', fontSize: '12px', fontWeight: 600, color: 'var(--text-main)' } },
                          c.tokens_formatted || (c.tokens ? `${c.tokens.toLocaleString()} tok` : (c.step ? `${(c.step * 1024).toLocaleString()} tok` : '—'))
                        ),
                        e('td', { style: { color: 'var(--text-muted)', fontSize: '12px' } }, c.mtime || '—'),
                        e('td', null,
                          isActive
                            ? e('span', { className: 'badge-ckpt-active' }, '● Loaded & Active')
                            : e('span', { className: 'badge-ckpt-idle' }, 'On Disk')
                        ),
                        e('td', { style: { textAlign: 'right' } },
                          e('div', { style: { display: 'flex', gap: '8px', justifyContent: 'flex-end' } },
                            e('button', {
                              className: 'btn-sec',
                              disabled: isActive || isLoadingCkpt,
                              style: {
                                padding: '4px 10px',
                                fontSize: '11px',
                                color: isActive ? 'var(--text-muted)' : 'var(--accent-sage)',
                                fontWeight: 600
                              },
                              onClick: () => handleLoadCheckpoint(c.path)
                            }, isActive ? 'Active' : [renderIcon('download', 11, { marginRight: '4px' }), 'Load into Engine']),
                            e('button', {
                              className: 'btn-sec',
                              style: { padding: '4px 8px', fontSize: '11px', color: 'var(--text-main)' },
                              title: 'Export this checkpoint file as SafeTensors',
                              onClick: () => handleExportCheckpoint(c.path)
                            }, [renderIcon('modules', 11, { marginRight: '4px' }), 'Export']),
                            e('button', {
                              className: 'btn-sec',
                              style: { padding: '4px 8px', fontSize: '11px', color: '#dc2626' },
                              onClick: () => handleDeleteCheckpoint(c.path)
                            }, renderIcon('trash', 13))
                          )
                        )
                      );
                    })
                  : e('tr', null,
                      e('td', { colSpan: 8, style: { textAlign: 'center', padding: '24px', color: 'var(--text-muted)' } },
                        'No checkpoints found on disk. Train some steps and click "Save Checkpoint".'
                      )
                    )
              )
              )
            )
          ),

          // Tab 4: Model Zoo & Architectural Presets
          activeTab === 'models' && e('div', { className: 'view-models' },
            e('div', { style: { marginBottom: '14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' } },
              e('div', null,
                e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', margin: '0 0 4px 0' } }, [renderIcon('models', 18, { marginRight: '8px' }), 'Triune Architecture Zoo & Model Presets']),
                e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', margin: 0 } },
                  'Pre-configured Mixture-of-Experts architectures with real dynamic parameter counts, hardware budgets, and one-click IDE synchronization.'
                )
              ),
              e('div', { className: 'badge-folder' }, `${MODEL_PRESETS.length} Verified Architectures`)
            ),
            e('div', { className: 'models-grid-rich' },
              MODEL_PRESETS.map((m, idx) => {
                const isCurrent = activeModel === m.id;
                return e('div', {
                  key: m.id || idx,
                  className: `card-model-rich ${isCurrent ? 'active-model' : ''}`
                },
                  // Card Top
                  e('div', null,
                    e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px', marginBottom: '6px' } },
                      e('div', null,
                        e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                          e('h3', { className: 'model-title', style: { margin: 0, fontSize: '16px' } }, m.name),
                          isCurrent && e('span', { className: 'badge-ckpt-active', style: { fontSize: '10.5px', padding: '1px 6px' } }, '● Active')
                        ),
                        e('span', { style: { fontSize: '11px', color: 'var(--accent-terracotta, #9a3412)', fontWeight: 600, fontFamily: 'var(--font-mono)' } }, m.tier)
                      ),
                      e('span', { className: 'badge', style: { background: 'var(--bg-surface)', borderColor: 'var(--border-color)', fontWeight: 700 } }, m.params)
                    ),
                    e('p', { style: { color: 'var(--text-muted)', fontSize: '12.5px', lineHeight: '1.4', margin: '6px 0 10px 0' } }, m.desc),

                    // Architectural Spec Chips
                    e('div', { className: 'model-spec-chips' },
                      e('span', { className: 'model-spec-chip' }, `${m.layers} Layers`),
                      e('span', { className: 'model-spec-chip' }, `${m.hidden_dim} Hidden Dim`),
                      e('span', { className: 'model-spec-chip' }, `${m.heads} Heads (${m.head_dim}d)`),
                      e('span', { className: 'model-spec-chip' }, `MoE: ${m.experts_routed}R / ${m.experts_shared}S`),
                      e('span', { className: 'model-spec-chip' }, `Exits: ${m.exits}`)
                    ),

                  ),

                  // Card Actions
                  e('div', { className: 'model-actions', style: { marginTop: '12px', display: 'flex', flexWrap: 'wrap', gap: '6px' } },
                    e('button', {
                      className: isCurrent ? 'btn-action' : 'btn-sec',
                      style: isCurrent ? { background: 'var(--accent-olive, #2b4c3f)', color: '#fff', flex: '1 1 100%' } : { flex: '1 1 100%', fontWeight: 600, borderColor: 'var(--accent-olive, #2b4c3f)', color: 'var(--accent-olive, #2b4c3f)' },
                      onClick: () => handleSelectModelPreset(m.id)
                    }, isCurrent ? [renderIcon('check', 13, { marginRight: '6px' }), 'Active Architecture (Synced)'] : [renderIcon('play', 13, { marginRight: '6px' }), 'Load Architecture & Sync IDE']),
                    e('button', {
                      className: 'btn-sec',
                      style: { flex: '1 1 45%' },
                      disabled: true,
                      title: 'GGUF conversion is not implemented for Triune architectures yet.',
                      'aria-label': 'GGUF export unavailable: converter not implemented'
                    }, 'GGUF Export Unavailable'),
                    e('button', {
                      className: 'btn-sec',
                      style: { flex: '1 1 45%' },
                      disabled: !isCurrent,
                      title: isCurrent ? 'Export weights currently loaded in the engine.' : 'Activate this architecture first; export always writes the active engine weights.',
                      onClick: () => handleExportModel(m.id, 'safetensors')
                    }, isCurrent ? 'Export Active SafeTensors' : 'Activate to Export')
                  )
                );
              })
            )
          ),

          // Tab 5: Dataset Explorer & Hugging Face Streaming Hub
          activeTab === 'datasets' && e('div', { className: 'view-datasets' },
            // CARD 1: Pull & Stream Any Dataset (Hugging Face / Direct Web URL)
            e('div', { className: 'card-dataset stream-hf-card' },
              e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px', marginBottom: '12px' } },
                e('div', null,
                  e('div', { style: { display: 'flex', alignItems: 'center', gap: '10px' } },
                    e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', margin: 0 } }, [renderIcon('hub', 18, { marginRight: '8px' }), 'Pull & Stream Any Dataset (Hugging Face / Web URL)']),
                    e('span', { className: 'stream-badge-pulse' }, 'Zero Disk Overhead')
                  ),
                  e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', margin: '6px 0 0 0' } },
                    'Stream any Hugging Face repo (e.g. roneneldan/TinyStories) or external web URL (.jsonl / .parquet / .csv / .txt) directly into the GPU autograd engine with 0 hardcoded dummy fallbacks.'
                  )
                ),
                byokKeys.huggingface
                  ? e('div', { className: 'badge-ckpt-active', style: { display: 'flex', alignItems: 'center', gap: '6px' } },
                      e('span', null, [renderIcon('byok', 13, { marginRight: '5px' }), 'Authenticated HF Token Active'])
                    )
                  : e('button', {
                      className: 'btn-sec',
                      style: { fontSize: '11px', padding: '4px 10px', color: '#78350f', borderColor: '#edd5a6', background: '#faeed7' },
                      onClick: () => setActiveTab('byok')
                    }, [renderIcon('alert', 12, { marginRight: '5px' }), 'Add HF Token (BYOK)'])
              ),

              // Quick Presets
              e('div', { className: 'hf-preset-container', style: { marginBottom: '14px' } },
                e('span', { style: { fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', marginRight: '6px' } }, 'Popular Presets:'),
                POPULAR_DATASETS.map((p, idx) =>
                  e('button', {
                    key: idx,
                    className: `preset-chip ${activeDataset === p.id ? 'active-preset' : ''}`,
                    style: { cursor: 'pointer', padding: '4px 10px', fontSize: '12px', borderRadius: '16px' },
                    onClick: () => {
                      setHfDatasetInput(p.id);
                      setHfConfigInput(p.config);
                      setHfSplitInput(p.split);
                      setHfTextColInput(p.col);
                      handleConnectHfStream(p.id, p.config, p.split, p.col);
                    }
                  }, p.name)
                )
              ),

              // Dataset Connection Form
              e('div', { className: 'hf-stream-form-grid' },
                e('div', { className: 'form-group', style: { flex: '2 1 280px' } },
                  e('label', { style: { fontSize: '12px', fontWeight: 600, color: 'var(--text-main)', display: 'block', marginBottom: '4px' } },
                    'Hugging Face Repository ID or Web URL:'
                  ),
                  e('input', {
                    style: { width: '100%', height: '38px', padding: '0 12px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px', color: 'var(--text-main)' },
                    placeholder: 'e.g., roneneldan/TinyStories, HuggingFaceFW/fineweb-edu, or https://.../data.jsonl',
                    value: hfDatasetInput,
                    onChange: ev => setHfDatasetInput(ev.target.value)
                  })
                ),
                e('div', { className: 'form-group', style: { flex: '1 1 140px' } },
                  e('label', { style: { fontSize: '12px', fontWeight: 600, color: 'var(--text-main)', display: 'block', marginBottom: '4px' } },
                    'Config / Subset:'
                  ),
                  e('input', {
                    style: { width: '100%', height: '38px', padding: '0 12px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px', color: 'var(--text-main)' },
                    placeholder: 'Optional (e.g. sample-10BT)',
                    value: hfConfigInput,
                    onChange: ev => setHfConfigInput(ev.target.value)
                  })
                ),
                e('div', { className: 'form-group', style: { flex: '0 1 90px' } },
                  e('label', { style: { fontSize: '12px', fontWeight: 600, color: 'var(--text-main)', display: 'block', marginBottom: '4px' } },
                    'Split:'
                  ),
                  e('input', {
                    style: { width: '100%', height: '38px', padding: '0 12px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px', color: 'var(--text-main)' },
                    placeholder: 'train',
                    value: hfSplitInput,
                    onChange: ev => setHfSplitInput(ev.target.value)
                  })
                ),
                e('div', { className: 'form-group', style: { flex: '1 1 120px' } },
                  e('label', { style: { fontSize: '12px', fontWeight: 600, color: 'var(--text-main)', display: 'block', marginBottom: '4px' } },
                    'Text Column:'
                  ),
                  e('input', {
                    style: { width: '100%', height: '38px', padding: '0 12px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px', color: 'var(--text-main)' },
                    placeholder: 'Auto-detect',
                    value: hfTextColInput,
                    onChange: ev => setHfTextColInput(ev.target.value)
                  })
                ),
                e('div', { style: { display: 'flex', alignItems: 'flex-end', flex: '0 0 auto' } },
                  e('button', {
                    className: 'btn-action',
                    disabled: isConnectingHf,
                    style: { height: '38px', padding: '0 18px', background: 'var(--accent-olive, #2b4c3f)', color: '#fff', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px' },
                    onClick: () => handleConnectHfStream()
                  }, isConnectingHf ? 'Connecting Stream...' : [renderIcon('play', 13, { marginRight: '6px' }), 'Connect & Stream'])
                )
              ),

              // Active Stream Telemetry & Live Verification Box
              e('div', { className: 'stream-ingestion-box', style: { marginTop: '16px' } },
                e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' } },
                  e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                    e('span', { className: 'stream-badge-pulse' },
                      datasetInfo.is_streaming || hfStreamResult ? 'Stream Live & Pre-Buffered' : 'Stream Ready'
                    ),
                    e('span', { style: { fontWeight: 700, fontSize: '13px' } },
                      `Source: ${datasetInfo.name || hfDatasetInput} (${datasetInfo.type || 'hf_streaming'})`
                    ),
                    (hfStreamResult && hfStreamResult.text_column) && e('span', { className: 'badge-folder' },
                      `Column: ${hfStreamResult.text_column}`
                    )
                  ),
                  e('span', { style: { fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' } },
                    `Buffer: ${hfStreamResult ? hfStreamResult.buffered_chunks || 20 : 20} chunks ready | Cumulative: ${(datasetInfo.total_tokens || metrics.tokens_trained || 0).toLocaleString()} tok`
                  )
                ),
                e('div', { style: { fontSize: '11px', color: 'var(--text-muted)' } },
                  'Live Ingested Micro-Batch Decoded Sample Preview (Zero dummy strings):'
                ),
                e('div', { className: 'stream-preview-text' },
                  activeBatchPreview || (hfStreamResult && hfStreamResult.sample_preview)
                    ? `"${cleanText(activeBatchPreview || (hfStreamResult && hfStreamResult.sample_preview))}"`
                    : '"Waiting for stream connection or next training step to decode batch..."'
                )
              )
            ),

            // CARD 2: Live BPE Tokenizer Sandbox & Registered Datasets Table
            e('div', { className: 'card-dataset', style: { marginTop: '20px' } },
              e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', marginBottom: '8px' } }, 'Dataset Explorer & Tokenizer Playground'),
              e('p', { style: { color: 'var(--text-muted)', fontSize: '13px' } }, 'Inspect dataset statistics and test live BPE tokenization.'),
              e('div', { style: { marginTop: '20px' } },
                e('label', { style: { fontSize: '12.5px', color: 'var(--text-muted)' } }, 'Live Tokenizer Input Sandbox:'),
                e('input', {
                  style: { width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-color)', color: 'var(--text-main)', padding: '10px', borderRadius: '6px', marginTop: '6px' },
                  value: sampleText,
                  onChange: ev => setSampleText(ev.target.value)
                }),
                e('div', { role: 'status', 'aria-live': 'polite', style: { margin: '7px 2px', fontSize: '11px', color: tokens.length ? 'var(--text-muted)' : 'var(--text-dim)' } }, tokenizerStatus),
                e('div', { className: 'token-chip-container' },
                  tokens.map(t => {
                    const raw = t.text || '';
                    const isPrefixed = raw.startsWith('Ġ') || raw.startsWith(' ');
                    const clean = raw.replace(/^[Ġ ]/, '');
                    return e('span', { key: t.id, className: 'token-chip' },
                      isPrefixed && e('span', { style: { opacity: 0.5, marginRight: '3px', fontWeight: 700 } }, '␣'),
                      `${clean || '␣'} [ID:${t.id}]`
                    );
                  })
                )
              ),
              e('div', { style: { marginTop: '20px' } },
                e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' } },
                  e('h4', { style: { margin: 0, fontFamily: 'Newsreader', fontSize: '16px' } }, 'Registered Training Corpora & Streams'),
                  e('span', { style: { fontSize: '11px', color: 'var(--text-muted)' } }, '1-Click Engine Streaming or Visual DAG Insertion')
                ),
                e('div', { className: 'local-dataset-register-bar', style: { display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', padding: '10px 14px', background: 'var(--bg-surface)', border: '1px solid var(--border-color)', borderRadius: '6px' } },
                  e('div', { style: { display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12.5px', fontWeight: 600, color: 'var(--text-main)' } },
                    renderIcon('folder', 14),
                    'Register Local Corpora File:'
                  ),
                  e('input', {
                    style: { flex: '1 1 240px', height: '34px', padding: '0 12px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '12px', color: 'var(--text-main)' },
                    placeholder: 'e.g. data/fineweb_sample.jsonl or datasets/corpus.parquet',
                    value: localDatasetInput,
                    onChange: ev => setLocalDatasetInput(ev.target.value),
                    onKeyDown: ev => { if (ev.key === 'Enter') handleRegisterLocalDataset(); }
                  }),
                  e('button', {
                    className: 'btn-action',
                    style: { height: '34px', padding: '0 14px', background: 'var(--accent-olive, #2b4c3f)', color: '#fff', fontSize: '12px', fontWeight: 600 },
                    onClick: handleRegisterLocalDataset
                  }, 'Register & Index')
                ),
                e('table', { className: 'table-data' },
                  e('thead', null,
                    e('tr', null,
                      e('th', null, 'ID'),
                      e('th', null, 'Dataset Name'),
                      e('th', null, 'Token Count'),
                      e('th', null, 'Status'),
                      e('th', { style: { textAlign: 'right' } }, 'Actions')
                    )
                  ),
                  e('tbody', null,
                    (datasetList.length > 0 ? datasetList : [
                      { id: '1', name: 'HuggingFaceFW/fineweb-edu', path: 'HuggingFaceFW/fineweb-edu', tokens: '10,000,000,000', status: 'Streaming Active' },
                      { id: '2', name: 'wikitext-103-raw-v1', path: 'wikitext-103-raw-v1', tokens: '103,000,000', status: 'Cached Local' },
                      { id: '3', name: 'roneneldan/TinyStories', path: 'roneneldan/TinyStories', tokens: '2,500,000,000', status: 'Ready to Stream' }
                    ]).map((ds, idx) => {
                      const isActive = (activeDataset === ds.name || activeDataset === ds.path || (ds.status && ds.status.includes('Active')));
                      return e('tr', { key: ds.id || idx, style: isActive ? { background: 'rgba(43, 76, 63, 0.05)' } : {} },
                        e('td', null, ds.id || (idx + 1)),
                        e('td', { style: { fontWeight: isActive ? 700 : 500 } }, ds.name),
                        e('td', null, ds.tokens),
                        e('td', null, e('span', { className: isActive ? 'pill online' : 'pill' }, isActive ? 'Active in Studio' : ds.status)),
                        e('td', { style: { textAlign: 'right' } },
                          e('div', { style: { display: 'flex', gap: '6px', justifyContent: 'flex-end', alignItems: 'center' } },
                            isActive
                              ? e('span', { style: { fontSize: '11px', color: 'var(--accent-sage)', fontWeight: 600, marginRight: '4px', display: 'inline-flex', alignItems: 'center', gap: '3px' } }, [renderIcon('check', 11), 'Active'])
                              : e('button', {
                                  className: 'btn-action',
                                  style: { padding: '4px 10px', fontSize: '11px', background: 'var(--accent-olive, #2b4c3f)', color: '#fff' },
                                  title: 'Stream into PyTorch CUDA engine immediately',
                                  onClick: () => {
                                    setHfDatasetInput(ds.path || ds.name);
                                    handleConnectHfStream(ds.path || ds.name, '', 'train', 'text');
                                  }
                                }, [renderIcon('play', 11, { marginRight: '4px' }), 'Stream in Engine']),
                            e('button', {
                              className: 'btn-sec',
                              style: { padding: '4px 10px', fontSize: '11px' },
                              title: 'Add as a streaming node to the Visual DAG canvas',
                              onClick: () => handleAddDatasetNodeToDAG(ds)
                            }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Add to DAG'])
                          )
                        )
                      );
                    })
                  )
                )
              )
            )
          ),

          // Tab 7: Python Sandbox
          activeTab === 'notebook' && e('div', { className: 'view-notebook' },
            e('div', { className: 'card-notebook' },
              e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', marginBottom: '6px' } }, 'Sandboxed Interactive Python Environment'),
              e('p', { style: { color: 'var(--text-muted)', fontSize: '13px' } },
                'Execute Python snippets safely to inspect model layers, VRAM stats, or run DAG pipelines.'
              ),
              e('textarea', {
                className: 'notebook-textarea',
                value: notebookCode,
                onChange: ev => setNotebookCode(ev.target.value)
              }),
              e('button', { className: 'btn-execute', onClick: handleExecuteNotebook, disabled: isExecutingNotebook },
                isExecutingNotebook ? 'Executing in PythonSandbox...' : 'Run Code in PythonSandbox'
              ),
              e('pre', { className: 'notebook-terminal' }, notebookOutput)
            )
          ),

          // Tab 8: BYOK Manager
          activeTab === 'byok' && e('div', { className: 'view-byok' },
            e('div', { className: 'card-byok' },
              e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', marginBottom: '6px' } }, 'Bring-Your-Own-Key (BYOK) Subscription Manager'),
              e('p', { style: { color: 'var(--text-muted)', fontSize: '13px' } },
                'Add provider credentials, verify them against the provider, then choose that provider in Chat & Playground. Credentials are stored in the operating system credential vault when available.'
              ),
              e('div', { className: 'byok-grid' },
                ['openai', 'anthropic', 'gemini'].map(prov =>
                  e('div', { key: prov, className: 'field-group' },
                    e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' } },
                      e('label', null, `${prov.toUpperCase()} API Key:`),
                      e('span', { style: { fontSize: '11px', color: byokStatus[prov] === 'Connection verified' ? 'var(--accent-sage)' : (byokStatus[prov] ? 'var(--accent-bronze)' : 'var(--text-muted)') } },
                        byokStatus[prov] || (byokConfigured[prov] ? 'Saved — not tested this session' : 'Not configured'))
                    ),
                    e('input', {
                      type: 'password',
                      placeholder: byokConfigured[prov] ? 'Saved key — enter a new key to replace' : `Enter ${prov} key...`,
                      value: byokKeys[prov],
                      onChange: ev => setByokKeys({ ...byokKeys, [prov]: ev.target.value })
                    }),
                    e('button', {
                      className: 'btn-sec',
                      style: { marginTop: '8px', width: '100%' },
                      onClick: () => testBYOKKey(prov)
                    }, 'Verify Provider Connection')
                  )
                )
              ),

              // Dedicated Hugging Face Token Card
              e('div', { className: 'hf-token-highlight-box', style: { marginTop: '20px' } },
                e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' } },
                  e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                    renderIcon('hub', 20, { color: 'var(--text-main)' }),
                    e('label', { style: { fontFamily: 'var(--font-serif)', fontWeight: 700, fontSize: '14px', color: 'var(--text-main)' } }, 'HUGGING FACE USER ACCESS TOKEN (HF_TOKEN)'),
                    e('span', { className: 'badge-folder' }, 'Full Gated Dataset Streaming Access')
                  ),
                  (byokStatus['huggingface'] || byokConfigured.huggingface) && e('span', {
                    style: {
                      fontSize: '12px',
                      fontWeight: 600,
                      color: byokStatus['huggingface'] === 'Connection verified' ? 'var(--accent-olive)' : 'var(--text-muted)'
                    }
                  }, byokStatus['huggingface'] || 'Saved — not tested this session')
                ),
                e('p', { style: { fontSize: '12.5px', color: 'var(--text-muted)', margin: '0 0 12px 0', lineHeight: '1.5' } },
                  'Authenticates directly against huggingface.co/api/whoami-v2. Unlocks gated datasets (e.g. StarCoder, LLaMA, gated fine-web) and eliminates Hugging Face Hub 429 rate limit throttling during streaming training.'
                ),
                e('div', { style: { display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' } },
                  e('input', {
                    type: 'password',
                    style: { flex: '1 1 280px', height: '40px', padding: '0 12px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px', color: 'var(--text-main)' },
                    placeholder: byokConfigured.huggingface ? 'Saved token — enter a new token to replace' : 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
                    value: byokKeys['huggingface'] || '',
                    onChange: ev => setByokKeys({ ...byokKeys, huggingface: ev.target.value })
                  }),
                  e('button', {
                    className: 'btn-action',
                    style: { height: '40px', padding: '0 16px', background: 'var(--accent-olive, #2b4c3f)', color: '#fff', fontWeight: 600 },
                    onClick: () => testBYOKKey('huggingface')
                  }, 'Verify Token with HF Hub'),
                  e('a', {
                    href: 'https://huggingface.co/settings/tokens',
                    target: '_blank',
                    rel: 'noreferrer',
                    className: 'btn-sec',
                    style: { height: '40px', padding: '0 12px', display: 'flex', alignItems: 'center', textDecoration: 'none', fontSize: '12px' }
                  }, 'Get HF Token ↗')
                )
              ),

              // Dedicated Weights & Biases Telemetry Card
              e('div', { className: 'hf-token-highlight-box', style: { marginTop: '20px', borderLeft: '4px solid #d97706' } },
                e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' } },
                  e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                    renderIcon('train', 20, { color: '#d97706' }),
                    e('label', { style: { fontFamily: 'var(--font-serif)', fontWeight: 700, fontSize: '14px', color: 'var(--text-main)' } }, 'WEIGHTS & BIASES API KEY (WANDB_API_KEY)'),
                    e('span', { className: 'badge-folder', style: { background: 'rgba(217, 119, 6, 0.15)', color: '#d97706', borderColor: 'rgba(217, 119, 6, 0.3)' } }, 'Live Cloud Telemetry & Experiment Dashboard')
                  ),
                  (byokStatus['wandb'] || byokConfigured.wandb) && e('span', {
                    style: {
                      fontSize: '12px',
                      fontWeight: 600,
                      color: byokStatus['wandb'] === 'Connection verified' ? 'var(--accent-sage)' : 'var(--text-muted)'
                    }
                  }, byokStatus['wandb'] || 'Saved — not tested this session')
                ),
                e('p', { style: { fontSize: '12.5px', color: 'var(--text-muted)', margin: '0 0 12px 0', lineHeight: '1.5' } },
                  'Synchronizes live training metrics to your Weights & Biases cloud dashboard (wandb.ai). Automatically logs step loss, LM loss, MoE router loss, balance loss, learning rate, GPU throughput (tok/s), VRAM usage, and depth exit distribution (Reflex/Limbic/Cortex).'
                ),
                e('div', { style: { display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' } },
                  e('input', {
                    type: 'password',
                    style: { flex: '1 1 280px', height: '40px', padding: '0 12px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px', color: 'var(--text-main)' },
                    placeholder: byokConfigured.wandb ? 'Saved W&B key — enter a new key to replace' : 'Enter Weights & Biases API key...',
                    value: byokKeys['wandb'] || '',
                    onChange: ev => setByokKeys({ ...byokKeys, wandb: ev.target.value })
                  }),
                  e('button', {
                    className: 'btn-action',
                    style: { height: '40px', padding: '0 16px', background: '#d97706', color: '#fff', fontWeight: 600 },
                    onClick: () => testBYOKKey('wandb')
                  }, 'Verify W&B Login'),
                  e('a', {
                    href: 'https://wandb.ai/authorize',
                    target: '_blank',
                    rel: 'noreferrer',
                    className: 'btn-sec',
                    style: { height: '40px', padding: '0 12px', display: 'flex', alignItems: 'center', textDecoration: 'none', fontSize: '12px' }
                  }, 'Get W&B Key ↗'),
                  wandbUrl && e('a', {
                    href: wandbUrl,
                    target: '_blank',
                    rel: 'noreferrer',
                    className: 'btn-sec',
                    style: { height: '40px', padding: '0 12px', display: 'flex', alignItems: 'center', textDecoration: 'none', fontSize: '12px', borderColor: '#d97706', color: '#d97706' }
                  }, 'Open W&B Run ↗')
                )
              ),

              e('button', {
                className: 'btn-send',
                style: { width: '100%', height: '46px', marginTop: '20px' },
                onClick: handleSaveBYOK
                  }, 'Save Provider Credentials')
            )
          ),

          activeTab === 'customize' && e('div', { className: 'view-appearance' },
            e('div', { className: 'appearance-hero' },
              e('div', null,
                e('span', { className: 'appearance-eyebrow' }, 'STUDIO PERSONALIZATION'),
                e('h2', null, 'Make the workspace yours'),
                e('p', null, 'Choose a visual edition and color theme. Your choice is saved on this device and applied across Studio.')
              ),
              e('div', { className: 'appearance-current' },
                e('span', null, 'CURRENT THEME'),
                e('strong', null, (APPEARANCE_THEMES.find(theme => theme.id === studioTheme) || APPEARANCE_THEMES[0]).name),
                e('small', null, `${studioTheme.startsWith('classic') ? 'Classic' : 'Modern'} · ${/dark|forest/.test(studioTheme) ? 'Dark' : 'Light'}`)
              )
            ),
            e('div', { className: 'appearance-section-heading' },
              e('div', null,
                e('h3', null, 'Theme marketplace'),
                e('p', null, 'Curated, built-in themes—no downloads, third-party CSS, or executable theme packages.')
              ),
              e('div', { className: 'appearance-mode-switch', role: 'group', 'aria-label': 'Filter themes by appearance' },
                ['All', 'Light', 'Dark'].map(mode => e('button', {
                  key: mode,
                  type: 'button',
                  className: `appearance-filter ${appearanceFilter === mode ? 'active' : ''}`,
                  onClick: () => setAppearanceFilter(mode)
                }, mode))
              )
            ),
            e('div', { className: 'appearance-theme-grid' },
              APPEARANCE_THEMES.filter(theme => appearanceFilter === 'All' || theme.mode === appearanceFilter).map(theme =>
                e('article', { key: theme.id, className: `appearance-theme-card ${studioTheme === theme.id ? 'selected' : ''}` },
                  e('div', { className: `appearance-preview preview-${theme.id}`, 'aria-hidden': 'true' },
                    e('div', { className: 'preview-sidebar' }, e('i', null), e('i', null), e('i', null), e('i', null)),
                    e('div', { className: 'preview-workspace' },
                      e('div', { className: 'preview-topbar' }),
                      e('div', { className: 'preview-content' }, e('i', null), e('i', null), e('i', null))
                    )
                  ),
                  e('div', { className: 'appearance-theme-info' },
                    e('div', { className: 'appearance-theme-title' },
                      e('h4', null, theme.name),
                      studioTheme === theme.id && e('span', { className: 'appearance-applied-badge' }, 'Applied')
                    ),
                    e('div', { className: 'appearance-theme-meta' }, `${theme.edition} edition · ${theme.mode} mode`),
                    e('p', null, theme.description),
                    e('button', {
                      className: studioTheme === theme.id ? 'appearance-apply applied' : 'appearance-apply',
                      type: 'button',
                      disabled: studioTheme === theme.id,
                      onClick: () => { setStudioTheme(theme.id); showToast(`${theme.name} applied.`); }
                    }, studioTheme === theme.id ? 'Currently applied' : 'Apply theme')
                  )
                )
              )
            ),
            e('section', { className: 'appearance-custom-card' },
              e('div', null,
                e('h3', null, 'Personal accent color'),
                e('p', null, 'Optional: choose an accent color on top of any theme. This only changes Studio appearance.')
              ),
              e('div', { className: 'appearance-color-controls' },
                e('label', { className: 'appearance-color-picker' },
                  e('input', {
                    type: 'color',
                    value: customAccent || (APPEARANCE_THEMES.find(theme => theme.id === studioTheme) || APPEARANCE_THEMES[0]).accent,
                    'aria-label': 'Choose personal accent color',
                    onChange: event => setCustomAccent(event.target.value)
                  }),
                  e('span', null, customAccent || 'Theme default')
                ),
                customAccent && e('button', { className: 'appearance-reset', type: 'button', onClick: () => setCustomAccent('') }, 'Reset to theme color')
              )
            ),
            e('section', { className: 'appearance-custom-card extension-marketplace-link' },
              e('div', null,
                e('h3', null, 'Looking for tools and integrations?'),
                e('p', null, 'Triune’s extension marketplace includes curated modules, local plugins, and GitHub or Hugging Face repositories. Remote plugins stay dormant until a signed manifest is verified and you explicitly approve activation.')
              ),
              e('button', {
                className: 'appearance-apply',
                type: 'button',
                onClick: () => { setMarketplaceSubTab('curated'); setActiveTab('modules'); fetchInstalledModules(); searchMarketplace('', 'all', 'all'); }
              }, 'Browse extensions')
            )
          ),

          // Tab 10: Extensions & Repositories Marketplace
          activeTab === 'modules' && e('div', { className: 'view-modules' },
            e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' } },
              e('div', null,
                e('h2', { style: { fontFamily: 'Newsreader', fontSize: '24px', margin: 0 } }, 'Extensions & Integrations'),
                e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', margin: '4px 0 0 0' } }, 'Browse curated Triune modules, community repositories, and the extensions installed in this workspace.')
              ),
              e('div', { style: { display: 'flex', gap: '8px' } },
                e('button', { className: 'btn-sec', onClick: () => { fetchInstalledModules(); searchMarketplace(moduleSearchQuery, moduleFilter, 'all'); checkModuleUpdates(); } }, [renderIcon('refresh', 13, { marginRight: '6px' }), 'Refresh All']),
                e('button', { className: 'btn-sec', onClick: checkModuleUpdates }, [renderIcon('alert', 13, { marginRight: '6px' }), 'Check Updates'])
              )
            ),

            e('details', { className: 'marketplace-advanced-tools' },
              e('summary', null,
                e('span', null, 'Add local or remote extensions'),
                e('small', null, 'Workspace scan · Git / Hugging Face clone')
              ),
              e('div', { className: 'marketplace-advanced-body' },
            // Local Workspace Plugin & Script Scanner Box
            e('div', { className: 'workspace-scanner-box' },
              e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' } },
                e('div', null,
                  e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                    renderIcon('search', 16),
                    e('h3', { style: { fontFamily: 'Newsreader', fontSize: '17px', margin: 0, fontWeight: '600' } }, 'Workspace Plugin & Script Scanner'),
                    e('span', { className: 'badge-update', style: { background: '#e3ebd8', color: '#2b4c3f', borderColor: '#c5d6b4', fontSize: '11px' } }, 'AUTO-LOAD')
                  ),
                  e('p', { style: { color: 'var(--text-muted)', fontSize: '12.5px', margin: '4px 0 0 0' } },
                    'Instantly scan your local Triune project repository (triune/plugins/, scripts/, root) for custom @register_node plugins and custom models.'
                  )
                ),
                e('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } },
                  e('button', {
                    className: 'btn-action',
                    style: { background: 'var(--accent-olive, #2b4c3f)', color: '#fff', fontSize: '12px', padding: '6px 14px' },
                    onClick: handleScanWorkspacePlugins
                  }, [renderIcon('search', 12, { marginRight: '5px' }), 'Scan Local Plugins']),
                  e('button', {
                    className: 'btn-sec',
                    style: { fontSize: '12px', padding: '6px 14px' },
                    onClick: () => setShowWorkspacePluginModal(true)
                  }, [renderIcon('plus', 12, { marginRight: '5px' }), 'Scaffold New Plugin'])
                )
              )
            ),

            // Direct Repository Cloner Box
            e('div', { className: 'repo-cloner-box' },
              e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
                renderIcon('download', 16),
                e('h3', { style: { fontFamily: 'Newsreader', fontSize: '17px', margin: 0, fontWeight: '600' } }, 'Clone Remote Repository (GitHub / Hugging Face)'),
                e('span', { className: 'repo-git-pill', style: { marginLeft: 'auto' } }, 'git clone --depth 1')
              ),
              e('p', { style: { color: 'var(--text-muted)', fontSize: '12.5px', margin: '4px 0 0 0' } }, 'Clone a repository for inspection. Remote Python plugins remain dormant until a trusted signed manifest passes validation and you explicitly approve activation.'),
              e('form', { className: 'repo-cloner-form', onSubmit: handleCloneCustomRepo },
                e('input', {
                  style: { flex: '2 1 280px', height: '38px', padding: '0 12px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px', color: 'var(--text-main)' },
                  placeholder: 'https://github.com/username/repo.git or https://huggingface.co/org/model',
                  value: repoCloneUrl,
                  onChange: ev => setRepoCloneUrl(ev.target.value)
                }),
                e('select', {
                  style: { flex: '1 1 150px', height: '38px', padding: '0 10px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '12.5px', color: 'var(--text-main)' },
                  value: repoCloneType,
                  onChange: ev => setRepoCloneType(ev.target.value)
                },
                  e('option', { value: 'auto' }, 'Auto Detect Type'),
                  e('option', { value: 'plugin' }, 'DAG Plugins'),
                  e('option', { value: 'model' }, 'Model Weights'),
                  e('option', { value: 'adapter' }, 'LoRA Adapters'),
                  e('option', { value: 'dataset' }, 'Datasets (.jsonl)'),
                  e('option', { value: 'general' }, 'General Repos')
                ),
                e('input', {
                  style: { flex: '0 1 120px', height: '38px', padding: '0 10px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '12.5px', color: 'var(--text-main)' },
                  placeholder: 'Branch (main)',
                  value: repoCloneBranch,
                  onChange: ev => setRepoCloneBranch(ev.target.value)
                }),
                e('button', {
                  type: 'submit',
                  className: 'btn-send',
                  disabled: isCloningRepo,
                  style: { height: '38px', padding: '0 18px', fontSize: '13px', whiteSpace: 'nowrap' }
                }, isCloningRepo ? [renderIcon('refresh', 12, { marginRight: '5px' }), 'Cloning...'] : [renderIcon('download', 12, { marginRight: '5px' }), 'Clone & Inspect'])
              )
            ),
              )
            ),

            // Subtabs Navigation
            e('div', { className: 'repo-subtabs' },
              e('button', {
                className: `repo-subtab-btn ${marketplaceSubTab === 'installed' ? 'active' : ''}`,
                onClick: () => setMarketplaceSubTab('installed')
              }, [renderIcon('checkpoints', 13, { marginRight: '6px' }), `Installed & Cloned Repos (${installedModules.length})`]),
              e('button', {
                className: `repo-subtab-btn ${marketplaceSubTab === 'curated' ? 'active' : ''}`,
                onClick: () => setMarketplaceSubTab('curated')
              }, [renderIcon('sparkle', 13, { marginRight: '6px' }), `Curated & Verified (${marketplaceData && marketplaceData.curated ? marketplaceData.curated.length : 0})`]),
              e('button', {
                className: `repo-subtab-btn ${marketplaceSubTab === 'huggingface' ? 'active' : ''}`,
                onClick: () => {
                  setMarketplaceSubTab('huggingface');
                  if (!marketplaceData || !marketplaceData.huggingface || marketplaceData.huggingface.length === 0) {
                    searchMarketplace(moduleSearchQuery || 'triune', moduleFilter, 'huggingface');
                  }
                }
              }, [renderIcon('hub', 13, { marginRight: '6px' }), `Hugging Face Hub (${marketplaceData && marketplaceData.huggingface ? marketplaceData.huggingface.length : 0})`]),
              e('button', {
                className: `repo-subtab-btn ${marketplaceSubTab === 'github' ? 'active' : ''}`,
                onClick: () => {
                  setMarketplaceSubTab('github');
                  if (!marketplaceData || !marketplaceData.github || marketplaceData.github.length === 0) {
                    searchMarketplace(moduleSearchQuery || 'transformer', moduleFilter, 'github');
                  }
                }
              }, [renderIcon('git', 13, { marginRight: '6px' }), `GitHub ML (${marketplaceData && marketplaceData.github ? marketplaceData.github.length : 0})`])
            ),

            // Modrinth & F-Droid Storefront Category Filter Pills
            e('div', { className: 'modules-filter-pills' },
              [
                { id: 'all', label: 'All Modules' },
                { id: 'plugin', label: 'DAG Plugins' },
                { id: 'model', label: 'MoE Weights' },
                { id: 'adapter', label: 'LoRA Adapters' },
                { id: 'dataset', label: 'Streaming Datasets' },
                { id: 'framework', label: 'Acceleration & Quant' }
              ].map(pill =>
                e('button', {
                  key: pill.id,
                  className: `filter-pill ${moduleFilter === pill.id ? 'active' : ''}`,
                  onClick: () => {
                    setModuleFilter(pill.id);
                    searchMarketplace(moduleSearchQuery, pill.id, marketplaceSubTab === 'installed' ? 'all' : marketplaceSubTab);
                  }
                }, pill.label)
              )
            ),

            // Search Bar & Filter Chips
            e('div', { style: { display: 'flex', gap: '10px', marginBottom: '18px', alignItems: 'center', flexWrap: 'wrap' } },
              e('input', {
                style: { flex: '1 1 260px', height: '40px', padding: '0 14px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px', color: 'var(--text-main)' },
                placeholder: marketplaceSubTab === 'huggingface' ? 'Search Hugging Face Hub (e.g. TinyStories, fineweb, gpt2, wikitext)...' : (marketplaceSubTab === 'github' ? 'Search GitHub ML repos (e.g. flash-attention, bitsandbytes)...' : 'Search modules, plugins, adapters, datasets...'),
                value: moduleSearchQuery,
                onChange: ev => setModuleSearchQuery(ev.target.value),
                onKeyDown: ev => { if (ev.key === 'Enter') searchMarketplace(moduleSearchQuery, moduleFilter, marketplaceSubTab === 'installed' ? 'all' : marketplaceSubTab); }
              }),
              e('button', {
                className: 'btn-sec',
                style: { height: '40px', padding: '0 16px', fontSize: '13px' },
                onClick: () => searchMarketplace(moduleSearchQuery, moduleFilter, marketplaceSubTab === 'installed' ? 'all' : marketplaceSubTab)
              }, isSearchingModules ? 'Searching...' : [renderIcon('search', 13, { marginRight: '5px' }), 'Search'])
            ),

            marketplaceError && e('div', { className: 'marketplace-error', role: 'alert' },
              e('span', null, marketplaceError),
              e('button', { type: 'button', onClick: () => searchMarketplace(moduleSearchQuery, moduleFilter, marketplaceSubTab === 'installed' ? 'all' : marketplaceSubTab) }, 'Retry')
            ),

            // Updates Available Banner
            availableUpdates && availableUpdates.length > 0 && e('div', { style: { background: '#faeed7', border: '1px solid #edd5a6', padding: '14px 18px', borderRadius: '8px', marginBottom: '18px' } },
              e('h4', { style: { color: '#78350f', margin: '0 0 6px 0', fontSize: '14px' } }, [renderIcon('alert', 13, { marginRight: '6px' }), `${availableUpdates.length} Module Update(s) Available`]),
              e('div', { style: { display: 'flex', gap: '10px', flexWrap: 'wrap' } },
                (availableUpdates || []).map(up =>
                  e('button', {
                    key: up.id,
                    className: 'btn-sec',
                    style: { fontSize: '12px', background: '#f5e3c3', color: '#78350f', borderColor: '#edd5a6' },
                    onClick: () => installModule(up)
                  }, `Update ${up.name} (${up.current_version} → ${up.latest_version})`)
                )
              )
            ),

            // Subtab 1: Installed & Cloned Repositories
            marketplaceSubTab === 'installed' && e('div', null,
              installedModules.length === 0 ?
                e('div', { style: { textAlign: 'center', padding: '40px 20px', background: 'var(--bg-card)', border: '1px dashed var(--border-color)', borderRadius: '8px' } },
                  renderIcon('modules', 32, { marginBottom: '8px', color: 'var(--text-muted)' }),
                  e('h4', { style: { margin: '0 0 6px 0', fontFamily: 'Newsreader', fontSize: '18px' } }, 'No Repositories or Modules Installed Yet'),
                  e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', maxWidth: '520px', margin: '0 auto 16px auto' } }, 'Use the Cloner bar above to clone any Git or Hugging Face repository, or switch to Curated & Verified to install custom loss DAG plugins, base weights, and datasets.')
                ) :
                e('div', { style: { display: 'flex', flexDirection: 'column', gap: '12px' } },
                  installedModules.map(m =>
                    e('div', { key: m.id, className: 'repo-card-installed' },
                      e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' } },
                        e('div', null,
                          e('div', { style: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' } },
                            e('span', { style: { fontWeight: '700', fontSize: '15px' } }, m.name),
                            e('span', { className: 'repo-git-pill' }, `${m.git_info && m.git_info.branch ? m.git_info.branch : 'local'} • ${(m.git_info && m.git_info.commit ? m.git_info.commit : (m.version || 'v1.0')).slice(0, 8)}`),
                            e('span', { className: 'badge-update', style: { background: 'var(--bg-surface)', color: 'var(--primary)', borderColor: 'var(--border-color)', fontSize: '11px' } }, (m.type || 'module').toUpperCase())
                          ),
                          e('div', { style: { fontSize: '12px', color: 'var(--text-muted)', marginTop: '3px' } }, `By ${m.author || 'User'} • ${m.installed_at || ''}`)
                        ),
                        e('div', { style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' } },
                          m.git_info && m.git_info.is_git && e('button', {
                            className: 'btn-sec',
                            style: { padding: '5px 12px', fontSize: '12px' },
                            disabled: activePullingModId === m.id,
                            onClick: () => handleGitPullRepo(m.id)
                          }, activePullingModId === m.id ? 'Pulling...' : [renderIcon('refresh', 11, { marginRight: '4px' }), 'Git Pull']),
                          ((m.registered_nodes && m.registered_nodes.length > 0) || m.type === 'plugin') && e('button', {
                            className: 'btn-sec',
                            style: { padding: '5px 12px', fontSize: '12px' },
                            onClick: () => handleAddModuleNodesToCanvas(m)
                          }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Add Nodes to Canvas']),
                          ((m.artifacts && m.artifacts.weights && m.artifacts.weights.length > 0) || m.type === 'model') && e('button', {
                            className: 'btn-sec',
                            style: { padding: '5px 12px', fontSize: '12px' },
                            onClick: () => handleLoadModuleWeights(m)
                          }, [renderIcon('models', 11, { marginRight: '4px' }), 'Load Weights']),
                          ((m.artifacts && m.artifacts.datasets && m.artifacts.datasets.length > 0) || m.type === 'dataset') && e('button', {
                            className: 'btn-sec',
                            style: { padding: '5px 12px', fontSize: '12px' },
                            onClick: () => handleStreamModuleDataset(m)
                          }, [renderIcon('play', 11, { marginRight: '4px' }), 'Stream in Engine']),
                          ((m.artifacts && m.artifacts.adapters && m.artifacts.adapters.length > 0) || m.type === 'adapter') && e('button', {
                            className: 'btn-sec',
                            style: { padding: '5px 12px', fontSize: '12px' },
                            onClick: () => handleAttachModuleAdapter(m)
                          }, [renderIcon('finetune', 11, { marginRight: '4px' }), 'Attach in LoRA']),
                          e('button', {
                            className: 'btn-action',
                            style: { padding: '5px 12px', fontSize: '12px', background: 'var(--accent-olive, #2b4c3f)', color: '#fff' },
                            disabled: activeActivatingModId === m.id,
                            onClick: () => handleActivateModule(m)
                          }, activeActivatingModId === m.id ? 'Activating...' : [renderIcon('bolt', 11, { marginRight: '4px' }), 'Activate in Studio']),
                          e('button', { className: 'btn-purge', style: { padding: '5px 10px', fontSize: '12px' }, onClick: () => uninstallModule(m.id) }, [renderIcon('trash', 11, { marginRight: '4px' }), 'Remove'])
                        )
                      ),
                      e('div', { style: { fontSize: '13px', color: 'var(--text-main)', lineHeight: '1.4' } }, m.description || 'Locally installed module repository.'),
                      // Detected Artifacts & Capabilities Row
                      e('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' } },
                        ((m.registered_nodes && m.registered_nodes.length > 0) || (m.artifacts && m.artifacts.nodes && m.artifacts.nodes.length > 0)) &&
                          e('span', { className: 'repo-artifact-badge plugin' }, `DAG Nodes: ${(m.registered_nodes || m.artifacts.nodes).join(', ')}`),
                        m.artifacts && m.artifacts.weights && m.artifacts.weights.length > 0 &&
                          e('span', { className: 'repo-artifact-badge model' }, `Weights: ${m.artifacts.weights.slice(0, 2).join(', ')}`),
                        m.artifacts && m.artifacts.adapters && m.artifacts.adapters.length > 0 &&
                          e('span', { className: 'repo-artifact-badge adapter' }, `Adapter: ${m.artifacts.adapters.join(', ')}`),
                        m.artifacts && m.artifacts.datasets && m.artifacts.datasets.length > 0 &&
                          e('span', { className: 'repo-artifact-badge dataset' }, `Dataset: ${m.artifacts.datasets.join(', ')}`),
                        m.artifacts && m.artifacts.requirements &&
                          e('span', { className: 'repo-git-pill' }, 'requirements.txt'),
                        e('span', { style: { fontSize: '11px', color: 'var(--text-dim)', marginLeft: 'auto' } }, `${m.file_count || 0} files • ${m.size_mb ? m.size_mb + ' MB' : '0 MB'}`)
                      )
                    )
                  )
                )
            ),

            // Subtab 2: Curated & Verified (Modrinth / F-Droid Storefront Style)
            marketplaceSubTab === 'curated' && e('div', null,
              e('div', { className: 'modules-grid' },
                ((marketplaceData && marketplaceData.curated ? marketplaceData.curated : []))
                  .filter(mod => moduleFilter === 'all' || mod.type === moduleFilter)
                  .map(mod => {
                    const icon = mod.type === 'plugin' ? renderIcon('nodes', 22) :
                                 mod.type === 'model' ? renderIcon('models', 22) :
                                 mod.type === 'adapter' ? renderIcon('finetune', 22) :
                                 mod.type === 'dataset' ? renderIcon('datasets', 22) :
                                 renderIcon('modules', 22);
                    return e('div', { key: mod.id, className: 'module-card-modrinth' },
                      e('div', null,
                        e('div', { className: 'modrinth-card-top' },
                          e('div', { className: 'modrinth-icon-box' }, icon),
                          e('div', { className: 'modrinth-meta-wrap' },
                            e('div', { className: 'modrinth-title-row' },
                              e('span', { className: 'modrinth-title' }, mod.name),
                              e('span', { className: 'modrinth-badge verified', style: { display: 'inline-flex', alignItems: 'center' } }, [renderIcon('check', 10, { marginRight: '3px' }), 'VERIFIED CORE']),
                              mod.type === 'plugin' && e('span', { className: 'modrinth-badge dag-ready', style: { display: 'inline-flex', alignItems: 'center' } }, [renderIcon('nodes', 10, { marginRight: '3px' }), 'DAG READY']),
                              mod.requires_cuda && e('span', { className: 'modrinth-badge cuda', style: { display: 'inline-flex', alignItems: 'center' } }, [renderIcon('bolt', 10, { marginRight: '3px' }), 'CUDA'])
                            ),
                            e('div', { className: 'modrinth-author-row' },
                              `By ${mod.author} • v${mod.version || '1.0.0'} • ${(mod.type || '').toUpperCase()}`
                            )
                          )
                        ),
                        e('p', { className: 'modrinth-desc' }, mod.description),
                        e('div', { className: 'module-tags', style: { marginTop: '8px' } },
                          mod.tags && mod.tags.map((t, idx) => e('span', { key: idx, className: 'module-tag' }, t))
                        )
                      ),
                      e('div', { className: 'modrinth-actions-footer' },
                        e('span', { style: { fontSize: '11px', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' } },
                          mod.size_mb ? `${mod.size_mb} MB` : 'Local Core'
                        ),
                        e('div', { className: 'modrinth-actions-group' },
                          (mod.type === 'plugin' || mod.type === 'dataset' || mod.registered_nodes) && e('button', {
                            className: 'btn-sec',
                            style: { fontSize: '11.5px', padding: '4px 10px', display: 'inline-flex', alignItems: 'center' },
                            title: 'Drop node onto Visual DAG Canvas',
                            onClick: () => handleAddModuleNodeToDAG(mod)
                          }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Add to DAG']),
                          mod.type === 'dataset' && e('button', {
                            className: 'btn-sec',
                            style: { fontSize: '11.5px', padding: '4px 10px', display: 'inline-flex', alignItems: 'center' },
                            title: 'Stream into PyTorch CUDA backward graph',
                            onClick: () => handleStreamModuleDataset(mod)
                          }, [renderIcon('play', 11, { marginRight: '4px' }), 'Stream in Engine']),
                          mod.type === 'model' && e('button', {
                            className: 'btn-sec',
                            style: { fontSize: '11.5px', padding: '4px 10px', display: 'inline-flex', alignItems: 'center' },
                            title: 'Load weights into model backbone',
                            onClick: () => handleLoadModuleWeights(mod)
                          }, [renderIcon('models', 11, { marginRight: '4px' }), 'Load Weights']),
                          mod.installed
                            ? e('div', { style: { display: 'flex', gap: '6px', alignItems: 'center' } },
                                e('span', { style: { fontSize: '11.5px', color: 'var(--accent-sage)', fontWeight: '600', display: 'inline-flex', alignItems: 'center' } }, [renderIcon('check', 11, { marginRight: '3px' }), 'Installed']),
                                e('button', { className: 'btn-purge', style: { padding: '4px 8px', fontSize: '11px' }, onClick: () => uninstallModule(mod.id) }, 'Remove')
                              )
                            : e('button', {
                                className: 'btn-send',
                                style: { height: '30px', padding: '0 14px', fontSize: '12px', display: 'inline-flex', alignItems: 'center' },
                                onClick: () => installModule(mod)
                              }, [renderIcon('download', 11, { marginRight: '4px' }), 'Install'])
                        )
                      )
                    );
                  })
              )
            ),

            // Subtab 3: Hugging Face Hub
            marketplaceSubTab === 'huggingface' && e('div', null,
              isSearchingModules ?
                e('div', { style: { textAlign: 'center', padding: '40px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', justifyContent: 'center' } },
                  [renderIcon('refresh', 14, { animation: 'spin 1.5s linear infinite', marginRight: '8px' }), 'Searching Hugging Face Hub...']
                ) :
                (!marketplaceData || !marketplaceData.huggingface || marketplaceData.huggingface.length === 0 ?
                  e('div', { style: { textAlign: 'center', padding: '40px 20px', background: 'var(--bg-card)', border: '1px dashed var(--border-color)', borderRadius: '8px' } },
                    e('div', { style: { marginBottom: '8px', color: 'var(--text-muted)' } }, renderIcon('hub', 36)),
                    e('h4', { style: { margin: '0 0 6px 0', fontFamily: 'Newsreader', fontSize: '18px' } }, 'Search the Hugging Face Hub'),
                    e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', maxWidth: '480px', margin: '0 auto 16px auto' } }, 'Enter a model or dataset keyword in the search bar above (e.g., fineweb, wikitext, starcoder, gpt2) to browse remote models and datasets.'),
                    e('div', { style: { display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' } },
                      ['fineweb-edu', 'wikitext', 'gpt2', 'open-web-math'].map(kw =>
                        e('button', {
                          key: kw,
                          className: 'btn-sec',
                          style: { fontSize: '12px' },
                          onClick: () => { setModuleSearchQuery(kw); searchMarketplace(kw, moduleFilter, 'huggingface'); }
                        }, `Search "${kw}"`)
                      )
                    )
                  ) :
                  e('div', { className: 'modules-grid' },
                    marketplaceData.huggingface.map(hf =>
                      e('div', { key: hf.id, className: 'module-card-modrinth' },
                        e('div', null,
                          e('div', { className: 'modrinth-card-top' },
                            e('div', { className: 'modrinth-icon-box' }, hf.type === 'dataset' ? renderIcon('datasets', 20) : renderIcon('hub', 20)),
                            e('div', { className: 'modrinth-meta-wrap' },
                              e('div', { className: 'modrinth-title-row' },
                                e('span', { className: 'modrinth-title', style: { wordBreak: 'break-all' } }, hf.name),
                                e('span', { className: 'modrinth-badge source-hf', style: { display: 'inline-flex', alignItems: 'center' } }, [renderIcon('hub', 10, { marginRight: '3px' }), 'HF Hub']),
                                e('span', { className: 'badge-update', style: { background: '#faeed7', color: '#78350f', borderColor: '#edd5a6', fontSize: '10.5px' } }, (hf.type || 'model').toUpperCase())
                              ),
                              e('div', { className: 'modrinth-author-row', style: { display: 'flex', alignItems: 'center', gap: '6px' } },
                                `Author: ${hf.author || 'HF'} • `,
                                renderIcon('heart', 11, { marginRight: '2px' }),
                                `${hf.likes || 0} • `,
                                renderIcon('download', 11, { marginRight: '2px' }),
                                `${(hf.downloads || 0).toLocaleString()} downloads`
                              )
                            )
                          ),
                          e('p', { className: 'modrinth-desc' }, hf.description || 'Hugging Face hub repository resource.'),
                          e('div', { className: 'module-tags', style: { marginTop: '8px' } },
                            hf.tags && hf.tags.slice(0, 4).map((t, idx) => e('span', { key: idx, className: 'module-tag' }, t))
                          )
                        ),
                        e('div', { className: 'modrinth-actions-footer' },
                          e('a', { href: hf.repo_url, target: '_blank', rel: 'noreferrer', style: { fontSize: '11.5px', color: 'var(--primary)', textDecoration: 'none', fontWeight: 600 } }, 'Hub Page ↗'),
                          e('div', { className: 'modrinth-actions-group' },
                            hf.type === 'dataset' ?
                              e('div', { style: { display: 'flex', gap: '6px' } },
                                e('button', {
                                  className: 'btn-action',
                                  style: { height: '30px', padding: '0 12px', fontSize: '11.5px', background: 'var(--accent-olive, #2b4c3f)', color: '#fff', display: 'inline-flex', alignItems: 'center' },
                                  title: 'Stream dataset immediately in PyTorch Engine',
                                  onClick: () => {
                                    setHfDatasetInput(hf.id);
                                    handleConnectHfStream(hf.id, '', 'train', 'text');
                                    setActiveTab('datasets');
                                  }
                                }, [renderIcon('play', 11, { marginRight: '4px' }), 'Stream in Engine']),
                                e('button', {
                                  className: 'btn-sec',
                                  style: { height: '30px', padding: '0 10px', fontSize: '11.5px', display: 'inline-flex', alignItems: 'center' },
                                  title: 'Add as a streaming node on Visual DAG canvas',
                                  onClick: () => handleAddDatasetNodeToDAG(hf)
                                }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Add to DAG'])
                              ) :
                              e('button', {
                                className: 'btn-sec',
                                style: { height: '30px', padding: '0 10px', fontSize: '11.5px', display: 'inline-flex', alignItems: 'center' },
                                onClick: () => handleAddModuleNodeToDAG(hf)
                              }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Add to DAG']),
                            hf.installed ?
                              e('span', { style: { fontSize: '12px', color: 'var(--accent-sage)', fontWeight: '600', alignSelf: 'center', display: 'inline-flex', alignItems: 'center' } }, [renderIcon('check', 11, { marginRight: '3px' }), 'Cloned']) :
                              e('button', {
                                className: 'btn-send',
                                style: { height: '30px', padding: '0 12px', fontSize: '11.5px', display: 'inline-flex', alignItems: 'center' },
                                onClick: () => {
                                  setRepoCloneUrl(hf.repo_url);
                                  setRepoCloneType(hf.type || 'model');
                                  handleCloneCustomRepo();
                                }
                              }, [renderIcon('download', 11, { marginRight: '4px' }), 'Clone Repo'])
                          )
                        )
                      )
                    )
                  )
                )
            ),

            // Subtab 4: GitHub ML Repositories
            marketplaceSubTab === 'github' && e('div', null,
              isSearchingModules ?
                e('div', { style: { textAlign: 'center', padding: '40px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', justifyContent: 'center' } },
                  [renderIcon('refresh', 14, { animation: 'spin 1.5s linear infinite', marginRight: '8px' }), 'Searching GitHub...']
                ) :
                (!marketplaceData || !marketplaceData.github || marketplaceData.github.length === 0 ?
                  e('div', { style: { textAlign: 'center', padding: '40px 20px', background: 'var(--bg-card)', border: '1px dashed var(--border-color)', borderRadius: '8px' } },
                    e('div', { style: { marginBottom: '8px', color: 'var(--text-muted)' } }, renderIcon('git', 36)),
                    e('h4', { style: { margin: '0 0 6px 0', fontFamily: 'Newsreader', fontSize: '18px' } }, 'Search GitHub ML Repositories'),
                    e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', maxWidth: '480px', margin: '0 auto 16px auto' } }, 'Enter keywords to find open-source PyTorch models, custom attention kernels, and optimizer tools from GitHub.'),
                    e('div', { style: { display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' } },
                      ['transformer', 'flash-attention', 'bitsandbytes', 'muon-optimizer'].map(kw =>
                        e('button', {
                          key: kw,
                          className: 'btn-sec',
                          style: { fontSize: '12px' },
                          onClick: () => { setModuleSearchQuery(kw); searchMarketplace(kw, moduleFilter, 'github'); }
                        }, `Search "${kw}"`)
                      )
                    )
                  ) :
                  e('div', { className: 'modules-grid' },
                    marketplaceData.github.map(gh =>
                      e('div', { key: gh.id, className: 'module-card-modrinth' },
                        e('div', null,
                          e('div', { className: 'modrinth-card-top' },
                            e('div', { className: 'modrinth-icon-box' }, renderIcon('git', 20)),
                            e('div', { className: 'modrinth-meta-wrap' },
                              e('div', { className: 'modrinth-title-row' },
                                e('span', { className: 'modrinth-title', style: { wordBreak: 'break-all' } }, gh.name),
                                e('span', { className: 'modrinth-badge source-gh', style: { display: 'inline-flex', alignItems: 'center' } }, [renderIcon('git', 10, { marginRight: '3px' }), 'GitHub']),
                                e('span', { className: 'modrinth-badge verified', style: { display: 'inline-flex', alignItems: 'center' } }, [renderIcon('sparkle', 10, { marginRight: '3px' }), `${(gh.stars || 0).toLocaleString()} stars`])
                              ),
                              e('div', { className: 'modrinth-author-row' },
                                `Author: ${gh.author || 'OpenSource'} • License: Permissive`
                              )
                            )
                          ),
                          e('p', { className: 'modrinth-desc' }, gh.description || 'Open source GitHub repository.'),
                          e('div', { className: 'module-tags', style: { marginTop: '8px' } },
                            gh.tags && gh.tags.slice(0, 4).map((t, idx) => e('span', { key: idx, className: 'module-tag' }, t))
                          )
                        ),
                        e('div', { className: 'modrinth-actions-footer' },
                          e('a', { href: gh.repo_url, target: '_blank', rel: 'noreferrer', style: { fontSize: '11.5px', color: 'var(--primary)', textDecoration: 'none', fontWeight: 600 } }, 'GitHub ↗'),
                          e('div', { className: 'modrinth-actions-group' },
                            e('button', {
                              className: 'btn-sec',
                              style: { height: '30px', padding: '0 10px', fontSize: '11.5px', display: 'inline-flex', alignItems: 'center' },
                              title: 'Add as custom module node to Visual DAG Canvas',
                              onClick: () => handleAddModuleNodeToDAG(gh)
                            }, [renderIcon('plus', 11, { marginRight: '4px' }), 'Add to DAG']),
                            gh.installed ?
                              e('span', { style: { fontSize: '12px', color: 'var(--accent-sage)', fontWeight: '600', alignSelf: 'center', display: 'inline-flex', alignItems: 'center' } }, [renderIcon('check', 11, { marginRight: '3px' }), 'Cloned']) :
                              e('button', {
                                className: 'btn-send',
                                style: { height: '30px', padding: '0 14px', fontSize: '12px', display: 'inline-flex', alignItems: 'center' },
                                onClick: () => {
                                  setRepoCloneUrl(gh.repo_url);
                                  setRepoCloneType('auto');
                                  handleCloneCustomRepo();
                                }
                              }, [renderIcon('download', 11, { marginRight: '4px' }), 'Clone & Install'])
                          )
                        )
                      )
                    )
                  )
                )
            )
          ),

          // Tab 10: System & Hardware Auto-Scanner
          activeTab === 'environment' && e('div', { className: 'view-environment' },
            e('h2', { style: { fontFamily: 'Newsreader', fontSize: '24px', marginBottom: '6px' } }, 'System & Hardware Auto-Scanner'),
            e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', marginBottom: '16px' } }, 'Auto-detects available GPU hardware, CUDA drivers, Python runtimes, and allows configuring custom workspace paths.'),
            e('div', {
              className: `metric-card ${systemDiagnostics.cuda_init_error ? 'warning' : ''}`,
              style: { marginBottom: '16px', borderLeft: `4px solid ${systemDiagnostics.cuda_initialized ? 'var(--success, #3b8f70)' : (systemDiagnostics.cuda_init_error ? 'var(--danger, #c94d3f)' : 'var(--border-color)')}` }
            },
              e('span', { className: 'metric-label' }, 'PyTorch GPU Runtime'),
              e('span', { className: 'metric-value', style: { fontSize: '15px' } }, systemDiagnostics.cuda_initialized
                ? `${systemDiagnostics.device_name || 'CUDA'} initialized${systemDiagnostics.model_devices?.length ? ` · model: ${systemDiagnostics.model_devices.join(', ')}` : ''}`
                : (systemDiagnostics.cuda_init_error ? 'GPU detected but initialization failed' : 'CUDA runtime unavailable; using CPU')),
              systemDiagnostics.cuda_init_error && e('small', { style: { display: 'block', marginTop: '6px', color: 'var(--danger, #c94d3f)', overflowWrap: 'anywhere' } }, systemDiagnostics.cuda_init_error)
            ),

            // Summary Cards Grid
            systemScan && e('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '20px' } },
              e('div', { className: 'metric-card' },
                e('span', { className: 'metric-label' }, 'GPU Hardware Acceleration'),
                e('span', { className: 'metric-value', style: { fontSize: '15px' } }, systemScan.gpu)
              ),
              e('div', { className: 'metric-card' },
                e('span', { className: 'metric-label' }, 'CUDA Toolkit Version'),
                e('span', { className: 'metric-value', style: { fontSize: '15px' } }, systemScan.cuda_version || 'None')
              ),
              e('div', { className: 'metric-card' },
                e('span', { className: 'metric-label' }, 'VRAM Memory'),
                e('span', { className: 'metric-value', style: { fontSize: '15px' } }, `${systemScan.vram_gb} GB`)
              ),
              e('div', { className: 'metric-card' },
                e('span', { className: 'metric-label' }, 'Python Environment'),
                e('span', { className: 'metric-value', style: { fontSize: '15px' } }, systemScan.python)
              )
            ),

            // Live Hardware Acceleration & GPU Offload Tuner (LM Studio / Ollama Style)
            renderHardwareTunerDeck(),

            // Installed Software Stack Checklist
            e('h3', { style: { fontFamily: 'Newsreader', fontSize: '18px', marginBottom: '8px' } }, 'Installed Software Stack & Frameworks'),
            systemScan && systemScan.packages && e('div', { className: 'software-stack-grid' },
              Object.keys(systemScan.packages).map(pkgName => {
                const info = systemScan.packages[pkgName];
                return e('div', { key: pkgName, className: 'software-item' },
                  e('div', null,
                    e('span', { style: { fontWeight: '600', fontSize: '13.5px' } }, pkgName),
                    e('span', { style: { fontSize: '11px', color: 'var(--text-dim)', display: 'block' } }, info.version)
                  ),
                  e('span', {
                    style: {
                      fontSize: '11px',
                      padding: '2px 8px',
                      borderRadius: '10px',
                      fontWeight: '600',
                      background: info.installed ? '#e3ebd8' : '#fae8e3',
                      color: info.installed ? '#2b4c3f' : '#7c2d12',
                      border: info.installed ? '1px solid #c5d6b4' : '1px solid #f0cac0'
                    }
                  }, info.installed ? 'Installed' : 'Missing')
                );
              })
            ),

            // Custom Paths & Configuration Panel
            e('h3', { style: { fontFamily: 'Newsreader', fontSize: '18px', marginTop: '24px', marginBottom: '8px' } }, 'Custom Paths & Workspace Directories'),
            e('div', { className: 'card-byok', style: { maxWidth: '100%', margin: '0' } },
              e('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '14px' } },
                e('div', { className: 'field-group' },
                  e('label', null, 'Installation Base Path:'),
                  e('input', {
                    value: systemConfig.installation_path || 'C:\\TriuneStudio',
                    onChange: ev => setSystemConfig({ ...systemConfig, installation_path: ev.target.value })
                  })
                ),
                e('div', { className: 'field-group' },
                  e('label', null, 'Models Directory:'),
                  e('input', {
                    value: systemConfig.models_path || 'C:\\TriuneStudio\\models',
                    onChange: ev => setSystemConfig({ ...systemConfig, models_path: ev.target.value })
                  })
                ),
                e('div', { className: 'field-group' },
                  e('label', null, 'Datasets Directory:'),
                  e('input', {
                    value: systemConfig.datasets_path || 'C:\\TriuneStudio\\datasets',
                    onChange: ev => setSystemConfig({ ...systemConfig, datasets_path: ev.target.value })
                  })
                ),
                e('div', { className: 'field-group' },
                  e('label', null, 'Checkpoints Directory:'),
                  e('input', {
                    value: systemConfig.checkpoints_path || 'C:\\TriuneStudio\\checkpoints',
                    onChange: ev => setSystemConfig({ ...systemConfig, checkpoints_path: ev.target.value })
                  })
                ),
                e('div', { className: 'field-group' },
                  e('label', null, 'Hardware Acceleration Engine:'),
                  e('select', {
                    style: { width: '100%', height: '38px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '0 8px', color: 'var(--text-main)' },
                    value: systemConfig.hardware_mode || 'Auto Detect',
                    onChange: ev => setSystemConfig({ ...systemConfig, hardware_mode: ev.target.value })
                  },
                    e('option', { value: 'Auto Detect' }, 'Auto Detect (Recommended)'),
                    e('option', { value: 'NVIDIA CUDA GPU' }, 'NVIDIA CUDA GPU Acceleration'),
                    e('option', { value: 'CPU Only' }, 'CPU Only (Fallback)'),
                    e('option', { value: 'Apple Metal' }, 'Apple Metal MPS Acceleration')
                  )
                ),
                e('div', { className: 'field-group', style: { display: 'flex', alignItems: 'center', gap: '10px', marginTop: '20px' } },
                  e('input', {
                    type: 'checkbox',
                    id: 'chk_updates',
                    checked: systemConfig.auto_check_updates !== false,
                    onChange: ev => setSystemConfig({ ...systemConfig, auto_check_updates: ev.target.checked })
                  }),
                  e('label', { htmlFor: 'chk_updates', style: { cursor: 'pointer', fontSize: '13px' } }, 'Automatically check for module & stack updates on startup')
                )
              ),
              e('button', {
                className: 'btn-send',
                style: { width: '100%', height: '44px', marginTop: '20px' },
                onClick: () => saveSystemConfig(systemConfig)
              }, 'Save System Configuration')
            )
          )
        )
      )
    );
  }

  function initApp() {
    const container = document.getElementById('root');
    if (container && !container._triune_mounted) {
      container._triune_mounted = true;
      if (typeof ReactDOM !== 'undefined' && ReactDOM.createRoot) {
        const root = ReactDOM.createRoot(container);
        root.render(e(TriuneStudio));
      }
    }
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(initApp, 1);
  } else {
    document.addEventListener('DOMContentLoaded', initApp);
  }
})();
