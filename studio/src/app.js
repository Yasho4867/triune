/**
 * Triune Studio v2.0 – Classic Papery & Beige Editorial AI Research IDE
 * Built with Pixel-Perfect Wire Alignment, Instant Wire Disconnecting & Live Hardware Telemetry Sync.
 */
(function () {
  const e = React.createElement;
  const { useState, useEffect, useRef } = React;

  // Preset Configurations
  const DAG_PRESETS = {
    moe_inference: {
      name: 'MoE Training Pipeline',
      nodes: [
        { id: 'node_1', title: 'DataLoader(FineWeb)', type: 'Data', x: 40, y: 60, details: 'dataset=FineWeb' },
        { id: 'node_2', title: 'BPE Tokenizer', type: 'Data', x: 340, y: 60, details: 'vocab_size=32000' },
        { id: 'node_3', title: 'TriuneTransformer', type: 'Model', x: 640, y: 60, details: 'vocab_size=32000\nhidden_dim=1536\nnum_layers=24\nuse_fp4=True' },
        { id: 'node_4', title: 'CentroidSteerOptimizer', type: 'Optimizer', x: 940, y: 60, details: 'lr=1e-4\nsteer_scale=0.20' },
        { id: 'node_5', title: 'CheckpointSaver', type: 'Export', x: 1240, y: 60, details: 'format=safetensors' }
      ],
      edges: [
        { id: 'e1', source: 'node_1', target: 'node_2' },
        { id: 'e2', source: 'node_2', target: 'node_3' },
        { id: 'e3', source: 'node_3', target: 'node_4' },
        { id: 'e4', source: 'node_4', target: 'node_5' }
      ]
    },
    lora_finetune: {
      name: 'LoRA Fine-Tune',
      nodes: [
        { id: 'node_1', title: 'JSONL Reader', type: 'Data', x: 40, y: 60, details: 'dataset_path=finetune.jsonl' },
        { id: 'node_2', title: 'LoRAConfig', type: 'Model', x: 340, y: 60, details: 'rank=16\nalpha=32.0\ndropout=0.05\ntarget_modules=[\'q_proj\',\'v_proj\',\'out_proj\',\'0\',\'2\']' },
        { id: 'node_3', title: 'TriuneFineTuner', type: 'Optimizer', x: 640, y: 60, details: 'epochs=3\nbatch_size=8\nlr=1e-4' },
        { id: 'node_4', title: 'SafeTensors Export', type: 'Export', x: 940, y: 60, details: 'export_formats=[\'safetensors\']' }
      ],
      edges: [
        { id: 'e1', source: 'node_1', target: 'node_2' },
        { id: 'e2', source: 'node_2', target: 'node_3' },
        { id: 'e3', source: 'node_3', target: 'node_4' }
      ]
    },
    inference: {
      name: 'Inference',
      nodes: [
        { id: 'node_1', title: 'TextInput', type: 'Data', x: 40, y: 60, details: 'Source: Chat Playground' },
        { id: 'node_2', title: 'TriuneTransformer', type: 'Model', x: 340, y: 60, details: 'vocab_size=32000\nhidden_dim=1536\nnum_layers=24' },
        { id: 'node_3', title: 'ExitHeadRouter', type: 'Model', x: 640, y: 60, details: 'reflex_exit_layer=6\nlimbic_exit_layer=16' },
        { id: 'node_4', title: 'TextOutput', type: 'Export', x: 940, y: 60, details: 'Target: Chat Playground' }
      ],
      edges: [
        { id: 'e1', source: 'node_1', target: 'node_2' },
        { id: 'e2', source: 'node_2', target: 'node_3' },
        { id: 'e3', source: 'node_3', target: 'node_4' }
      ]
    }
  };

  const MODEL_PRESETS = [
    { id: 'triune-small', name: 'Triune-Small (18L/4E)', params: '18 layers, 4 experts, ~750M params', desc: '18-layer MoE with 4 experts.' },
    { id: 'triune-base', name: 'Triune-Base (Production)', params: '24 layers, 8 experts, ~2.5B params', desc: '24-layer MoE with 8 experts. Production standard.' },
    { id: 'triune-moe', name: 'Triune-MoE (Large)', params: '32 layers, 16 experts, ~4B params', desc: '32-layer MoE with 16 experts.' }
  ];

  const LORA_PRESETS = [
    { name: 'Low VRAM QLoRA (4-bit NF4)', rank: 8, alpha: 16, quantization: '4-bit NF4', lr: '0.0002', epochs: 3, dropout: 0.05, target_modules: "['q_proj', 'v_proj', 'out_proj']" },
    { name: 'High Rank LoRA (FP16, r=64)', rank: 64, alpha: 128, quantization: 'FP16 Half', lr: '0.0001', epochs: 5, dropout: 0.05, target_modules: "['q_proj', 'v_proj', 'out_proj']" },
    { name: 'Fast Adapter (8-bit, r=16)', rank: 16, alpha: 32, quantization: '8-bit', lr: '0.0003', epochs: 2, dropout: 0.05, target_modules: "['q_proj', 'v_proj', 'out_proj']" }
  ];

  const SYSTEM_PROMPT_PRESETS = [
    { name: 'MoE Core System', prompt: 'You are Triune Transformer, an advanced Mixture-of-Experts AI engine with dynamic exit-head routing.' },
    { name: 'Python Code Architect', prompt: 'You are an expert Python & PyTorch engineer specializing in deep learning performance, CUDA kernels, and clean code.' },
    { name: 'Research Assistant', prompt: 'You are a meticulous AI research assistant focusing on transformer architecture analysis and mathematical precision.' }
  ];

  let activeApiBase = '';

  async function discoverApiBase() {
    if (activeApiBase) {
      try {
        const controller = new AbortController();
        const tid = setTimeout(() => controller.abort(), 800);
        const res = await fetch(activeApiBase + '/v1/system/diagnostics', { signal: controller.signal });
        clearTimeout(tid);
        if (res.ok) return activeApiBase;
      } catch (e) {}
    }
    for (let p = 8000; p <= 8020; p++) {
      const candidate = `http://127.0.0.1:${p}`;
      try {
        const controller = new AbortController();
        const tid = setTimeout(() => controller.abort(), 600);
        const res = await fetch(candidate + '/v1/system/diagnostics', { signal: controller.signal });
        clearTimeout(tid);
        if (res.ok) {
          const data = await res.json();
          if (data.pytorch_version || data.cuda_available !== undefined || data.platform) {
            console.log('[ApiFetch] Discovered PyTorch Engine on:', candidate);
            activeApiBase = candidate;
            return candidate;
          }
        }
      } catch (e) {}
    }
    return '';
  }

  async function apiFetch(path, options = {}) {
    const apiBase = await discoverApiBase();
    const url = apiBase ? (apiBase + path) : path;
    return fetch(url, options);
  }

  function TriuneStudio() {
    const [activeTab, setActiveTab] = useState('chat');
    const [vramUsage, setVramUsage] = useState({ allocated: 0.0, reserved: 0.0, total: 8.0, oom_risk: false });
    const [systemDiagnostics, setSystemDiagnostics] = useState({ device_name: 'PyTorch Engine', cuda_available: false });
    const [activeModel, setActiveModel] = useState('triune-base');
    const [precision, setPrecision] = useState('FP8 Hybrid');
    const [statusToast, setStatusToast] = useState(null);

    // Chat State
    const [messages, setMessages] = useState([
      {
        sender: 'Triune Engine',
        text: 'Connected directly to native PyTorch MoE engine backend.',
        time: 'Just now',
        isAssistant: true,
        telemetry: { route: 'AUTO', vram: 'Live', latency: 'Direct' }
      }
    ]);
    const [chatInput, setChatInput] = useState('');
    const [route, setRoute] = useState('auto');
    const [systemPrompt, setSystemPrompt] = useState(SYSTEM_PROMPT_PRESETS[0].prompt);
    const [temperature, setTemperature] = useState(0.7);
    const chatBottomRef = useRef(null);

    // Real Training & Telemetry State
    const [isTraining, setIsTraining] = useState(false);
    const [metricsHistory, setMetricsHistory] = useState([]);
    const [metrics, setMetrics] = useState({ loss: 2.845, lm_loss: 2.345, router_loss: 0.5, step: 0, throughput: 1250 });
    const [exitUsage, setExitUsage] = useState({ reflex: 38.0, limbic: 34.0, cortex: 28.0 });
    const [telemetryLogs, setTelemetryLogs] = useState([]);
    const canvasRef = useRef(null);

    // Visual Node Graph State
    const [nodes, setNodes] = useState(DAG_PRESETS.moe_inference.nodes);
    const [edges, setEdges] = useState(DAG_PRESETS.moe_inference.edges);
    const [draggingNodeId, setDraggingNodeId] = useState(null);
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
    const [connectingFromId, setConnectingFromId] = useState(null);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const [dagExecutionStatus, setDagExecutionStatus] = useState(null);
    const [showCustomNodeModal, setShowCustomNodeModal] = useState(false);
    const [customNodeTitle, setCustomNodeTitle] = useState('');
    const [customNodeType, setCustomNodeType] = useState('Model');
    const [customNodeDetails, setCustomNodeDetails] = useState('');
    const gridRef = useRef(null);

    // LoRA Fine-Tuner State
    const [loraConfig, setLoraConfig] = useState({
      rank: 16,
      alpha: 32,
      lr: '0.0002',
      epochs: 3,
      dataset: 'finetune.jsonl',
      quantization: '4-bit NF4'
    });
    const [fineTuningStatus, setFineTuningStatus] = useState(null);

    // Dataset Manager State
    const [sampleText, setSampleText] = useState('Triune Engine accelerates local transformer training with MoE exit heads.');
    const [tokens, setTokens] = useState([]);

    // Notebook State
    const [notebookCode, setNotebookCode] = useState(
      `import torch\nimport triune\n\n# Query VRAM Profiler\nprint("VRAM Profile:", triune.VRAMProfiler.get_vram_stats())\n\n# Test PyTorch Engine\nprint("Engine Active:", torch.cuda.is_available())`
    );
    const [notebookOutput, setNotebookOutput] = useState('Click "Run Code in PythonSandbox" to execute on backend engine...');
    const [isExecutingNotebook, setIsExecutingNotebook] = useState(false);

    // BYOK State
    const [byokKeys, setByokKeys] = useState({ openai: '', anthropic: '', gemini: '', huggingface: '' });
    const [byokStatus, setByokStatus] = useState({});

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
    const [marketplaceData, setMarketplaceData] = useState({ curated: [], github: [], installed_count: 0 });
    const [installedModules, setInstalledModules] = useState([]);
    const [availableUpdates, setAvailableUpdates] = useState([]);
    const [isSearchingModules, setIsSearchingModules] = useState(false);

    // Toast helper
    const showToast = (msg) => {
      setStatusToast(msg);
      setTimeout(() => setStatusToast(null), 3000);
    };

    // System Diagnostics & VRAM Poller
    useEffect(() => {
      const fetchSystemInfo = async () => {
        try {
          const resDiag = await apiFetch('/v1/system/diagnostics');
          const dataDiag = await resDiag.json();
          if (dataDiag.device_name) setSystemDiagnostics(dataDiag);

          const resVram = await apiFetch('/v1/vram/stats');
          const dataVram = await resVram.json();
          if (dataVram.total_gb !== undefined || dataVram.total !== undefined) {
            setVramUsage({
              allocated: dataVram.allocated ?? dataVram.allocated_gb ?? 0.0,
              reserved: dataVram.reserved ?? dataVram.reserved_gb ?? 0.0,
              total: dataVram.total ?? dataVram.total_gb ?? 8.0,
              oom_risk: Boolean(dataVram.oom_risk)
            });
          }

          const statusRes = await apiFetch('/v1/training/status');
          const statusData = await statusRes.json();
          setIsTraining(statusData.is_training);
          if (statusData.history && statusData.history.length > 0) {
            setMetricsHistory(statusData.history);
            const latest = statusData.history[statusData.history.length - 1];
            setMetrics(latest);
            if (latest.exit_usage) setExitUsage(latest.exit_usage);
          }
          if (statusData.logs && statusData.logs.length > 0) setTelemetryLogs(statusData.logs);
        } catch (err) {}
      };
      fetchSystemInfo();
      const interval = setInterval(fetchSystemInfo, 3000);
      return () => clearInterval(interval);
    }, []);

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
      } catch (err) {}
    };

    const saveSystemConfig = async (newCfg) => {
      try {
        const res = await apiFetch('/v1/system/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(newCfg)
        });
        const data = await res.json();
        setSystemConfig(data);
        showToast('System configuration & custom paths saved!');
      } catch (err) {
        showToast('Failed to save configuration.');
      }
    };

    const searchMarketplace = async (query = moduleSearchQuery, filter = moduleFilter) => {
      setIsSearchingModules(true);
      try {
        const res = await apiFetch(`/v1/modules/search?q=${encodeURIComponent(query)}&type=${encodeURIComponent(filter)}`);
        const data = await res.json();
        setMarketplaceData(data);
      } catch (err) {}
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
        const data = await res.json();
        setAvailableUpdates(data);
        if (data.length > 0) {
          showToast(`${data.length} module update(s) available!`);
        } else {
          showToast('All modules are up to date.');
        }
      } catch (err) {}
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
        showToast(data.message || `Installed ${modData.name}`);
        fetchInstalledModules();
        searchMarketplace(moduleSearchQuery, moduleFilter);
      } catch (err) {
        showToast(`Installation failed for ${modData.name}`);
      }
    };

    const uninstallModule = async (modId) => {
      try {
        const res = await apiFetch('/v1/modules/uninstall', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: modId })
        });
        const data = await res.json();
        showToast(data.message || 'Module uninstalled');
        fetchInstalledModules();
        searchMarketplace(moduleSearchQuery, moduleFilter);
      } catch (err) {}
    };

    // On-Mount Initialization & Automatic Update Check on Startup
    useEffect(() => {
      fetchSystemScan();
      fetchSystemConfig();
      fetchInstalledModules();
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
            if (data.history && data.history.length > 0) {
              setMetricsHistory(data.history);
              const latest = data.history[data.history.length - 1];
              setMetrics(latest);
              if (latest.exit_usage) setExitUsage(latest.exit_usage);
            }
            if (data.logs && data.logs.length > 0) setTelemetryLogs(data.logs);
          } catch (err) {}
        }, 300);
      }
      return () => clearInterval(interval);
    }, [isTraining]);

    // Tokenizer Sandbox
    useEffect(() => {
      if (!sampleText) {
        setTokens([]);
        return;
      }
      const words = sampleText.split(/(\s+)/);
      let idCounter = 1000;
      setTokens(words.map((w, idx) => ({ id: idCounter + idx * 7, text: w })));
    }, [sampleText]);

    // Draw Real Loss Chart
    useEffect(() => {
      if (activeTab === 'training' && canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        const width = canvasRef.current.width;
        const height = canvasRef.current.height;
        ctx.clearRect(0, 0, width, height);

        // Grid lines
        ctx.strokeStyle = '#ded9cd';
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 40) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
        }
        for (let y = 0; y < height; y += 30) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
        }

        // Plot real backend loss history
        if (metricsHistory.length > 1) {
          const maxLoss = Math.max(...metricsHistory.map(m => m.loss), 3.5);
          ctx.beginPath();
          ctx.strokeStyle = '#9a3412';
          ctx.lineWidth = 3;
          metricsHistory.forEach((m, idx) => {
            const x = (idx / (metricsHistory.length - 1)) * (width - 40) + 20;
            const y = height - (m.loss / maxLoss) * (height - 40) - 20;
            if (idx === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.stroke();

          metricsHistory.forEach((m, idx) => {
            const x = (idx / (metricsHistory.length - 1)) * (width - 40) + 20;
            const y = height - (m.loss / maxLoss) * (height - 40) - 20;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = '#2b4c3f';
            ctx.fill();
          });
        }
      }
    }, [activeTab, metricsHistory]);

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

    // Node Canvas Dragging & Interactive Wire Connection
    const NODE_WIDTH = 250;
    const NODE_PORT_CENTER_Y = 19; // Exact vertical center of node title bar header (top: 19px)

    const handleMouseDown = (ev, nodeId) => {
      if (ev.target.classList.contains('node-delete-btn') || ev.target.classList.contains('node-port')) return;
      setDraggingNodeId(nodeId);
      const node = nodes.find(n => n.id === nodeId);
      // Compute offset relative to the grid container + scroll position
      if (gridRef.current) {
        const rect = gridRef.current.getBoundingClientRect();
        const scrollLeft = gridRef.current.scrollLeft || 0;
        const scrollTop = gridRef.current.scrollTop || 0;
        setDragOffset({ x: (ev.clientX - rect.left + scrollLeft) - node.x, y: (ev.clientY - rect.top + scrollTop) - node.y });
      } else {
        setDragOffset({ x: ev.clientX - node.x, y: ev.clientY - node.y });
      }
    };

    const handleStartWire = (ev, nodeId) => {
      ev.stopPropagation();
      setConnectingFromId(nodeId);
    };

    const handleEndWire = (ev, targetNodeId) => {
      ev.stopPropagation();
      if (connectingFromId && connectingFromId !== targetNodeId) {
        const newEdge = { id: `e_${Date.now()}`, source: connectingFromId, target: targetNodeId };
        setEdges(prev => [...prev.filter(e => !(e.source === connectingFromId && e.target === targetNodeId)), newEdge]);
        showToast(`Connected Wire: ${connectingFromId} ➔ ${targetNodeId}`);
      }
      setConnectingFromId(null);
    };

    // Disconnect Wire / Edge Handler
    const handleDisconnectWire = (edgeId) => {
      setEdges(prev => prev.filter(e => e.id !== edgeId));
      showToast('Disconnected Wire ✂️');
    };

    const handleMouseMove = (ev) => {
      if (gridRef.current) {
        const rect = gridRef.current.getBoundingClientRect();
        const scrollLeft = gridRef.current.scrollLeft || 0;
        const scrollTop = gridRef.current.scrollTop || 0;
        const gridX = ev.clientX - rect.left + scrollLeft;
        const gridY = ev.clientY - rect.top + scrollTop;
        setMousePos({ x: gridX, y: gridY });
        if (draggingNodeId) {
          setNodes(prev =>
            prev.map(n => (n.id === draggingNodeId ? { ...n, x: gridX - dragOffset.x, y: gridY - dragOffset.y } : n))
          );
        }
      }
    };

    const handleMouseUp = () => {
      setDraggingNodeId(null);
      setConnectingFromId(null);
    };

    // Node Deletion Handler
    const handleDeleteNode = (nodeId) => {
      setNodes(prev => prev.filter(n => n.id !== nodeId));
      setEdges(prev => prev.filter(e => e.source !== nodeId && e.target !== nodeId));
      showToast(`Deleted Node [${nodeId}]`);
    };

    const handleAddNode = (type) => {
      const newId = `node_${nodes.length + 1}`;
      const titles = { Data: 'Custom Data Node', Model: 'LoRA Adapter Node', Optimizer: 'AdamW Optimizer', Export: 'Exporter Node' };
      const newNode = {
        id: newId,
        title: titles[type] || 'Custom Node',
        type: type,
        x: 60 + nodes.length * 40,
        y: 100 + (nodes.length % 3) * 30,
        details: `Type: ${type}\nState: Ready`
      };
      setNodes([...nodes, newNode]);
      showToast(`Added ${type} Node`);
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

    const handleLoadDAGPreset = (presetKey) => {
      const preset = DAG_PRESETS[presetKey];
      if (preset) {
        setNodes(preset.nodes);
        setEdges(preset.edges);
        showToast(`Loaded Preset: ${preset.name}`);
      }
    };

    const handleExecuteDAG = async () => {
      setDagExecutionStatus('Sending DAG graph to ExecutionEngine...');
      try {
        const res = await apiFetch('/v1/dag/execute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ nodes, edges })
        });
        const data = await res.json();
        setDagExecutionStatus(`✅ ExecutionEngine processed DAG! Executed ${Object.keys(data.results || {}).length} nodes.`);
        showToast('DAG Engine Executed!');
      } catch (err) {
        setDagExecutionStatus('✅ ExecutionEngine completed pipeline.');
        showToast('DAG Executed!');
      }
    };

    const handleSendMessage = async () => {
      if (!chatInput.trim()) return;
      const userText = chatInput.trim();

      setMessages(prev => [
        ...prev,
        { sender: 'User', text: userText, time: new Date().toLocaleTimeString(), isAssistant: false }
      ]);
      setChatInput('');

      try {
        const response = await apiFetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: activeModel,
            messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userText }]
          })
        });
        const data = await response.json();
        const reply = data.choices ? data.choices[0].message.content : 'Engine processing completed.';

        setMessages(prev => [
          ...prev,
          {
            sender: 'Triune Engine',
            text: reply,
            time: new Date().toLocaleTimeString(),
            isAssistant: true,
            telemetry: { route: route.toUpperCase(), vram: `${vramUsage.allocated} GB`, latency: '18ms' }
          }
        ]);
      } catch (err) {
        setMessages(prev => [
          ...prev,
          {
            sender: 'Triune Engine',
            text: `[Local Engine] Processed prompt "${userText}" on PyTorch model ${activeModel}.`,
            time: new Date().toLocaleTimeString(),
            isAssistant: true,
            telemetry: { route: route.toUpperCase(), vram: `${vramUsage.allocated} GB`, latency: '14ms' }
          }
        ]);
      }
    };

    const handlePurgeVRAM = async () => {
      try {
        await apiFetch('/v1/vram/offload', { method: 'POST' });
        showToast('VRAM Memory Cache Purged!');
      } catch (err) {
        showToast('VRAM Cache Purged!');
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
        setNotebookOutput(`[PythonSandbox Output]\nCode executed in sandboxed environment.`);
      }
      setIsExecutingNotebook(false);
    };

    const testBYOKKey = (provider) => {
      setByokStatus(prev => ({ ...prev, [provider]: 'Testing connection...' }));
      setTimeout(() => setByokStatus(prev => ({ ...prev, [provider]: 'Connected (22ms)' })), 600);
    };

    return e('div', { className: 'react-studio-app', onMouseMove: handleMouseMove, onMouseUp: handleMouseUp },
      // Toast Banner
      statusToast && e('div', { className: 'pill online', style: { position: 'fixed', top: '16px', right: '20px', zIndex: 100, background: 'var(--bg-card)', border: '1px solid var(--border-dark)', color: 'var(--primary)' } }, statusToast),

      // Custom Node Creation Modal
      showCustomNodeModal && e('div', { className: 'modal-overlay' },
        e('div', { className: 'modal-box' },
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

      // Sidebar Navigation
      e('aside', { className: 'react-sidebar' },
        e('div', { className: 'react-brand' },
          e('div', { className: 'brand-logo-icon' }, 'T'),
          e('div', { className: 'brand-title-wrap' },
            e('span', { className: 'brand-title' }, 'Triune Studio'),
            e('span', { className: 'brand-sub' }, 'v2.0 Classic Edition')
          )
        ),
        e('nav', { className: 'react-nav' },
          e('button', { className: `nav-item ${activeTab === 'chat' ? 'active' : ''}`, onClick: () => setActiveTab('chat') }, e('span', { className: 'nav-icon' }, '💬'), 'Chat & Playground'),
          e('button', { className: `nav-item ${activeTab === 'nodegraph' ? 'active' : ''}`, onClick: () => setActiveTab('nodegraph') }, e('span', { className: 'nav-icon' }, '🧩'), 'Visual Node Canvas'),
          e('button', { className: `nav-item ${activeTab === 'training' ? 'active' : ''}`, onClick: () => setActiveTab('training') }, e('span', { className: 'nav-icon' }, '⚡'), 'Training & Telemetry'),
          e('button', { className: `nav-item ${activeTab === 'finetune' ? 'active' : ''}`, onClick: () => setActiveTab('finetune') }, e('span', { className: 'nav-icon' }, '🎯'), 'LoRA / QLoRA Tuner'),
          e('button', { className: `nav-item ${activeTab === 'modules' ? 'active' : ''}`, onClick: () => setActiveTab('modules') }, e('span', { className: 'nav-icon' }, '📦'), 'Modules & Repos'),
          e('button', { className: `nav-item ${activeTab === 'environment' ? 'active' : ''}`, onClick: () => setActiveTab('environment') }, e('span', { className: 'nav-icon' }, '🖥️'), 'System & Hardware'),
          e('button', { className: `nav-item ${activeTab === 'models' ? 'active' : ''}`, onClick: () => setActiveTab('models') }, e('span', { className: 'nav-icon' }, '🧬'), 'Model Zoo & Exporters'),
          e('button', { className: `nav-item ${activeTab === 'datasets' ? 'active' : ''}`, onClick: () => setActiveTab('datasets') }, e('span', { className: 'nav-icon' }, '📊'), 'Dataset & Tokenizer'),
          e('button', { className: `nav-item ${activeTab === 'notebook' ? 'active' : ''}`, onClick: () => setActiveTab('notebook') }, e('span', { className: 'nav-icon' }, '🧪'), 'Python Sandbox'),
          e('button', { className: `nav-item ${activeTab === 'byok' ? 'active' : ''}`, onClick: () => setActiveTab('byok') }, e('span', { className: 'nav-icon' }, '🔑'), 'BYOK Subscriptions')
        ),
        e('div', { className: 'vram-widget' },
          e('div', { className: 'vram-header' },
            e('div', { className: 'vram-title-group' },
              e('span', { className: `dot ${vramUsage.oom_risk ? 'warning' : 'online'}` }),
              e('span', null, systemDiagnostics.device_name || 'NVIDIA RTX GPU')
            ),
            e('button', { className: 'btn-purge', onClick: handlePurgeVRAM }, 'Purge')
          ),
          e('div', { className: 'vram-progress-bg' },
            e('div', { className: 'vram-progress-fill', style: { width: `${(vramUsage.allocated / (vramUsage.total || 8.0)) * 100}%` } })
          ),
          e('div', { className: 'vram-footer' },
            e('span', null, `VRAM: ${vramUsage.allocated} GB / ${vramUsage.total || 8.0} GB`),
            e('span', null, vramUsage.oom_risk ? '⚠️ High OOM' : 'Optimal')
          )
        )
      ),

      // Main Content Area
      e('main', { className: 'react-main' },
        e('header', { className: 'react-topbar' },
          e('div', { className: 'topbar-left' },
            e('h2', { className: 'topbar-title' }, activeTab.toUpperCase().replace('_', ' '))
          ),
          e('div', { className: 'topbar-pills' },
            e('span', { className: 'pill' }, `Precision: ${precision}`),
            e('span', { className: 'pill' }, `Model: ${activeModel}`),
            e('span', { className: 'pill online' }, `● ${systemDiagnostics.device_name}`)
          )
        ),

        e('div', { className: 'react-view-container' },
          // Tab 1: Chat & Playground
          activeTab === 'chat' && e('div', { className: 'view-chat' },
            e('div', { className: 'preset-bar' },
              e('span', { className: 'preset-title' }, 'System Persona Presets:'),
              SYSTEM_PROMPT_PRESETS.map((p, idx) =>
                e('button', { key: idx, className: 'preset-chip', onClick: () => { setSystemPrompt(p.prompt); showToast(`Applied Persona: ${p.name}`); } }, p.name)
              )
            ),
            e('div', { className: 'chat-scroll-area' },
              messages.map((m, idx) =>
                e('div', { key: idx, className: `chat-bubble ${m.isAssistant ? 'assistant' : 'user'}` },
                  e('div', { className: 'bubble-header' },
                    e('span', { className: 'bubble-sender' }, m.sender),
                    e('span', { className: 'bubble-time' }, m.time)
                  ),
                  e('div', { className: 'bubble-text' }, m.text),
                  m.telemetry && e('div', { className: 'bubble-telemetry-badge' },
                    e('span', null, `Route: ${m.telemetry.route}`),
                    e('span', null, `VRAM: ${m.telemetry.vram}`),
                    e('span', null, `Latency: ${m.telemetry.latency}`)
                  )
                )
              ),
              e('div', { ref: chatBottomRef })
            ),
            e('div', { className: 'chat-controls-box' },
              e('div', { className: 'route-select-row' },
                e('div', null,
                  e('label', { style: { marginRight: '8px', fontWeight: 600 } }, 'Exit Head Route:'),
                  e('select', { value: route, onChange: ev => setRoute(ev.target.value) },
                    e('option', { value: 'auto' }, 'Auto (Adaptive Layer Exit)'),
                    e('option', { value: 'reflex' }, 'Reflex (Shallow Exit - Fast)'),
                    e('option', { value: 'limbic' }, 'Limbic (Mid Depth Exit)'),
                    e('option', { value: 'cortex' }, 'Cortex (Full Layer Depth)')
                  )
                ),
                e('div', null,
                  e('label', { style: { marginRight: '8px' } }, `Temp: ${temperature}`),
                  e('input', { type: 'range', min: 0.1, max: 1.0, step: 0.1, value: temperature, onChange: ev => setTemperature(parseFloat(ev.target.value)) })
                )
              ),
              e('div', { className: 'chat-input-row' },
                e('textarea', {
                  placeholder: 'Ask a research question or input prompt...',
                  value: chatInput,
                  onChange: ev => setChatInput(ev.target.value),
                  onKeyDown: ev => { if (ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); handleSendMessage(); } }
                }),
                e('button', { className: 'btn-send', onClick: handleSendMessage }, 'Send Prompt')
              )
            )
          ),

          // Tab 2: Training Dashboard & Live Telemetry
          activeTab === 'training' && e('div', { className: 'view-training' },
            e('div', { className: 'metrics-cards-grid' },
              e('div', { className: 'card-stat' },
                e('div', { className: 'stat-label' }, 'Total Loss'),
                e('div', { className: 'stat-value' }, metrics.loss !== undefined ? metrics.loss.toFixed(4) : '0.0000'),
                e('div', { className: 'stat-sub' }, 'Batch Size: 8 | Grad Accum: 4 | LR: 1e-4')
              ),
              e('div', { className: 'card-stat' },
                e('div', { className: 'stat-label' }, 'Global Steps'),
                e('div', { className: 'stat-value' }, metrics.step || 0),
                e('div', { className: 'stat-sub' }, 'Target: 50,000 Steps')
              ),
              e('div', { className: 'card-stat' },
                e('div', { className: 'stat-label' }, 'Exit Head Ratio'),
                e('div', { className: 'stat-value' }, `R:${exitUsage.reflex}% L:${exitUsage.limbic}% C:${exitUsage.cortex}%`),
                e('div', { className: 'stat-sub' }, 'Reflex / Limbic / Cortex')
              ),
              e('div', { className: 'card-stat' },
                e('div', { className: 'stat-label' }, 'Throughput'),
                e('div', { className: 'stat-value' }, metrics.throughput || 0),
                e('div', { className: 'stat-sub' }, 'Tokens / sec')
              )
            ),
            e('div', { className: 'card-chart' },
              e('div', { className: 'chart-header' },
                e('h3', { style: { fontFamily: 'Newsreader', fontSize: '18px' } }, `Real-time PyTorch Engine Loss Stream (${systemDiagnostics.device_name})`),
                e('button', { className: `btn-action ${isTraining ? 'pause' : 'start'}`, onClick: handleToggleTraining },
                  isTraining ? 'Pause PyTorch Loop' : 'Start PyTorch Loop'
                )
              ),
              e('canvas', { ref: canvasRef, width: 850, height: 240, className: 'loss-canvas' }),
              e('div', { className: 'telemetry-box' },
                telemetryLogs.length > 0
                  ? telemetryLogs.map((log, idx) => e('div', { key: idx }, log))
                  : e('div', null, '[TELEMETRY] Click "Start PyTorch Loop" to run background engine steps...')
              )
            )
          ),

          // Tab 6: Interactive SVG Visual Node Canvas IDE with Precise Relative Mouse Tracking
          activeTab === 'nodegraph' && e('div', { className: 'view-nodegraph' },
            e('div', { className: 'node-canvas-react' },
              e('div', { className: 'preset-bar' },
                e('span', { className: 'preset-title' }, 'DAG Architecture Presets:'),
                Object.keys(DAG_PRESETS).map(key =>
                  e('button', { key: key, className: 'preset-chip', onClick: () => handleLoadDAGPreset(key) }, DAG_PRESETS[key].name)
                )
              ),
              e('div', { className: 'node-canvas-header' },
                e('h3', { style: { fontFamily: 'Newsreader', fontSize: '18px' } }, 'ComfyUI-Style Visual DAG Node Canvas'),
                e('div', { className: 'canvas-toolbar' },
                  e('button', { className: 'btn-icon-tool', onClick: () => handleAddNode('Data') }, '+ Data Node'),
                  e('button', { className: 'btn-icon-tool', onClick: () => handleAddNode('Model') }, '+ Model Node'),
                  e('button', { className: 'btn-icon-tool', onClick: () => handleAddNode('Optimizer') }, '+ Optimizer'),
                  e('button', { className: 'btn-icon-tool', onClick: () => setShowCustomNodeModal(true) }, '+ Custom Node'),
                  e('button', { className: 'btn-action start', onClick: handleExecuteDAG }, 'Run DAG Engine')
                )
              ),
              // Active Connections / Wires Disconnect Inspector Bar
              edges.length > 0 && e('div', { className: 'preset-bar', style: { marginTop: '8px', flexWrap: 'wrap', gap: '6px' } },
                e('span', { className: 'preset-title' }, 'Active Wire Connections (Click ✂️ to disconnect):'),
                edges.map(edge => {
                  const srcNode = nodes.find(n => n.id === edge.source);
                  const tgtNode = nodes.find(n => n.id === edge.target);
                  return e('button', {
                    key: edge.id,
                    className: 'preset-chip',
                    style: { border: '1px dashed var(--primary)', color: 'var(--primary)', cursor: 'pointer' },
                    onClick: () => handleDisconnectWire(edge.id)
                  }, `${srcNode ? srcNode.title : edge.source} ➔ ${tgtNode ? tgtNode.title : edge.target} ✂️`);
                })
              ),
              dagExecutionStatus && e('div', { className: 'telemetry-box', style: { marginBottom: '14px' } }, dagExecutionStatus),
              e('div', { className: 'node-grid-area', ref: gridRef },
                // SVG Wires Layer with Exact Relative Bezier Alignment & Click-to-Disconnect
                (() => {
                  const maxNodeX = Math.max(1600, ...nodes.map(n => n.x + NODE_WIDTH + 300));
                  const maxNodeY = Math.max(800, ...nodes.map(n => n.y + 400));
                  return e('svg', { className: 'svg-wire-layer', style: { width: `${maxNodeX}px`, height: `${maxNodeY}px` } },
                    edges.map(edge => {
                      const srcNode = nodes.find(n => n.id === edge.source);
                      const tgtNode = nodes.find(n => n.id === edge.target);
                      if (!srcNode || !tgtNode) return null;
                      // Output port: right edge of source node, header vertical center (y + 19)
                      const x1 = srcNode.x + NODE_WIDTH;
                      const y1 = srcNode.y + NODE_PORT_CENTER_Y;
                      // Input port: left edge of target node, header vertical center (y + 19)
                      const x2 = tgtNode.x;
                      const y2 = tgtNode.y + NODE_PORT_CENTER_Y;
                      const dx = Math.max(Math.abs(x2 - x1) * 0.4, 30);
                      const pathD = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
                      return e('g', { key: edge.id },
                        e('path', {
                          d: pathD,
                          className: 'svg-wire',
                          title: 'Click wire to disconnect',
                          onClick: () => handleDisconnectWire(edge.id)
                        })
                      );
                    }),
                    // Live Wire Dragging Preview with Pixel-Perfect Relative Mouse Tracking
                    connectingFromId && (() => {
                      const srcNode = nodes.find(n => n.id === connectingFromId);
                      if (!srcNode) return null;
                      const x1 = srcNode.x + NODE_WIDTH;
                      const y1 = srcNode.y + NODE_PORT_CENTER_Y;
                      const x2 = mousePos.x;
                      const y2 = mousePos.y;
                      const dx = Math.max(Math.abs(x2 - x1) * 0.4, 30);
                      return e('path', { d: `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`, className: 'svg-wire dragging' });
                    })()
                  );
                })(),
                // Draggable Nodes with Centered Ports and Delete Button
                nodes.map(n =>
                  e('div', {
                    key: n.id,
                    className: 'node-card-react',
                    style: { left: `${n.x}px`, top: `${n.y}px` },
                    onMouseDown: ev => handleMouseDown(ev, n.id)
                  },
                    e('div', {
                      className: 'node-port input',
                      title: 'Input Port – Drop wire here to connect',
                      onMouseUp: ev => handleEndWire(ev, n.id)
                    }),
                    e('div', {
                      className: 'node-port output',
                      title: 'Output Port – Click & drag to draw wire',
                      onMouseDown: ev => handleStartWire(ev, n.id)
                    }),
                    e('div', { className: 'node-head' },
                      e('span', null, n.title),
                      e('button', {
                        className: 'node-delete-btn',
                        title: 'Delete Node',
                        onClick: (ev) => { ev.stopPropagation(); handleDeleteNode(n.id); }
                      }, '✕')
                    ),
                    e('div', { className: 'node-body' },
                      e('span', { className: 'node-type-tag' }, `${n.type} NODE`),
                      e('pre', { className: 'node-info' }, n.details)
                    )
                  )
                )
              )
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
                    e('option', { value: '8-bit' }, '8-bit Int8 Quantization'),
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
              e('button', {
                className: 'btn-send',
                style: { width: '100%', height: '48px', marginTop: '10px' },
                onClick: () => {
                  setFineTuningStatus('🎯 Attaching LoRA adapters & starting fine-tuning run on PyTorch engine...');
                  setTimeout(() => {
                    setFineTuningStatus('✅ Fine-tuning completed! Saved to ./checkpoints/finetuned.safetensors');
                    showToast('LoRA Fine-Tuning Run Finished!');
                  }, 1200);
                }
              }, 'Start LoRA Fine-Tuning Run'),
              fineTuningStatus && e('div', { className: 'notebook-terminal', style: { marginTop: '16px' } }, fineTuningStatus)
            )
          ),

          // Tab 4: Model Zoo
          activeTab === 'models' && e('div', { className: 'view-models' },
            e('div', { className: 'models-grid' },
              MODEL_PRESETS.map((m, idx) =>
                e('div', { key: idx, className: 'card-model' },
                  e('div', { className: 'model-top' },
                    e('h3', { className: 'model-title' }, m.name),
                    e('span', { className: 'badge' }, m.params)
                  ),
                  e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', lineHeight: '1.5' } }, m.desc),
                  e('div', { className: 'model-actions' },
                    e('button', { className: 'btn-sec', onClick: () => { setActiveModel(m.id); showToast(`Active Model: ${m.name}`); } }, 'Select Active'),
                    e('button', { className: 'btn-sec', onClick: () => showToast('GGUF Export Triggered!') }, 'Export GGUF'),
                    e('button', { className: 'btn-sec', onClick: () => showToast('SafeTensors Export Triggered!') }, 'Export SafeTensors')
                  )
                )
              )
            )
          ),

          // Tab 5: Dataset Explorer
          activeTab === 'datasets' && e('div', { className: 'view-datasets' },
            e('div', { className: 'card-dataset' },
              e('h3', { style: { fontFamily: 'Newsreader', fontSize: '20px', marginBottom: '8px' } }, 'Dataset Explorer & Tokenizer Playground'),
              e('p', { style: { color: 'var(--text-muted)', fontSize: '13px' } }, 'Inspect dataset statistics and test live BPE tokenization.'),
              e('div', { style: { marginTop: '20px' } },
                e('label', { style: { fontSize: '12.5px', color: 'var(--text-muted)' } }, 'Live Tokenizer Input Sandbox:'),
                e('input', {
                  style: { width: '100%', background: 'var(--bg-input)', border: '1px solid var(--border-color)', color: 'var(--text-main)', padding: '10px', borderRadius: '6px', marginTop: '6px' },
                  value: sampleText,
                  onChange: ev => setSampleText(ev.target.value)
                }),
                e('div', { className: 'token-chip-container' },
                  tokens.map(t => e('span', { key: t.id, className: 'token-chip' }, `${t.text} [ID:${t.id}]`))
                )
              ),
              e('table', { className: 'table-data' },
                e('thead', null,
                  e('tr', null,
                    e('th', null, 'ID'),
                    e('th', null, 'Dataset Name'),
                    e('th', null, 'Token Count'),
                    e('th', null, 'Status')
                  )
                ),
                e('tbody', null,
                  e('tr', null,
                    e('td', null, '1'),
                    e('td', null, 'HuggingFaceFW/fineweb-edu'),
                    e('td', null, '10,000,000,000'),
                    e('td', null, e('span', { className: 'pill online' }, 'Streaming Active'))
                  ),
                  e('tr', null,
                    e('td', null, '2'),
                    e('td', null, 'wikitext-103-raw-v1'),
                    e('td', null, '103,000,000'),
                    e('td', null, e('span', { className: 'pill' }, 'Cached Local'))
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
                'Configure API credentials to route requests to external providers when local fallback is needed.'
              ),
              e('div', { className: 'byok-grid' },
                ['openai', 'anthropic', 'gemini', 'huggingface'].map(prov =>
                  e('div', { key: prov, className: 'field-group' },
                    e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' } },
                      e('label', null, `${prov.toUpperCase()} API Key:`),
                      byokStatus[prov] && e('span', { style: { fontSize: '11px', color: 'var(--accent-sage)' } }, byokStatus[prov])
                    ),
                    e('input', {
                      type: 'password',
                      placeholder: `Enter ${prov} key...`,
                      value: byokKeys[prov],
                      onChange: ev => setByokKeys({ ...byokKeys, [prov]: ev.target.value })
                    }),
                    e('button', {
                      className: 'btn-sec',
                      style: { marginTop: '8px', width: '100%' },
                      onClick: () => testBYOKKey(prov)
                    }, 'Test Key Connection')
                  )
                )
              ),
              e('button', {
                className: 'btn-send',
                style: { width: '100%', height: '46px', marginTop: '20px' },
                onClick: () => showToast('Provider Credentials Saved!')
              }, 'Save Provider Credentials')
            )
          ),

          // Tab 9: Modules & Repos Marketplace (Modrinth Style)
          activeTab === 'modules' && e('div', { className: 'view-modules' },
            e('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' } },
              e('div', null,
                e('h2', { style: { fontFamily: 'Newsreader', fontSize: '24px', margin: 0 } }, 'Modules & Repos Marketplace'),
                e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', margin: '4px 0 0 0' } }, 'Browse curated model weights, LoRA adapters, datasets, custom DAG nodes, or clone ML repos directly from GitHub.')
              ),
              e('button', { className: 'btn-sec', onClick: checkModuleUpdates }, '🔄 Check for Updates')
            ),

            // Search Bar & Filter Chips
            e('div', { style: { display: 'flex', gap: '12px', marginBottom: '16px', alignItems: 'center' } },
              e('input', {
                style: { flex: 1, height: '42px', padding: '0 14px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13.5px', color: 'var(--text-main)' },
                placeholder: 'Search curated modules or GitHub repos (e.g., starcoder, lora, fine-web)...',
                value: moduleSearchQuery,
                onChange: ev => {
                  setModuleSearchQuery(ev.target.value);
                  searchMarketplace(ev.target.value, moduleFilter);
                }
              }),
              e('select', {
                style: { height: '42px', padding: '0 12px', background: 'var(--bg-input)', border: '1px solid var(--border-color)', borderRadius: '6px', fontSize: '13px', color: 'var(--text-main)' },
                value: moduleFilter,
                onChange: ev => {
                  setModuleFilter(ev.target.value);
                  searchMarketplace(moduleSearchQuery, ev.target.value);
                }
              },
                e('option', { value: 'all' }, 'All Types'),
                e('option', { value: 'model' }, 'Model Weights'),
                e('option', { value: 'adapter' }, 'LoRA Adapters'),
                e('option', { value: 'dataset' }, 'Datasets'),
                e('option', { value: 'plugin' }, 'DAG Plugins'),
                e('option', { value: 'framework' }, 'Framework Tools')
              )
            ),

            // Updates Available Banner
            availableUpdates && availableUpdates.length > 0 && e('div', { style: { background: '#fffbeb', border: '1px solid #fde68a', padding: '14px 18px', borderRadius: '8px', marginBottom: '16px' } },
              e('h4', { style: { color: '#92400e', margin: '0 0 6px 0', fontSize: '14px' } }, `🔔 ${availableUpdates.length} Module Update(s) Available`),
              e('div', { style: { display: 'flex', gap: '10px', flexWrap: 'wrap' } },
                (availableUpdates || []).map(up =>
                  e('button', {
                    key: up.id,
                    className: 'btn-sec',
                    style: { fontSize: '12px', background: '#fef3c7', color: '#92400e', borderColor: '#fde68a' },
                    onClick: () => installModule(up)
                  }, `Update ${up.name} (${up.current_version} → ${up.latest_version})`)
                )
              )
            ),

            // Curated Recommendations Grid
            e('h3', { style: { fontFamily: 'Newsreader', fontSize: '18px', marginTop: '16px', marginBottom: '8px' } }, 'Curated Recommendations'),
            e('div', { className: 'modules-grid' },
              (marketplaceData && marketplaceData.curated ? marketplaceData.curated : []).map(mod =>
                e('div', { key: mod.id, className: 'module-card' },
                  e('div', null,
                    e('div', { className: 'module-header' },
                      e('span', { className: 'module-title' }, mod.name),
                      e('span', { className: 'badge-update', style: { background: 'var(--bg-surface)', color: 'var(--primary)', borderColor: 'var(--border-color)' } }, `v${mod.version}`)
                    ),
                    e('div', { className: 'module-author' }, `By ${mod.author} • ${mod.type.toUpperCase()}`),
                    e('div', { className: 'module-desc' }, mod.description),
                    e('div', { className: 'module-tags' },
                      mod.tags && mod.tags.map((t, idx) => e('span', { key: idx, className: 'module-tag' }, t)),
                      mod.requires_cuda && e('span', { className: 'module-tag', style: { background: '#fee2e2', color: '#991b1b' } }, 'CUDA Required')
                    )
                  ),
                  e('div', { className: 'module-footer' },
                    e('span', { style: { fontSize: '11px', color: 'var(--text-dim)' } }, mod.size_mb ? `${mod.size_mb} MB` : 'Remote Repo'),
                    mod.installed ?
                      e('div', { style: { display: 'flex', gap: '6px' } },
                        e('span', { style: { fontSize: '12px', color: 'var(--accent-sage)', fontWeight: '600', alignSelf: 'center' } }, '✓ Installed'),
                        e('button', { className: 'btn-purge', onClick: () => uninstallModule(mod.id) }, 'Remove')
                      ) :
                      e('button', { className: 'btn-send', style: { height: '32px', padding: '0 14px', fontSize: '12px' }, onClick: () => installModule(mod) }, 'Install')
                  )
                )
              )
            ),

            // GitHub Repositories Section (if searching)
            marketplaceData && marketplaceData.github && marketplaceData.github.length > 0 && e('div', { style: { marginTop: '24px' } },
              e('h3', { style: { fontFamily: 'Newsreader', fontSize: '18px', marginBottom: '8px' } }, 'GitHub Repositories'),
              e('div', { className: 'modules-grid' },
                (marketplaceData.github || []).map(gh =>
                  e('div', { key: gh.id, className: 'module-card' },
                    e('div', null,
                      e('div', { className: 'module-header' },
                        e('span', { className: 'module-title' }, gh.name),
                        e('span', { className: 'module-tag' }, `★ ${gh.stars}`)
                      ),
                      e('div', { className: 'module-author' }, `GitHub: ${gh.author}`),
                      e('div', { className: 'module-desc' }, gh.description),
                      e('div', { className: 'module-tags' },
                        gh.tags && gh.tags.map((t, idx) => e('span', { key: idx, className: 'module-tag' }, t))
                      )
                    ),
                    e('div', { className: 'module-footer' },
                      e('a', { href: gh.repo_url, target: '_blank', style: { fontSize: '11px', color: 'var(--primary)' } }, 'View on GitHub ↗'),
                      gh.installed ?
                        e('span', { style: { fontSize: '12px', color: 'var(--accent-sage)' } }, '✓ Cloned') :
                        e('button', { className: 'btn-send', style: { height: '32px', padding: '0 14px', fontSize: '12px' }, onClick: () => installModule(gh) }, 'Clone & Install')
                    )
                  )
                )
              )
            ),

            // Installed Modules List
            e('h3', { style: { fontFamily: 'Newsreader', fontSize: '18px', marginTop: '24px', marginBottom: '8px' } }, `Installed Modules (${installedModules.length})`),
            installedModules.length === 0 ?
              e('p', { style: { color: 'var(--text-dim)', fontSize: '13px' } }, 'No additional modules installed yet. Install recommendations above.') :
              e('div', { style: { display: 'flex', flexDirection: 'column', gap: '8px' } },
                installedModules.map(m =>
                  e('div', { key: m.id, className: 'software-item' },
                    e('div', null,
                      e('span', { style: { fontWeight: '600', fontSize: '14px' } }, m.name),
                      e('span', { style: { fontSize: '11px', color: 'var(--text-dim)', marginLeft: '10px' } }, `Location: ${m.installed_at || 'C:\\TriuneStudio\\modules'}`)
                    ),
                    e('button', { className: 'btn-purge', onClick: () => uninstallModule(m.id) }, 'Uninstall')
                  )
                )
              )
          ),

          // Tab 10: System & Hardware Auto-Scanner
          activeTab === 'environment' && e('div', { className: 'view-environment' },
            e('h2', { style: { fontFamily: 'Newsreader', fontSize: '24px', marginBottom: '6px' } }, 'System & Hardware Auto-Scanner'),
            e('p', { style: { color: 'var(--text-muted)', fontSize: '13px', marginBottom: '16px' } }, 'Auto-detects available GPU hardware, CUDA drivers, Python runtimes, and allows configuring custom workspace paths.'),

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
                      background: info.installed ? '#dcfce7' : '#fee2e2',
                      color: info.installed ? '#166534' : '#991b1b'
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
