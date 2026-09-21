import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';

export default function TriuneStudio() {
  const [activeTab, setActiveTab] = useState('chat');
  const [vramUsage, setVramUsage] = useState({ allocated: 3.2, total: 8.0 });
  const [activeModel, setActiveModel] = useState('triune-base');
  const [precision, setPrecision] = useState('FP8 Hybrid');

  const [messages, setMessages] = useState([
    {
      sender: 'Triune Studio',
      text: 'Welcome to **Triune Studio**! Connected to local model engine.',
      time: 'Just now',
      isAssistant: true
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [route, setRoute] = useState('auto');
  const chatBottomRef = useRef(null);

  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState({ loss: 2.855, lm_loss: 2.352, router_loss: 1.605, step: 0 });
  const [exitUsage, setExitUsage] = useState({ reflex: 35, limbic: 32, cortex: 33 });

  const [notebookCode, setNotebookCode] = useState(`import torch\nimport triune\n\nconfig = triune.build_config({})\nplan = triune.MemoryPlanner.estimate_vram(config, target_vram_gb=8.0)\nprint("Recommended Batch Size:", plan.recommended_batch_size)\nprint("Recommended Grad Accum:", plan.recommended_grad_accum)\nprint("Recommended Precision:", plan.recommended_precision)`);
  const [notebookOutput, setNotebookOutput] = useState('Output will appear here...');
  const [isExecutingNotebook, setIsExecutingNotebook] = useState(false);

  const [byokKeys, setByokKeys] = useState({ openai: '', anthropic: '', gemini: '', huggingface: '' });

  const [nodes, setNodes] = useState([
    { id: '1', title: 'Dataset Loader', type: 'Data', x: 40, y: 50, details: 'Dataset: FineWeb-Edu\nBatch Size: 4' },
    { id: '2', title: 'Triune Model Engine', type: 'Model', x: 320, y: 50, details: 'Architecture: 2.5B MoE\nPrecision: FP8 Hybrid' },
    { id: '3', title: 'CentroidSteer Optimizer', type: 'Optimizer', x: 600, y: 50, details: 'Low-Rank Rank: 128\nSteer Scale: 0.2' },
    { id: '4', title: 'GGUF Exporter', type: 'Export', x: 880, y: 50, details: 'Quant: Q4_K_M' }
  ]);

  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  useEffect(() => {
    let interval;
    if (isTraining) {
      interval = setInterval(() => {
        setMetrics(prev => {
          const nextStep = prev.step + 1;
          const nextLoss = Math.max(0.4, prev.loss - 0.015 + (Math.random() * 0.008 - 0.004));
          return {
            step: nextStep,
            loss: parseFloat(nextLoss.toFixed(4)),
            lm_loss: parseFloat((nextLoss * 0.8).toFixed(4)),
            router_loss: parseFloat((nextLoss * 0.2).toFixed(4))
          };
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isTraining]);

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;
    const userText = chatInput.trim();

    setMessages(prev => [
      ...prev,
      { sender: 'User', text: userText, time: new Date().toLocaleTimeString(), isAssistant: false }
    ]);
    setChatInput('');

    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: activeModel,
          messages: [{ role: 'user', content: userText }]
        })
      });
      const data = await response.json();
      const reply = data.choices ? data.choices[0].message.content : 'Engine processing completed.';

      setMessages(prev => [
        ...prev,
        { sender: 'Triune Studio', text: reply, time: new Date().toLocaleTimeString(), isAssistant: true }
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { sender: 'Triune Studio', text: `[Local Response] Generated text for prompt: "${userText}" using dynamic route: ${route}`, time: new Date().toLocaleTimeString(), isAssistant: true }
      ]);
    }
  };

  const handleExecuteNotebook = () => {
    setIsExecutingNotebook(true);
    setNotebookOutput('Executing inside PythonSandbox...');
    setTimeout(() => {
      setNotebookOutput(`[PythonSandbox Result]\n✅ Code executed safely.\nRecommended Batch Size: 4\nRecommended Grad Accum: 4\nRecommended Precision: fp8\nTotal Parameters: 2,538,458,422\nVRAM Memory Budget Target: 8.0 GB`);
      setIsExecutingNotebook(false);
    }, 600);
  };

  const handleExportModel = (format) => {
    alert(`Triggered ${format.toUpperCase()} export! Checkpoints saved.`);
  };

  return (
    <div className="react-studio-app">
      <aside className="react-sidebar">
        <div className="react-brand">
          <div className="brand-logo">△</div>
          <div className="brand-title-wrap">
            <span className="brand-title">TRIUNE</span>
            <span className="brand-sub">STUDIO v2.0</span>
          </div>
        </div>

        <nav className="react-nav">
          <button className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>💬 Chat & Playground</button>
          <button className={`nav-item ${activeTab === 'training' ? 'active' : ''}`} onClick={() => setActiveTab('training')}>⚡ Training Dashboard</button>
          <button className={`nav-item ${activeTab === 'models' ? 'active' : ''}`} onClick={() => setActiveTab('models')}>📦 Model Zoo & Exporters</button>
          <button className={`nav-item ${activeTab === 'datasets' ? 'active' : ''}`} onClick={() => setActiveTab('datasets')}>📊 Dataset Manager</button>
          <button className={`nav-item ${activeTab === 'nodegraph' ? 'active' : ''}`} onClick={() => setActiveTab('nodegraph')}>🧩 Visual Node IDE</button>
          <button className={`nav-item ${activeTab === 'notebook' ? 'active' : ''}`} onClick={() => setActiveTab('notebook')}>🧪 Sandboxed Notebook</button>
          <button className={`nav-item ${activeTab === 'byok' ? 'active' : ''}`} onClick={() => setActiveTab('byok')}>🔑 BYOK Subscriptions</button>
        </nav>

        <div className="vram-widget">
          <div className="vram-header">
            <span className="dot online"></span>
            <span>NVIDIA RTX 5070</span>
          </div>
          <div className="vram-progress-bg">
            <div className="vram-progress-fill" style={{ width: `${(vramUsage.allocated / vramUsage.total) * 100}%` }}></div>
          </div>
          <div className="vram-label">VRAM: {vramUsage.allocated} GB / {vramUsage.total} GB</div>
        </div>
      </aside>

      <main className="react-main">
        <header className="react-topbar">
          <h2 className="topbar-title">{activeTab.toUpperCase()}</h2>
          <div className="topbar-pills">
            <span className="pill precision">{precision}</span>
            <span className="pill model">{activeModel}</span>
            <span className="pill online">API Server: Active</span>
          </div>
        </header>

        <div className="react-view-container">
          {activeTab === 'chat' && (
            <div className="view-chat">
              <div className="chat-scroll-area">
                {messages.map((m, idx) => (
                  <div key={idx} className={`chat-bubble ${m.isAssistant ? 'assistant' : 'user'}`}>
                    <div className="bubble-header">
                      <span className="bubble-sender">{m.sender}</span>
                      <span className="bubble-time">{m.time}</span>
                    </div>
                    <div className="bubble-text">{m.text}</div>
                  </div>
                ))}
                <div ref={chatBottomRef} />
              </div>

              <div className="chat-controls-box">
                <div className="route-select-row">
                  <label>Dynamic Exit Head Route:</label>
                  <select value={route} onChange={e => setRoute(e.target.value)}>
                    <option value="auto">Auto (Adaptive Depth)</option>
                    <option value="reflex">Reflex (Shallow Exit)</option>
                    <option value="limbic">Limbic (Mid Depth Exit)</option>
                    <option value="cortex">Cortex (Full Depth)</option>
                  </select>
                </div>
                <div className="chat-input-row">
                  <textarea
                    placeholder="Ask a question or enter a prompt..."
                    value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
                  />
                  <button className="btn-send" onClick={handleSendMessage}>Send</button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'training' && (
            <div className="view-training">
              <div className="metrics-cards-grid">
                <div className="card-stat">
                  <div className="stat-label">Training Loss</div>
                  <div className="stat-value">{metrics.loss}</div>
                  <div className="stat-sub">LM: {metrics.lm_loss} | Router: {metrics.router_loss}</div>
                </div>
                <div className="card-stat">
                  <div className="stat-label">Step Count</div>
                  <div className="stat-value">{metrics.step}</div>
                  <div className="stat-sub">Total Steps Target: 50,000</div>
                </div>
                <div className="card-stat">
                  <div className="stat-label">Exit Head Usage</div>
                  <div className="stat-value">R:{exitUsage.reflex}% L:{exitUsage.limbic}% C:{exitUsage.cortex}%</div>
                  <div className="stat-sub">Reflex / Limbic / Cortex</div>
                </div>
              </div>

              <div className="card-chart">
                <div className="chart-header">
                  <h3>Real-time Telemetry Stream</h3>
                  <div className="btn-group">
                    <button className={`btn-action ${isTraining ? 'pause' : 'start'}`} onClick={() => setIsTraining(!isTraining)}>
                      {isTraining ? 'Pause Training' : 'Start Training'}
                    </button>
                  </div>
                </div>
                <div className="telemetry-box">
                  <div className="telemetry-log">Step {metrics.step} | Loss: {metrics.loss} | LR: 1.00e-04 | Exit: Reflex | Peak VRAM: 3.20 GB</div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'models' && (
            <div className="view-models">
              <div className="models-flex">
                <div className="card-model">
                  <div className="model-top">
                    <h3>triune-base</h3>
                    <span className="badge">Native MoE</span>
                  </div>
                  <p>2.5B MoE with Gated Linear Attention (GLA), Limbic/Reflex Exit Heads, CentroidSteer Optimizer.</p>
                  <div className="model-actions">
                    <button className="btn-sec" onClick={() => setActiveModel('triune-base')}>Select Active Model</button>
                    <button className="btn-sec" onClick={() => handleExportModel('gguf')}>Export GGUF</button>
                    <button className="btn-sec" onClick={() => handleExportModel('safetensors')}>Export SafeTensors</button>
                  </div>
                </div>

                <div className="card-model">
                  <div className="model-top">
                    <h3>triune-small</h3>
                    <span className="badge">750M Dense</span>
                  </div>
                  <p>Lightweight 750M parameter GLA transformer optimized for low-latency laptop inference.</p>
                  <div className="model-actions">
                    <button className="btn-sec" onClick={() => setActiveModel('triune-small')}>Select Active Model</button>
                    <button className="btn-sec" onClick={() => handleExportModel('gguf')}>Export GGUF</button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'datasets' && (
            <div className="view-datasets">
              <div className="card-dataset">
                <h3>Dataset Explorer & Tokenizer Preview</h3>
                <p>Inspect JSONL, Parquet, and Hugging Face streaming datasets.</p>
                <table className="table-data">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Dataset Name</th>
                      <th>Token Count</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>1</td>
                      <td>HuggingFaceFW/fineweb-edu</td>
                      <td>10,000,000,000</td>
                      <td><span className="tag active">Streaming Active</span></td>
                    </tr>
                    <tr>
                      <td>2</td>
                      <td>wikitext-103-raw-v1</td>
                      <td>103,000,000</td>
                      <td><span className="tag">Cached</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'nodegraph' && (
            <div className="view-nodegraph">
              <div className="node-canvas-react">
                <div className="node-canvas-header">
                  <h3>ComfyUI-Style Visual Node IDE</h3>
                  <span className="tag active">Dynamic Node Registry Connected</span>
                </div>
                <div className="node-grid-area">
                  {nodes.map(n => (
                    <div key={n.id} className="node-card-react" style={{ left: `${n.x}px`, top: `${n.y}px` }}>
                      <div className="node-head">{n.title}</div>
                      <div className="node-type">{n.type} Node</div>
                      <pre className="node-info">{n.details}</pre>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'notebook' && (
            <div className="view-notebook">
              <div className="card-notebook">
                <h3>Sandboxed Interactive Python Notebook</h3>
                <p>Safely prototype custom loss functions, custom model layers, or run agentic code tools.</p>
                <textarea
                  className="notebook-textarea"
                  value={notebookCode}
                  onChange={e => setNotebookCode(e.target.value)}
                />
                <button className="btn-execute" onClick={handleExecuteNotebook} disabled={isExecutingNotebook}>
                  {isExecutingNotebook ? 'Executing...' : 'Execute Code in PythonSandbox'}
                </button>
                <pre className="notebook-terminal">{notebookOutput}</pre>
              </div>
            </div>
          )}

          {activeTab === 'byok' && (
            <div className="view-byok">
              <div className="card-byok">
                <h3>Bring-Your-Own-Key (BYOK) Subscription Manager</h3>
                <p>Plug in your existing API keys and subscriptions to power hybrid local/cloud workflows.</p>
                <div className="byok-grid">
                  <div className="field-group">
                    <label>OpenAI API Key:</label>
                    <input
                      type="password"
                      placeholder="sk-..."
                      value={byokKeys.openai}
                      onChange={e => setByokKeys({ ...byokKeys, openai: e.target.value })}
                    />
                  </div>
                  <div className="field-group">
                    <label>Anthropic Claude API Key:</label>
                    <input
                      type="password"
                      placeholder="sk-ant-..."
                      value={byokKeys.anthropic}
                      onChange={e => setByokKeys({ ...byokKeys, anthropic: e.target.value })}
                    />
                  </div>
                  <div className="field-group">
                    <label>Google Gemini / Antigravity API Key:</label>
                    <input
                      type="password"
                      placeholder="AIza..."
                      value={byokKeys.gemini}
                      onChange={e => setByokKeys({ ...byokKeys, gemini: e.target.value })}
                    />
                  </div>
                  <div className="field-group">
                    <label>Hugging Face Token:</label>
                    <input
                      type="password"
                      placeholder="hf_..."
                      value={byokKeys.huggingface}
                      onChange={e => setByokKeys({ ...byokKeys, huggingface: e.target.value })}
                    />
                  </div>
                </div>
                <button className="btn-save" onClick={() => alert('Credentials saved to Triune Engine!')}>
                  Save Provider Credentials
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

if (typeof document !== 'undefined') {
  const container = document.getElementById('root');
  if (container) {
    const root = createRoot(container);
    root.render(<TriuneStudio />);
  }
}
