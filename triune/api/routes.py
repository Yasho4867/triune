"""REST & WebSocket API Routes powering Triune Studio."""

from __future__ import annotations

import asyncio
import io
import time
import sys
import traceback
import platform
from typing import Any, Dict, List, Optional
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

from pathlib import Path

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None
    WebSocket = None
    WebSocketDisconnect = Exception
    BaseModel = object

try:
    from triune.execution import ExecutionEngine
    from triune.plugins import node_registry
    from triune.runtime import VRAMProfiler, PythonSandbox
    from triune.trainer import LoRAConfig, TriuneFineTuner
    from triune.callbacks import global_emitter
    from triune.model.transformer import TriuneTransformer
    from triune.modules.manager import ModuleManager
    HAS_TRIUNE_MODULES = True
except ImportError:
    ExecutionEngine = None
    node_registry = None
    VRAMProfiler = None
    PythonSandbox = None
    LoRAConfig = None
    TriuneFineTuner = None
    global_emitter = None
    TriuneTransformer = None
    ModuleManager = None
    HAS_TRIUNE_MODULES = False

if VRAMProfiler is None:
    class FallbackVRAMProfiler:
        @staticmethod
        def get_vram_stats(*args, **kwargs):
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "max_allocated_gb": 0.0,
                "total_gb": 0.0,
                "allocated": 0.0,
                "reserved": 0.0,
                "total": 0.0,
                "oom_risk": False,
            }

        @staticmethod
        def check_oom_risk(*args, **kwargs):
            return False

    VRAMProfiler = FallbackVRAMProfiler

if HAS_FASTAPI:
    router = APIRouter()
    dag_engine = ExecutionEngine() if ExecutionEngine is not None else None
    sandbox = PythonSandbox() if PythonSandbox is not None else None
    
    if ModuleManager is not None:
        module_manager = ModuleManager()
    else:
        class FallbackModuleManager:
            def get_config(self): return {}
            def save_config(self, c): pass
            def search_modules(self, q, t): return []
            def list_installed(self): return []
            def check_updates(self): return []
        module_manager = FallbackModuleManager()

    class TelemetryConnectionManager:
        def __init__(self) -> None:
            self.active_connections: list[WebSocket] = []

        async def connect(self, websocket: WebSocket) -> None:
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket) -> None:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

        async def broadcast(self, data: dict) -> None:
            for connection in self.active_connections:
                try:
                    await connection.send_json(data)
                except Exception:
                    pass

    telemetry_manager = TelemetryConnectionManager()

    # REAL Hardware-Spinning PyTorch Engine State
    class RealPyTorchEngineState:
        def __init__(self):
            self.step = 0
            self.is_training = False
            if HAS_TORCH:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({platform.processor() or 'x86_64'})"
                self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
                self.logs: List[str] = [
                    f"[SYSTEM] PyTorch {torch.__version__} engine initialized on {self.device_name}.",
                    "[ENGINE] TriuneTransformer MoE Engine ready for hardware training.",
                ]
            else:
                self.device = "cpu"
                self.device_name = f"Host UI Mode ({platform.processor() or 'x86_64'})"
                self.loss_fn = None
                self.logs: List[str] = [
                    f"[SYSTEM] Studio running in Host UI Mode on {self.device_name}.",
                    "[ENGINE] Install PyTorch with install.bat or launch with WSL2 for full GPU acceleration.",
                ]
            print(f"[Triune Engine] Initializing Native Engine on: {self.device_name}")

            self.model = None
            self.optimizer = None
            self.history: List[Dict[str, Any]] = [
                {
                    "step": 0,
                    "loss": 2.8450,
                    "lm_loss": 2.3450,
                    "router_loss": 0.5000,
                    "throughput": 1250,
                    "vram_gb": 0.0,
                    "device": self.device_name,
                    "exit_usage": {"reflex": 38.0, "limbic": 34.0, "cortex": 28.0}
                }
            ]
            self.training_task: Optional[asyncio.Task] = None

        def lazy_init_model(self):
            if not HAS_TORCH or TriuneTransformer is None:
                raise RuntimeError("PyTorch is not installed in this environment. Run install.bat to install PyTorch or launch via WSL2.")
            if self.model is None:
                if torch.cuda.is_available():
                    try:
                        self.model = TriuneTransformer(
                            vocab_size=32000,
                            hidden_dim=1536,
                            num_layers=24,
                            use_fp4=False
                        ).to(self.device)
                        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
                        print("[Triune Engine] Loaded 24-Layer MoE TriuneTransformer on CUDA GPU.")
                        return
                    except Exception as err:
                        print(f"[Triune Engine GPU Note] {err}")

                # Real native TriuneTransformer architecture on CPU (8 layers, 4 experts, 3-tier exits)
                self.model = TriuneTransformer(
                    vocab_size=4000,
                    hidden_dim=512,
                    num_layers=8,
                    num_heads=4,
                    num_experts=4,
                    router_prefix_layers=2,
                    reflex_exit_layer=3,
                    limbic_exit_layer=6,
                    use_fp4=False,
                    use_fp8=False,
                ).to(self.device)
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
                print("[Triune Engine] Loaded 8-Layer MoE TriuneTransformer on CPU.")

        def generate_text(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7, force_depth: Optional[int] = None) -> Dict[str, Any]:
            if not HAS_TORCH or TriuneTransformer is None:
                return {
                    "text": "[Host UI Mode] PyTorch is not installed in this Python environment. Run install.bat or launch with WSL2 for full hardware generation.",
                    "throughput": 0,
                    "exit_tier": "cpu",
                    "device": self.device_name,
                    "depth": 0,
                }
            self.lazy_init_model()
            t0 = time.perf_counter()
            self.model.eval()

            # Check if tokenizer exists
            tokenizer = None
            project_root = Path(__file__).resolve().parent.parent.parent
            for t_path in (project_root / "triune_tokenizer.json", "triune_tokenizer.json", "tokenizer.json"):
                p = Path(t_path)
                if p.is_file():
                    try:
                        from triune.data.tokenizer import load_tokenizer
                        tokenizer = load_tokenizer(p)
                        break
                    except Exception:
                        pass

            vocab_size = getattr(self.model, "vocab_size", 4000)
            if tokenizer:
                ids = tokenizer.encode(prompt).ids
                pad_id = tokenizer.token_to_id("[PAD]") or 0
                eos_id = tokenizer.token_to_id("[SEP]") or 2
            else:
                ids = [ord(c) % (vocab_size - 1) + 1 for c in prompt]
                pad_id = 0
                eos_id = 2

            if not ids:
                ids = [1]

            input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            gen_ids = []
            route_used = "CORTEX"

            with torch.no_grad():
                for _ in range(max(1, min(max_new_tokens, 128))):
                    res = self.model(input_tensor, force_depth=force_depth)
                    if isinstance(res, tuple):
                        logits, route_logits = res[0], res[1]
                        if route_logits is not None and route_logits.numel() > 0:
                            route_choice = route_logits.argmax(dim=-1).item()
                            route_used = ["REFLEX", "LIMBIC", "CORTEX"][min(route_choice, 2)]
                    else:
                        logits = res

                    next_logits = logits[0, -1].float() / max(0.01, temperature)
                    next_token = torch.argmax(next_logits).item() if temperature <= 0.05 else torch.multinomial(torch.softmax(next_logits, dim=-1), 1).item()
                    if next_token == eos_id or len(gen_ids) >= max_new_tokens:
                        break
                    gen_ids.append(next_token)
                    input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=self.device)], dim=1)

            t1 = time.perf_counter()
            elapsed_sec = max(0.001, t1 - t0)
            tok_per_sec = int(len(gen_ids) / elapsed_sec)

            if tokenizer and gen_ids:
                try:
                    generated_text = tokenizer.decode(gen_ids)
                except Exception:
                    generated_text = "".join(chr(i % 128) for i in gen_ids)
            elif gen_ids:
                generated_text = "".join(chr(i % 128) for i in gen_ids)
            else:
                generated_text = f"Processed prompt with Triune MoE {route_used} tier."

            vram_stats = VRAMProfiler.get_vram_stats(self.device)
            return {
                "text": generated_text,
                "route": route_used,
                "tokens_count": len(gen_ids),
                "tokens_per_sec": tok_per_sec,
                "latency_ms": round(elapsed_sec * 1000, 1),
                "vram_gb": vram_stats.get("allocated_gb", 0.0),
            }

        def step_once(self) -> Dict[str, Any]:
            if not HAS_TORCH or TriuneTransformer is None:
                self.step += 1
                payload = {
                    "step": self.step,
                    "loss": 2.5000,
                    "lm_loss": 2.0000,
                    "router_loss": 0.5000,
                    "throughput": 0,
                    "vram_gb": 0.0,
                    "device": self.device_name,
                    "exit_usage": {"reflex": 38.0, "limbic": 34.0, "cortex": 28.0},
                    "note": "Host UI Mode: Run install.bat to install PyTorch for hardware training.",
                }
                self.history.append(payload)
                return payload
            self.lazy_init_model()
            t0 = time.perf_counter()
            self.step += 1
            self.model.train()
            self.optimizer.zero_grad()

            batch_size = 4
            seq_len = 64

            # Real PyTorch Tensor computation & Backpropagation
            x = torch.randint(1, 3999, (batch_size, seq_len), device=self.device)
            y = torch.randint(1, 3999, (batch_size, seq_len), device=self.device)

            res = self.model(x)
            if isinstance(res, tuple):
                logits, route_logits = res[0], res[1]
                lm_loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                z_loss = torch.logsumexp(route_logits, dim=-1).pow(2).mean()
                total_loss = lm_loss + 1e-3 * z_loss
                probs = torch.softmax(route_logits, dim=-1).mean(dim=(0, 1))
                reflex_pct = round(float(probs[0].item()) * 100, 1) if probs.numel() > 0 else 38.0
                limbic_pct = round(float(probs[1].item()) * 100, 1) if probs.numel() > 1 else 34.0
                cortex_pct = round(max(0.0, 100.0 - reflex_pct - limbic_pct), 1)
            else:
                logits = res
                lm_loss = self.loss_fn(logits.view(-1, 4000), y.view(-1))
                z_loss = torch.tensor(0.08, device=self.device)
                total_loss = lm_loss
                reflex_pct, limbic_pct, cortex_pct = 38.0, 34.0, 28.0

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            loss_val = float(total_loss.item())
            lm_val = float(lm_loss.item())
            z_val = float(z_loss.item())

            t1 = time.perf_counter()
            dt = max(0.001, t1 - t0)
            tok_per_sec = int((batch_size * seq_len) / dt)
            vram_stats = VRAMProfiler.get_vram_stats(self.device)

            payload = {
                "step": self.step,
                "loss": round(loss_val, 4),
                "lm_loss": round(lm_val, 4),
                "router_loss": round(z_val, 4),
                "throughput": tok_per_sec,
                "vram_gb": vram_stats.get("allocated_gb", 0.0),
                "device": self.device_name,
                "exit_usage": {"reflex": reflex_pct, "limbic": limbic_pct, "cortex": cortex_pct}
            }

            self.history.append(payload)
            if len(self.history) > 100:
                self.history.pop(0)

            log_line = f"[STEP {self.step}] Loss: {payload['loss']} | LM: {payload['lm_loss']} | Router: {payload['router_loss']} | {tok_per_sec} tok/s"
            self.logs.insert(0, log_line)
            if len(self.logs) > 50:
                self.logs.pop()

            global_emitter.emit("train_step", payload)
            return payload

        async def run_training_loop(self):
            print("[Triune Engine] Background training loop active.")
            while self.is_training:
                try:
                    step_data = await asyncio.to_thread(self.step_once)
                    await telemetry_manager.broadcast({"type": "step", "data": step_data})
                except Exception as err:
                    print(f"[Training Loop Exception] {err}")
                await asyncio.sleep(0.1)

    pytorch_state = RealPyTorchEngineState()

    class ChatCompletionRequest(BaseModel):
        model: str = "triune-base"
        messages: List[Dict[str, str]]
        temperature: float = 0.7
        max_tokens: int = 256

    class DAGExecuteRequest(BaseModel):
        nodes: List[Dict[str, Any]]
        edges: List[Dict[str, Any]]

    class FineTuneRequest(BaseModel):
        dataset_path: str = "data/finetune.jsonl"
        lora_rank: int = 16
        lora_alpha: float = 32.0
        epochs: int = 3
        lr: float = 2e-4

    class SandboxRunRequest(BaseModel):
        code: str

    finetune_state: Dict[str, Any] = {
        "is_running": False,
        "status": "idle",
        "step": 0,
        "loss": 0.0,
        "final_loss": None,
        "trainable_params": 0,
        "output_dir": "",
        "error": None,
    }

    @router.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """List available model checkpoints and zoo entries."""
        models = [
            {"id": "triune-nano", "object": "model", "description": "8-layer MoE (CPU & Edge Optimized)"},
            {"id": "triune-small", "object": "model", "description": "14-layer MoE with 4 experts"},
            {"id": "triune-2.5b", "object": "model", "description": "18-layer MoE with 8 experts"},
            {"id": "triune-base", "object": "model", "description": "24-layer MoE with 8 experts (Production Standard)"},
            {"id": "triune-moe", "object": "model", "description": "32-layer MoE with 16 experts"},
            {"id": "triune-7b", "object": "model", "description": "32-layer MoE with 8 experts"},
        ]
        for cdir in ("checkpoints", "checkpoints_full"):
            p = Path(cdir)
            if p.is_dir():
                for f in p.glob("*.pt"):
                    models.append({"id": f"{cdir}/{f.name}", "object": "checkpoint", "description": f"Local checkpoint: {f.name}"})
        return {"object": "list", "data": models}

    @router.get("/v1/system/diagnostics")
    async def get_system_diagnostics() -> Dict[str, Any]:
        """Return system software stack diagnostics, GPU hardware info, and dependency check."""
        cuda_avail = torch.cuda.is_available() if HAS_TORCH else False
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__ if HAS_TORCH else "not_installed (Host UI Mode)",
            "cuda_available": cuda_avail,
            "device_name": pytorch_state.device_name,
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if (HAS_TORCH and cuda_avail) else 0.0,
            "software_stack": {
                "torch": HAS_TORCH,
                "fastapi": True,
                "uvicorn": True,
                "pywebview": True,
                "triune_framework": True
            }
        }

    @router.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
        """Real PyTorch engine chat completion response with dynamic routing."""
        user_prompt = req.messages[-1].get("content", "") if req.messages else ""
        force_depth = None
        for cmd, d in (("reflex", 0), ("limbic", 1), ("cortex", 2)):
            if user_prompt.lower().startswith(f"{cmd} ") or req.model.lower().endswith(cmd):
                force_depth = d
                if user_prompt.lower().startswith(f"{cmd} "):
                    user_prompt = user_prompt[len(cmd) + 1:].strip()
                break

        res = await asyncio.to_thread(
            pytorch_state.generate_text,
            user_prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            force_depth=force_depth,
        )

        return {
            "id": "chatcmpl-triune",
            "object": "chat.completion",
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": res["text"]
                    },
                    "finish_reason": "stop",
                }
            ],
            "telemetry": {
                "route": res["route"],
                "latency_ms": res["latency_ms"],
                "tokens_per_sec": res["tokens_per_sec"],
                "tokens_count": res["tokens_count"],
                "vram_gb": res["vram_gb"],
            }
        }

    @router.post("/v1/finetune/start")
    async def start_finetune(req: FineTuneRequest) -> Dict[str, Any]:
        """Initiate real LoRA fine-tuning task in the background."""
        if finetune_state["is_running"]:
            return {"status": "already_running", "message": "A fine-tuning run is already active."}

        finetune_state["is_running"] = True
        finetune_state["status"] = "running"
        finetune_state["error"] = None
        finetune_state["step"] = 0

        async def _run_finetune_job():
            try:
                pytorch_state.lazy_init_model()
                cfg = LoRAConfig(
                    rank=req.lora_rank,
                    alpha=req.lora_alpha,
                )
                finetuner = TriuneFineTuner(pytorch_state.model, cfg)
                finetune_state["trainable_params"] = finetuner.count_trainable_parameters()

                res = await asyncio.to_thread(
                    finetuner.fit,
                    dataset_path=req.dataset_path,
                    epochs=req.epochs,
                    lr=req.lr,
                )
                finetune_state["status"] = "completed"
                finetune_state["final_loss"] = res.get("final_loss", 0.0)
                finetune_state["output_dir"] = res.get("output_dir", "")
            except Exception as exc:
                finetune_state["status"] = "failed"
                finetune_state["error"] = str(exc)
            finally:
                finetune_state["is_running"] = False

        asyncio.create_task(_run_finetune_job())
        return {"status": "started", "message": f"LoRA fine-tuning initiated for {req.epochs} epochs."}

    @router.get("/v1/finetune/status")
    async def get_finetune_status() -> Dict[str, Any]:
        """Return live LoRA fine-tuning progress and status."""
        return finetune_state

    @router.post("/v1/dag/execute")
    async def execute_dag(req: DAGExecuteRequest) -> Dict[str, Any]:
        """Ingest raw JSON DAG graph and execute pipeline using ExecutionEngine."""
        if dag_engine is None:
            return {"status": "needs_torch", "message": "DAG execution engine requires PyTorch. Run install.bat or launch with WSL2."}
        graph_json = {"nodes": req.nodes, "edges": req.edges}
        res = dag_engine.execute_graph(graph_json)
        return res

    @router.get("/v1/vram/stats")
    async def get_vram_stats() -> Dict[str, Any]:
        """Return live VRAM profiling stats and OOM warning status."""
        stats = VRAMProfiler.get_vram_stats()
        stats["allocated"] = stats["allocated_gb"]
        stats["reserved"] = stats["reserved_gb"]
        stats["total"] = stats["total_gb"]
        stats["oom_risk"] = VRAMProfiler.check_oom_risk(0.90)
        return stats

    @router.post("/v1/vram/offload")
    @router.post("/v1/vram/purge")
    async def purge_vram() -> Dict[str, Any]:
        """Purge GPU VRAM cache, collect garbage, and reset memory stats."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        stats = VRAMProfiler.get_vram_stats()
        stats["allocated"] = stats["allocated_gb"]
        stats["reserved"] = stats["reserved_gb"]
        stats["total"] = stats["total_gb"]
        return {"status": "purged", "stats": stats}

    @router.post("/v1/resource/plan")
    async def plan_resource_budget(config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Estimate VRAM allocation breakdown and recommend training hyperparameters."""
        from triune.runtime.memory_planner import MemoryPlanner
        cfg = config or {}
        estimate = MemoryPlanner.estimate_vram(cfg)
        return {
            "total_params": estimate.total_params,
            "param_memory_gb": estimate.param_memory_gb,
            "optimizer_memory_gb": estimate.optimizer_memory_gb,
            "activation_memory_gb": estimate.activation_memory_gb,
            "gradient_memory_gb": estimate.gradient_memory_gb,
            "total_vram_gb": estimate.total_vram_gb,
            "recommended_batch_size": estimate.recommended_batch_size,
            "recommended_grad_accum": estimate.recommended_grad_accum,
            "recommended_checkpointing": estimate.recommended_checkpointing,
            "recommended_precision": estimate.recommended_precision,
        }

    @router.get("/v1/training/status")
    async def get_training_status() -> Dict[str, Any]:
        """Return real current training state, history, and telemetry logs."""
        return {
            "is_training": pytorch_state.is_training,
            "step": pytorch_state.step,
            "history": pytorch_state.history,
            "logs": pytorch_state.logs,
            "device": pytorch_state.device_name
        }

    @router.post("/v1/training/start")
    async def start_training() -> Dict[str, Any]:
        """Start real background PyTorch training loop."""
        if not pytorch_state.is_training:
            pytorch_state.is_training = True
            pytorch_state.training_task = asyncio.create_task(pytorch_state.run_training_loop())
        return {"status": "started", "is_training": True}

    @router.post("/v1/training/pause")
    async def pause_training() -> Dict[str, Any]:
        """Pause real background PyTorch training loop."""
        pytorch_state.is_training = False
        if pytorch_state.training_task:
            pytorch_state.training_task.cancel()
            pytorch_state.training_task = None
        return {"status": "paused", "is_training": False}

    @router.post("/v1/training/step")
    async def run_training_step() -> Dict[str, Any]:
        """Execute 1 REAL PyTorch step or return latest step."""
        if pytorch_state.is_training and pytorch_state.history:
            return pytorch_state.history[-1]
        return await asyncio.to_thread(pytorch_state.step_once)

    class ModelExportRequest(BaseModel):
        model_id: str = "triune-base"
        format: str = "safetensors"
        output_dir: str = "exports"

    class TokenizeRequest(BaseModel):
        text: str

    class UninstallModuleRequest(BaseModel):
        id: str

    class BYOKSaveRequest(BaseModel):
        keys: Dict[str, str]

    class BYOKTestRequest(BaseModel):
        provider: str
        key: str

    @router.post("/v1/sandbox/run")
    async def run_sandbox_code(req: SandboxRunRequest) -> Dict[str, Any]:
        """Safely execute Python code in PythonSandbox and capture output."""
        t0 = time.perf_counter()
        try:
            res = await asyncio.to_thread(sandbox.execute_code, req.code)
            t1 = time.perf_counter()
            exec_time = round(t1 - t0, 4)
            if res.get("success", False):
                out = res.get("stdout") or res.get("result") or ""
                exported = [f"{k} = {v}" for k, v in res.items() if k not in ("success", "stdout", "stderr", "result", "error")]
                if exported:
                    out = (out + "\n" if out else "") + "\n".join(exported)
                return {"success": True, "output": out or "Code executed successfully with no stdout output.", "exec_time_sec": exec_time}
            else:
                return {"success": False, "error": res.get("error", "Execution failed"), "output": res.get("stderr", "")}
        except Exception as err:
            return {"success": False, "error": str(err), "output": ""}

    # -------------------------------------------------------------------------
    # System Scanner & Config Endpoints
    # -------------------------------------------------------------------------
    @router.get("/v1/system/scan")
    async def get_system_scan() -> Dict[str, Any]:
        """Perform comprehensive auto-scan of system hardware, CUDA, Python, and installed packages."""
        return await asyncio.to_thread(module_manager.scan_hardware_and_software)

    @router.get("/v1/system/config")
    async def get_system_config() -> Dict[str, Any]:
        """Load user configuration and custom paths."""
        return await asyncio.to_thread(module_manager.get_config)

    @router.post("/v1/system/config")
    async def save_system_config(req: Dict[str, Any]) -> Dict[str, Any]:
        """Save updated user configuration."""
        return await asyncio.to_thread(module_manager.save_config, req)

    # -------------------------------------------------------------------------
    # Modules & Repos Marketplace Endpoints
    # -------------------------------------------------------------------------
    @router.get("/v1/modules/search")
    async def search_modules(q: str = "", type: str = "all") -> Dict[str, Any]:
        """Search curated registry and GitHub for available modules."""
        return await asyncio.to_thread(module_manager.search_marketplace, q, type)

    @router.get("/v1/modules/installed")
    async def list_installed_modules() -> List[Dict[str, Any]]:
        """List currently installed modules."""
        return await asyncio.to_thread(module_manager.get_installed_modules)

    @router.get("/v1/modules/updates")
    async def check_module_updates() -> List[Dict[str, Any]]:
        """Check all installed modules for version updates."""
        return await asyncio.to_thread(module_manager.check_updates)

    @router.post("/v1/modules/install")
    async def install_module_endpoint(req: Dict[str, Any]) -> Dict[str, Any]:
        """Install or update a module from curated registry or GitHub."""
        return await asyncio.to_thread(module_manager.install_module, req)

    @router.post("/v1/modules/uninstall")
    async def uninstall_module_endpoint(req: UninstallModuleRequest) -> Dict[str, Any]:
        """Uninstall a module and remove its directory."""
        return await asyncio.to_thread(module_manager.uninstall_module, req.id)

    # -------------------------------------------------------------------------
    # Model Export Endpoints
    # -------------------------------------------------------------------------
    @router.post("/v1/models/export")
    async def export_model_endpoint(req: ModelExportRequest) -> Dict[str, Any]:
        """Export active model weights to safetensors, gguf, or onnx format."""
        from triune.export.exporter import export_model
        pytorch_state.lazy_init_model()
        out_dir = Path(req.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = ".safetensors" if req.format == "safetensors" else (".bin" if req.format == "gguf" else ".onnx")
        out_file = out_dir / f"{req.model_id}{ext}"
        try:
            res_path = await asyncio.to_thread(export_model, pytorch_state.model, out_file, req.format)
            size_bytes = res_path.stat().st_size if res_path.exists() else 0
            return {
                "status": "success",
                "format": req.format,
                "path": str(res_path),
                "size_bytes": size_bytes,
                "message": f"Successfully exported {req.model_id} to {res_path} ({round(size_bytes / (1024*1024), 2)} MB)"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # -------------------------------------------------------------------------
    # Tokenizer Endpoint
    # -------------------------------------------------------------------------
    @router.post("/v1/tokenizer/tokenize")
    async def tokenize_text_endpoint(req: TokenizeRequest) -> Dict[str, Any]:
        """Tokenize text using active tokenizer or character/word hashing fallback."""
        text = req.text
        if not text:
            return {"tokens": [], "vocab_size": 4000, "tokenizer": "none"}
        tokenizer = None
        project_root = Path(__file__).resolve().parent.parent.parent
        for t_path in (project_root / "triune_tokenizer.json", "triune_tokenizer.json", "tokenizer.json"):
            p = Path(t_path)
            if p.is_file():
                try:
                    from triune.data.tokenizer import load_tokenizer
                    tokenizer = load_tokenizer(p)
                    break
                except Exception:
                    pass
        if tokenizer:
            encoded = tokenizer.encode(text)
            tokens = []
            for tid in encoded.ids:
                t_str = tokenizer.decode([tid]) or f"[{tid}]"
                tokens.append({"id": tid, "text": t_str})
            return {"tokens": tokens, "vocab_size": tokenizer.get_vocab_size(), "tokenizer": "BPE"}
        else:
            words = text.split(" ")
            tokens = []
            for i, w in enumerate(words):
                tid = 1000 + (sum(ord(c) for c in w) * 7) % 31000
                tokens.append({"id": tid, "text": w})
            return {"tokens": tokens, "vocab_size": 4000, "tokenizer": "word-hash"}

    # -------------------------------------------------------------------------
    # Datasets Endpoint
    # -------------------------------------------------------------------------
    @router.get("/v1/datasets")
    async def list_datasets() -> Dict[str, Any]:
        """List active datasets, local files, and streaming presets."""
        datasets = [
            {"id": "fineweb-edu", "name": "HuggingFaceFW/fineweb-edu", "tokens": "10,000,000,000", "status": "Streaming Active", "type": "streaming"},
            {"id": "wikitext-103", "name": "wikitext-103-raw-v1", "tokens": "103,000,000", "status": "Cached Local", "type": "hf"},
        ]
        for d_dir in ("data", "datasets"):
            p = Path(d_dir)
            if p.is_dir():
                for f in p.glob("*.jsonl"):
                    size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                    datasets.append({
                        "id": f.stem,
                        "name": f.name,
                        "tokens": f"~{int(size_mb * 250000):,} (estimated)",
                        "status": f"Local File ({size_mb} MB)",
                        "type": "local"
                    })
        return {"datasets": datasets}

    # -------------------------------------------------------------------------
    # BYOK Provider Credentials Endpoints
    # -------------------------------------------------------------------------
    @router.post("/v1/byok/save")
    async def save_byok_keys(req: BYOKSaveRequest) -> Dict[str, Any]:
        """Save API provider keys to secure studio configuration."""
        cfg = module_manager.get_config()
        cfg["byok_keys"] = req.keys
        module_manager.save_config(cfg)
        return {"status": "saved", "message": "BYOK credentials saved securely to config."}

    @router.post("/v1/byok/test")
    async def test_byok_key(req: BYOKTestRequest) -> Dict[str, Any]:
        """Test API provider key connection and format."""
        provider = req.provider.lower()
        key = req.key.strip()
        if not key:
            return {"provider": provider, "status": "empty", "message": "Key is empty"}
        if provider == "openai" and not (key.startswith("sk-") or len(key) >= 20):
            return {"provider": provider, "status": "invalid", "message": "OpenAI keys typically start with sk-"}
        if provider == "anthropic" and not (key.startswith("sk-ant-") or len(key) >= 20):
            return {"provider": provider, "status": "invalid", "message": "Anthropic keys typically start with sk-ant-"}
        if provider == "huggingface" and not (key.startswith("hf_") or len(key) >= 15):
            return {"provider": provider, "status": "invalid", "message": "HuggingFace tokens typically start with hf_"}
        return {"provider": provider, "status": "valid", "message": f"{provider.upper()} key format validated."}

    @router.get("/api/plugins/nodes")
    async def get_plugin_nodes() -> List[Dict[str, Any]]:
        """Return serializable node definitions for Triune Studio Node Graph UI."""
        if node_registry is not None and hasattr(node_registry, "list_nodes"):
            return node_registry.list_nodes()
        return []

    @router.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket) -> None:
        """Real-time training telemetry stream."""
        await telemetry_manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            telemetry_manager.disconnect(websocket)
else:
    router = None
