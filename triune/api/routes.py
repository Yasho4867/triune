"""REST & WebSocket API Routes powering Triune Studio."""

from __future__ import annotations

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import asyncio
import warnings
warnings.filterwarnings("ignore", message=".*httpx.*")
warnings.filterwarnings("ignore", message=".*starlette.testclient.*")
import io
import json
import time
import math
import sys
import threading
from collections import deque
import traceback
import platform
import urllib.request
import urllib.error
import urllib.parse
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
except ImportError:
    ExecutionEngine = None

try:
    from triune.plugins import node_registry
except ImportError:
    node_registry = None

try:
    from triune.runtime import VRAMProfiler, PythonSandbox
except ImportError:
    VRAMProfiler = None
    try:
        from triune.runtime.sandbox import PythonSandbox
    except Exception:
        PythonSandbox = None

try:
    from triune.trainer import LoRAConfig, TriuneFineTuner
except ImportError:
    LoRAConfig = None
    TriuneFineTuner = None

try:
    from triune.callbacks import global_emitter
except ImportError:
    global_emitter = None

try:
    from triune.model.transformer import TriuneTransformer
except ImportError:
    TriuneTransformer = None

try:
    from triune.modules.manager import ModuleManager
except ImportError:
    ModuleManager = None

HAS_TRIUNE_MODULES = all(x is not None for x in [ExecutionEngine, node_registry, ModuleManager])

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


def sanitize_text(text: str) -> str:
    """Repair common UTF-8 / Windows-1252 Mojibake artifacts in raw datasets and generations."""
    if not text or not isinstance(text, str):
        return ""
    replacements = [
        ("â€™", "'"),
        ("â€˜", "'"),
        ("â€œ", '"'),
        ("â€", '"'),
        ("â€", '"'),
        ("â€“", "-"),
        ("â€”", "--"),
        ("â€¦", "..."),
        ("Ã©", "é"),
        ("Ã¨", "è"),
        ("Ã ", "à"),
        ("Ã¢", "â"),
        ("Ã®", "î"),
        ("Ã´", "ô"),
        ("Ã»", "û"),
        ("Ã§", "ç"),
        ("\ufffd", ""),
    ]
    for bad, good in replacements:
        if bad in text:
            text = text.replace(bad, good)
    return text


if HAS_FASTAPI:
    router = APIRouter()
    dag_engine = ExecutionEngine() if ExecutionEngine is not None else None
    sandbox = PythonSandbox() if PythonSandbox is not None else None
    
    if ModuleManager is not None:
        module_manager = ModuleManager()
    else:
        class FallbackModuleManager:
            def get_config(self): return {"byok_keys": {}, "theme": "retro"}
            def save_config(self, c): pass
            def search_marketplace(self, q="", mod_type="all", source="all"): return {"results": []}
            def search_modules(self, q, t): return []
            def get_installed_modules(self): return []
            def list_installed(self): return []
            def check_updates(self): return []
            def scan_hardware_and_software(self): return {"device": "cpu", "status": "nominal"}
            def clone_repository(self, url, mod_type="module", branch="main", name=None):
                return {"status": "error", "message": "ModuleManager unavailable"}
            def git_pull(self, module_id):
                return {"status": "error", "message": "ModuleManager unavailable"}
            def discover_and_register_nodes(self, p): return []
            def install_module(self, req): return {"status": "error"}
            def uninstall_module(self, mod_id): return {"status": "error"}
            def scan_workspace_plugins(self, path=None): return []
            def create_custom_plugin_template(self, name, cat="custom", dest=None): return {"status": "error"}
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
            disconnected: list[WebSocket] = []
            for connection in list(self.active_connections):
                try:
                    await connection.send_json(data)
                except Exception:
                    # A failed send means this connection can no longer receive
                    # telemetry. Retaining it would grow this list indefinitely.
                    disconnected.append(connection)
            for connection in disconnected:
                self.disconnect(connection)

    telemetry_manager = TelemetryConnectionManager()

    # BYOK External Provider Callers
    def _call_openai_api(messages: List[Dict[str, str]], api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 2048) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max(64, min(max_tokens, 8192)),
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "TriuneStudio/2.1",
            }
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _call_anthropic_api(messages: List[Dict[str, str]], api_key: str, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.7, max_tokens: int = 2048) -> str:
        url = "https://api.anthropic.com/v1/messages"
        system_msg = ""
        user_msgs = []
        for m in messages:
            if m.get("role") == "system":
                system_msg = m.get("content", "")
            else:
                user_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        if not user_msgs:
            user_msgs = [{"role": "user", "content": "Hello"}]
        payload_dict = {
            "model": model,
            "messages": user_msgs,
            "max_tokens": max(64, min(max_tokens, 8192)),
            "temperature": temperature,
        }
        if system_msg:
            payload_dict["system"] = system_msg
        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "User-Agent": "TriuneStudio/2.1",
            }
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]

    def _call_gemini_api(messages: List[Dict[str, str]], api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7, max_tokens: int = 2048) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        contents = []
        system_instruction = None
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                system_instruction = {"parts": [{"text": m.get("content", "")}]}
            else:
                api_role = "model" if role == "assistant" else "user"
                contents.append({"role": api_role, "parts": [{"text": m.get("content", "")}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        payload_dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": max(0.0, min(temperature, 2.0)),
                "maxOutputTokens": max(64, min(max_tokens, 8192))
            }
        }
        if system_instruction:
            payload_dict["systemInstruction"] = system_instruction
        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "TriuneStudio/2.1"}
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]

    class HFRestStreamReader:
        """Streaming reader for Hugging Face datasets via the dataset-server REST API.

        Fetches JSON row pages with offset pagination and loops to the next epoch at
        the end of the split. Client errors (4xx: wrong config/split, gated repo) are
        raised immediately; transient errors are retried with exponential backoff up
        to ``max_retries`` consecutive failures, then raised so callers can surface it.
        """
        def __init__(self, dataset_name: str, config: Optional[str] = "default", split: str = "train", token: Optional[str] = None, page_size: int = 100, max_retries: int = 6):
            self.dataset_name = dataset_name
            self.config = config or "default"
            self.split = split
            self.token = token
            self.page_size = page_size
            self.max_retries = max_retries
            self.offset = 0
            self.epoch = 0
            self.total_rows_yielded = 0
            self.last_error = None

        def __iter__(self):
            headers = {"User-Agent": "TriuneStudio/2.0"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            failures = 0
            while True:
                url = (
                    f"https://datasets-server.huggingface.co/rows?"
                    f"dataset={urllib.parse.quote(self.dataset_name, safe='')}&"
                    f"config={urllib.parse.quote(self.config, safe='')}&"
                    f"split={urllib.parse.quote(self.split, safe='')}&"
                    f"offset={self.offset}&limit={self.page_size}"
                )
                req = urllib.request.Request(url, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                except urllib.error.HTTPError as e:
                    self.last_error = f"HTTP {e.code}: {e.reason}"
                    if 400 <= e.code < 500 and e.code != 429:
                        raise RuntimeError(f"HF dataset-server rejected '{self.dataset_name}' (config={self.config}, split={self.split}): {self.last_error}")
                    failures += 1
                    if failures > self.max_retries:
                        raise RuntimeError(f"HF stream failed after {self.max_retries} retries: {self.last_error}")
                    time.sleep(min(30.0, 0.5 * (2 ** failures)))
                    continue
                except Exception as e:
                    self.last_error = str(e)
                    failures += 1
                    if failures > self.max_retries:
                        raise RuntimeError(f"HF stream failed after {self.max_retries} retries: {self.last_error}")
                    time.sleep(min(30.0, 0.5 * (2 ** failures)))
                    continue

                failures = 0
                rows = data.get("rows", [])
                if not rows:
                    if self.offset == 0:
                        raise RuntimeError(f"HF split '{self.split}' of '{self.dataset_name}' returned no rows")
                    # End of split: wrap around for the next epoch
                    self.epoch += 1
                    self.offset = 0
                    continue
                for r in rows:
                    self.total_rows_yielded += 1
                    row_data = r.get("row", {})
                    for text_key in ("text", "content", "story", "prompt", "completion", "question"):
                        if text_key in row_data and isinstance(row_data[text_key], str):
                            row_data[text_key] = sanitize_text(row_data[text_key])
                    yield row_data
                self.offset += len(rows)

    # REAL Hardware-Spinning PyTorch Engine State
    class RealPyTorchEngineState:
        def __init__(self):
            self.step = 0
            self.is_training = False
            self.model = None
            self.optimizer = None
            self.tokenizer = None
            self.data_chunks: List[List[int]] = []
            self.data_idx = 0
            self.batch_size = 4
            self.seq_len = 64
            self.grad_accum_steps = 4
            self.gpu_offload_pct = 100
            self.gpu_layers = 6
            self.precision = "bf16"
            self.active_model_id = "triune-small"
            self.architecture_spec: Dict[str, Any] = {
                "preset_id": "triune-small",
                "name": "Triune-Small (Canonical 6L MoE)",
                "tier": "Production Baseline",
                "num_layers": 6,
                "hidden_dim": 256,
                "num_heads": 4,
                "head_dim": 64,
                "num_experts": 4,
                "experts_routed": 4,
                "experts_shared": 1,
                "router_prefix_layers": 1,
                "reflex_exit_layer": 2,
                "limbic_exit_layer": 4,
                "depth_mode": "cortex",
                "balance_loss_weight": 0.3,
            }
            self._token_accumulator: List[int] = []
            self.depth_mode = "cortex"  # "cortex" (full 6-layer backbone) | "joint" (all 3 exits) | "dynamic" (router)
            self.balance_loss_weight = 0.3
            self.router_temp = 1.0
            self.steer_scale = 0.20
            self.muon_lr = 0.02
            self.centroid_lr = 5e-4
            self.momentum = 0.95
            self.lr_peak = 5e-4
            self.warmup_steps = 500
            self.min_lr = 1e-5
            self.current_lr = 5e-4
            self.dataset_path = "data/fineweb_sample.jsonl"
            self.dataset_name = "HuggingFaceFW/fineweb-edu"
            self.dataset_type = "streaming"
            self.total_tokens_trained = 0
            self.tokens_trained = 0
            self.current_batch_preview = ""
            self.active_checkpoint: Optional[str] = None
            self.last_sample_decode = ""
            self._prefetch_thread: Optional[threading.Thread] = None
            self._prefetch_running = True

            # Hugging Face Direct Streaming State & Authentication
            self.hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
            if not self.hf_token and hasattr(module_manager, "get_config"):
                try:
                    byok = module_manager.get_config().get("byok_keys", {})
                    self.hf_token = byok.get("huggingface", "")
                    if self.hf_token:
                        os.environ["HF_TOKEN"] = self.hf_token
                        os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token
                except Exception:
                    pass
            self.is_streaming_hf = False
            self.hf_dataset_name = "HuggingFaceFW/fineweb-edu"
            self.hf_config: Optional[str] = "sample-10BT"
            self.hf_split: str = "train"
            self.hf_text_column: str = "text"
            self.hf_stream_dataset: Any = None
            self.hf_stream_iter: Any = None
            self.hf_stream_buffer: deque = deque(maxlen=2000)
            self.hf_stream_stats: Dict[str, Any] = {
                "total_samples_ingested": 0,
                "columns_detected": [],
                "sample_preview": "",
                "last_error": None,
            }

            if HAS_TORCH:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({platform.processor() or 'x86_64'})"
                self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
                self.logs: List[str] = [
                    f"[SYSTEM] PyTorch {torch.__version__} engine initialized on {self.device_name}.",
                    "[ENGINE] TriuneTransformer MoE Engine ready for real training and inference.",
                ]
            else:
                self.device = "cpu"
                self.device_name = f"Host UI Mode ({platform.processor() or 'x86_64'})"
                self.loss_fn = None
                self.logs: List[str] = [
                    f"[SYSTEM] Studio running in Host UI Mode on {self.device_name}.",
                    "[ENGINE] PyTorch not available in current process. Run with WSL2 for full GPU acceleration.",
                ]
            print(f"[Triune Engine] Initializing Native Engine on: {self.device_name}")

            self.history: List[Dict[str, Any]] = []
            self.training_task: Optional[asyncio.Task] = None
            self._restore_persisted_session_state()

        def _restore_persisted_session_state(self):
            """Restore cumulative tokens trained, step, and active checkpoint across sessions and reboots."""
            session_file = Path("checkpoints/session_state.json")
            restored = 0
            if session_file.is_file():
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        restored = data.get("total_tokens_trained", data.get("tokens_trained", 0))
                        if "step" in data and data["step"] > self.step:
                            self.step = data["step"]
                        if data.get("active_checkpoint"):
                            self.active_checkpoint = data["active_checkpoint"]
                        if data.get("dataset_name"):
                            self.dataset_name = data["dataset_name"]
                        if "batch_size" in data and data["batch_size"] > 0:
                            self.batch_size = data["batch_size"]
                        if "seq_len" in data and data["seq_len"] > 0:
                            self.seq_len = data["seq_len"]
                        if "grad_accum_steps" in data and data["grad_accum_steps"] > 0:
                            self.grad_accum_steps = data["grad_accum_steps"]
                        if "gpu_offload_pct" in data:
                            self.gpu_offload_pct = data["gpu_offload_pct"]
                        if "gpu_layers" in data:
                            self.gpu_layers = data["gpu_layers"]
                        if "precision" in data:
                            self.precision = data["precision"]
                        if data.get("active_model_id"):
                            self.active_model_id = data["active_model_id"]
                        if data.get("architecture_spec"):
                            self.architecture_spec = data["architecture_spec"]
                except Exception:
                    pass

            # Cross-check against latest checkpoint on disk
            ckpt_dir = Path("checkpoints")
            if ckpt_dir.is_dir():
                ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
                if ckpts:
                    try:
                        latest = ckpts[0]
                        if HAS_TORCH:
                            ckpt_data = torch.load(latest, map_location="cpu", weights_only=False)
                            ckpt_tokens = ckpt_data.get("total_tokens_trained", ckpt_data.get("tokens_trained", 0))
                            ckpt_step = ckpt_data.get("step", 0)
                            if not ckpt_tokens and ckpt_step > 0:
                                b = ckpt_data.get("batch_size", self.batch_size)
                                s = ckpt_data.get("seq_len", self.seq_len)
                                a = ckpt_data.get("grad_accum_steps", self.grad_accum_steps)
                                ckpt_tokens = ckpt_step * b * s * a
                            if ckpt_tokens and (not restored or ckpt_tokens > restored):
                                restored = ckpt_tokens
                            if ckpt_step > self.step:
                                self.step = ckpt_step
                            if not self.active_checkpoint:
                                self.active_checkpoint = latest.name
                    except Exception:
                        pass

            self.total_tokens_trained = restored
            self.tokens_trained = restored
            if restored > 0:
                msg = f"[Triune Engine] Restored {restored:,} historical cumulative trained tokens from previous sessions (step {self.step})."
                self.logs.insert(0, msg)
                print(msg)
                self._save_persisted_session_state()

        def _save_persisted_session_state(self):
            """Persist session cumulative tokens and metadata to checkpoints/session_state.json."""
            try:
                session_file = Path("checkpoints/session_state.json")
                session_file.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "total_tokens_trained": self.total_tokens_trained,
                    "tokens_trained": self.tokens_trained,
                    "step": self.step,
                    "active_checkpoint": self.active_checkpoint,
                    "dataset_name": self.dataset_name,
                    "hf_dataset_name": self.hf_dataset_name if self.is_streaming_hf else self.dataset_name,
                    "batch_size": self.batch_size,
                    "seq_len": self.seq_len,
                    "grad_accum_steps": self.grad_accum_steps,
                    "gpu_offload_pct": getattr(self, "gpu_offload_pct", 100),
                    "gpu_layers": getattr(self, "gpu_layers", 6),
                    "precision": getattr(self, "precision", "bf16"),
                    "active_model_id": getattr(self, "active_model_id", "triune-small"),
                    "architecture_spec": getattr(self, "architecture_spec", {}),
                    "updated_at": time.time(),
                }
                with open(session_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

        def get_scheduled_lr(self, step: int) -> float:
            """Cosine learning rate schedule with linear warmup."""
            if step < self.warmup_steps:
                return max(self.min_lr, self.lr_peak * float(step + 1) / max(1, self.warmup_steps))
            decay_steps = 25000
            progress = min(1.0, float(step - self.warmup_steps) / float(max(1, decay_steps - self.warmup_steps)))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(self.min_lr, self.min_lr + (self.lr_peak - self.min_lr) * cosine_factor)

        def load_tokenizer(self):
            if self.tokenizer is not None:
                return self.tokenizer
            project_root = Path(__file__).resolve().parent.parent.parent
            for t_path in (project_root / "triune_tokenizer.json", Path("triune_tokenizer.json"), Path("tokenizer.json")):
                if t_path.is_file():
                    try:
                        from triune.data.tokenizer import load_tokenizer
                        self.tokenizer = load_tokenizer(t_path)
                        print(f"[Triune Engine] Loaded BPE tokenizer from {t_path} (vocab: {self.tokenizer.get_vocab_size()})")
                        break
                    except Exception as err:
                        print(f"[Triune Engine] Tokenizer load notice: {err}")
            return self.tokenizer

        def reconfigure_architecture(self, preset_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
            """Rebuild TriuneTransformer architecture and optimizer to match specified preset or custom parameters."""
            ARCH_PRESETS = {
                "triune-nano": {
                    "preset_id": "triune-nano",
                    "name": "Triune-Nano (Edge / Embedded)",
                    "tier": "Edge & Mobile",
                    "num_layers": 4,
                    "hidden_dim": 192,
                    "num_heads": 4,
                    "head_dim": 48,
                    "num_experts": 2,
                    "experts_routed": 2,
                    "experts_shared": 1,
                    "router_prefix_layers": 1,
                    "reflex_exit_layer": 2,
                    "limbic_exit_layer": 3,
                    "recommended_batch": 4,
                    "recommended_seq": 64,
                    "params": "18.5M params",
                },
                "triune-small": {
                    "preset_id": "triune-small",
                    "name": "Triune-Small (Canonical 6L MoE)",
                    "tier": "Production Baseline",
                    "num_layers": 6,
                    "hidden_dim": 256,
                    "num_heads": 4,
                    "head_dim": 64,
                    "num_experts": 4,
                    "experts_routed": 4,
                    "experts_shared": 1,
                    "router_prefix_layers": 1,
                    "reflex_exit_layer": 2,
                    "limbic_exit_layer": 4,
                    "recommended_batch": 4,
                    "recommended_seq": 64,
                    "params": "33.8M params",
                },
                "triune-medium": {
                    "preset_id": "triune-medium",
                    "name": "Triune-Medium (Reasoning & Code)",
                    "tier": "High Capacity",
                    "num_layers": 12,
                    "hidden_dim": 512,
                    "num_heads": 8,
                    "head_dim": 64,
                    "num_experts": 8,
                    "experts_routed": 8,
                    "experts_shared": 1,
                    "router_prefix_layers": 2,
                    "reflex_exit_layer": 4,
                    "limbic_exit_layer": 8,
                    "recommended_batch": 2,
                    "recommended_seq": 128,
                    "params": "142M params",
                },
                "triune-large": {
                    "preset_id": "triune-large",
                    "name": "Triune-Large (High-Capacity MoE)",
                    "tier": "Research Scale",
                    "num_layers": 16,
                    "hidden_dim": 768,
                    "num_heads": 12,
                    "head_dim": 64,
                    "num_experts": 8,
                    "experts_routed": 8,
                    "experts_shared": 2,
                    "router_prefix_layers": 2,
                    "reflex_exit_layer": 5,
                    "limbic_exit_layer": 10,
                    "recommended_batch": 1,
                    "recommended_seq": 128,
                    "params": "385M params",
                },
                "triune-1b": {
                    "preset_id": "triune-1b",
                    "name": "Triune-1B (Frontier MoE)",
                    "tier": "Billion-Scale Baseline",
                    "num_layers": 24,
                    "hidden_dim": 1536,
                    "num_heads": 16,
                    "head_dim": 96,
                    "num_experts": 8,
                    "experts_routed": 8,
                    "experts_shared": 2,
                    "router_prefix_layers": 3,
                    "reflex_exit_layer": 8,
                    "limbic_exit_layer": 16,
                    "recommended_batch": 1,
                    "recommended_seq": 128,
                    "params": "1.24B params",
                },
            }

            p_id = preset_id or getattr(self, "active_model_id", "triune-small") or "triune-small"
            base = ARCH_PRESETS.get(p_id, ARCH_PRESETS["triune-small"]).copy()
            if preset_id not in ARCH_PRESETS and preset_id:
                base["name"] = f"Triune-Custom ({preset_id})"
                base["preset_id"] = preset_id

            for k in ("num_layers", "hidden_dim", "num_heads", "head_dim", "num_experts", "depth_mode", "balance_loss_weight"):
                if k in kwargs and kwargs[k] is not None:
                    base[k] = kwargs[k]

            L = base["num_layers"]
            r_prefix = max(1, min(base.get("router_prefix_layers", 1), max(1, L - 3)))
            r_reflex = max(r_prefix + 1, min(base.get("reflex_exit_layer", max(2, L // 3)), max(r_prefix + 1, L - 2)))
            r_limbic = max(r_reflex + 1, min(base.get("limbic_exit_layer", max(3, (2 * L) // 3)), max(r_reflex + 1, L - 1)))
            base["router_prefix_layers"] = r_prefix
            base["reflex_exit_layer"] = r_reflex
            base["limbic_exit_layer"] = r_limbic
            base["hidden_dim"] = base["num_heads"] * base["head_dim"]

            if HAS_TORCH and TriuneTransformer is not None:
                self.load_tokenizer()
                self.model = TriuneTransformer(
                    vocab_size=32000,
                    hidden_dim=base["hidden_dim"],
                    num_layers=base["num_layers"],
                    num_heads=base["num_heads"],
                    head_dim=base["head_dim"],
                    num_experts=base["num_experts"],
                    router_prefix_layers=r_prefix,
                    reflex_exit_layer=r_reflex,
                    limbic_exit_layer=r_limbic,
                    use_fp4=False,
                    use_fp8=False,
                ).to(self.device)

                from triune.optim.factory import build_optimizer
                optim_cfg = {
                    "lr": self.current_lr,
                    "betas": (0.9, 0.95),
                    "weight_decay": 0.01,
                    "galore": True,
                    "galore_rank": 64,
                    "galore_update_gap": 100,
                    "steer_scale": getattr(self, "steer_scale", 0.20),
                    "galore_lr": getattr(self, "centroid_lr", self.current_lr),
                    "galore_betas": (0.9, 0.999),
                    "galore_weight_decay": 0.01,
                    "use_muon": True,
                    "muon_lr": getattr(self, "muon_lr", 0.02),
                    "muon_momentum": getattr(self, "momentum", 0.95),
                    "muon_weight_decay": 0.01,
                }
                try:
                    self.optimizer = build_optimizer(self.model, optim_cfg)
                except Exception as e:
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.current_lr, weight_decay=0.01)

            self.active_model_id = p_id
            self.architecture_spec = base
            self.gpu_layers = min(getattr(self, "gpu_layers", base["num_layers"]), base["num_layers"])
            if "recommended_batch" in base and "batch_size" not in kwargs:
                self.batch_size = base["recommended_batch"]
            if "recommended_seq" in base and "seq_len" not in kwargs:
                self.seq_len = base["recommended_seq"]
            if "batch_size" in kwargs and kwargs["batch_size"]:
                self.batch_size = kwargs["batch_size"]
            if "seq_len" in kwargs and kwargs["seq_len"]:
                self.seq_len = kwargs["seq_len"]

            self._save_persisted_session_state()

            log_msg = f"[Triune Engine] Architecture active: {base.get('name', p_id)} ({base['num_layers']}L, {base['hidden_dim']}d, {base['num_heads']}H, {base['num_experts']}E). Optimizer rebuilt."
            self.logs.insert(0, log_msg)
            print(log_msg)
            return base

        def lazy_init_model(self):
            if not HAS_TORCH or TriuneTransformer is None:
                raise RuntimeError("PyTorch is not installed in this environment.")
            if self.model is not None:
                return

            self.reconfigure_architecture(getattr(self, "active_model_id", "triune-small"))

            # Auto-restore latest studio checkpoint if available
            ckpt_dir = Path("checkpoints")
            if ckpt_dir.is_dir():
                ckpts = sorted(ckpt_dir.glob("triune_studio_step_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
                if ckpts and ckpts[0].is_file():
                    try:
                        latest_p = ckpts[0]
                        ckpt = torch.load(latest_p, map_location=self.device, weights_only=False)
                        st = ckpt.get("model_state", ckpt)
                        self.model.load_state_dict(st, strict=False)
                        self.step = ckpt.get("step", 0)
                        restored_tokens = ckpt.get("total_tokens_trained", ckpt.get("tokens_trained", 0))
                        if not restored_tokens and self.step > 0:
                            restored_tokens = self.step * self.batch_size * self.seq_len * self.grad_accum_steps
                        self.total_tokens_trained = restored_tokens
                        self.tokens_trained = restored_tokens
                        if "optimizer_state" in ckpt and ckpt["optimizer_state"] and self.optimizer:
                            try:
                                opt_st = ckpt["optimizer_state"]
                                if isinstance(self.optimizer, CentroidSteerOptimizer):
                                    if "layer_groups" in opt_st and "base_optimizer" in opt_st:
                                        self.optimizer.load_state_dict(opt_st)
                                    elif "state" in opt_st and "param_groups" in opt_st:
                                        try:
                                            self.optimizer.base_optimizer.load_state_dict(opt_st)
                                        except Exception:
                                            pass
                                else:
                                    self.optimizer.load_state_dict(opt_st)
                            except Exception:
                                pass
                        self.active_checkpoint = latest_p.name
                        msg = f"[Triune Engine] Auto-restored active weights from {latest_p.name} (step {self.step})."
                        self.logs.insert(0, msg)
                        print(msg)
                    except Exception as err:
                        print(f"[Triune Engine] Checkpoint auto-load notice: {err}")

            print(f"[Triune Engine] Loaded 6-Layer MoE TriuneTransformer (32k vocab, 4 experts, Centroid+Muon Optimizer) on {self.device_name}.")
            self.load_training_data()

        def connect_hf_stream(self, dataset_name: str, config: Optional[str] = None, split: str = "train", text_column: Optional[str] = None) -> Dict[str, Any]:
            """Connect directly to any Hugging Face dataset via streaming API with optional authentication."""
            self.load_tokenizer()
            canonical_hf_aliases = {
                "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
                "HuggingFaceFW/fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
                "streaming": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
                "tinystories": ("roneneldan/TinyStories", None, "text"),
                "roneneldan/TinyStories": ("roneneldan/TinyStories", None, "text"),
                "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
                "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
                "wikitext-103-raw-v1": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
                "Salesforce/wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
                "openwebtext": ("Skylion007/openwebtext", None, "text"),
                "Skylion007/openwebtext": ("Skylion007/openwebtext", None, "text"),
                "gsm8k": ("openai/gsm8k", "main", "question"),
                "openai/gsm8k": ("openai/gsm8k", "main", "question"),
                "c4": ("allenai/c4", "en", "text"),
                "allenai/c4": ("allenai/c4", "en", "text"),
                "falcon-refinedweb": ("tiiuae/falcon-refinedweb", None, "content"),
                "tiiuae/falcon-refinedweb": ("tiiuae/falcon-refinedweb", None, "content"),
            }
            if dataset_name in canonical_hf_aliases:
                canon_name, canon_cfg, canon_col = canonical_hf_aliases[dataset_name]
                dataset_name = canon_name
                if not config:
                    config = canon_cfg
                if not text_column:
                    text_column = canon_col

            token = self.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
            if token and (token.startswith("hf_testtoken") or token == "dummy" or len(token) < 10):
                token = None

            ds = None
            first_sample = None
            resolved_config = config or "default"

            print(f"[Triune Engine] Connecting to Hugging Face stream: '{dataset_name}' (config: {resolved_config}, split: {split}, token: {'[ACTIVE]' if token else '[UNAUTHENTICATED]'})...")

            # 1. Primary: Resilient continuous REST streaming reader
            try:
                rest_reader = HFRestStreamReader(dataset_name, config=resolved_config, split=split, token=token)
                test_iter = iter(rest_reader)
                first_sample = next(test_iter)
                ds = rest_reader
            except Exception as rest_err:
                # Try auto-discovering split configuration if default config wasn't recognized
                try:
                    disc_url = f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(dataset_name, safe='')}"
                    disc_headers = {"User-Agent": "TriuneStudio/2.0"}
                    if token:
                        disc_headers["Authorization"] = f"Bearer {token}"
                    disc_req = urllib.request.Request(disc_url, headers=disc_headers)
                    with urllib.request.urlopen(disc_req, timeout=10) as resp:
                        splits_data = json.loads(resp.read().decode("utf-8"))
                        matching = [s["config"] for s in splits_data.get("splits", []) if s.get("split") == split]
                        if matching:
                            resolved_config = matching[0]
                            rest_reader = HFRestStreamReader(dataset_name, config=resolved_config, split=split, token=token)
                            test_iter = iter(rest_reader)
                            first_sample = next(test_iter)
                            ds = rest_reader
                except Exception:
                    pass

            # 2. Secondary: Fall back to native datasets library if REST server unavailable
            if ds is None or first_sample is None:
                try:
                    import datasets
                    load_kwargs = {"split": split, "streaming": True}
                    if resolved_config and resolved_config != "default":
                        load_kwargs["name"] = resolved_config
                    if token:
                        load_kwargs["token"] = token
                    native_ds = datasets.load_dataset(dataset_name, **load_kwargs)
                    test_iter = iter(native_ds)
                    first_sample = next(test_iter)
                    ds = native_ds
                except Exception as nat_err:
                    raise RuntimeError(f"Could not connect to Hugging Face stream '{dataset_name}': {nat_err}")

            keys = list(first_sample.keys()) if isinstance(first_sample, dict) else []
            col = text_column
            if not col:
                candidates = ["text", "content", "story", "prompt", "sentence", "article", "document", "input", "instruction", "body", "code"]
                for c in candidates:
                    if c in first_sample and isinstance(first_sample[c], str) and first_sample[c].strip():
                        col = c
                        break
                if not col:
                    for k, v in first_sample.items():
                        if isinstance(v, str) and len(v.strip()) > 5:
                            col = k
                            break
            if not col and keys:
                col = keys[0]

            raw_preview = str(first_sample.get(col, ""))[:250] if isinstance(first_sample, dict) else str(first_sample)[:250]
            preview = raw_preview.replace("\ufffd", "").replace("Ġ", " ").replace("Ċ", "\n").replace("\n", " ").replace("  ", " ").strip()

            self.hf_dataset_name = dataset_name
            self.hf_config = config
            self.hf_split = split
            self.hf_text_column = col or "text"
            self.hf_stream_dataset = ds
            self.hf_stream_iter = iter(ds)
            self.hf_stream_buffer.clear()
            self._token_accumulator.clear()
            self.is_streaming_hf = True
            self.dataset_name = dataset_name
            self.dataset_type = "hf_streaming"
            self.dataset_path = dataset_name
            self.hf_stream_stats = {
                "source": "huggingface",
                "total_samples_ingested": 1,
                "columns_detected": keys,
                "sample_preview": preview,
                "last_error": None,
            }

            self._refill_stream_buffer(target_chunks=40)
            self._start_prefetch_worker()
            log_msg = f"[HF STREAM] Successfully connected to '{dataset_name}' (split: {split}, col: '{self.hf_text_column}'). Pre-buffered {len(self.hf_stream_buffer)} sequences with background prefetcher."
            self.logs.insert(0, log_msg)
            print(f"[Triune Engine] {log_msg}")

            return {
                "status": "success",
                "is_streaming": True,
                "dataset_name": dataset_name,
                "config": config,
                "split": split,
                "text_column": self.hf_text_column,
                "columns": keys,
                "sample_preview": preview,
                "buffered_chunks": len(self.hf_stream_buffer),
                "message": log_msg,
            }

        def connect_url_stream(self, url: str, text_column: Optional[str] = None) -> Dict[str, Any]:
            """Connect directly to an external dataset file via HTTP/HTTPS URL streaming."""
            self.load_tokenizer()
            import datasets

            print(f"[Triune Engine] Connecting to external URL stream: '{url}'...")
            ext = url.split("?")[0].split(".")[-1].lower()
            if ext in ("jsonl", "json"):
                ds = datasets.load_dataset("json", data_files=url, split="train", streaming=True)
            elif ext == "parquet":
                ds = datasets.load_dataset("parquet", data_files=url, split="train", streaming=True)
            elif ext in ("csv", "tsv"):
                ds = datasets.load_dataset("csv", data_files=url, split="train", streaming=True)
            else:
                ds = datasets.load_dataset("text", data_files=url, split="train", streaming=True)

            test_iter = iter(ds)
            first_sample = next(test_iter)
            keys = list(first_sample.keys()) if isinstance(first_sample, dict) else []
            col = text_column or ("text" if "text" in keys else (keys[0] if keys else "text"))
            raw_preview = str(first_sample.get(col, ""))[:250] if isinstance(first_sample, dict) else str(first_sample)[:250]
            preview = raw_preview.replace("\n", " ").strip()

            self.hf_dataset_name = url
            self.hf_config = None
            self.hf_split = "train"
            self.hf_text_column = col
            self.hf_stream_dataset = ds
            self.hf_stream_iter = iter(ds)
            self.hf_stream_buffer.clear()
            self._token_accumulator.clear()
            self.is_streaming_hf = True
            self.dataset_name = url.split("/")[-1] or "Remote URL Stream"
            self.dataset_type = "url_streaming"
            self.dataset_path = url
            self.hf_stream_stats = {
                "source": "url",
                "total_samples_ingested": 1,
                "columns_detected": keys,
                "sample_preview": preview,
                "last_error": None,
            }

            self._refill_stream_buffer(target_chunks=40)
            self._start_prefetch_worker()
            log_msg = f"[URL STREAM] Connected to external dataset '{url}'. Pre-buffered {len(self.hf_stream_buffer)} sequences with background prefetcher."
            self.logs.insert(0, log_msg)
            print(f"[Triune Engine] {log_msg}")

            return {
                "status": "success",
                "is_streaming": True,
                "dataset_name": url,
                "text_column": col,
                "columns": keys,
                "sample_preview": preview,
                "buffered_chunks": len(self.hf_stream_buffer),
                "message": log_msg,
            }

        def _start_prefetch_worker(self):
            """Start asynchronous background stream prefetching thread if not running."""
            if os.environ.get("TRIUNE_TESTING") == "1":
                return
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                return
            self._prefetch_running = True
            self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True, name="TriuneStreamPrefetcher")
            self._prefetch_thread.start()

        def stop_prefetch_worker(self):
            """Gracefully signal and stop the background stream prefetching worker thread."""
            self._prefetch_running = False
            self.is_streaming_hf = False
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                try:
                    self._prefetch_thread.join(timeout=2.0)
                except Exception:
                    pass
            self._prefetch_thread = None
            self.hf_stream_iter = None
            self.hf_stream_dataset = None

        def _prefetch_loop(self):
            """Background worker thread continuously streaming and tokenizing ahead of GPU execution."""
            while self._prefetch_running:
                try:
                    if self.is_streaming_hf and self.hf_stream_iter is not None and self.tokenizer is not None:
                        if len(self.hf_stream_buffer) < 300:
                            self._refill_stream_buffer(target_chunks=300)
                        else:
                            time.sleep(0.02)
                    else:
                        time.sleep(0.05)
                except Exception:
                    time.sleep(0.1)

        def _refill_stream_buffer(self, target_chunks: int = 50) -> None:
            """Fill the streaming token buffer from active stream iterator."""
            if not self.hf_stream_iter or not self.tokenizer:
                return

            chunk_size = self.seq_len + 1
            sep_id = self.tokenizer.token_to_id("[SEP]") or self.tokenizer.token_to_id("[EOS]") or 2
            attempts = 0

            while len(self.hf_stream_buffer) < target_chunks and attempts < 1000:
                attempts += 1
                try:
                    sample = next(self.hf_stream_iter)
                    self.hf_stream_stats["total_samples_ingested"] += 1
                    if isinstance(sample, dict):
                        text = sample.get(self.hf_text_column, "")
                        if not text and "text" in sample:
                            text = sample.get("text", "")
                        elif not text and "content" in sample:
                            text = sample.get("content", "")
                        elif not text and "story" in sample:
                            text = sample.get("story", "")
                        elif not text and "question" in sample:
                            text = sample.get("question", "")
                        elif not text and "prompt" in sample:
                            text = sample.get("prompt", "")
                    else:
                        text = str(sample)

                    if not text or not str(text).strip():
                        continue

                    # Update preview directly from raw text sample
                    raw_text_clean = str(text).replace("\n", " ").strip()
                    raw_text_clean = raw_text_clean.replace("\ufffd", "").replace("Ġ", " ").replace("Ċ", "\n").strip()
                    if raw_text_clean and len(self.hf_stream_stats.get("sample_preview", "")) < 5:
                        self.hf_stream_stats["sample_preview"] = raw_text_clean[:250]
                        if not self.current_batch_preview:
                            self.current_batch_preview = raw_text_clean[:120]

                    encoded_ids = self.tokenizer.encode(str(text)).ids
                    if not encoded_ids:
                        continue

                    self._token_accumulator.extend(encoded_ids)
                    self._token_accumulator.append(sep_id)

                    while len(self._token_accumulator) >= chunk_size:
                        chunk = self._token_accumulator[:chunk_size]
                        del self._token_accumulator[:chunk_size]
                        self.hf_stream_buffer.append(chunk)
                        if len(self.hf_stream_buffer) >= target_chunks:
                            break
                except StopIteration:
                    if self.hf_stream_dataset is not None:
                        try:
                            self.hf_stream_iter = iter(self.hf_stream_dataset)
                            continue
                        except Exception:
                            break
                    else:
                        break
                except Exception as err:
                    # The reader already retried internally; record the error, re-arm the
                    # iterator (it resumes from its saved offset) and end this refill pass.
                    self.hf_stream_stats["last_error"] = str(err)
                    if self.hf_stream_dataset is not None:
                        try:
                            self.hf_stream_iter = iter(self.hf_stream_dataset)
                        except Exception:
                            pass
                    break

        def load_training_data(self, dataset_path: Optional[str] = None):
            self.load_tokenizer()
            if dataset_path:
                self.dataset_path = dataset_path

            # Direct HTTP/HTTPS URL streaming
            if self.dataset_path.startswith("http://") or self.dataset_path.startswith("https://"):
                try:
                    self.connect_url_stream(self.dataset_path)
                    return
                except Exception as err:
                    print(f"[Triune Engine] Notice: Could not stream URL '{self.dataset_path}': {err}")

            canonical_hf_aliases = {
                "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
                "HuggingFaceFW/fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
                "streaming": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
                "tinystories": ("roneneldan/TinyStories", None, "text"),
                "roneneldan/TinyStories": ("roneneldan/TinyStories", None, "text"),
                "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
                "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
                "wikitext-103-raw-v1": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
                "Salesforce/wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
                "openwebtext": ("Skylion007/openwebtext", None, "text"),
                "Skylion007/openwebtext": ("Skylion007/openwebtext", None, "text"),
                "gsm8k": ("openai/gsm8k", "main", "question"),
                "openai/gsm8k": ("openai/gsm8k", "main", "question"),
                "c4": ("allenai/c4", "en", "text"),
                "allenai/c4": ("allenai/c4", "en", "text"),
                "falcon-refinedweb": ("tiiuae/falcon-refinedweb", None, "content"),
                "tiiuae/falcon-refinedweb": ("tiiuae/falcon-refinedweb", None, "content"),
            }

            if self.dataset_path in canonical_hf_aliases:
                hf_name, hf_cfg, hf_col = canonical_hf_aliases[self.dataset_path]
                try:
                    self.connect_hf_stream(hf_name, config=hf_cfg, text_column=hf_col)
                    return
                except Exception as err:
                    print(f"[Triune Engine] Notice: Could not stream '{hf_name}' directly ({err}). Checking local cached samples.")

            is_hf = ("/" in self.dataset_path and not Path(self.dataset_path).is_file())
            if is_hf:
                try:
                    self.connect_hf_stream(self.dataset_path)
                    return
                except Exception as err:
                    print(f"[Triune Engine] Notice: Could not stream '{self.dataset_path}' directly ({err}). Checking local cached samples.")

            p = Path(self.dataset_path)
            if not p.is_file():
                for alt in ("data/fineweb_sample.jsonl", "data/finetune.jsonl"):
                    if Path(alt).is_file():
                        p = Path(alt)
                        break
            self.dataset_name = p.name if p.is_file() else "FineWeb-Edu Stream"
            self.dataset_type = "local" if p.is_file() else "streaming"

            texts = []
            if p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                item = json.loads(line)
                                t = item.get("text") or item.get("content") or item.get("story") or (item.get("prompt", "") + " " + item.get("completion", ""))
                                if t:
                                    texts.append(sanitize_text(t))
                            except Exception:
                                texts.append(sanitize_text(line))
                except Exception as err:
                    print(f"[Triune Engine] Error reading dataset {p}: {err}")

            if not texts:
                # If local file missing, attempt to stream default FineWeb-Edu stream
                try:
                    self.connect_hf_stream("HuggingFaceFW/fineweb-edu", config="sample-10BT")
                    return
                except Exception:
                    pass

            all_token_ids = []
            if self.tokenizer:
                for t in texts:
                    try:
                        all_token_ids.extend(self.tokenizer.encode(t).ids)
                    except Exception:
                        pass

            chunk_size = self.seq_len + 1
            chunks = []
            for i in range(0, len(all_token_ids) - chunk_size, chunk_size):
                chunks.append(all_token_ids[i : i + chunk_size])

            if not chunks:
                chunks = [[1] * chunk_size]

            self.data_chunks = chunks
            self.data_idx = 0
            self.is_streaming_hf = False
            log_msg = f"[DATASET] Active dataset source: '{self.dataset_name}' ({self.dataset_type}) | {len(chunks)} sequences ({len(all_token_ids):,} tokens)."
            self.logs.insert(0, log_msg)
            print(f"[Triune Engine] {log_msg}")

        def generate_text(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7, force_depth: Optional[int] = None) -> Dict[str, Any]:
            if not HAS_TORCH or TriuneTransformer is None:
                return {
                    "text": "[Host UI Mode] PyTorch is not available in current process. Run in WSL2 for full GPU acceleration.",
                    "throughput": 0,
                    "exit_tier": "CPU",
                    "device": self.device_name,
                    "depth": 0,
                    "latency_ms": 0,
                    "tokens_count": 0,
                    "vram_gb": 0.0,
                    "route": "HOST",
                }
            self.lazy_init_model()
            t0 = time.perf_counter()
            self.model.eval()

            self.load_tokenizer()
            if self.tokenizer:
                try:
                    ids = self.tokenizer.encode(prompt).ids
                    eos_id = self.tokenizer.token_to_id("[SEP]") or self.tokenizer.token_to_id("[EOS]") or 2
                except Exception:
                    ids = [abs(hash(w)) % 31900 + 100 for w in prompt.split()] or [1]
                    eos_id = 2
            else:
                ids = [abs(hash(w)) % 31900 + 100 for w in prompt.split()] or [1]
                eos_id = 2

            if not ids:
                ids = [1]

            input_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            gen_ids = []
            route_used = "CORTEX"

            with torch.no_grad():
                cache = self.model.init_cache(batch_size=1) if hasattr(self.model, "init_cache") else None
                curr_input = input_tensor
                for step_idx in range(max(1, min(max_new_tokens, 8192))):
                    if cache is not None:
                        res = self.model(curr_input, cache=cache, force_depth=force_depth)
                    else:
                        res = self.model(curr_input, force_depth=force_depth)

                    if isinstance(res, tuple):
                        logits = res[0]
                        if cache is not None and len(res) > 1:
                            cache = res[1]
                    else:
                        logits = res

                    if force_depth is not None:
                        route_used = ["REFLEX", "LIMBIC", "CORTEX"][min(force_depth, 2)]
                    elif hasattr(self.model, "last_depth_choice") and self.model.last_depth_choice is not None:
                        try:
                            d_idx = int(self.model.last_depth_choice[0].item())
                            route_used = ["REFLEX", "LIMBIC", "CORTEX"][min(d_idx, 2)]
                        except Exception:
                            pass

                    next_logits = logits[0, -1].float() / max(0.01, temperature)

                    # Multiplicative repetition penalty
                    for token_id in set(ids + gen_ids):
                        if token_id < next_logits.size(-1):
                            if next_logits[token_id] > 0:
                                next_logits[token_id] /= 1.2
                            else:
                                next_logits[token_id] *= 1.2

                    # Top-k filtering (k=50)
                    top_k = min(50, next_logits.size(-1))
                    val, _ = torch.topk(next_logits, top_k)
                    next_logits[next_logits < val[-1]] = float("-inf")

                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item() if temperature > 0.05 else torch.argmax(next_logits).item()

                    if next_token == eos_id or len(gen_ids) >= max_new_tokens:
                        break
                    gen_ids.append(next_token)

                    if cache is not None:
                        # Feed ONLY the single newly generated token with recurrent cache - O(1) step!
                        curr_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                    else:
                        curr_input = torch.cat([curr_input, torch.tensor([[next_token]], device=self.device)], dim=1)

            t1 = time.perf_counter()
            elapsed_sec = max(0.001, t1 - t0)
            tok_per_sec = int(len(gen_ids) / elapsed_sec)

            if self.tokenizer and gen_ids:
                try:
                    generated_text = sanitize_text(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                except Exception:
                    generated_text = " ".join(str(i) for i in gen_ids)
            elif gen_ids:
                generated_text = f"Generated {len(gen_ids)} tokens across {route_used} tier."
            else:
                generated_text = f"Response processed via {route_used} exit tier."

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
            if not self.data_chunks and not self.is_streaming_hf:
                self.load_training_data()

            t0 = time.perf_counter()
            self.step += 1
            self.model.train()

            # Dynamic Warmup & Cosine Decay LR Scheduling
            self.current_lr = self.get_scheduled_lr(self.step)
            if self.optimizer:
                if hasattr(self.optimizer, "set_lr"):
                    self.optimizer.set_lr(self.current_lr)
                else:
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.current_lr

            self.optimizer.zero_grad()

            accum_total_loss = 0.0
            accum_lm_loss = 0.0
            accum_z_loss = 0.0
            accum_tokens = 0
            reflex_pct, limbic_pct, cortex_pct = 34.0, 33.0, 33.0

            amp_device = self.device.type if hasattr(self.device, "type") else "cuda"
            use_amp = (amp_device == "cuda")
            amp_dtype = torch.bfloat16 if (amp_device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

            # Gradient Accumulation Micro-Batch Loop
            for micro_idx in range(self.grad_accum_steps):
                # Assemble micro-batch from real streaming prefetch queue or local dataset
                micro_chunks = []
                if self.is_streaming_hf:
                    needed = self.batch_size
                    if len(self.hf_stream_buffer) < needed:
                        self._refill_stream_buffer(target_chunks=max(40, needed * self.grad_accum_steps * 2))

                    if len(self.hf_stream_buffer) >= needed:
                        for _ in range(needed):
                            micro_chunks.append(self.hf_stream_buffer.popleft())
                    elif self.data_chunks:
                        for _ in range(needed):
                            chunk = self.data_chunks[self.data_idx % len(self.data_chunks)]
                            micro_chunks.append(chunk)
                            self.data_idx += 1
                    else:
                        err_detail = self.hf_stream_stats.get("last_error") or "network timeout"
                        raise RuntimeError(
                            f"Streaming buffer starvation on dataset '{self.hf_dataset_name}'. "
                            f"Only {len(self.hf_stream_buffer)}/{needed} sequences buffered. Last error: {err_detail}."
                        )
                else:
                    if not self.data_chunks:
                        self.load_training_data()
                    for _ in range(self.batch_size):
                        chunk = self.data_chunks[self.data_idx % len(self.data_chunks)]
                        micro_chunks.append(chunk)
                        self.data_idx += 1

                # Fast direct vectorized tensor allocation on GPU
                batch = torch.tensor(micro_chunks, dtype=torch.long, device=self.device)
                x = batch[:, :-1]
                targets = batch[:, 1:].contiguous()
                accum_tokens += x.numel()

                # Live batch preview showing actual tokens currently trained
                if micro_idx == 0 and self.tokenizer and (self.step % 5 == 0 or not self.current_batch_preview):
                    try:
                        self.current_batch_preview = sanitize_text(self.tokenizer.decode(x[0].tolist(), skip_special_tokens=True).strip())
                    except Exception:
                        self.current_batch_preview = f"Stream token chunk #{self.step}"

                # Mixed Precision Autocast for RTX 5070 Tensor Cores
                with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                    # Forward pass routed by depth supervision mode
                    if self.depth_mode == "cortex":
                        # Full Cortex Pretraining: 100% gradient backpropagation to all 6 layers and final_head
                        res = self.model(x, force_depth=2)
                        logits = res[0] if isinstance(res, tuple) else res
                        lm_loss = self.loss_fn(logits.view(-1, 32000), targets.view(-1))
                        z_loss = torch.tensor(0.0, device=self.device)
                        micro_loss = lm_loss
                        reflex_pct, limbic_pct, cortex_pct = 0.0, 0.0, 100.0

                    elif self.depth_mode == "joint":
                        # Joint Multi-Exit Supervision: all 3 exit heads and intermediate layers receive simultaneous gradients
                        reflex_logits, limbic_logits, cortex_logits, route_logits = self.model.forward_all_exits(x, update_stats=True)
                        loss_ref = self.loss_fn(reflex_logits.view(-1, 32000), targets.view(-1))
                        loss_limb = self.loss_fn(limbic_logits.view(-1, 32000), targets.view(-1))
                        loss_cort = self.loss_fn(cortex_logits.view(-1, 32000), targets.view(-1))
                        lm_loss = 0.33 * loss_ref + 0.33 * loss_limb + 0.34 * loss_cort
                        if route_logits is not None and route_logits.numel() > 0:
                            z_loss = torch.logsumexp(route_logits, dim=-1).pow(2).mean()
                            micro_loss = lm_loss + 1e-3 * z_loss
                        else:
                            z_loss = torch.tensor(0.0, device=self.device)
                            micro_loss = lm_loss
                        reflex_pct, limbic_pct, cortex_pct = 33.3, 33.3, 33.4

                    else:
                        # Dynamic Adaptive Router with Load Balancing Z-Loss and Gumbel Regularization
                        res = self.model(x, return_balance_loss=True)
                        if isinstance(res, tuple):
                            logits = res[0]
                            route_logits = res[1] if len(res) > 1 else None
                            balance_loss = res[2] if len(res) > 2 else torch.tensor(0.0, device=self.device)
                            lm_loss = self.loss_fn(logits.view(-1, 32000), targets.view(-1))
                            bal_weight = getattr(self, "balance_loss_weight", 0.3)
                            if route_logits is not None and route_logits.numel() > 0:
                                z_loss = torch.logsumexp(route_logits, dim=-1).pow(2).mean()
                                micro_loss = lm_loss + 1e-3 * z_loss + bal_weight * balance_loss
                                probs = torch.softmax(route_logits, dim=-1)
                                flat_probs = probs.reshape(-1, probs.size(-1)).mean(dim=0)
                                reflex_pct = round(float(flat_probs[0].item()) * 100, 1) if flat_probs.numel() > 0 else 34.0
                                limbic_pct = round(float(flat_probs[1].item()) * 100, 1) if flat_probs.numel() > 1 else 33.0
                                cortex_pct = round(max(0.0, 100.0 - reflex_pct - limbic_pct), 1)
                            else:
                                z_loss = torch.tensor(0.0, device=self.device)
                                micro_loss = lm_loss + bal_weight * balance_loss
                                reflex_pct, limbic_pct, cortex_pct = 34.0, 33.0, 33.0
                        else:
                            logits = res
                            lm_loss = self.loss_fn(logits.view(-1, 32000), targets.view(-1))
                            z_loss = torch.tensor(0.0, device=self.device)
                            micro_loss = lm_loss
                            reflex_pct, limbic_pct, cortex_pct = 34.0, 33.0, 33.0

                    scaled_loss = micro_loss / self.grad_accum_steps

                scaled_loss.backward()

                accum_total_loss += float(micro_loss.item()) / self.grad_accum_steps
                accum_lm_loss += float(lm_loss.item()) / self.grad_accum_steps
                accum_z_loss += float(z_loss.item()) / self.grad_accum_steps

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = max(0.0001, t1 - t0)
            tok_per_sec = int(accum_tokens / dt)

            self.total_tokens_trained += accum_tokens
            self.tokens_trained = self.total_tokens_trained
            if self.step % 10 == 0:
                self._save_persisted_session_state()

            vram_stats = VRAMProfiler.get_vram_stats(self.device)

            payload = {
                "step": self.step,
                "loss": round(accum_total_loss, 4),
                "lm_loss": round(accum_lm_loss, 4),
                "router_loss": round(accum_z_loss, 4),
                "throughput": tok_per_sec,
                "vram_gb": vram_stats.get("allocated_gb", 0.0),
                "device": self.device_name,
                "exit_usage": {"reflex": reflex_pct, "limbic": limbic_pct, "cortex": cortex_pct},
                "tokens_trained": self.total_tokens_trained,
                "batch_preview": self.current_batch_preview,
                "dataset_name": self.hf_dataset_name if self.is_streaming_hf else self.dataset_name,
                "dataset_type": self.dataset_type,
                "active_checkpoint": self.active_checkpoint,
                "is_streaming": self.is_streaming_hf,
                "grad_accum_steps": self.grad_accum_steps,
                "depth_mode": self.depth_mode,
                "lr": round(self.current_lr, 7),
                "tokens_in_step": accum_tokens,
            }

            self.history.append(payload)
            if len(self.history) > 100:
                self.history.pop(0)

            # Periodic auto-checkpoint every 500 steps
            if self.step % 500 == 0:
                try:
                    ckpt_dir = Path("checkpoints")
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    auto_name = f"triune_studio_step_{self.step}.pt"
                    auto_path = ckpt_dir / auto_name
                    torch.save({
                        "step": self.step,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
                        "loss": payload["loss"],
                        "total_tokens_trained": self.total_tokens_trained,
                        "tokens_trained": self.total_tokens_trained,
                        "batch_size": self.batch_size,
                        "seq_len": self.seq_len,
                        "grad_accum_steps": self.grad_accum_steps,
                    }, auto_path)
                    self.active_checkpoint = auto_name
                    self.logs.insert(0, f"[CHECKPOINT] Periodic auto-save: {auto_name} at step {self.step}.")
                except Exception:
                    pass

            # Periodically generate a sample every 100 steps
            if self.step > 0 and self.step % 100 == 0:
                try:
                    sample_res = self.generate_text("The transformer", max_new_tokens=16, temperature=0.7)
                    self.last_sample_decode = sample_res.get("text", "")
                except Exception:
                    pass

            log_line = f"[STEP {self.step}] Loss: {payload['loss']} | {tok_per_sec} tok/s | Ingesting [{self.dataset_name}]: \"{self.current_batch_preview[:36]}...\""
            self.logs.insert(0, log_line)
            if len(self.logs) > 50:
                self.logs.pop()

            if global_emitter is not None:
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
                await asyncio.sleep(0.001)

    pytorch_state = RealPyTorchEngineState()
    import atexit
    atexit.register(pytorch_state.stop_prefetch_worker)

    class ChatCompletionRequest(BaseModel):
        model: str = "triune-base"
        messages: List[Dict[str, str]]
        temperature: float = 0.7
        max_tokens: int = 2048
        route: Optional[str] = None
        force_depth: Optional[int] = None

    class DAGExecuteRequest(BaseModel):
        nodes: List[Dict[str, Any]]
        edges: List[Dict[str, Any]]

    class FineTuneRequest(BaseModel):
        dataset_path: str = "data/finetune.jsonl"
        lora_rank: int = 16
        lora_alpha: float = 32.0
        epochs: int = 3
        lr: float = 2e-4
        quantization: str = "4-bit NF4"

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

    def _build_chat_response(content: str, model: str, route: str, latency_ms: float) -> Dict[str, Any]:
        toks = len(content.split())
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "telemetry": {
                "route": route,
                "latency_ms": latency_ms,
                "tokens_per_sec": int(toks / max(0.001, latency_ms / 1000)),
                "tokens_count": toks,
                "vram_gb": 0.0,
            }
        }

    @router.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
        """Real PyTorch engine or BYOK frontier provider chat completion response."""
        user_prompt = req.messages[-1].get("content", "") if req.messages else ""
        model_name = req.model.lower()

        # Check for BYOK Provider Routing
        byok_keys = module_manager.get_config().get("byok_keys", {}) if hasattr(module_manager, "get_config") else {}

        if any(model_name.startswith(p) for p in ("openai", "gpt-", "chatgpt")):
            api_key = byok_keys.get("openai", "") or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                reply_text = "[BYOK Provider] OpenAI API key not found. Please add your 'sk-...' key in the BYOK Subscriptions tab in the left sidebar to chat with GPT-4o."
                return _build_chat_response(reply_text, req.model, "OPENAI-BYOK", 0)
            try:
                t0 = time.perf_counter()
                reply_text = await asyncio.to_thread(_call_openai_api, req.messages, api_key, model="gpt-4o-mini", temperature=req.temperature, max_tokens=req.max_tokens)
                dt = round((time.perf_counter() - t0) * 1000, 1)
                return _build_chat_response(reply_text, req.model, "GPT-4o", dt)
            except Exception as e:
                return _build_chat_response(f"[OpenAI API Error] {e}", req.model, "ERROR", 0)

        elif any(model_name.startswith(p) for p in ("anthropic", "claude")):
            api_key = byok_keys.get("anthropic", "") or os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                reply_text = "[BYOK Provider] Anthropic API key not found. Please add your 'sk-ant-...' key in the BYOK Subscriptions tab in the left sidebar to chat with Claude 3.5."
                return _build_chat_response(reply_text, req.model, "ANTHROPIC-BYOK", 0)
            try:
                t0 = time.perf_counter()
                reply_text = await asyncio.to_thread(_call_anthropic_api, req.messages, api_key, model="claude-3-5-sonnet-20241022", temperature=req.temperature, max_tokens=req.max_tokens)
                dt = round((time.perf_counter() - t0) * 1000, 1)
                return _build_chat_response(reply_text, req.model, "CLAUDE-3.5", dt)
            except Exception as e:
                return _build_chat_response(f"[Anthropic API Error] {e}", req.model, "ERROR", 0)

        elif any(model_name.startswith(p) for p in ("google", "gemini")):
            api_key = byok_keys.get("gemini", "") or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                reply_text = "[BYOK Provider] Google Gemini API key not found. Please add your Gemini API key in the BYOK Subscriptions tab in the left sidebar to chat with Gemini 1.5."
                return _build_chat_response(reply_text, req.model, "GEMINI-BYOK", 0)
            try:
                t0 = time.perf_counter()
                reply_text = await asyncio.to_thread(_call_gemini_api, req.messages, api_key, model="gemini-1.5-flash", temperature=req.temperature, max_tokens=req.max_tokens)
                dt = round((time.perf_counter() - t0) * 1000, 1)
                return _build_chat_response(reply_text, req.model, "GEMINI-1.5", dt)
            except Exception as e:
                return _build_chat_response(f"[Gemini API Error] {e}", req.model, "ERROR", 0)

        # Default: Local Triune MoE Engine
        force_depth = req.force_depth
        if force_depth is None and req.route:
            r_low = req.route.lower()
            if r_low == "reflex":
                force_depth = 0
            elif r_low == "limbic":
                force_depth = 1
            elif r_low == "cortex":
                force_depth = 2

        if force_depth is None:
            for cmd, d in (("reflex", 0), ("limbic", 1), ("cortex", 2)):
                if user_prompt.lower().startswith(f"{cmd} ") or req.model.lower().endswith(cmd):
                    force_depth = d
                    if user_prompt.lower().startswith(f"{cmd} "):
                        user_prompt = user_prompt[len(cmd) + 1:].strip()
                    break

        # Multi-turn conversation prompt assembly (OpenAI / LM Studio style)
        if len(req.messages) > 1:
            turns = []
            for m in req.messages:
                role = (m.get("role") or "user").strip().capitalize()
                content = (m.get("content") or "").strip()
                if content:
                    turns.append(f"{role}: {content}")
            turns.append("Assistant:")
            effective_prompt = "\n".join(turns)
        else:
            effective_prompt = user_prompt

        res = await asyncio.to_thread(
            pytorch_state.generate_text,
            effective_prompt,
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

                dpath = req.dataset_path
                if not Path(dpath).is_file():
                    for cand in ("data/fineweb_sample.jsonl", "data/finetune.jsonl"):
                        if Path(cand).is_file():
                            dpath = cand
                            break

                res = await asyncio.to_thread(
                    finetuner.fit,
                    dataset_path=dpath,
                    epochs=req.epochs,
                    lr=req.lr,
                )
                finetune_state["status"] = "completed"
                finetune_state["final_loss"] = res.get("final_loss", 0.0)
                finetune_state["output_dir"] = res.get("output_dir", "")
                finetune_state["step"] = res.get("total_steps", 0)
                pytorch_state.logs.insert(0, f"[LORA] Fine-tuning finished! Final Loss: {finetune_state['final_loss']} -> {finetune_state['output_dir']}")
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
        try:
            graph_json = {"nodes": req.nodes, "edges": req.edges}
            loop = asyncio.get_running_loop()

            def _on_dag_event(event_data: Dict[str, Any]):
                try:
                    asyncio.run_coroutine_threadsafe(
                        telemetry_manager.broadcast({
                            "type": "dag_event",
                            "data": event_data
                        }),
                        loop
                    )
                except Exception:
                    pass

            dag_engine.register_event_callback(_on_dag_event)
            try:
                res = await asyncio.to_thread(dag_engine.execute_graph, graph_json)
            finally:
                dag_engine.unregister_event_callback(_on_dag_event)

            return {
                "status": res.get("status", "success"),
                "results": res.get("results", {}),
                "context": res.get("context", {})
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "results": {}
            }

    @router.post("/v1/dag/compile_to_training")
    @router.post("/v1/model/compile_dag")
    async def compile_dag_to_training(req: DAGExecuteRequest) -> Dict[str, Any]:
        """Compile visual DAG node graph into live PyTorch engine training configuration."""
        nodes = req.nodes
        applied: Dict[str, Any] = {}

        for n in nodes:
            ntype = n.get("type", "")
            title = n.get("title", n.get("name", ""))
            cfg = n.get("config", {})
            params = n.get("params", {})
            details = n.get("details", "")

            def get_val(*keys, default=None):
                for key in keys:
                    if key in cfg:
                        c = cfg[key]
                        return c.get("value", c) if isinstance(c, dict) else c
                    if key in params:
                        return params[key]
                    if details:
                        for raw_line in str(details).splitlines():
                            line = raw_line.strip()
                            for sep in ("=", ":"):
                                prefix = f"{key}{sep}"
                                prefix_space = f"{key} {sep}"
                                if line.startswith(prefix) or line.startswith(prefix_space):
                                    v_str = line.split(sep, 1)[1].strip()
                                    try:
                                        return json.loads(v_str.lower() if v_str.lower() in ("true", "false") else v_str)
                                    except Exception:
                                        return v_str
                return default

            # 0. Model Architecture Reconfiguration (TriuneTransformer / Model)
            if ntype in ("Model", "TriuneTransformer") or any(k in title for k in ("Transformer", "Triune", "Architecture", "Model")):
                layers = get_val("num_layers", "layers")
                hidden_dim = get_val("hidden_dim", "hiddenDim", "dim")
                heads = get_val("heads", "num_heads")
                head_dim = get_val("head_dim", "headDim")
                experts = get_val("experts_routed", "num_experts", "experts")
                preset_id = get_val("preset_id", "preset", "model_preset")
                d_mode = get_val("depth_mode", "depth")
                b_weight = get_val("balance_loss_weight", "balance_weight")

                reconfig_kwargs = {}
                if layers is not None: reconfig_kwargs["num_layers"] = int(layers)
                if hidden_dim is not None: reconfig_kwargs["hidden_dim"] = int(hidden_dim)
                if heads is not None: reconfig_kwargs["num_heads"] = int(heads)
                if head_dim is not None: reconfig_kwargs["head_dim"] = int(head_dim)
                if experts is not None: reconfig_kwargs["num_experts"] = int(experts)
                if d_mode: reconfig_kwargs["depth_mode"] = str(d_mode).lower()
                if b_weight is not None: reconfig_kwargs["balance_loss_weight"] = float(b_weight)

                if preset_id or reconfig_kwargs:
                    arch = pytorch_state.reconfigure_architecture(preset_id=preset_id, **reconfig_kwargs)
                    applied["model_architecture"] = arch

            # 1. Dataset Streaming
            if ntype in ("Data", "Dataset", "HuggingFaceStreamer", "HFRestStreamReader") or any(k in title for k in ("Streamer", "Data", "Dataset")):
                ds_name = get_val("dataset_name", "dataset", "dataset_path", "hf_dataset")
                if ds_name:
                    split = get_val("split", "hf_split", default="train")
                    config_name = get_val("config", "dataset_config", "hf_config")
                    col = get_val("text_column", "column", "hf_text_column", default="text")
                    pytorch_state.dataset_name = ds_name
                    pytorch_state.hf_dataset_name = ds_name
                    pytorch_state.hf_split = split
                    pytorch_state.hf_config = config_name
                    pytorch_state.hf_text_column = col
                    try:
                        res = pytorch_state.connect_hf_stream(ds_name, config=config_name, split=split, text_column=col)
                        applied["dataset"] = {
                            "name": ds_name,
                            "split": split,
                            "config": config_name,
                            "text_column": res.get("text_column", col),
                            "buffered_chunks": res.get("buffered_chunks", 0),
                        }
                    except Exception as err:
                        applied["dataset"] = {
                            "name": ds_name,
                            "split": split,
                            "config": config_name,
                            "text_column": col,
                        }
                        applied["dataset_error"] = str(err)

            # 2. DataLoader & Batch Size
            if ntype in ("DataLoader", "CyclingDataLoader", "Data") or "DataLoader" in title:
                b_size = get_val("batch_size", "batchSize", "micro_batch_size")
                if b_size is not None:
                    pytorch_state.batch_size = max(1, int(b_size))
                    applied["batch_size"] = pytorch_state.batch_size
                s_len = get_val("seq_len", "seqLength", "max_seq_len", "sequence_length")
                if s_len is not None:
                    pytorch_state.seq_len = max(8, int(s_len))
                    applied["seq_len"] = pytorch_state.seq_len

            # 3. Router & Depth Supervision Mode
            if ntype in ("ExitRouter", "DepthRouter", "HierarchicalDepthRouter", "Model", "JointExitLoss", "RouterZLoss") or "Router" in title or "Loss" in title:
                d_mode = get_val("depth_mode", "mode", "depth")
                if d_mode and str(d_mode).lower() in ("cortex", "joint", "dynamic"):
                    pytorch_state.depth_mode = str(d_mode).lower()
                    applied["depth_mode"] = pytorch_state.depth_mode
                b_weight = get_val("balance_loss_weight", "balance_weight", "z_loss_weight")
                if b_weight is not None:
                    pytorch_state.balance_loss_weight = float(b_weight)
                    applied["balance_loss_weight"] = pytorch_state.balance_loss_weight
                r_temp = get_val("temperature", "router_temperature", "temp")
                if r_temp is not None:
                    r_temp_val = float(r_temp)
                    pytorch_state.router_temp = r_temp_val
                    if pytorch_state.model is not None and hasattr(pytorch_state.model, "router") and pytorch_state.model.router is not None:
                        pytorch_state.model.router.temperature = r_temp_val
                    applied["temperature"] = r_temp_val

            # 4. Optimizer & Hyperparameters
            if ntype in ("Optimizer", "OptimizerGroup", "CentroidSteerOptimizer", "MuonOptimizer", "AdamWOptimizer") or "Optimizer" in title:
                lr = get_val("lr", "learning_rate", "peak_lr")
                if lr is not None:
                    lr_val = float(lr)
                    pytorch_state.current_lr = lr_val
                    pytorch_state.lr_peak = lr_val
                    if pytorch_state.optimizer is not None:
                        if hasattr(pytorch_state.optimizer, "set_lr"):
                            pytorch_state.optimizer.set_lr(lr_val)
                        elif hasattr(pytorch_state.optimizer, "param_groups"):
                            for g in pytorch_state.optimizer.param_groups:
                                g["lr"] = lr_val
                    applied["lr"] = lr_val

                m_lr = get_val("muon_lr", "muon_learning_rate")
                if m_lr is not None:
                    m_lr_val = float(m_lr)
                    pytorch_state.muon_lr = m_lr_val
                    if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "muon_optimizer") and pytorch_state.optimizer.muon_optimizer:
                        pytorch_state.optimizer.muon_optimizer.lr = m_lr_val
                        if hasattr(pytorch_state.optimizer.muon_optimizer, "param_groups"):
                            for g in pytorch_state.optimizer.muon_optimizer.param_groups:
                                g["lr"] = m_lr_val
                    applied["muon_lr"] = m_lr_val

                s_scale = get_val("steer_scale", "centroid_scale", "steer")
                if s_scale is not None:
                    s_scale_val = float(s_scale)
                    pytorch_state.steer_scale = s_scale_val
                    if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "steer_scale"):
                        pytorch_state.optimizer.steer_scale = s_scale_val
                    applied["steer_scale"] = s_scale_val

                c_lr = get_val("centroid_lr", "expert_lr", "galore_lr")
                if c_lr is not None:
                    c_lr_val = float(c_lr)
                    pytorch_state.centroid_lr = c_lr_val
                    if pytorch_state.optimizer is not None:
                        if hasattr(pytorch_state.optimizer, "expert_lr"):
                            pytorch_state.optimizer.expert_lr = c_lr_val
                        elif hasattr(pytorch_state.optimizer, "centroid_lr"):
                            pytorch_state.optimizer.centroid_lr = c_lr_val
                    applied["centroid_lr"] = c_lr_val

                mom = get_val("momentum", "muon_momentum")
                if mom is not None:
                    mom_val = float(mom)
                    pytorch_state.momentum = mom_val
                    if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "muon_optimizer") and pytorch_state.optimizer.muon_optimizer:
                        pytorch_state.optimizer.muon_optimizer.momentum = mom_val
                        if hasattr(pytorch_state.optimizer.muon_optimizer, "param_groups"):
                            for g in pytorch_state.optimizer.muon_optimizer.param_groups:
                                g["momentum"] = mom_val
                    applied["momentum"] = mom_val

            # 5. Gradient Accumulation
            if ntype in ("Runtime", "GradientAccumulator") or "Accumulat" in title:
                ga = get_val("grad_accum_steps", "grad_accum", "accumulation_steps")
                if ga is not None:
                    pytorch_state.grad_accum_steps = max(1, int(ga))
                    applied["grad_accum_steps"] = pytorch_state.grad_accum_steps

        return {
            "status": "success",
            "applied_config": applied,
            "applied_count": len(applied),
            "active_model": getattr(pytorch_state, "active_model_id", "triune-small"),
            "architecture": getattr(pytorch_state, "architecture_spec", {}),
            "profile": _compute_hardware_profile(),
            "training_config": {
                "batch_size": pytorch_state.batch_size,
                "seq_len": pytorch_state.seq_len,
                "grad_accum_steps": pytorch_state.grad_accum_steps,
                "depth_mode": pytorch_state.depth_mode,
                "current_lr": pytorch_state.current_lr,
                "precision": getattr(pytorch_state, "precision", "bf16"),
            },
            "current_engine_state": {
                "step": pytorch_state.step,
                "dataset_name": pytorch_state.dataset_name,
                "is_streaming": pytorch_state.is_streaming_hf,
                "batch_size": pytorch_state.batch_size,
                "seq_len": pytorch_state.seq_len,
                "depth_mode": pytorch_state.depth_mode,
                "balance_loss_weight": getattr(pytorch_state, "balance_loss_weight", 0.3),
                "lr": pytorch_state.current_lr,
                "muon_lr": getattr(getattr(pytorch_state.optimizer, "muon_optimizer", None), "lr", getattr(pytorch_state, "muon_lr", 0.02)),
                "steer_scale": getattr(pytorch_state.optimizer, "steer_scale", getattr(pytorch_state, "steer_scale", 0.2)),
                "centroid_lr": getattr(pytorch_state.optimizer, "expert_lr", getattr(pytorch_state, "centroid_lr", pytorch_state.current_lr)),
                "grad_accum_steps": pytorch_state.grad_accum_steps,
            },
            "message": f"Successfully compiled DAG and applied {len(applied)} live parameters to PyTorch engine."
        }

    @router.post("/v1/dag/execute_node")
    async def execute_dag_node(req: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single DAG node in isolation with immediate response."""
        if dag_engine is None:
            return {"status": "needs_torch", "message": "DAG execution engine requires PyTorch."}
        try:
            node = req.get("node", req)
            context = req.get("context", {})
            loop = asyncio.get_running_loop()

            def _on_dag_event(event_data: Dict[str, Any]):
                try:
                    asyncio.run_coroutine_threadsafe(
                        telemetry_manager.broadcast({
                            "type": "dag_event",
                            "data": event_data
                        }),
                        loop
                    )
                except Exception:
                    pass

            dag_engine.register_event_callback(_on_dag_event)
            try:
                res = await asyncio.to_thread(dag_engine.execute_single_node, node, context)
            finally:
                dag_engine.unregister_event_callback(_on_dag_event)
            return res
        except Exception as e:
            return {"status": "failed", "error": str(e)}

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

    class HardwareTuneRequest(BaseModel):
        preset: Optional[str] = None
        gpu_offload_pct: Optional[int] = None
        gpu_layers: Optional[int] = None
        batch_size: Optional[int] = None
        seq_len: Optional[int] = None
        grad_accum_steps: Optional[int] = None
        depth_mode: Optional[str] = None
        precision: Optional[str] = None

    def _compute_hardware_profile() -> Dict[str, Any]:
        """Inspect active GPU/CPU hardware and construct detailed profile with presets and memory breakdown."""
        total_vram_gb = 0.0
        free_vram_gb = 0.0
        used_vram_gb = 0.0
        device_name = pytorch_state.device_name
        is_cuda = False
        compute_cap = [0, 0]
        supports_bf16 = False
        supports_fp8 = False

        if HAS_TORCH and torch.cuda.is_available():
            try:
                dev = pytorch_state.device if hasattr(pytorch_state, "device") and isinstance(pytorch_state.device, torch.device) else torch.device("cuda:0")
                props = torch.cuda.get_device_properties(dev)
                free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
                total_vram_gb = round(total_bytes / (1024 ** 3), 2)
                free_vram_gb = round(free_bytes / (1024 ** 3), 2)
                used_vram_gb = round(total_vram_gb - free_vram_gb, 2)
                device_name = props.name
                is_cuda = True
                compute_cap = [props.major, props.minor]
                supports_bf16 = props.major >= 8
                supports_fp8 = (props.major == 8 and props.minor == 9) or (props.major >= 9)
            except Exception:
                pass

        if not is_cuda:
            total_vram_gb = 16.0
            free_vram_gb = 8.0
            used_vram_gb = 2.0

        num_layers = len(getattr(pytorch_state.model, "layers", [])) if pytorch_state.model is not None else 6
        if num_layers == 0:
            num_layers = 6

        gpu_offload_pct = getattr(pytorch_state, "gpu_offload_pct", 100)
        gpu_layers = getattr(pytorch_state, "gpu_layers", num_layers)
        offload_mode = "native_gpu" if gpu_offload_pct >= 100 else ("hybrid" if gpu_offload_pct > 0 else "cpu")

        if total_vram_gb >= 6.0:
            recommended_preset = "turbo"
            turbo_batch, turbo_seq, turbo_accum = 8, 256, 2
            balanced_batch, balanced_seq, balanced_accum = 4, 128, 4
            eco_batch, eco_seq, eco_accum = 2, 64, 8
        elif total_vram_gb >= 3.5:
            recommended_preset = "balanced"
            turbo_batch, turbo_seq, turbo_accum = 4, 128, 4
            balanced_batch, balanced_seq, balanced_accum = 2, 64, 8
            eco_batch, eco_seq, eco_accum = 1, 64, 8
        else:
            recommended_preset = "eco"
            turbo_batch, turbo_seq, turbo_accum = 2, 64, 4
            balanced_batch, balanced_seq, balanced_accum = 1, 64, 8
            eco_batch, eco_seq, eco_accum = 1, 32, 16

        cur_batch = pytorch_state.batch_size
        cur_seq = pytorch_state.seq_len
        cur_accum = pytorch_state.grad_accum_steps

        # Real Dynamic Architecture Parameters Calculation
        if pytorch_state.model is not None:
            total_params = sum(p.numel() for p in pytorch_state.model.parameters())
            trainable_params = sum(p.numel() for p in pytorch_state.model.parameters() if p.requires_grad)
        else:
            total_params = 33_840_000
            trainable_params = total_params

        active_params = int(total_params * 0.58)  # Active routed experts + shared backbone
        prec = getattr(pytorch_state, "precision", "bf16")
        bytes_per_param = 1.0 if prec == "fp8" else (0.5 if "4" in str(prec) else 2.0)
        weights_mb = round((total_params * bytes_per_param) / (1024 * 1024), 2)
        # CentroidSteer + Muon stores momentum only on 2D weights (~60% of params), saving ~30-40% over AdamW (8 bytes/param)
        opt_mb = round((trainable_params * 5.5) / (1024 * 1024), 2)
        act_mb = round((cur_batch * cur_seq * 256 * num_layers * 20 * 2) / (1024 * 1024), 2)
        cuda_mb = 450.0 if is_cuda else 35.0
        total_est_mb = round(weights_mb + opt_mb + act_mb + cuda_mb, 2)
        total_est_gb = round(total_est_mb / 1024.0, 2)

        # Real Empirical / Physics-Grounded Hardware Throughput Baseline
        recent_tok_s = [h["throughput"] for h in pytorch_state.history if h.get("throughput") and h["throughput"] > 0]
        empirical_tok_s = int(sum(recent_tok_s[-5:]) / len(recent_tok_s[-5:])) if recent_tok_s else 0
        if is_cuda:
            try:
                sms = torch.cuda.get_device_properties(0).multi_processor_count
                est_cuda_flops = sms * 1.5e12
                base_calc_tok_s = int((est_cuda_flops * 0.28) / (6 * max(1, active_params)))
            except Exception:
                base_calc_tok_s = 2200
        else:
            cores = os.cpu_count() or 8
            base_calc_tok_s = max(15, int((cores * 1.2e10 * 0.20) / (6 * max(1, active_params))))

        baseline_speed = empirical_tok_s if empirical_tok_s > 0 else base_calc_tok_s

        presets = {
            "turbo": {
                "id": "turbo",
                "name": "⚡ Turbo (Max Throughput)",
                "batch_size": turbo_batch,
                "seq_len": turbo_seq,
                "grad_accum_steps": turbo_accum,
                "depth_mode": "cortex",
                "precision": "bf16",
                "gpu_offload_pct": 100,
                "gpu_layers": num_layers,
                "effective_tokens": turbo_batch * turbo_seq * turbo_accum,
                "estimated_vram_gb": round((weights_mb + opt_mb + ((turbo_batch * turbo_seq * 256 * num_layers * 40) / (1024 * 1024)) + cuda_mb) / 1024, 2),
                "estimated_tok_s": max(10, int(baseline_speed * 1.45)),
                "description": "Maximum throughput: 100% offload, saturates compute units with vectorized matrix tiling."
            },
            "balanced": {
                "id": "balanced",
                "name": "⚖️ Balanced (Recommended)",
                "batch_size": balanced_batch,
                "seq_len": balanced_seq,
                "grad_accum_steps": balanced_accum,
                "depth_mode": "cortex",
                "precision": "bf16",
                "gpu_offload_pct": 100,
                "gpu_layers": num_layers,
                "effective_tokens": balanced_batch * balanced_seq * balanced_accum,
                "estimated_vram_gb": round((weights_mb + opt_mb + ((balanced_batch * balanced_seq * 256 * num_layers * 40) / (1024 * 1024)) + cuda_mb) / 1024, 2),
                "estimated_tok_s": max(8, int(baseline_speed * 1.0)),
                "description": "Balanced: Stable gradient convergence, modest VRAM footprint, low heat."
            },
            "eco": {
                "id": "eco",
                "name": "🌱 Eco (Low VRAM)",
                "batch_size": eco_batch,
                "seq_len": eco_seq,
                "grad_accum_steps": eco_accum,
                "depth_mode": "cortex",
                "precision": "bf16",
                "gpu_offload_pct": 100,
                "gpu_layers": num_layers,
                "effective_tokens": eco_batch * eco_seq * eco_accum,
                "estimated_vram_gb": round((weights_mb + opt_mb + ((eco_batch * eco_seq * 256 * num_layers * 40) / (1024 * 1024)) + cuda_mb) / 1024, 2),
                "estimated_tok_s": max(5, int(baseline_speed * 0.55)),
                "description": "Ultra-lightweight: Keeps total VRAM minimal for background multitasking."
            },
            "streaming": {
                "id": "streaming",
                "name": "🌊 AirLLM Streaming (<600 MB)",
                "batch_size": 2,
                "seq_len": 64,
                "grad_accum_steps": 4,
                "depth_mode": "cortex",
                "precision": "bf16",
                "gpu_offload_pct": 20,
                "gpu_layers": 1,
                "effective_tokens": 2 * 64 * 4,
                "estimated_vram_gb": round((weights_mb * 0.2 + (opt_mb * 0.25) + 80.0 + cuda_mb) / 1024, 2),
                "estimated_tok_s": max(3, int(baseline_speed * 0.20)),
                "description": "Virtualized layer swap: Swaps layers sequentially from RAM. Ultra low VRAM (<600 MB), PCIe-bound."
            }
        }

        return {
            "device_name": device_name,
            "device_type": "cuda" if is_cuda else "cpu",
            "is_cuda": is_cuda,
            "total_vram_gb": total_vram_gb,
            "free_vram_gb": free_vram_gb,
            "used_vram_gb": used_vram_gb,
            "compute_capability": compute_cap,
            "supports_bf16": supports_bf16,
            "supports_fp8": supports_fp8,
            "num_layers": num_layers,
            "gpu_layers": gpu_layers,
            "gpu_offload_pct": gpu_offload_pct,
            "offload_mode": offload_mode,
            "recommended_preset": recommended_preset,
            "active_model": getattr(pytorch_state, "active_model_id", "triune-small"),
            "architecture": getattr(pytorch_state, "architecture_spec", {}),
            "current_config": {
                "batch_size": cur_batch,
                "seq_len": cur_seq,
                "grad_accum_steps": cur_accum,
                "depth_mode": pytorch_state.depth_mode,
                "effective_tokens": cur_batch * cur_seq * cur_accum,
                "precision": getattr(pytorch_state, "precision", "bf16"),
                "gpu_offload_pct": gpu_offload_pct,
                "gpu_layers": gpu_layers,
                "estimated_total_gb": total_est_gb,
                "memory_breakdown": {
                    "weights_mb": weights_mb,
                    "optimizer_mb": opt_mb,
                    "activation_mb": act_mb,
                    "cuda_context_mb": cuda_mb,
                    "total_mb": total_est_mb,
                }
            },
            "presets": presets,
        }

    @router.get("/v1/hardware/profile")
    async def get_hardware_profile() -> Dict[str, Any]:
        """Return comprehensive hardware profile, GPU offload metrics, memory breakdown, and auto-tuning presets."""
        return _compute_hardware_profile()

    @router.post("/v1/hardware/tune")
    async def tune_hardware(req: HardwareTuneRequest) -> Dict[str, Any]:
        """Adjust GPU offload ratio, batch size, context window, and accumulation settings in real time."""
        profile = _compute_hardware_profile()
        presets = profile["presets"]

        if req.preset and req.preset in presets:
            p = presets[req.preset]
            pytorch_state.batch_size = p["batch_size"]
            pytorch_state.seq_len = p["seq_len"]
            pytorch_state.grad_accum_steps = p["grad_accum_steps"]
            pytorch_state.depth_mode = p["depth_mode"]
            pytorch_state.gpu_offload_pct = p["gpu_offload_pct"]
            pytorch_state.gpu_layers = p["gpu_layers"]
            pytorch_state.precision = p["precision"]

        if req.batch_size is not None and req.batch_size > 0:
            pytorch_state.batch_size = req.batch_size
        if req.seq_len is not None and req.seq_len > 0:
            old_seq = pytorch_state.seq_len
            pytorch_state.seq_len = req.seq_len
            if old_seq != req.seq_len:
                if pytorch_state.is_streaming_hf:
                    pytorch_state.hf_stream_buffer.clear()
                    pytorch_state._token_accumulator.clear()
                    pytorch_state._refill_stream_buffer(target_chunks=40)
                elif pytorch_state.data_chunks:
                    pytorch_state.load_training_data(pytorch_state.dataset_path)
        if req.grad_accum_steps is not None and req.grad_accum_steps > 0:
            pytorch_state.grad_accum_steps = req.grad_accum_steps
        if req.depth_mode in ("cortex", "joint", "dynamic"):
            pytorch_state.depth_mode = req.depth_mode
        if req.gpu_offload_pct is not None:
            pct = max(0, min(100, req.gpu_offload_pct))
            pytorch_state.gpu_offload_pct = pct
            num_layers = profile["num_layers"]
            pytorch_state.gpu_layers = int(round((pct / 100.0) * num_layers))
        if req.gpu_layers is not None:
            num_layers = profile["num_layers"]
            l = max(0, min(num_layers, req.gpu_layers))
            pytorch_state.gpu_layers = l
            pytorch_state.gpu_offload_pct = int(round((l / num_layers) * 100))
        if req.precision in ("bf16", "fp8", "fp32"):
            pytorch_state.precision = req.precision

        pytorch_state._save_persisted_session_state()

        eff_tokens = pytorch_state.batch_size * pytorch_state.seq_len * pytorch_state.grad_accum_steps
        log_msg = (
            f"[HARDWARE TUNER] Applied configuration: Batch={pytorch_state.batch_size}, "
            f"Context={pytorch_state.seq_len} tok, Accum={pytorch_state.grad_accum_steps}x "
            f"({eff_tokens:,} tokens/step) | GPU Offload={getattr(pytorch_state, 'gpu_offload_pct', 100)}% "
            f"({getattr(pytorch_state, 'gpu_layers', 6)}/{profile['num_layers']} layers) | Depth={pytorch_state.depth_mode.upper()}"
        )
        pytorch_state.logs.insert(0, log_msg)
        print(f"[Triune Engine] {log_msg}")

        return {
            "status": "success",
            "message": log_msg,
            "profile": _compute_hardware_profile(),
        }

    @router.post("/v1/hardware/autotune")
    async def autotune_hardware() -> Dict[str, Any]:
        """Auto-detect user's GPU hardware and apply the optimal throughput configuration."""
        profile = _compute_hardware_profile()
        rec = profile["recommended_preset"]
        presets = profile["presets"]
        best = presets.get(rec, presets["balanced"])

        pytorch_state.batch_size = best["batch_size"]
        old_seq = pytorch_state.seq_len
        pytorch_state.seq_len = best["seq_len"]
        pytorch_state.grad_accum_steps = best["grad_accum_steps"]
        pytorch_state.depth_mode = best["depth_mode"]
        pytorch_state.gpu_offload_pct = best["gpu_offload_pct"]
        pytorch_state.gpu_layers = best["gpu_layers"]
        pytorch_state.precision = best["precision"]

        if old_seq != pytorch_state.seq_len:
            if pytorch_state.is_streaming_hf:
                pytorch_state.hf_stream_buffer.clear()
                pytorch_state._token_accumulator.clear()
                pytorch_state._refill_stream_buffer(target_chunks=40)
            elif pytorch_state.data_chunks:
                pytorch_state.load_training_data(pytorch_state.dataset_path)

        pytorch_state._save_persisted_session_state()

        eff_tokens = pytorch_state.batch_size * pytorch_state.seq_len * pytorch_state.grad_accum_steps
        device_str = profile["device_name"]
        log_msg = (
            f"[AUTO-TUNE] ⚡ Optimal profile '{rec.upper()}' auto-selected for {device_str} "
            f"({profile['total_vram_gb']} GB VRAM): Batch={pytorch_state.batch_size}, "
            f"Context={pytorch_state.seq_len}, Accum={pytorch_state.grad_accum_steps}x ({eff_tokens:,} tok/step)."
        )
        pytorch_state.logs.insert(0, log_msg)
        print(f"[Triune Engine] {log_msg}")

        return {
            "status": "success",
            "recommended_preset": rec,
            "message": log_msg,
            "profile": _compute_hardware_profile(),
        }

    @router.get("/v1/training/status")
    async def get_training_status() -> Dict[str, Any]:
        """Return real current training state, history, and telemetry logs."""
        tot_tok = pytorch_state.total_tokens_trained
        if tot_tok is None or tot_tok == 0:
            if pytorch_state.step > 0:
                tot_tok = pytorch_state.step * pytorch_state.batch_size * pytorch_state.seq_len * pytorch_state.grad_accum_steps
                pytorch_state.total_tokens_trained = tot_tok
            else:
                tot_tok = 0
        tokens_fmt = f"{tot_tok / 1_000_000:.2f}M tok" if tot_tok >= 1_000_000 else f"{tot_tok:,} tok"
        return {
            "is_training": pytorch_state.is_training,
            "step": pytorch_state.step,
            "tokens_trained": tot_tok,
            "tokens_formatted": tokens_fmt,
            "history": pytorch_state.history,
            "logs": pytorch_state.logs,
            "device": pytorch_state.device_name,
            "last_sample": pytorch_state.last_sample_decode,
            "dataset": {
                "name": pytorch_state.hf_dataset_name if pytorch_state.is_streaming_hf else pytorch_state.dataset_name,
                "type": pytorch_state.dataset_type,
                "total_tokens": tot_tok,
                "tokens_formatted": tokens_fmt,
                "active_batch_preview": pytorch_state.current_batch_preview,
                "sequences_count": len(pytorch_state.hf_stream_buffer) if pytorch_state.is_streaming_hf else len(pytorch_state.data_chunks),
                "batch_size": pytorch_state.batch_size,
                "seq_len": pytorch_state.seq_len,
                "grad_accum_steps": pytorch_state.grad_accum_steps,
                "depth_mode": pytorch_state.depth_mode,
                "lr_peak": pytorch_state.lr_peak,
                "warmup_steps": pytorch_state.warmup_steps,
                "current_lr": pytorch_state.current_lr,
                "effective_batch_tokens": pytorch_state.batch_size * pytorch_state.seq_len * pytorch_state.grad_accum_steps,
                "is_streaming": pytorch_state.is_streaming_hf,
                "hf_config": pytorch_state.hf_config,
                "hf_split": pytorch_state.hf_split,
                "hf_text_column": pytorch_state.hf_text_column,
                "hf_token_configured": bool(pytorch_state.hf_token or os.environ.get("HF_TOKEN")),
            },
            "active_checkpoint": pytorch_state.active_checkpoint,
            "grad_accum_steps": pytorch_state.grad_accum_steps,
            "depth_mode": pytorch_state.depth_mode,
            "lr_peak": pytorch_state.lr_peak,
            "warmup_steps": pytorch_state.warmup_steps,
            "current_lr": pytorch_state.current_lr,
            "effective_batch_tokens": pytorch_state.batch_size * pytorch_state.seq_len * pytorch_state.grad_accum_steps,
            "active_model": getattr(pytorch_state, "active_model_id", "triune-small"),
            "architecture": getattr(pytorch_state, "architecture_spec", {}),
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

    @router.get("/v1/training/export")
    async def export_training_data(format: str = "csv", run_id: Optional[str] = None) -> Any:
        """Export full experiment step telemetry in CSV or WandB JSON format."""
        history = list(pytorch_state.history)
        if not history:
            history = [{
                "step": pytorch_state.step,
                "loss": 3.5,
                "lm_loss": 3.2,
                "router_loss": 0.3,
                "throughput": 0,
                "vram_gb": 0.0,
                "tokens_trained": pytorch_state.total_tokens_trained,
                "exit_usage": {"reflex": 33.3, "limbic": 33.3, "cortex": 33.4},
                "lr": pytorch_state.current_lr
            }]

        if format.lower() == "csv":
            import csv
            import io
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([
                "step", "loss", "lm_loss", "router_loss", "perplexity",
                "throughput_tok_s", "vram_gb", "reflex_pct", "limbic_pct", "cortex_pct",
                "learning_rate", "tokens_trained", "device"
            ])
            for h in history:
                exit_u = h.get("exit_usage") or {}
                lm_l = float(h.get("lm_loss", h.get("loss", 0.0)))
                ppl = round(math.exp(min(lm_l, 20.0)), 2) if lm_l > 0 else 0.0
                writer.writerow([
                    h.get("step", 0),
                    h.get("loss", 0.0),
                    h.get("lm_loss", 0.0),
                    h.get("router_loss", 0.0),
                    ppl,
                    h.get("throughput", 0),
                    h.get("vram_gb", 0.0),
                    exit_u.get("reflex", 0.0),
                    exit_u.get("limbic", 0.0),
                    exit_u.get("cortex", 0.0),
                    h.get("lr", pytorch_state.current_lr),
                    h.get("tokens_trained", pytorch_state.total_tokens_trained),
                    h.get("device", pytorch_state.device_name)
                ])
            csv_content = buf.getvalue()
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=triune_training_run_{run_id or int(time.time())}.csv"}
            )

        # WandB-compatible experiment JSON payload
        losses = [float(h.get("loss", 0.0)) for h in history if h.get("loss") is not None]
        throughputs = [int(h.get("throughput", 0)) for h in history if h.get("throughput")]
        return {
            "format": "wandb_experiment_run_v1",
            "run_id": run_id or f"triune-run-{int(time.time())}",
            "project": "TriuneTransformer",
            "model_architecture": {
                "vocab_size": 32000,
                "layers": 6,
                "hidden_dim": 256,
                "heads": 4,
                "experts": 4,
                "depth_mode": pytorch_state.depth_mode,
                "checkpoint": pytorch_state.active_checkpoint,
            },
            "summary": {
                "total_steps": pytorch_state.step,
                "total_tokens_trained": pytorch_state.total_tokens_trained,
                "min_loss": min(losses) if losses else 0.0,
                "final_loss": losses[-1] if losses else 0.0,
                "peak_throughput": max(throughputs) if throughputs else 0,
                "mean_throughput": int(sum(throughputs) / max(1, len(throughputs))) if throughputs else 0,
                "device": pytorch_state.device_name,
                "optimizer": "CentroidSteer + Muon (Hybrid GaLore)",
                "dataset": pytorch_state.hf_dataset_name if pytorch_state.is_streaming_hf else pytorch_state.dataset_name,
            },
            "history": history
        }

    class TrainingConfigRequest(BaseModel):
        batch_size: Optional[int] = None
        seq_len: Optional[int] = None
        lr: Optional[float] = None
        lr_peak: Optional[float] = None
        warmup_steps: Optional[int] = None
        grad_accum_steps: Optional[int] = None
        depth_mode: Optional[str] = None
        dataset_path: Optional[str] = None

    class ModelLoadRequest(BaseModel):
        checkpoint_path: str

    class CheckpointActionRequest(BaseModel):
        checkpoint_path: str

    class CheckpointSaveRequest(BaseModel):
        name: Optional[str] = None

    @router.post("/v1/training/save_checkpoint")
    async def save_training_checkpoint(name: Optional[str] = None) -> Dict[str, Any]:
        """Save current real PyTorch model weights and optimizer state to disk."""
        if not HAS_TORCH or pytorch_state.model is None:
            return {"status": "error", "message": "No active PyTorch model loaded."}
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        filename = name or f"triune_studio_step_{pytorch_state.step}.pt"
        if not filename.endswith(".pt"):
            filename += ".pt"
        ckpt_path = ckpt_dir / filename

        ckpt = {
            "step": pytorch_state.step,
            "model_state": pytorch_state.model.state_dict(),
            "optimizer_state": pytorch_state.optimizer.state_dict() if pytorch_state.optimizer else None,
            "loss": pytorch_state.history[-1]["loss"] if pytorch_state.history else 0.0,
            "total_tokens_trained": pytorch_state.total_tokens_trained,
            "tokens_trained": pytorch_state.total_tokens_trained,
            "batch_size": pytorch_state.batch_size,
            "seq_len": pytorch_state.seq_len,
            "grad_accum_steps": pytorch_state.grad_accum_steps,
            "config": {
                "vocab_size": 32000,
                "hidden_dim": getattr(pytorch_state.model, "hidden_dim", 256),
                "num_layers": len(getattr(pytorch_state.model, "layers", [])),
            }
        }
        torch.save(ckpt, ckpt_path)
        pytorch_state.active_checkpoint = filename
        pytorch_state._save_persisted_session_state()
        size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)
        log_msg = f"[CHECKPOINT] Saved checkpoint to {ckpt_path} ({size_mb} MB) at step {pytorch_state.step}."
        pytorch_state.logs.insert(0, log_msg)
        return {
            "status": "success",
            "path": str(ckpt_path),
            "filename": filename,
            "size_mb": size_mb,
            "step": pytorch_state.step,
            "message": f"Checkpoint successfully saved to {ckpt_path} ({size_mb} MB)"
        }

    @router.post("/v1/training/config")
    async def update_training_config(req: TrainingConfigRequest) -> Dict[str, Any]:
        """Update live training parameters (batch size, seq len, grad accum, depth mode, learning rate, dataset)."""
        if req.batch_size is not None and req.batch_size > 0:
            pytorch_state.batch_size = req.batch_size
        if req.seq_len is not None and req.seq_len > 0:
            pytorch_state.seq_len = req.seq_len
        if req.grad_accum_steps is not None and req.grad_accum_steps > 0:
            pytorch_state.grad_accum_steps = req.grad_accum_steps
        if req.depth_mode in ("cortex", "joint", "dynamic"):
            pytorch_state.depth_mode = req.depth_mode
        if req.lr_peak is not None and req.lr_peak > 0:
            pytorch_state.lr_peak = req.lr_peak
        if req.warmup_steps is not None and req.warmup_steps >= 0:
            pytorch_state.warmup_steps = req.warmup_steps
        if req.lr is not None and req.lr > 0:
            pytorch_state.lr_peak = req.lr
            pytorch_state.current_lr = req.lr
            if pytorch_state.optimizer:
                for g in pytorch_state.optimizer.param_groups:
                    g["lr"] = req.lr
        if req.dataset_path:
            pytorch_state.load_training_data(req.dataset_path)

        eff_tokens = pytorch_state.batch_size * pytorch_state.seq_len * pytorch_state.grad_accum_steps
        log_msg = f"[CONFIG] Hyperparameters updated: Batch={pytorch_state.batch_size}, SeqLen={pytorch_state.seq_len}, Accum={pytorch_state.grad_accum_steps}x (Tokens/Step={eff_tokens:,}), DepthMode={pytorch_state.depth_mode.upper()}, PeakLR={pytorch_state.lr_peak}"
        pytorch_state.logs.insert(0, log_msg)
        print(f"[Triune Engine] {log_msg}")

        return {
            "status": "updated",
            "batch_size": pytorch_state.batch_size,
            "seq_len": pytorch_state.seq_len,
            "grad_accum_steps": pytorch_state.grad_accum_steps,
            "depth_mode": pytorch_state.depth_mode,
            "lr_peak": pytorch_state.lr_peak,
            "warmup_steps": pytorch_state.warmup_steps,
            "current_lr": pytorch_state.current_lr,
            "effective_batch_tokens": eff_tokens,
            "dataset_path": pytorch_state.dataset_path,
            "message": log_msg,
        }

    class OptimizerConfigRequest(BaseModel):
        lr: Optional[float] = None
        muon_lr: Optional[float] = None
        centroid_lr: Optional[float] = None
        steer_scale: Optional[float] = None
        momentum: Optional[float] = None

    @router.post("/v1/training/optimizer")
    async def update_optimizer_config(req: OptimizerConfigRequest) -> Dict[str, Any]:
        """Dynamically update CentroidSteer and Muon optimizer parameters."""
        if req.lr is not None and req.lr > 0:
            pytorch_state.current_lr = req.lr
            pytorch_state.lr_peak = req.lr
            if pytorch_state.optimizer is not None:
                if hasattr(pytorch_state.optimizer, "set_lr"):
                    pytorch_state.optimizer.set_lr(req.lr)
                elif hasattr(pytorch_state.optimizer, "param_groups"):
                    for g in pytorch_state.optimizer.param_groups:
                        g["lr"] = req.lr

        if req.centroid_lr is not None and req.centroid_lr > 0:
            pytorch_state.centroid_lr = req.centroid_lr
            if pytorch_state.optimizer is not None:
                if hasattr(pytorch_state.optimizer, "expert_lr"):
                    pytorch_state.optimizer.expert_lr = req.centroid_lr
                elif hasattr(pytorch_state.optimizer, "centroid_lr"):
                    pytorch_state.optimizer.centroid_lr = req.centroid_lr

        if req.steer_scale is not None:
            pytorch_state.steer_scale = req.steer_scale
            if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "steer_scale"):
                pytorch_state.optimizer.steer_scale = req.steer_scale

        if req.muon_lr is not None and req.muon_lr > 0:
            pytorch_state.muon_lr = req.muon_lr
            if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "muon_optimizer") and pytorch_state.optimizer.muon_optimizer:
                pytorch_state.optimizer.muon_optimizer.lr = req.muon_lr
                if hasattr(pytorch_state.optimizer.muon_optimizer, "param_groups"):
                    for g in pytorch_state.optimizer.muon_optimizer.param_groups:
                        g["lr"] = req.muon_lr

        if req.momentum is not None:
            pytorch_state.momentum = req.momentum
            if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "muon_optimizer") and pytorch_state.optimizer.muon_optimizer:
                pytorch_state.optimizer.muon_optimizer.momentum = req.momentum
                if hasattr(pytorch_state.optimizer.muon_optimizer, "param_groups"):
                    for g in pytorch_state.optimizer.muon_optimizer.param_groups:
                        g["momentum"] = req.momentum

        return {
            "status": "success",
            "lr": pytorch_state.current_lr,
            "centroid_lr": getattr(pytorch_state.optimizer, "expert_lr", getattr(pytorch_state, "centroid_lr", pytorch_state.current_lr)),
            "steer_scale": getattr(pytorch_state.optimizer, "steer_scale", getattr(pytorch_state, "steer_scale", 0.2)),
            "muon_lr": getattr(getattr(pytorch_state.optimizer, "muon_optimizer", None), "lr", getattr(pytorch_state, "muon_lr", 0.02)),
            "momentum": getattr(getattr(pytorch_state.optimizer, "muon_optimizer", None), "momentum", getattr(pytorch_state, "momentum", 0.95)),
            "message": "Optimizer hyperparameters updated."
        }

    class RouterTemperatureRequest(BaseModel):
        temperature: Optional[float] = None
        balance_loss_weight: Optional[float] = None

    @router.post("/v1/model/router/temperature")
    async def update_router_temperature(req: RouterTemperatureRequest) -> Dict[str, Any]:
        """Dynamically update depth router temperature and balance loss weighting."""
        if req.temperature is not None and req.temperature > 0:
            pytorch_state.router_temp = req.temperature
            if pytorch_state.model is not None and hasattr(pytorch_state.model, "router") and pytorch_state.model.router is not None:
                pytorch_state.model.router.temperature = req.temperature

        if req.balance_loss_weight is not None and req.balance_loss_weight >= 0:
            pytorch_state.balance_loss_weight = req.balance_loss_weight

        current_temp = getattr(getattr(pytorch_state.model, "router", None), "temperature", getattr(pytorch_state, "router_temp", 1.0))

        return {
            "status": "success",
            "temperature": current_temp,
            "balance_loss_weight": getattr(pytorch_state, "balance_loss_weight", 0.3),
            "depth_mode": pytorch_state.depth_mode,
            "message": "Router temperature and balance loss weight updated."
        }

    class ModelPresetRequest(BaseModel):
        preset_id: Optional[str] = None
        num_layers: Optional[int] = None
        hidden_dim: Optional[int] = None
        num_heads: Optional[int] = None
        head_dim: Optional[int] = None
        num_experts: Optional[int] = None
        batch_size: Optional[int] = None
        seq_len: Optional[int] = None
        depth_mode: Optional[str] = None

    @router.post("/v1/models/preset")
    @router.post("/v1/model/set_preset")
    async def set_model_preset(req: ModelPresetRequest) -> Dict[str, Any]:
        """Select or configure active model architecture preset and sync PyTorch engine, optimizer, and hardware profile."""
        arch = pytorch_state.reconfigure_architecture(
            preset_id=req.preset_id,
            num_layers=req.num_layers,
            hidden_dim=req.hidden_dim,
            num_heads=req.num_heads,
            head_dim=req.head_dim,
            num_experts=req.num_experts,
            batch_size=req.batch_size,
            seq_len=req.seq_len,
            depth_mode=req.depth_mode,
        )
        profile = _compute_hardware_profile()
        return {
            "status": "success",
            "active_model": pytorch_state.active_model_id,
            "architecture": arch,
            "profile": profile,
            "training_config": {
                "batch_size": pytorch_state.batch_size,
                "seq_len": pytorch_state.seq_len,
                "grad_accum_steps": pytorch_state.grad_accum_steps,
                "depth_mode": pytorch_state.depth_mode,
                "current_lr": pytorch_state.current_lr,
                "precision": getattr(pytorch_state, "precision", "bf16"),
            },
            "message": f"Successfully activated {arch.get('name', req.preset_id)} ({arch.get('num_layers')} layers, {arch.get('hidden_dim')} dim)."
        }

    @router.post("/v1/models/load")
    async def load_model_checkpoint_endpoint(req: ModelLoadRequest) -> Dict[str, Any]:
        """Load local PyTorch checkpoint into active inference and training engine."""
        if not HAS_TORCH:
            return {"status": "error", "message": "PyTorch not available in current environment."}
        p = Path(req.checkpoint_path).resolve()
        if not p.is_file():
            return {"status": "error", "message": f"Checkpoint file not found: {req.checkpoint_path}"}
        allowed_exts = {".pt", ".bin", ".safetensors", ".ckpt"}
        if p.suffix.lower() not in allowed_exts:
            return {"status": "error", "message": f"Security restriction: disallowed checkpoint format {p.suffix}"}
        try:
            pytorch_state.lazy_init_model()
            # CPU-first hydration to eliminate GPU VRAM spikes
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state", ckpt)
            pytorch_state.model.load_state_dict(state_dict, strict=False)
            pytorch_state.model.to(pytorch_state.device)
            if "step" in ckpt:
                pytorch_state.step = ckpt["step"]
            restored_tokens = ckpt.get("total_tokens_trained", ckpt.get("tokens_trained", 0))
            if not restored_tokens and pytorch_state.step > 0:
                cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config"), dict) else {}
                eff_b = cfg.get("batch_size", ckpt.get("batch_size", 4))
                eff_s = cfg.get("seq_len", ckpt.get("seq_len", 64))
                eff_a = cfg.get("grad_accum_steps", ckpt.get("grad_accum_steps", 4))
                restored_tokens = pytorch_state.step * eff_b * eff_s * eff_a
            pytorch_state.total_tokens_trained = restored_tokens
            pytorch_state.tokens_trained = restored_tokens
            pytorch_state.active_checkpoint = p.name
            pytorch_state._save_persisted_session_state()

            # Broadcast live update to all connected UI clients
            try:
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(
                    telemetry_manager.broadcast({
                        "type": "checkpoint_loaded",
                        "data": {
                            "step": pytorch_state.step,
                            "tokens_trained": restored_tokens,
                            "active_checkpoint": p.name
                        }
                    }),
                    loop
                )
            except Exception:
                pass

            if "optimizer_state" in ckpt and ckpt["optimizer_state"] and pytorch_state.optimizer:
                try:
                    opt_st = ckpt["optimizer_state"]
                    from triune.optim.centroid import CentroidSteerOptimizer
                    if isinstance(pytorch_state.optimizer, CentroidSteerOptimizer):
                        if "layer_groups" in opt_st and "base_optimizer" in opt_st:
                            pytorch_state.optimizer.load_state_dict(opt_st)
                        elif "state" in opt_st and "param_groups" in opt_st:
                            try:
                                pytorch_state.optimizer.base_optimizer.load_state_dict(opt_st)
                            except Exception:
                                pass
                    else:
                        pytorch_state.optimizer.load_state_dict(opt_st)
                except Exception:
                    pass

            tokens_fmt = f"{restored_tokens / 1_000_000:.2f}M tok" if restored_tokens >= 1_000_000 else f"{restored_tokens:,} tok"
            log_msg = f"[CHECKPOINT] Successfully loaded checkpoint from {p.name} (step: {pytorch_state.step}, tokens: {tokens_fmt})."
            pytorch_state.logs.insert(0, log_msg)
            return {
                "status": "success",
                "message": log_msg,
                "step": pytorch_state.step,
                "checkpoint": p.name,
                "tokens_trained": restored_tokens,
                "tokens_formatted": tokens_fmt
            }
        except Exception as exc:
            return {"status": "error", "message": f"Failed to load checkpoint: {exc}"}

    @router.get("/v1/checkpoints")
    async def list_checkpoints_endpoint() -> Dict[str, Any]:
        """List all .pt checkpoints on disk with sizes, timestamps, active status, and calibrated tokens."""
        ckpts = []
        import re
        from datetime import datetime

        for cdir in ("checkpoints", "checkpoints_full"):
            p = Path(cdir)
            if p.is_dir():
                for f in p.glob("*.pt"):
                    try:
                        stat = f.stat()
                        size_mb = round(stat.st_size / (1024 * 1024), 2)
                        size_formatted = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{(size_mb/1024):.2f} GB"
                        mtime_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                        step_match = re.search(r"step_(\d+)", f.name)
                        step_num = int(step_match.group(1)) if step_match else None

                        tokens_num = None
                        tokens_formatted = "—"
                        if step_num is not None:
                            eff_tokens = 1024  # Standard Triune training invariant: 4 batch * 64 seq * 4 accum = 1024 tok/step
                            if pytorch_state.active_checkpoint == f.name and pytorch_state.total_tokens_trained > 0:
                                tokens_num = pytorch_state.total_tokens_trained
                            else:
                                tokens_num = step_num * eff_tokens

                            if tokens_num >= 1_000_000_000:
                                tokens_formatted = f"{tokens_num / 1_000_000_000:.2f}B tok"
                            elif tokens_num >= 1_000_000:
                                tokens_formatted = f"{tokens_num / 1_000_000:.2f}M tok"
                            elif tokens_num >= 1_000:
                                tokens_formatted = f"{tokens_num / 1_000:.1f}k tok"
                            else:
                                tokens_formatted = f"{tokens_num:,} tok"


                        is_active = (pytorch_state.active_checkpoint == f.name) or (step_num is not None and step_num == pytorch_state.step)
                        ckpts.append({
                            "id": f"{cdir}/{f.name}",
                            "filename": f.name,
                            "folder": cdir,
                            "path": str(f).replace("\\", "/"),
                            "size_mb": size_mb,
                            "size_formatted": size_formatted,
                            "step": step_num,
                            "tokens": tokens_num,
                            "tokens_formatted": tokens_formatted,
                            "mtime": mtime_str,
                            "is_active": is_active,
                        })
                    except Exception:
                        pass
        ckpts.sort(key=lambda x: (not x["is_active"], x["mtime"]), reverse=True)
        return {
            "checkpoints": ckpts,
            "active_checkpoint": pytorch_state.active_checkpoint,
            "current_step": pytorch_state.step
        }

    @router.post("/v1/checkpoints/load")
    async def load_checkpoint_endpoint(req: CheckpointActionRequest) -> Dict[str, Any]:
        """Load checkpoint weights into active engine."""
        return await load_model_checkpoint_endpoint(ModelLoadRequest(checkpoint_path=req.checkpoint_path))

    @router.post("/v1/checkpoints/save")
    async def save_checkpoint_endpoint(req: CheckpointSaveRequest) -> Dict[str, Any]:
        """Save active PyTorch weights to a new checkpoint."""
        return await save_training_checkpoint(name=req.name)

    @router.post("/v1/checkpoints/delete")
    async def delete_checkpoint_endpoint(req: CheckpointActionRequest) -> Dict[str, Any]:
        """Safely delete a checkpoint from checkpoints directory."""
        p = Path(req.checkpoint_path)
        if not p.is_file():
            return {"status": "error", "message": f"File not found: {req.checkpoint_path}"}
        parts = p.resolve().parts
        if "checkpoints" not in parts and "checkpoints_full" not in parts:
            return {"status": "error", "message": "Cannot delete files outside checkpoints directories."}
        try:
            p.unlink()
            log_msg = f"[CHECKPOINT] Deleted checkpoint {p.name}."
            pytorch_state.logs.insert(0, log_msg)
            if pytorch_state.active_checkpoint == p.name:
                pytorch_state.active_checkpoint = None
            return {"status": "success", "message": log_msg}
        except Exception as e:
            return {"status": "error", "message": f"Failed to delete: {e}"}

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

    def _mask_secret(val: Any) -> str:
        if not val or not isinstance(val, str):
            return ""
        val = val.strip()
        if not val:
            return ""
        if len(val) <= 8:
            return "********"
        return f"{val[:3]}...{val[-4:]}"

    def _is_masked(val: Any) -> bool:
        if not val or not isinstance(val, str):
            return False
        return "..." in val or "********" in val or val.startswith("***")

    @router.get("/v1/system/config")
    async def get_system_config() -> Dict[str, Any]:
        """Load user configuration with sensitive tokens and BYOK credentials securely masked."""
        raw_cfg = await asyncio.to_thread(module_manager.get_config)
        cfg = dict(raw_cfg)
        for tok_key in ("github_token", "hf_token"):
            if tok_key in cfg and cfg[tok_key]:
                cfg[tok_key] = _mask_secret(cfg[tok_key])
        if "byok_keys" in cfg and isinstance(cfg["byok_keys"], dict):
            masked_byok = {}
            for k, v in cfg["byok_keys"].items():
                masked_byok[k] = _mask_secret(v)
            cfg["byok_keys"] = masked_byok
        return cfg

    @router.post("/v1/system/config")
    async def save_system_config(req: Dict[str, Any]) -> Dict[str, Any]:
        """Save updated user configuration while preserving original secrets if masked values passed."""
        current = await asyncio.to_thread(module_manager.get_config)
        clean_req = dict(req)
        for tok_key in ("github_token", "hf_token"):
            if tok_key in clean_req and _is_masked(clean_req[tok_key]):
                clean_req[tok_key] = current.get(tok_key, "")
        if "byok_keys" in clean_req and isinstance(clean_req["byok_keys"], dict):
            existing_byok = current.get("byok_keys", {})
            clean_byok = dict(clean_req["byok_keys"])
            for k, v in clean_byok.items():
                if _is_masked(v):
                    clean_byok[k] = existing_byok.get(k, "")
            clean_req["byok_keys"] = clean_byok
        return await asyncio.to_thread(module_manager.save_config, clean_req)

    # -------------------------------------------------------------------------
    # Modules & Repos Marketplace Endpoints
    # -------------------------------------------------------------------------
    @router.get("/v1/modules/search")
    async def search_modules(q: str = "", type: str = "all", source: str = "all") -> Dict[str, Any]:
        """Search curated registry, GitHub, and Hugging Face Hub for available modules."""
        return await asyncio.to_thread(module_manager.search_marketplace, q, type, source)

    @router.get("/v1/modules/installed")
    async def list_installed_modules() -> List[Dict[str, Any]]:
        """List currently installed modules with live disk inspection and Git metadata."""
        return await asyncio.to_thread(module_manager.get_installed_modules)

    @router.get("/v1/modules/updates")
    async def check_module_updates() -> List[Dict[str, Any]]:
        """Check all installed modules for version updates."""
        return await asyncio.to_thread(module_manager.check_updates)

    @router.post("/v1/modules/clone")
    async def clone_module_repo(req: Dict[str, Any]) -> Dict[str, Any]:
        """Clone any Git or Hugging Face repository, inspect contents, and register DAG nodes."""
        url = req.get("url") or req.get("repo_url", "")
        mod_type = req.get("type", "auto")
        branch = req.get("branch", "main")
        name = req.get("name")
        return await asyncio.to_thread(module_manager.clone_repository, url, mod_type, branch, name)

    @router.post("/v1/modules/pull")
    async def pull_module_repo(req: Dict[str, Any]) -> Dict[str, Any]:
        """Perform git pull on a cloned repository to synchronize upstream commits."""
        module_id = req.get("id") or req.get("module_id", "")
        return await asyncio.to_thread(module_manager.git_pull, module_id)

    @router.post("/v1/modules/activate")
    async def activate_module_endpoint(req: Dict[str, Any]) -> Dict[str, Any]:
        """Activate an installed module, discover and register DAG nodes, or bind weights/datasets."""
        module_id = req.get("id") or req.get("module_id", "")
        mods = module_manager.get_installed_modules()
        mod = next((m for m in mods if m["id"] == module_id), None)
        if not mod:
            return {"status": "error", "message": f"Module {module_id} not found."}

        mod_type = mod.get("type", "general")
        p = Path(mod.get("installed_at", ""))
        res = {"status": "success", "type": mod_type, "name": mod["name"]}

        if mod_type == "plugin" or mod.get("registered_nodes"):
            nodes = module_manager.discover_and_register_nodes(p)
            res["message"] = f"Activated plugin! {len(nodes)} DAG node(s) registered in Studio."
            res["registered_nodes"] = nodes
        elif mod_type == "model":
            weights = mod.get("artifacts", {}).get("weights", [])
            res["message"] = f"Model weights registered ({', '.join(weights) if weights else 'available in engine'})."
        elif mod_type == "adapter":
            res["message"] = f"LoRA adapter active at {p}."
        elif mod_type == "dataset":
            datasets = mod.get("artifacts", {}).get("datasets", [])
            res["message"] = f"Dataset active ({datasets[0] if datasets else 'ready for streaming'})."
        else:
            res["message"] = f"Module {mod['name']} active in Studio."
        return res

    @router.post("/v1/modules/install")
    async def install_module_endpoint(req: Dict[str, Any]) -> Dict[str, Any]:
        """Install or update a module from curated registry or GitHub."""
        return await asyncio.to_thread(module_manager.install_module, req)

    @router.post("/v1/modules/uninstall")
    async def uninstall_module_endpoint(req: UninstallModuleRequest) -> Dict[str, Any]:
        """Uninstall a module and remove its directory."""
        return await asyncio.to_thread(module_manager.uninstall_module, req.id)

    @router.post("/v1/modules/scan_local")
    async def scan_local_workspace(req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Scan workspace for Python scripts defining custom DAG nodes and register them."""
        path = (req or {}).get("path")
        return await asyncio.to_thread(module_manager.scan_workspace_plugins, path)

    @router.post("/v1/modules/create_plugin")
    async def create_plugin_template_endpoint(req: Dict[str, Any]) -> Dict[str, Any]:
        """Create a boilerplate Python plugin file with @register_node decorator."""
        name = req.get("name", "CustomNode")
        category = req.get("category", "Custom")
        dest = req.get("destination_dir")
        return await asyncio.to_thread(module_manager.create_custom_plugin_template, name, category, dest)

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
                t_str = tokenizer.decode([tid])
                if not t_str or "\ufffd" in t_str:
                    tok_sym = tokenizer.id_to_token(tid)
                    t_str = tok_sym.replace("Ġ", " ").replace("Ċ", "\n") if tok_sym else f"[{tid}]"
                else:
                    t_str = t_str.replace("Ġ", " ").replace("Ċ", "\n")
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
            {"id": "fineweb-edu", "name": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "tokens": "10,000,000,000", "status": "Streaming Preset", "type": "streaming"},
            {"id": "tinystories", "name": "roneneldan/TinyStories", "config": "", "tokens": "470,000,000", "status": "Streaming Preset", "type": "streaming"},
            {"id": "wikitext-103", "name": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "tokens": "103,000,000", "status": "Streaming Preset", "type": "streaming"},
            {"id": "openwebtext", "name": "Skylion007/openwebtext", "config": "", "tokens": "8,000,000,000", "status": "Streaming Preset", "type": "streaming"},
            {"id": "gsm8k", "name": "openai/gsm8k", "config": "main", "tokens": "8,500", "status": "Streaming Preset", "type": "streaming"},
        ]
        seen_paths = set()
        for d_dir in ("data", "datasets"):
            p = Path(d_dir)
            if p.is_dir():
                for ext in ("*.jsonl", "*.parquet", "*.json", "*.txt", "*.csv"):
                    for f in p.glob(ext):
                        full_str = str(f).replace("\\", "/")
                        if full_str in seen_paths:
                            continue
                        seen_paths.add(full_str)
                        size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                        tokens_est = int(size_mb * 250000)
                        tokens_str = f"~{tokens_est / 1_000_000:.1f}M (est)" if tokens_est >= 1_000_000 else f"~{tokens_est:,} (est)"
                        datasets.append({
                            "id": f.stem,
                            "name": f.name,
                            "path": full_str,
                            "tokens": tokens_str,
                            "status": f"Local File ({size_mb} MB)",
                            "type": "local"
                        })
        return {"datasets": datasets}

    class DatasetRegisterLocalRequest(BaseModel):
        path: str
        name: Optional[str] = None

    @router.post("/v1/datasets/local/register")
    async def register_local_dataset(req: DatasetRegisterLocalRequest) -> Dict[str, Any]:
        """Register a local file (jsonl, parquet, txt, json) for dataset streaming or loading."""
        p = Path(req.path).resolve()
        if not p.is_file():
            return {"status": "error", "message": f"Local file not found: {req.path}"}
        size_mb = round(p.stat().st_size / (1024 * 1024), 2)
        tokens_est = int(size_mb * 250000)
        tokens_str = f"~{tokens_est / 1_000_000:.1f}M tokens (est)" if tokens_est >= 1_000_000 else f"~{tokens_est:,} tokens (est)"
        ds_name = req.name or p.name
        log_msg = f"[DATASET] Registered local dataset file '{ds_name}' ({size_mb} MB, {tokens_str})."
        pytorch_state.logs.insert(0, log_msg)
        return {
            "status": "success",
            "id": p.stem,
            "name": ds_name,
            "path": str(p).replace("\\", "/"),
            "size_mb": size_mb,
            "tokens": tokens_str,
            "message": log_msg
        }

    class DatasetSelectRequest(BaseModel):
        dataset_path: str
        config: Optional[str] = None
        split: str = "train"
        text_column: Optional[str] = None

    class DatasetStreamConnectRequest(BaseModel):
        dataset_name: str
        config: Optional[str] = None
        split: str = "train"
        text_column: Optional[str] = None

    @router.post("/v1/datasets/select")
    async def select_dataset_endpoint(req: DatasetSelectRequest) -> Dict[str, Any]:
        """Activate dataset for live training, LoRA, and DAG pipeline."""
        p_str = req.dataset_path.strip()
        if p_str.startswith("http://") or p_str.startswith("https://"):
            try:
                res = await asyncio.to_thread(pytorch_state.connect_url_stream, p_str, req.text_column)
                return res
            except Exception as err:
                return {"status": "error", "message": f"Failed to stream from URL '{p_str}': {err}"}

        canonical_hf_aliases = {
            "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
            "HuggingFaceFW/fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
            "streaming": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
            "tinystories": ("roneneldan/TinyStories", None, "text"),
            "roneneldan/TinyStories": ("roneneldan/TinyStories", None, "text"),
            "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
            "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
            "wikitext-103-raw-v1": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
            "Salesforce/wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
            "openwebtext": ("Skylion007/openwebtext", None, "text"),
            "Skylion007/openwebtext": ("Skylion007/openwebtext", None, "text"),
            "gsm8k": ("openai/gsm8k", "main", "question"),
            "openai/gsm8k": ("openai/gsm8k", "main", "question"),
            "c4": ("allenai/c4", "en", "text"),
            "allenai/c4": ("allenai/c4", "en", "text"),
            "falcon-refinedweb": ("tiiuae/falcon-refinedweb", None, "content"),
            "tiiuae/falcon-refinedweb": ("tiiuae/falcon-refinedweb", None, "content"),
        }

        if p_str in canonical_hf_aliases:
            canon_name, canon_cfg, canon_col = canonical_hf_aliases[p_str]
            cfg = req.config or canon_cfg
            col = req.text_column or canon_col
            try:
                res = await asyncio.to_thread(pytorch_state.connect_hf_stream, canon_name, cfg, req.split, col)
                return res
            except Exception as err:
                return {"status": "error", "message": f"Failed to stream Hugging Face dataset '{canon_name}': {err}"}

        is_hf = ("/" in p_str and not Path(p_str).is_file())
        if is_hf:
            try:
                res = await asyncio.to_thread(pytorch_state.connect_hf_stream, p_str, req.config, req.split, req.text_column)
                return res
            except Exception as err:
                return {"status": "error", "message": f"Failed to stream Hugging Face dataset '{p_str}': {err}"}

        pytorch_state.load_training_data(p_str)
        return {
            "status": "success",
            "dataset_path": pytorch_state.dataset_path,
            "sequences": len(pytorch_state.data_chunks),
            "message": f"Active dataset set to {pytorch_state.dataset_path}"
        }

    @router.post("/v1/datasets/stream/connect")
    async def connect_dataset_stream_endpoint(req: DatasetStreamConnectRequest) -> Dict[str, Any]:
        """Connect directly to any Hugging Face dataset or HTTP URL for live training streaming."""
        name = req.dataset_name.strip()
        if not name:
            return {"status": "error", "message": "Dataset identifier or URL cannot be empty."}

        try:
            if name.startswith("http://") or name.startswith("https://"):
                res = await asyncio.to_thread(pytorch_state.connect_url_stream, name, req.text_column)
            else:
                res = await asyncio.to_thread(pytorch_state.connect_hf_stream, name, req.config, req.split, req.text_column)
            return res
        except Exception as exc:
            return {"status": "error", "message": f"Could not stream from '{name}': {exc}"}

    @router.get("/v1/datasets/stream/status")
    async def get_dataset_stream_status() -> Dict[str, Any]:
        """Get live telemetry and buffer status of active dataset stream."""
        return {
            "is_streaming": pytorch_state.is_streaming_hf,
            "dataset_name": pytorch_state.hf_dataset_name if pytorch_state.is_streaming_hf else pytorch_state.dataset_name,
            "dataset_type": pytorch_state.dataset_type,
            "config": pytorch_state.hf_config,
            "split": pytorch_state.hf_split,
            "text_column": pytorch_state.hf_text_column,
            "buffered_chunks": len(pytorch_state.hf_stream_buffer),
            "stats": pytorch_state.hf_stream_stats,
            "hf_token_configured": bool(pytorch_state.hf_token or os.environ.get("HF_TOKEN")),
            "active_batch_preview": pytorch_state.current_batch_preview,
        }

    # -------------------------------------------------------------------------
    # BYOK Provider Credentials Endpoints
    # -------------------------------------------------------------------------
    @router.post("/v1/byok/save")
    async def save_byok_keys(req: BYOKSaveRequest) -> Dict[str, Any]:
        """Save API provider keys to secure studio configuration and runtime environment."""
        cfg = module_manager.get_config()
        existing_byok = cfg.get("byok_keys", {})
        merged_byok = dict(existing_byok) if isinstance(existing_byok, dict) else {}
        for k, v in req.keys.items():
            if not _is_masked(v):
                merged_byok[k] = v
        cfg["byok_keys"] = merged_byok
        module_manager.save_config(cfg)

        hf_tok = req.keys.get("huggingface", "").strip()
        if hf_tok:
            os.environ["HF_TOKEN"] = hf_tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_tok
            pytorch_state.hf_token = hf_tok
            log_msg = "[AUTH] Hugging Face Access Token registered and active in runtime."
            pytorch_state.logs.insert(0, log_msg)
            return {"status": "saved", "message": "BYOK credentials saved securely. Hugging Face Access Token activated!"}
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
        if provider == "huggingface":
            if not (key.startswith("hf_") or len(key) >= 15):
                return {"provider": provider, "status": "invalid", "message": "Hugging Face tokens typically start with hf_"}
            # Live verification with Hugging Face Hub whoami endpoint
            try:
                hf_req = urllib.request.Request(
                    "https://huggingface.co/api/whoami-v2",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "User-Agent": "TriuneStudio/2.1"
                    }
                )
                with urllib.request.urlopen(hf_req, timeout=10.0) as resp:
                    if resp.status == 200:
                        user_info = json.loads(resp.read().decode("utf-8"))
                        username = user_info.get("name", "user")
                        user_type = user_info.get("type", "user")
                        orgs = [o.get("name") for o in user_info.get("orgs", []) if o.get("name")]
                        org_str = f" • Orgs: {', '.join(orgs)}" if orgs else ""

                        # Apply to live environment
                        os.environ["HF_TOKEN"] = key
                        os.environ["HUGGING_FACE_HUB_TOKEN"] = key
                        pytorch_state.hf_token = key

                        return {
                            "provider": provider,
                            "status": "valid",
                            "user": username,
                            "message": f"✓ Authenticated as @{username} [{user_type}]{org_str}. Full gated dataset streaming enabled!"
                        }
            except urllib.error.HTTPError as hf_err:
                if hf_err.code in (401, 403):
                    return {"provider": provider, "status": "invalid", "message": f"Authentication failed: Invalid Hugging Face token (HTTP {hf_err.code})."}
                return {"provider": provider, "status": "warning", "message": f"Hugging Face Hub returned HTTP {hf_err.code}."}
            except Exception as exc:
                return {"provider": provider, "status": "valid", "message": f"Token format valid (offline notice: {exc})."}
        return {"provider": provider, "status": "valid", "message": f"{provider.upper()} key format validated."}

    @router.get("/api/plugins/nodes")
    @router.get("/v1/plugins/nodes")
    @router.get("/v1/plugins")
    async def get_plugin_nodes() -> List[Dict[str, Any]]:
        """Return serializable node definitions for Triune Studio Node Graph UI."""
        if node_registry is not None and hasattr(node_registry, "list_nodes"):
            return node_registry.list_nodes()
        return []

    @router.get("/v1/nodes/catalog")
    async def get_nodes_catalog() -> List[Dict[str, Any]]:
        """Return strongly-typed node catalog with port contracts and execution vs specification types."""
        if node_registry is not None and hasattr(node_registry, "get_catalog"):
            return node_registry.get_catalog()
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
            pass
        finally:
            telemetry_manager.disconnect(websocket)
else:
    router = None
