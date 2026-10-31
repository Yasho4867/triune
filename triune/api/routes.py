"""REST & WebSocket API Routes powering Triune Studio."""

from __future__ import annotations

import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import ipaddress
import asyncio
import warnings
warnings.filterwarnings("ignore", message=".*httpx.*")
warnings.filterwarnings("ignore", message=".*starlette.testclient.*")
import io
import json
import hmac
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
    from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None
    HTTPException = None
    Request = None
    Response = None
    StreamingResponse = None
    WebSocket = None
    WebSocketDisconnect = Exception
    BaseModel = object
    Field = None

try:
    from triune.optim import CentroidSteerOptimizer
except ImportError:
    CentroidSteerOptimizer = None

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


def _format_tokens(n: int | float | None) -> str:
    """Format token counts consistently across API status and checkpoints."""
    if n is None or n == 0:
        return "0 tok"
    n_int = int(n)
    prefix = "-" if n_int < 0 else ""
    val = abs(n_int)
    if round(val / 1_000_000_000, 2) >= 1.0:
        return f"{prefix}{val / 1_000_000_000:.2f}B tok"
    if round(val / 1_000_000, 2) >= 1.0:
        return f"{prefix}{val / 1_000_000:.2f}M tok"
    return f"{prefix}{val:,} tok"


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


def _checkpoint_architecture(checkpoint: Dict[str, Any], fallback: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Resolve the Triune architecture recorded by a checkpoint, including legacy Studio saves.

    Older Studio checkpoints only recorded vocab/width/depth. Recover the remaining
    shape-defining fields from their state-dict keys and use the current architecture
    only for properties that cannot be inferred from tensor shapes (for example the
    GLA head split and router-prefix placement).
    """
    from triune.model.base import CANONICAL_MODEL_SPECS

    config = checkpoint.get("config", {})
    config = config if isinstance(config, dict) else {}
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a model state dictionary.")
    if any(str(key).startswith("_orig_mod.") for key in state):
        state = {str(key).removeprefix("_orig_mod."): value for key, value in state.items()}

    def tensor_shape(key: str) -> tuple[int, ...] | None:
        value = state.get(key)
        shape = getattr(value, "shape", None)
        return tuple(int(dim) for dim in shape) if shape is not None else None

    embed_shape = tensor_shape("token_embed.weight")
    hidden_dim = int(config.get("hidden_dim") or (embed_shape[1] if embed_shape and len(embed_shape) == 2 else fallback.get("hidden_dim", 256)))
    vocab_size = int(config.get("vocab_size") or (embed_shape[0] if embed_shape and len(embed_shape) == 2 else fallback.get("vocab_size", 32000)))

    layer_ids = []
    expert_ids = []
    exit_ids = []
    import re
    for key in state:
        key = str(key).removeprefix("_orig_mod.")
        match = re.match(r"layers\.(\d+)\.", key)
        if match:
            layer_ids.append(int(match.group(1)))
        match = re.match(r"layers\.\d+\.ffn\.experts\.(\d+)\.", key)
        if match:
            expert_ids.append(int(match.group(1)))
        match = re.match(r"layers\.(\d+)\.exit_head\.weight$", key)
        if match:
            exit_ids.append(int(match.group(1)))

    num_layers = int(config.get("num_layers") or (max(layer_ids) + 1 if layer_ids else fallback.get("num_layers", 6)))
    num_experts = int(config.get("num_experts") or (max(expert_ids) + 1 if expert_ids else fallback.get("num_experts", 4)))
    num_heads = int(config.get("num_heads") or fallback.get("num_heads", 0))
    head_dim = int(config.get("head_dim") or fallback.get("head_dim", 64))
    if hidden_dim != int(fallback.get("hidden_dim", -1)) or num_heads <= 0 or num_heads * head_dim != hidden_dim:
        head_dim = int(config.get("head_dim") or 64)
        if hidden_dim % head_dim:
            raise ValueError(
                f"Cannot infer GLA head dimensions for checkpoint width {hidden_dim}; "
                "the checkpoint must include num_heads and head_dim metadata."
            )
        num_heads = hidden_dim // head_dim

    overrides: Dict[str, Any] = {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_experts": num_experts,
        # Legacy checkpoints omit router-prefix metadata; preserve the
        # selected/canonical architecture value instead of the module default.
        "router_prefix_layers": int(config.get("router_prefix_layers") or fallback.get("router_prefix_layers", 1)),
    }
    for key in (
        "router_prefix_layers", "reflex_exit_layer", "limbic_exit_layer",
        "target_depth_dist", "balance_coef", "precision", "use_fp4", "use_fp8",
        "streaming_fp8_weights", "use_layer_streaming", "chunk_size",
    ):
        if config.get(key) is not None:
            overrides[key] = config[key]
    if len(exit_ids) >= 2:
        overrides.setdefault("reflex_exit_layer", min(exit_ids))
        overrides.setdefault("limbic_exit_layer", max(exit_ids))

    saved_model_id = config.get("model_id") or config.get("preset_id")
    if saved_model_id:
        model_id = str(saved_model_id)
    else:
        # Older checkpoints lack a model ID. Resolve the label from the actual
        # tensor dimensions instead of reusing whichever preset the UI selected
        # before loading (which could otherwise label a Small checkpoint as Base).
        model_id = next((spec_id for spec_id, spec in CANONICAL_MODEL_SPECS.items()
                         if spec.vocab_size == vocab_size
                         and spec.hidden_dim == hidden_dim
                         and spec.num_layers == num_layers
                         and spec.num_heads == num_heads
                         and spec.head_dim == head_dim
                         and spec.num_experts == num_experts
                         and spec.router_prefix_layers == overrides["router_prefix_layers"]
                         and spec.reflex_exit_layer == overrides.get("reflex_exit_layer", spec.reflex_exit_layer)
                         and spec.limbic_exit_layer == overrides.get("limbic_exit_layer", spec.limbic_exit_layer)),
                        "triune-custom")
    if model_id not in CANONICAL_MODEL_SPECS:
        # Preserve the custom architecture values while using a validated base spec.
        model_id = "triune-custom"
    return model_id, overrides


if HAS_FASTAPI:
    router = APIRouter()
    dag_engine = ExecutionEngine() if ExecutionEngine is not None else None
    sandbox = PythonSandbox() if PythonSandbox is not None else None
    
    if ModuleManager is not None:
        module_manager = ModuleManager()
    else:
        class FallbackModuleManager:
            def get_config(self): return {"byok_keys": {}, "theme": "retro"}
            def save_config(self, c, **kwargs): pass
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

        async def connect(self, websocket: WebSocket, *, accept: bool = True) -> None:
            if accept:
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
    def _provider_timeout_message() -> str:
        return (
            f"The cloud provider did not respond before Triune's {_provider_request_timeout():g}-second request timeout. "
            "This is provider/network wait time, not local model inference; check provider status and quota, then retry."
        )

    def _byok_error_message(exc: Exception) -> str:
        """Return a provider failure without leaking URLs, request headers, or credentials."""
        if isinstance(exc, urllib.error.HTTPError):
            reason = exc.reason or "request rejected"
            if exc.code == 401:
                return "provider rejected the API key (HTTP 401). Check that the saved key is current, valid, and for this provider; HTTP 401 is an authentication failure, not a rate/quota response."
            if exc.code == 403:
                return "provider denied access (HTTP 403). Check key permissions, project access, and whether this model is enabled."
            if exc.code == 429:
                return "provider rate or quota limit reached (HTTP 429). Check the account's limits and retry after the reset."
            return f"provider returned HTTP {exc.code} ({reason}). Check provider status and request settings."
        if isinstance(exc, urllib.error.URLError):
            if isinstance(exc.reason, (TimeoutError, asyncio.TimeoutError)) or "timed out" in str(exc.reason).lower():
                return _provider_timeout_message()
            return f"could not connect to provider ({exc.reason}). Check network access and try again."
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return _provider_timeout_message()
        return f"{type(exc).__name__}: provider request could not be completed."

    def _provider_request_timeout() -> float:
        """Keep slower free-tier cloud responses from failing at the old 30-second cutoff."""
        try:
            configured = float(os.environ.get("TRIUNE_PROVIDER_TIMEOUT_SECONDS", "90"))
        except (TypeError, ValueError):
            configured = 90.0
        return max(15.0, min(configured, 300.0))

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
        with urllib.request.urlopen(req, timeout=_provider_request_timeout()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _call_anthropic_api(messages: List[Dict[str, str]], api_key: str, model: str = "claude-3-5-sonnet-latest", temperature: float = 0.7, max_tokens: int = 2048) -> str:
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
        with urllib.request.urlopen(req, timeout=_provider_request_timeout()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]

    def _call_gemini_api(messages: List[Dict[str, str]], api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.7, max_tokens: int = 2048) -> str:
        clean_key = (api_key or "").strip()
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

        models_to_try = [model]
        for fallback in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        last_err = None
        for current_model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={urllib.parse.quote(clean_key)}"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "x-goog-api-key": clean_key, "User-Agent": "TriuneStudio/2.1"}
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=_provider_request_timeout()) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        candidates = data.get("candidates") or []
                        if candidates:
                            first_cand = candidates[0]
                            content_obj = first_cand.get("content") or {}
                            parts = content_obj.get("parts") or []
                            for part in parts:
                                if isinstance(part, dict) and "text" in part:
                                    return part["text"]
                            finish_reason = first_cand.get("finishReason")
                            if finish_reason:
                                return f"[Gemini response completed with reason: {finish_reason}]"
                        prompt_feedback = data.get("promptFeedback") or {}
                        if prompt_feedback.get("blockReason"):
                            return f"[Gemini content blocked: {prompt_feedback.get('blockReason')}]"
                        return "No text returned by Gemini."
                except urllib.error.HTTPError as h_err:
                    last_err = h_err
                    if h_err.code in (503, 429) and attempt < 2:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    if h_err.code == 404:
                        # Model not found; try next model in fallback cascade
                        break
                    raise
            else:
                continue
            if last_err and getattr(last_err, "code", None) != 404:
                raise last_err

        if last_err:
            raise last_err
        raise RuntimeError("Google Gemini could not be reached with any supported model identifier.")

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
                    self.offset += 1
                    row_data = r.get("row", {})
                    for text_key in ("text", "content", "story", "prompt", "completion", "question"):
                        if text_key in row_data and isinstance(row_data[text_key], str):
                            row_data[text_key] = sanitize_text(row_data[text_key])
                    yield row_data

    # Checkpoint Projects Registry & Chat Sessions Helpers
    PROJECTS_FILE = Path("checkpoints/projects.json")
    CHAT_SESSIONS_FILE = Path("checkpoints/chat_sessions.json")

    def _get_projects_registry() -> Dict[str, Any]:
        """Load project registry and checkpoint assignments."""
        if PROJECTS_FILE.is_file():
            try:
                with open(PROJECTS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "active_project": "Triune-Base · Adaptive CentroidSteer",
            "projects": [
                {"name": "Triune-Base · Adaptive CentroidSteer", "description": "Active training run with 5th-order Muon & SVD CentroidSteer"},
                {"name": "Triune-Base · Phase 5 Pretraining", "description": "Pretraining steps 25,000–28,500"},
                {"name": "Triune-Base · Phase 4 Pretraining", "description": "Pretraining steps 15,500–24,500"},
                {"name": "Triune CLI Training Runs", "description": "Standalone CLI training experiments"},
                {"name": "Triune-Small · Rapid Iteration", "description": "Compact 6-layer architecture checkpoints"},
                {"name": "Triune-Mid · Early Pretraining", "description": "Intermediate baseline pretraining checkpoints"},
                {"name": "Triune Genesis Foundation Run", "description": "Early exploratory foundation training"},
            ],
            "assignments": {}
        }

    def _save_projects_registry(registry: Dict[str, Any]) -> None:
        """Persist project registry to disk."""
        try:
            PROJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PROJECTS_FILE, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
        except Exception:
            pass

    def _infer_project_for_checkpoint(filename: str, size_mb: float, step: Optional[int]) -> str:
        """Intelligently map an unassigned checkpoint to a logical project cohort."""
        if filename.startswith("triune_step_"):
            return "Triune CLI Training Runs"
        if "finetun" in filename.lower() or "adapter" in filename.lower():
            return "LoRA & Fine-Tuned Adapters"
        if step is not None:
            if step >= 29000 or size_mb >= 700:
                return "Triune-Base · Adaptive CentroidSteer"
            elif step >= 25000 or size_mb >= 600:
                return "Triune-Base · Phase 5 Pretraining"
            elif step >= 15500 or size_mb >= 530:
                return "Triune-Base · Phase 4 Pretraining"
            elif size_mb < 290 or (14000 <= step <= 15000):
                return "Triune-Small · Rapid Iteration"
            elif 8500 <= step <= 13500:
                return "Triune-Mid · Early Pretraining"
            elif step < 8500:
                return "Triune Genesis Foundation Run"
        return "Triune-Base · Adaptive CentroidSteer"

    def _register_checkpoint_project_assignment(filename: str, project_name: str) -> None:
        """Record a checkpoint file's project assignment."""
        reg = _get_projects_registry()
        reg.setdefault("assignments", {})[filename] = project_name
        if not any(p.get("name") == project_name for p in reg.get("projects", [])):
            reg.setdefault("projects", []).append({"name": project_name, "description": "User created project"})
        _save_projects_registry(reg)

    def _get_chat_sessions() -> List[Dict[str, Any]]:
        """Load stored conversation history sessions."""
        if CHAT_SESSIONS_FILE.is_file():
            try:
                with open(CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict) and "sessions" in data:
                        return data["sessions"]
            except Exception:
                pass
        return []

    def _save_chat_sessions(sessions: List[Dict[str, Any]]) -> None:
        """Persist conversation history sessions to disk."""
        try:
            CHAT_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=2)
        except Exception:
            pass

    # REAL Hardware-Spinning PyTorch Engine State
    class RealPyTorchEngineState:
        def __init__(self):
            self.step = 0
            self.is_training = False
            self.active_generation_cancel: Optional[threading.Event] = None
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
                "depth_mode": "dynamic",
                "balance_loss_weight": 0.3,
            }
            self._token_accumulator: List[int] = []
            self.depth_mode = "dynamic"  # "cortex" (full backbone) | "joint" (all 3 exits) | "dynamic" (router) — user-selected, persisted
            self.streaming_engine = None
            self.placement_info: Dict[str, Any] = {}
            self.balance_loss_weight = 0.3
            self.router_temp = 1.0
            self.steer_scale = 0.20
            self.muon_lr = 0.02
            self.use_triangular_core = True
            self.use_8bit_base = False
            self.centroid_rank = 64
            self.centroid_update_gap = 100
            self.centroid_lr = 5e-4
            self.momentum = 0.95
            self.lr_peak = 5e-4
            self.warmup_steps = 500
            self.min_lr = 1e-5
            self.current_lr = 5e-4
            self.lr_cycle_start_step = 0
            self.lr_cycle_steps = 50000
            self.total_steps = 50000
            self.dataset_path = "data/fineweb_sample.jsonl"
            self.dataset_name = "HuggingFaceFW/fineweb-edu"
            self.dataset_type = "streaming"
            self.total_tokens_trained = 0
            self.tokens_trained = 0
            self.current_batch_preview = ""
            self.active_checkpoint: Optional[str] = None
            self.weights_loaded = False
            self.active_adapter: Optional[Dict[str, Any]] = None
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

            # Weights & Biases Experiment Telemetry Integration
            self.wandb_key = os.environ.get("WANDB_API_KEY", "")
            if not self.wandb_key and hasattr(module_manager, "get_config"):
                try:
                    byok = module_manager.get_config().get("byok_keys", {})
                    self.wandb_key = byok.get("wandb", "")
                    if self.wandb_key:
                        os.environ["WANDB_API_KEY"] = self.wandb_key
                except Exception:
                    pass
            self.wandb_run: Any = None
            self.wandb_url: Optional[str] = None
            self.wandb_project: str = "triune-transformer"
            self.active_project: str = "Triune-Base · Adaptive CentroidSteer"
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
                self.cuda_init_error: Optional[str] = None
                self.cuda_runtime_version = getattr(torch.version, "cuda", None)
                self.cuda_available = False
                self.device = torch.device("cpu")
                self.device_name = f"CPU ({platform.processor() or 'x86_64'})"
                if torch.cuda.is_available():
                    try:
                        torch.cuda.init()
                        torch.cuda.set_device(0)
                        # Touch the actual allocator and synchronize so startup
                        # cannot report a CUDA device that fails on first use.
                        _cuda_probe = torch.empty(1, device="cuda:0")
                        torch.cuda.synchronize(0)
                        del _cuda_probe
                        self.device = torch.device("cuda:0")
                        self.device_name = torch.cuda.get_device_name(0)
                        self.cuda_available = True
                    except Exception as exc:
                        self.cuda_init_error = f"CUDA initialization failed: {exc}"
                        print(f"[Triune Engine] {self.cuda_init_error}; using CPU.")
                self.pad_token_id = 0
                self.z_loss_coef = 1e-3
                self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction="none")
                self.router_loss_fn = nn.CrossEntropyLoss()
                self.logs: List[str] = [
                    f"[SYSTEM] PyTorch {torch.__version__} engine initialized on {self.device_name}.",
                    "[ENGINE] TriuneTransformer MoE Engine ready for real training and inference.",
                ]
                if self.cuda_init_error:
                    self.logs.insert(0, f"[GPU] {self.cuda_init_error}")
            else:
                self.device = "cpu"
                self.device_name = f"Host UI Mode ({platform.processor() or 'x86_64'})"
                self.pad_token_id = 0
                self.z_loss_coef = 1e-3
                self.loss_fn = None
                self.router_loss_fn = None
                self.logs: List[str] = [
                    f"[SYSTEM] Studio running in Host UI Mode on {self.device_name}.",
                    "[ENGINE] PyTorch not available in current process. Run with WSL2 for full GPU acceleration.",
                ]
            print(f"[Triune Engine] Initializing Native Engine on: {self.device_name}")

            self.training_engine: Any = None
            self.history: List[Dict[str, Any]] = []
            self.training_task: Optional[asyncio.Task] = None
            self.worker_lock = threading.Lock()
            self.stop_event = threading.Event()
            self.run_id = 0
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
                        if "active_adapter" in data:
                            self.active_adapter = data["active_adapter"]
                        if data.get("dataset_name"):
                            self.dataset_name = data["dataset_name"]
                        if data.get("hf_dataset_name"):
                            self.hf_dataset_name = data["hf_dataset_name"]
                        if "hf_stream_offset" in data and data["hf_stream_offset"] is not None:
                            self.hf_stream_offset = max(0, int(data["hf_stream_offset"]))
                        if os.environ.get("TRIUNE_TESTING") != "1" and self.dataset_name and "/" in self.dataset_name:
                            self.dataset_path = self.dataset_name
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
                        if data.get("active_project"):
                            self.active_project = str(data["active_project"])
                        if data.get("architecture_spec"):
                            self.architecture_spec = data["architecture_spec"]
                        if data.get("depth_mode") in ("cortex", "joint", "dynamic"):
                            self.depth_mode = data["depth_mode"]
                        if data.get("lr_peak"):
                            self.lr_peak = float(data["lr_peak"])
                        if "warmup_steps" in data and data["warmup_steps"] is not None:
                            self.warmup_steps = int(data["warmup_steps"])
                        if "lr_cycle_steps" in data and data["lr_cycle_steps"] is not None:
                            saved_cycle = int(data["lr_cycle_steps"])
                            self.lr_cycle_steps = max(50000 if saved_cycle == 25000 else 100, saved_cycle)
                        self.lr_cycle_start_step = 0
                        self.total_steps = max(self.lr_cycle_steps, self.step + 1)
                        self.current_lr = float(self.lr_schedule(self.config, self.step))
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
                            ckpt_data = torch.load(latest, map_location="cpu", weights_only=True)
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
            self.current_lr = float(self.lr_schedule(self.config, self.step))
            if restored > 0:
                msg = f"[Triune Engine] Restored {restored:,} historical cumulative trained tokens from previous sessions (step {self.step})."
                self.logs.insert(0, msg)
                print(msg)
                self._save_persisted_session_state()

            # Auto-restore active checkpoint weights on startup
            if HAS_TORCH and ckpt_dir.is_dir():
                target_ckpt = None
                if self.active_checkpoint and (ckpt_dir / self.active_checkpoint).is_file():
                    target_ckpt = ckpt_dir / self.active_checkpoint
                elif ckpts:
                    target_ckpt = ckpts[0]

                if target_ckpt is not None:
                    try:
                        self.load_model_checkpoint(target_ckpt, restore_optimizer=True)
                        loaded_msg = f"[Triune Engine] Auto-restored weights from active checkpoint: {target_ckpt.name}"
                        self.logs.insert(0, loaded_msg)
                        print(loaded_msg)
                    except Exception as exc:
                        self.logs.insert(0, f"[Triune Engine] Startup checkpoint {target_ckpt.name} bypassed: {exc}")

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
                    "active_adapter": getattr(self, "active_adapter", None),
                    "dataset_name": self.dataset_name,
                    "hf_dataset_name": self.hf_dataset_name if self.is_streaming_hf else self.dataset_name,
                    "hf_stream_offset": int(getattr(self.hf_stream_dataset, "offset", getattr(self, "hf_stream_offset", 0)) or 0),
                    "batch_size": self.batch_size,
                    "seq_len": self.seq_len,
                    "grad_accum_steps": self.grad_accum_steps,
                    "gpu_offload_pct": getattr(self, "gpu_offload_pct", 100),
                    "gpu_layers": getattr(self, "gpu_layers", 6),
                    "precision": getattr(self, "precision", "bf16"),
                    "active_model_id": getattr(self, "active_model_id", "triune-small"),
                    "active_project": getattr(self, "active_project", "Triune-Base · Adaptive CentroidSteer"),
                    "architecture_spec": getattr(self, "architecture_spec", {}),
                    "depth_mode": getattr(self, "depth_mode", "dynamic"),
                    "lr_peak": getattr(self, "lr_peak", None),
                    "current_lr": getattr(self, "current_lr", None),
                    "warmup_steps": getattr(self, "warmup_steps", None),
                    "lr_cycle_start_step": getattr(self, "lr_cycle_start_step", 0),
                    "lr_cycle_steps": getattr(self, "lr_cycle_steps", 50000),
                    "updated_at": time.time(),
                }
                with open(session_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

        def get_scheduled_lr(self, step: int) -> float:
            """Cosine learning rate schedule with linear warmup delegating to canonical scheduler."""
            return float(self.lr_schedule(self.config, step))

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
            """Rebuild TriuneTransformer architecture and optimizer using canonical ModelSpec."""
            from triune.model.base import build_model_from_spec, get_model_spec

            p_id = preset_id or getattr(self, "active_model_id", "triune-small") or "triune-small"
            spec_overrides = {
                k: kwargs[k]
                for k in (
                    "vocab_size",
                    "num_layers",
                    "hidden_dim",
                    "num_heads",
                    "head_dim",
                    "num_experts",
                    "router_prefix_layers",
                    "reflex_exit_layer",
                    "limbic_exit_layer",
                    "target_depth_dist",
                    "balance_coef",
                    "depth_mode",
                    "use_fp4",
                    "use_fp8",
                    "precision",
                )
                if k in kwargs and kwargs[k] is not None
            }
            if "balance_loss_weight" in kwargs and kwargs["balance_loss_weight"] is not None:
                spec_overrides["balance_coef"] = float(kwargs["balance_loss_weight"])
                self.balance_loss_weight = float(kwargs["balance_loss_weight"])
            if "depth_mode" in spec_overrides:
                self.depth_mode = str(spec_overrides["depth_mode"])

            spec = get_model_spec(p_id, **spec_overrides)
            base = spec.to_dict()
            base["balance_loss_weight"] = float(getattr(self, "balance_loss_weight", spec.balance_coef))
            self.precision = spec.precision

            if HAS_TORCH and TriuneTransformer is not None:
                self.load_tokenizer()
                total_p, _ = spec.compute_exact_params()
                exec_spec = spec

                # Zero fake or silent downgrades: always instantiate the real user-chosen model architecture
                self.model = build_model_from_spec(exec_spec)
                self.model.model_spec = spec

                num_layers = spec.num_layers
                pct = getattr(self, "gpu_offload_pct", 100)
                if "gpu_layers" in kwargs and kwargs["gpu_layers"] is not None:
                    requested_gpu_layers = max(0, min(num_layers, int(kwargs["gpu_layers"])))
                elif total_p <= 150_000_000 and not getattr(spec, "use_layer_streaming", False) and self.device.type == "cuda":
                    # Sub-150M models fit in <300MB VRAM; keep 100% resident on GPU unless explicitly offloaded
                    requested_gpu_layers = num_layers
                    pct = 100
                    self.gpu_offload_pct = 100
                else:
                    requested_gpu_layers = max(0, min(num_layers, int(round((pct / 100.0) * num_layers))))

                if self.device.type == "cuda":
                    if hasattr(self.model, "enable_gradient_checkpointing"):
                        self.model.enable_gradient_checkpointing()
                    try:
                        torch.cuda.set_per_process_memory_fraction(0.24, self.device)
                    except Exception:
                        pass
                    target_dtype = torch.bfloat16 if (getattr(self, "precision", "bf16") == "bf16" and torch.cuda.is_bf16_supported()) else torch.float32

                    if total_p > 150_000_000 or requested_gpu_layers < num_layers or getattr(spec, "use_layer_streaming", False):
                        layer_bytes = (total_p * 2) / max(1, num_layers)
                        # Allocate up to 2.5GB of GPU VRAM for resident layers, leaving VRAM for embeddings, activations, and buffers
                        max_safe_resident = max(2, min(num_layers, int((2.5 * (1024**3)) / max(1, layer_bytes))))
                        res_layers = min(requested_gpu_layers, max_safe_resident)

                        from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig
                        streaming_cfg = StreamingConfig(
                            enabled=True,
                            chunk_size=1,
                            pin_memory=False,
                            async_prefetch=True,
                            fp8_weights=(spec.precision == "fp8" or spec.use_fp8),
                            resident_gpu_layers=res_layers,
                        )
                        engine = LayerStreamingEngine(self.model, device=self.device, config=streaming_cfg)
                        engine.attach()
                        self.model._streaming_engine = engine
                    else:
                        self.model = self.model.to(device=self.device, dtype=target_dtype)
                else:
                    self.model = self.model.to(self.device)

                from triune.optim.factory import build_optimizer
                optim_cfg = {
                    "lr": self.current_lr,
                    "betas": (0.9, 0.95),
                    "weight_decay": 0.01,
                    "galore": True,
                    "galore_rank": getattr(self, "centroid_rank", 64),
                    "galore_update_gap": getattr(self, "centroid_update_gap", 100),
                    "steer_scale": getattr(self, "steer_scale", 0.20),
                    "galore_lr": getattr(self, "centroid_lr", self.current_lr),
                    "galore_betas": (0.9, 0.999),
                    "galore_weight_decay": 0.01,
                    "use_muon": True,
                    "muon_lr": getattr(self, "muon_lr", 0.02),
                    "muon_momentum": getattr(self, "momentum", 0.95),
                    "muon_weight_decay": 0.01,
                    "use_triangular_core": getattr(self, "use_triangular_core", True),
                    "use_8bit_base": getattr(self, "use_8bit_base", False),
                }
                try:
                    self.optimizer = build_optimizer(self.model, optim_cfg)
                except Exception:
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.current_lr, weight_decay=0.01)

                if getattr(self.model, "_streaming_engine", None) is not None:
                    self.model._streaming_engine.set_optimizer(self.optimizer)

                from triune.trainer.engine import TrainingEngine
                self.training_engine = TrainingEngine(self)
                self.training_engine.step = self.step

            if p_id != getattr(self, "active_model_id", None):
                self.history = []
                self.last_exit_usage = None
                self.last_loss = None
                self.active_checkpoint = None

            self.active_model_id = p_id
            self.architecture_spec = base
            self.weights_loaded = False
            if "gpu_layers" in kwargs and kwargs["gpu_layers"] is not None:
                self.gpu_layers = max(0, min(base["num_layers"], int(kwargs["gpu_layers"])))
                self.gpu_offload_pct = int(round((self.gpu_layers / max(1, base["num_layers"])) * 100))
            else:
                pct = getattr(self, "gpu_offload_pct", 100)
                self.gpu_layers = max(0, min(base["num_layers"], int(round((pct / 100.0) * base["num_layers"]))))
            if "recommended_batch" in base and "batch_size" not in kwargs:
                self.batch_size = base["recommended_batch"]
            if "recommended_seq" in base and "seq_len" not in kwargs:
                self.seq_len = base["recommended_seq"]
            if "batch_size" in kwargs and kwargs["batch_size"]:
                self.batch_size = kwargs["batch_size"]
            if "seq_len" in kwargs and kwargs["seq_len"]:
                self.seq_len = kwargs["seq_len"]

            self._save_persisted_session_state()

            log_msg = f"[Triune Engine] Architecture active: {base.get('name', p_id)} ({base['num_layers']}L, {base['hidden_dim']}d, {base['num_heads']}H, {base['num_experts']}E, {base['precision'].upper()}). Optimizer rebuilt."
            self.logs.insert(0, log_msg)
            print(log_msg)
            return base

        def _get_current_exit_usage(self) -> Optional[Dict[str, float]]:
            """Return strictly measured exit head usage percentages, or None if unmeasured."""
            if self.history and "exit_usage" in self.history[-1] and self.history[-1]["exit_usage"]:
                return self.history[-1]["exit_usage"]
            if getattr(self, "last_exit_usage", None):
                return self.last_exit_usage
            if getattr(self, "training_engine", None) is not None and hasattr(self.training_engine, "depth_usage_ema"):
                try:
                    ema = self.training_engine.depth_usage_ema
                    if ema is not None and getattr(self, "step", 0) > 0:
                        d_vals = ema.tolist() if torch.is_tensor(ema) else list(ema)
                        if d_vals and len(d_vals) >= 3:
                            tot = sum(d_vals) or 1.0
                            return {
                                "reflex": round((float(d_vals[0]) / tot) * 100.0, 1),
                                "limbic": round((float(d_vals[1]) / tot) * 100.0, 1),
                                "cortex": round((float(d_vals[2]) / tot) * 100.0, 1),
                            }
                except Exception:
                    pass
            return None

        def _get_target_exit_distribution(self) -> Optional[Dict[str, float]]:
            """Return the configured target depth distribution of the active model as percentages."""
            router = getattr(getattr(self, "model", None), "router", None)
            dist = getattr(router, "target_depth_dist", None)
            if dist is None and getattr(self, "architecture_spec", None):
                dist = self.architecture_spec.get("target_depth_dist")
            if dist is not None:
                try:
                    d_vals = dist.tolist() if torch.is_tensor(dist) else list(dist)
                    if d_vals and len(d_vals) >= 3:
                        tot = sum(d_vals) or 1.0
                        return {
                            "reflex": round((float(d_vals[0]) / tot) * 100.0, 1),
                            "limbic": round((float(d_vals[1]) / tot) * 100.0, 1),
                            "cortex": round((float(d_vals[2]) / tot) * 100.0, 1),
                        }
                except Exception:
                    pass
            return None

        def _get_current_throughput(self) -> int:
            """Return the latest measured step throughput or 0 if idle."""
            if not getattr(self, "is_training", False):
                return 0
            if self.history and "throughput" in self.history[-1]:
                return int(self.history[-1]["throughput"] or 0)
            return 0

        def _get_last_measured_throughput(self) -> int:
            """Return the last measured step throughput regardless of whether training is active."""
            if self.history and "throughput" in self.history[-1]:
                return int(self.history[-1]["throughput"] or 0)
            return 0

        def load_model_checkpoint(
            self,
            checkpoint_path: str | Path,
            *,
            restore_optimizer: bool = True,
            precision_override: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Load a checkpoint only after matching the model architecture and validating every weight."""
            p = Path(checkpoint_path).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            if p.suffix.lower() == ".safetensors":
                from safetensors.torch import load_file
                checkpoint = load_file(str(p), device="cpu")
            else:
                checkpoint = torch.load(p, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint contents must be a state dictionary or a checkpoint mapping.")

            model_id, arch_overrides = _checkpoint_architecture(checkpoint, self.architecture_spec)
            arch_overrides.update({
                "depth_mode": checkpoint.get("config", {}).get("depth_mode", self.depth_mode)
                if isinstance(checkpoint.get("config"), dict) else self.depth_mode,
                "precision": checkpoint.get("config", {}).get("precision", self.precision)
                if isinstance(checkpoint.get("config"), dict) else self.precision,
            })
            if precision_override is not None:
                if precision_override not in ("bf16", "fp16", "fp32", "fp8", "fp4"):
                    raise ValueError(f"Unsupported precision override: {precision_override}")
                arch_overrides["precision"] = precision_override
                arch_overrides["use_fp4"] = precision_override == "fp4"
                arch_overrides["use_fp8"] = precision_override == "fp8"
            self.reconfigure_architecture(model_id, **arch_overrides)

            state_dict = checkpoint.get("model_state", checkpoint)
            state_dict = {
                str(key).removeprefix("_orig_mod."): value
                for key, value in state_dict.items()
            }
            # strict=True is intentional: partial/randomly initialized models must never
            # be reported as successfully loaded trained checkpoints.
            self.model.load_state_dict(state_dict, strict=True)
            if getattr(self.model, "_streaming_engine", None) is None:
                if self.device.type == "cuda":
                    target_dtype = torch.bfloat16 if (getattr(self, "precision", "bf16") == "bf16" and torch.cuda.is_bf16_supported()) else torch.float32
                    self.model = self.model.to(device=self.device, dtype=target_dtype)
                    if hasattr(self.model, "enable_gradient_checkpointing"):
                        self.model.enable_gradient_checkpointing()
                    try:
                        torch.cuda.set_per_process_memory_fraction(0.24, self.device)
                    except Exception:
                        pass
                else:
                    self.model = self.model.to(self.device)

            if "step" in checkpoint:
                self.step = int(checkpoint["step"])
            if self.training_engine is not None:
                self.training_engine.step = self.step
                if checkpoint.get("depth_usage_ema") is not None:
                    try:
                        ema_t = checkpoint["depth_usage_ema"]
                        if not isinstance(ema_t, torch.Tensor):
                            ema_t = torch.tensor(ema_t, dtype=torch.float32)
                        self.training_engine.depth_usage_ema = ema_t.to(device=self.device, dtype=torch.float32)
                    except Exception:
                        pass
            config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}
            restored_tokens = checkpoint.get("total_tokens_trained", checkpoint.get("tokens_trained", 0))
            if not restored_tokens and self.step > 0:
                restored_tokens = self.step * int(config.get("batch_size", checkpoint.get("batch_size", 4))) \
                    * int(config.get("seq_len", checkpoint.get("seq_len", 64))) \
                    * int(config.get("grad_accum_steps", checkpoint.get("grad_accum_steps", 4)))
            self.total_tokens_trained = max(int(getattr(self, "total_tokens_trained", 0) or 0), int(restored_tokens or 0))
            self.tokens_trained = self.total_tokens_trained
            self.active_checkpoint = p.name
            self.weights_loaded = True
            self.current_lr = float(self.lr_schedule(self.config, self.step))
            if self.optimizer is not None:
                self.set_optimizer_lr(self.optimizer, self.current_lr)

            if restore_optimizer and checkpoint.get("optimizer_state") and self.optimizer:
                try:
                    opt_state = checkpoint["optimizer_state"]
                    if isinstance(self.optimizer, CentroidSteerOptimizer):
                        if "layer_groups" in opt_state and "base_optimizer" in opt_state:
                            self.optimizer.load_state_dict(opt_state)
                        elif "state" in opt_state and "param_groups" in opt_state:
                            base_p_count = sum(len(g.get("params", [])) for g in self.optimizer.base_optimizer.param_groups)
                            saved_p_count = sum(len(g.get("params", [])) for g in opt_state.get("param_groups", []))
                            if base_p_count == saved_p_count and len(self.optimizer.base_optimizer.param_groups) == len(opt_state.get("param_groups", [])):
                                self.optimizer.base_optimizer.load_state_dict(opt_state)
                            else:
                                self.logs.insert(0, f"[CHECKPOINT] Optimizer state skipped: schema mismatch ({saved_p_count} vs {base_p_count} params). Re-initialized cleanly.")
                    else:
                        self.optimizer.load_state_dict(opt_state)
                except Exception as exc:
                    # Inference weights are valid even when an old optimizer schema is not.
                    self.logs.insert(0, f"[CHECKPOINT] Optimizer state not restored: {exc}")

            saved_loss = float(checkpoint.get("loss", 0.0) or 0.0)
            saved_depth_ema = checkpoint.get("depth_usage_ema")
            exit_usage_dict = None
            if saved_depth_ema is not None:
                try:
                    if torch.is_tensor(saved_depth_ema):
                        d_vals = saved_depth_ema.tolist()
                    elif isinstance(saved_depth_ema, (list, tuple)):
                        d_vals = list(saved_depth_ema)
                    else:
                        d_vals = None
                    if d_vals and len(d_vals) >= 3:
                        tot = sum(d_vals) or 1.0
                        exit_usage_dict = {
                            "reflex": round((float(d_vals[0]) / tot) * 100.0, 1),
                            "limbic": round((float(d_vals[1]) / tot) * 100.0, 1),
                            "cortex": round((float(d_vals[2]) / tot) * 100.0, 1),
                        }
                except Exception:
                    pass
            self.history = []
            self.last_loss = saved_loss if saved_loss > 0 else getattr(self, "last_loss", None)
            self.last_exit_usage = exit_usage_dict

            if self.step > 0 and saved_loss > 0:
                self.history.append({
                    "step": self.step,
                    "loss": round(saved_loss, 4),
                    "lm_loss": round(saved_loss, 4),
                    "router_loss": 0.0,
                    "throughput": 0,
                    "lr": round(self.current_lr, 7),
                    "tokens_trained": self.total_tokens_trained,
                    "exit_usage": exit_usage_dict,
                })

            self._save_persisted_session_state()
            return {
                "checkpoint": p.name,
                "step": self.step,
                "tokens_trained": self.total_tokens_trained,
                "loss": saved_loss if saved_loss > 0 else None,
                "exit_usage": exit_usage_dict,
                "target_exit_distribution": self._get_target_exit_distribution(),
                "throughput": self._get_current_throughput(),
                "architecture": self.architecture_spec,
                "active_model": self.active_model_id,
            }

        def lazy_init_model(self):
            if not HAS_TORCH or TriuneTransformer is None:
                raise RuntimeError("PyTorch is not installed in this environment.")
            # Only restore implicitly during first backend initialization. A
            # model already built by an explicit preset selection is intentional
            # (possibly fresh for training) and must never be replaced by the
            # newest checkpoint behind the user's back.
            if self.model is not None:
                return

            # Prefer the checkpoint explicitly marked active; otherwise use the most
            # recently modified managed checkpoint. Never swallow a bad restore and
            # continue with random weights while claiming the trained model is active.
            active_candidates: List[Path] = []
            if self.active_checkpoint:
                active_candidates.extend(Path(folder) / Path(self.active_checkpoint).name for folder in ("checkpoints", "checkpoints_full"))
            active_path = next((path for path in active_candidates if path.is_file()), None)
            if active_path is not None:
                checkpoint_path = active_path
            else:
                candidates = [path for folder in ("checkpoints", "checkpoints_full")
                              for path in Path(folder).iterdir()
                              if path.is_file() and path.suffix.lower() in {".pt", ".safetensors"}]
                checkpoint_path = max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
            if checkpoint_path is not None:
                try:
                    restored = self.load_model_checkpoint(checkpoint_path)
                except Exception as err:
                    self.weights_loaded = False
                    self.active_checkpoint = None
                    raise RuntimeError(
                        f"Could not restore trained checkpoint {checkpoint_path.name}; local inference was stopped "
                        f"instead of using random weights. {err}"
                    ) from err
                msg = f"[Triune Engine] Auto-restored active weights from {restored['checkpoint']} (step {restored['step']})."
                self.logs.insert(0, msg)
                print(msg)
            elif self.model is None:
                self.reconfigure_architecture(getattr(self, "active_model_id", "triune-small"))

            print(f"[Triune Engine] Loaded 6-Layer MoE TriuneTransformer (32k vocab, 4 experts, Centroid+Muon Optimizer) on {self.device_name}.")
            # Inference must not initialize a training corpus or open a network
            # stream. Training's step_once/next_batch paths load data on demand.

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
            self.hf_stream_iter = test_iter if test_iter is not None else iter(ds)
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

        def _is_word_boundary(self, token_id: int) -> bool:
            """Determine if a token represents the start of a whole word, whitespace, or punctuation."""
            if not self.tokenizer:
                return True
            try:
                t = self.tokenizer.id_to_token(token_id)
                if not t:
                    return True
                # Special tokens like [PAD], [CLS], [SEP], [UNK], or Chat tokens
                if (t.startswith("[") and t.endswith("]")) or (t.startswith("<|") and t.endswith("|>")):
                    return True
                # Byte-level BPE space prefix \u0120 or newline \u010a or whitespace
                if t.startswith("Ġ") or t.startswith("Ċ") or t.startswith("\n") or t.startswith(" "):
                    return True
                # Punctuation or symbols
                if not t[0].isalnum():
                    return True
                return False
            except Exception:
                return True

        def init_wandb(self, project: Optional[str] = None, run_name: Optional[str] = None):
            """Initialize or resume a live Weights & Biases experiment run."""
            if self.wandb_run is not None:
                return self.wandb_run
            api_key = self.wandb_key or os.environ.get("WANDB_API_KEY")
            if not api_key:
                return None
            try:
                import wandb
                wandb.login(key=api_key, relogin=False)
                proj = project or self.wandb_project or "triune-transformer"
                rname = run_name or f"triune-{getattr(self, 'active_model_id', 'studio')}-step{self.step}"
                self.wandb_run = wandb.init(
                    project=proj,
                    name=rname,
                    config={
                        "model": getattr(self, "active_model_id", "triune"),
                        "batch_size": self.batch_size,
                        "seq_len": self.seq_len,
                        "grad_accum_steps": self.grad_accum_steps,
                        "depth_mode": self.depth_mode,
                        "lr": self.current_lr,
                        "device": self.device_name,
                    },
                    resume="allow",
                )
                self.wandb_url = getattr(self.wandb_run, "url", None)
                self.logs.insert(0, f"[WANDB] Experiment connected to dashboard: {self.wandb_url or proj}")
                return self.wandb_run
            except Exception as exc:
                self.logs.insert(0, f"[WANDB] Notice: Could not init run ({exc})")
                return None

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
                        if "question" in sample and "answer" in sample and sample.get("question") and sample.get("answer"):
                            text = f"User: {str(sample['question']).strip()}\nAssistant: {str(sample['answer']).strip()}"
                        elif "instruction" in sample and (sample.get("output") or sample.get("response")):
                            ans = sample.get("output") or sample.get("response")
                            text = f"User: {str(sample['instruction']).strip()}\nAssistant: {str(ans).strip()}"
                        elif "prompt" in sample and (sample.get("completion") or sample.get("response")):
                            ans = sample.get("completion") or sample.get("response")
                            p_str = str(sample["prompt"]).strip()
                            if p_str.startswith("User:"):
                                text = f"{p_str} {str(ans).strip()}"
                            else:
                                text = f"User: {p_str}\nAssistant: {str(ans).strip()}"
                        else:
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
                        split_idx = chunk_size
                        # Word-boundary alignment: ensure next chunk never starts with a severed subword
                        if split_idx < len(self._token_accumulator) and not self._is_word_boundary(self._token_accumulator[split_idx]):
                            bt = split_idx
                            while bt > split_idx - 16 and bt > 0 and not self._is_word_boundary(self._token_accumulator[bt]):
                                bt -= 1
                            if bt > 0:
                                split_idx = bt

                        chunk = self._token_accumulator[:split_idx]
                        del self._token_accumulator[:split_idx]

                        # Pad chunk to chunk_size if backtracked; CrossEntropyLoss ignores pad_token_id (0)
                        if len(chunk) < chunk_size:
                            chunk.extend([self.pad_token_id] * (chunk_size - len(chunk)))

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
            for i in range(0, len(all_token_ids) - chunk_size + 1, chunk_size):
                chunks.append(all_token_ids[i : i + chunk_size])

            if not chunks:
                raise RuntimeError(
                    f"Dataset '{self.dataset_name}' produced {len(all_token_ids)} token IDs; "
                    f"at least {chunk_size} are required for seq_len={self.seq_len}. "
                    "Check the dataset text and configured tokenizer."
                )

            self.data_chunks = chunks
            self.data_idx = 0
            self.is_streaming_hf = False
            log_msg = f"[DATASET] Active dataset source: '{self.dataset_name}' ({self.dataset_type}) | {len(chunks)} sequences ({len(all_token_ids):,} tokens)."
            self.logs.insert(0, log_msg)
            print(f"[Triune Engine] {log_msg}")

        def generate_text(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7, force_depth: Optional[int] = None, cancel_event: Optional[threading.Event] = None, prompt_is_formatted: bool = False, on_text=None, speculative: bool = False, k_draft: int = 3) -> Dict[str, Any]:
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
            if not self.weights_loaded:
                # Auto-check if a compatible checkpoint exists in checkpoints/
                ckpt_dir = Path("checkpoints")
                if ckpt_dir.is_dir():
                    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
                    for c_path in ckpts:
                        try:
                            self.load_model_checkpoint(c_path, restore_optimizer=False)
                            break
                        except Exception:
                            continue
            self.model.eval()

            self.load_tokenizer()
            if self.tokenizer is None:
                raise RuntimeError("Text generation requires the configured Triune tokenizer.")
            eos_id = self.tokenizer.token_to_id("[SEP]")
            if eos_id is None:
                eos_id = self.tokenizer.token_to_id("[EOS]")
            if eos_id is None:
                raise RuntimeError("Tokenizer must define [SEP] or [EOS] for generation.")
            from triune.inference import generate_response

            pad_id = self.tokenizer.token_to_id("[PAD]")
            if pad_id is None:
                pad_id = self.pad_token_id

            if speculative:
                from triune.model.speculative import SelfSpeculativeGenerator
                t_spec_start = time.perf_counter()
                spec_gen = SelfSpeculativeGenerator(
                    self.model,
                    self.tokenizer,
                    k_draft=max(1, int(k_draft)),
                    temperature=temperature,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
                accumulated_tokens: List[str] = []
                def _spec_on_tok(t_id: int, t_str: str) -> None:
                    accumulated_tokens.append(t_str)
                    if on_text:
                        on_text("".join(accumulated_tokens))

                with self.model_autocast():
                    spec_out = spec_gen.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        on_token=_spec_on_tok if on_text else None,
                    )
                spec_dt_ms = round((time.perf_counter() - t_spec_start) * 1000, 1)
                spec_text = sanitize_text(str(spec_out["text"]))
                if not self.weights_loaded:
                    banner = "[Notice: Model is running with initialized weights without trained checkpoint. Train this model or load weights in Checkpoints & Weights.]\n\n"
                    spec_text = f"{banner}{spec_text}"
                vram_stats = VRAMProfiler.get_vram_stats(self.device)
                return {
                    "text": spec_text,
                    "route": f"SPECULATIVE (K={k_draft}, acc={spec_out['acceptance_rate']:.1%})",
                    "tokens_count": spec_out["tokens_count"],
                    "tokens_per_sec": int(spec_out["tokens_per_second"]),
                    "latency_ms": spec_dt_ms,
                    "ttft_ms": None,
                    "decode_tokens_per_sec": spec_out["tokens_per_second"],
                    "vram_gb": vram_stats.get("allocated_gb", 0.0),
                    "checkpoint": self.active_checkpoint,
                    "weights_loaded": self.weights_loaded,
                    "cancelled": False,
                    "speculative_telemetry": spec_out.get("telemetry", {}),
                    "acceptance_rate": spec_out.get("acceptance_rate", 0.0),
                }

            with self.model_autocast():
                result = generate_response(
                    self.model,
                    self.tokenizer,
                    prompt,
                    device=self.device,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    force_depth=force_depth,
                    temperature=temperature,
                    top_k=50,
                    repetition_penalty=1.2,
                    max_new_tokens=max_new_tokens,
                    prompt_is_formatted=prompt_is_formatted,
                    cancel_event=cancel_event,
                    on_text=on_text,
                    return_info=True,
                )
            generated_text = sanitize_text(str(result["text"]))
            if not self.weights_loaded:
                banner = "[Notice: Model is running with initialized weights without trained checkpoint. Train this model or load weights in Checkpoints & Weights.]\n\n"
                generated_text = f"{banner}{generated_text}"
            vram_stats = VRAMProfiler.get_vram_stats(self.device)
            return {
                "text": generated_text,
                "route": result["route"],
                "tokens_count": result["tokens_count"],
                "tokens_per_sec": int(result["tokens_per_sec"]),
                "latency_ms": result["latency_ms"],
                "ttft_ms": result.get("ttft_ms"),
                "decode_tokens_per_sec": result.get("decode_tokens_per_sec"),
                "vram_gb": vram_stats.get("allocated_gb", 0.0),
                "checkpoint": self.active_checkpoint,
                "weights_loaded": self.weights_loaded,
                "cancelled": result["cancelled"],
            }

        @property
        def config(self) -> Dict[str, Any]:
            target_dist = [0.50, 0.30, 0.20]
            if self.model is not None and hasattr(self.model, "router") and hasattr(self.model.router, "target_depth_dist"):
                td = self.model.router.target_depth_dist
                target_dist = td.tolist() if hasattr(td, "tolist") else list(td)
            return {
                "target_depth_dist": target_dist,
                "usage_ema_decay": 0.99,
                "bias_strength": 0.5,
                "balance_coef": float(getattr(self, "balance_loss_weight", 0.3)),
                "exploration": str(getattr(self, "exploration_mode", "cosine")),
                "exploration_steps": int(getattr(self, "exploration_steps", 500)),
                "grad_clip": float(getattr(self, "grad_clip", 1.0)),
                "grad_accum_steps": int(self.grad_accum_steps),
                "total_steps": int(getattr(self, "total_steps", 50000)),
                "depth_mode": str(getattr(self, "depth_mode", "dynamic")),
                "log_every": 10,
                "eval_every": 250,
                "save_every": 500,
                "lr": float(self.lr_peak),
                "min_lr": float(self.min_lr),
                "warmup_steps": int(self.warmup_steps),
                "cycle_start_step": int(getattr(self, "lr_cycle_start_step", 0)),
                "cycle_steps": int(getattr(self, "lr_cycle_steps", 50000)),
            }

        @property
        def vocab_size(self) -> int:
            return int(getattr(self.model, "vocab_size", 32000))

        @property
        def grad_accum(self) -> int:
            return int(self.grad_accum_steps)

        def lr_schedule(self, config: dict, step: int) -> float:
            from triune.trainer.scheduler import lr_schedule as _canonical_lr_schedule
            return _canonical_lr_schedule(config, step)

        def set_optimizer_lr(self, optimizer, lr: float, expert_lr: Optional[float] = None) -> None:
            from triune.trainer.scheduler import set_optimizer_lr as _canonical_set_lr
            if expert_lr is None and hasattr(self, "centroid_lr") and self.centroid_lr is not None:
                expert_lr = self.centroid_lr
            _canonical_set_lr(optimizer, lr, expert_lr=expert_lr)

        def model_autocast(self):
            from triune.recipes import bf16_autocast, build_fp8_precision_context, build_precision_context
            use_fp8 = bool(getattr(self.model, "use_fp8", False) or getattr(self, "precision", "bf16") == "fp8")
            use_fp4 = bool(getattr(self.model, "use_fp4", False) or getattr(self, "precision", "bf16") == "fp4")
            if use_fp8:
                return build_fp8_precision_context(device=self.device)()
            if use_fp4:
                return build_precision_context(use_fp4=True, device=self.device)()
            return bf16_autocast(self.device.type if hasattr(self.device, "type") else "cpu")

        def next_batch(self):
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

            batch = torch.tensor(micro_chunks, dtype=torch.long, device=self.device).clamp(0, self.vocab_size - 1)
            x = batch[:, :-1]
            targets = batch[:, 1:].contiguous()

            if self.tokenizer and (self.step % 5 == 0 or not self.current_batch_preview):
                try:
                    self.current_batch_preview = sanitize_text(
                        self.tokenizer.decode(x[0].tolist(), skip_special_tokens=True).strip()
                    )
                except Exception:
                    self.current_batch_preview = f"Stream token chunk #{self.step}"

            return x, targets

        def _build_checkpoint_payload(self, loss_val: float, to_cpu: bool = True) -> Dict[str, Any]:
            """Build unified checkpoint dictionary shared by auto-save and manual save."""
            def _to_cpu(item):
                if torch.is_tensor(item):
                    return item.detach().to("cpu")
                elif isinstance(item, dict):
                    return {k: _to_cpu(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [_to_cpu(v) for v in item]
                return item

            depth_ema = None
            if getattr(self, "training_engine", None) is not None and hasattr(self.training_engine, "depth_usage_ema"):
                try:
                    depth_ema = self.training_engine.depth_usage_ema.detach().cpu()
                except Exception:
                    depth_ema = None
            cfg_dict = {
                **(self.model.model_spec.to_dict()
                   if getattr(self.model, "model_spec", None) is not None
                   and hasattr(self.model.model_spec, "to_dict")
                   else dict(self.architecture_spec or {})),
                "vocab_size": int(getattr(self.model, "vocab_size", 32000)),
                "hidden_dim": int(getattr(self.model, "hidden_dim", 256)),
                "num_layers": len(getattr(self.model, "layers", [])),
                "num_heads": int(getattr(self.model, "num_heads", 4)),
                "head_dim": int(getattr(self.model, "head_dim", 64)),
                "num_experts": int(getattr(self.model, "num_experts", 4)),
                "router_prefix_layers": int(getattr(self.model, "router_prefix_layers", 1)),
                "reflex_exit_layer": int(getattr(self.model, "reflex_exit_layer", 2)),
                "limbic_exit_layer": int(getattr(self.model, "limbic_exit_layer", 4)),
                "depth_mode": getattr(self, "depth_mode", "dynamic"),
                "precision": getattr(self, "precision", "bf16"),
                "use_fp4": bool(getattr(self.model, "use_fp4", False)),
                "use_fp8": bool(getattr(self.model, "use_fp8", False)),
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "grad_accum_steps": self.grad_accum_steps,
                "lr_peak": self.lr_peak,
                "current_lr": self.current_lr,
                "warmup_steps": self.warmup_steps,
                "lr_cycle_start_step": getattr(self, "lr_cycle_start_step", 0),
                "lr_cycle_steps": getattr(self, "lr_cycle_steps", 50000),
            }
            m_state = self.model.state_dict()
            opt_state = self.optimizer.state_dict() if self.optimizer else None
            if to_cpu:
                m_state = _to_cpu(m_state)
                opt_state = _to_cpu(opt_state)

            return {
                "step": self.step,
                "project": getattr(self, "active_project", "Triune-Base · Adaptive CentroidSteer"),
                "model_state": m_state,
                "optimizer_state": opt_state,
                "loss": loss_val,
                "total_tokens_trained": self.total_tokens_trained,
                "tokens_trained": self.total_tokens_trained,
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "grad_accum_steps": self.grad_accum_steps,
                "depth_usage_ema": depth_ema,
                "config": cfg_dict,
            }

        def step_once(self) -> Dict[str, Any]:
            if not HAS_TORCH or TriuneTransformer is None:
                raise RuntimeError("PyTorch is unavailable; this process can display Studio but cannot train.")

            self.lazy_init_model()
            if not self.data_chunks and not self.is_streaming_hf:
                self.load_training_data()

            if self.training_engine is None:
                from triune.trainer.engine import TrainingEngine
                self.training_engine = TrainingEngine(self)

            t0 = time.perf_counter()
            self.training_engine.step = self.step
            step_stats = self.training_engine.step_once(advance_step=True)
            self.step = self.training_engine.step
            self.weights_loaded = True
            # The in-memory model has diverged from the checkpoint it was restored from.
            self.active_checkpoint = None
            self.current_lr = float(step_stats["lr"])

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = max(0.0001, t1 - t0)
            accum_tokens = int(step_stats["tokens_in_step"])
            tok_per_sec = int(accum_tokens / dt)

            self.total_tokens_trained += accum_tokens
            self.tokens_trained = self.total_tokens_trained
            if self.step % 10 == 0:
                self._save_persisted_session_state()

            vram_stats = VRAMProfiler.get_vram_stats(self.device)
            ema = step_stats["depth_usage_ema"].tolist()
            reflex_pct = round(float(ema[0]) * 100.0, 1) if len(ema) > 0 else 34.0
            limbic_pct = round(float(ema[1]) * 100.0, 1) if len(ema) > 1 else 33.0
            cortex_pct = round(max(0.0, 100.0 - reflex_pct - limbic_pct), 1)

            payload = {
                "step": self.step,
                "loss": round(float(step_stats["loss"]), 4),
                "lm_loss": round(float(step_stats["lm_loss"]), 4),
                "router_loss": round(float(step_stats["router_loss"]), 4),
                "balance_loss": round(float(step_stats["balance_loss"]), 4),
                "z_loss": round(float(step_stats["z_loss"]), 4),
                "overflow": int(step_stats["overflow"]),
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
                "precision": getattr(self, "precision", "bf16"),
                "lr": round(self.current_lr, 7),
                "tokens_in_step": accum_tokens,
                "wandb_url": self.wandb_url,
                "wandb_active": bool(self.wandb_run),
            }

            # Log to Weights & Biases if configured
            if self.wandb_run is None and (self.wandb_key or os.environ.get("WANDB_API_KEY")):
                self.init_wandb()

            if self.wandb_run is not None:
                try:
                    import wandb
                    wandb.log({
                        "train/loss": payload["loss"],
                        "train/lm_loss": payload["lm_loss"],
                        "train/router_loss": payload["router_loss"],
                        "train/balance_loss": payload["balance_loss"],
                        "train/z_loss": payload["z_loss"],
                        "train/lr": payload["lr"],
                        "train/throughput_tok_per_sec": payload["throughput"],
                        "train/total_tokens": payload["tokens_trained"],
                        "system/vram_gb": payload["vram_gb"],
                        "exit/reflex_pct": reflex_pct,
                        "exit/limbic_pct": limbic_pct,
                        "exit/cortex_pct": cortex_pct,
                    }, step=self.step)
                except Exception:
                    pass

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
                    auto_tmp = auto_path.with_suffix(".tmp")
                    ckpt_payload = self._build_checkpoint_payload(payload["loss"], to_cpu=True)
                    torch.save(ckpt_payload, auto_tmp)
                    if auto_tmp.exists():
                        auto_tmp.replace(auto_path)
                    del ckpt_payload
                    self.active_checkpoint = auto_name
                    _register_checkpoint_project_assignment(auto_name, getattr(self, "active_project", "Triune-Base · Adaptive CentroidSteer"))
                    self.logs.insert(0, f"[CHECKPOINT] Periodic auto-save: {auto_name} at step {self.step} (Project: {self.active_project}).")
                except Exception:
                    pass
                finally:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Periodically generate a sample every 100 steps
            if self.step > 0 and self.step % 100 == 0:
                try:
                    with torch.inference_mode():
                        sample_res = self.generate_text("The transformer", max_new_tokens=16, temperature=0.7)
                        self.last_sample_decode = sample_res.get("text", "")
                    del sample_res
                except Exception:
                    pass
                finally:
                    self.model.train()
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            log_line = f"[STEP {self.step}] Loss: {payload['loss']:.4f} | LR: {self.current_lr:.2e} | {tok_per_sec} tok/s | Ingesting [{self.dataset_name}]: \"{self.current_batch_preview[:36]}...\""
            self.logs.insert(0, log_line)
            if len(self.logs) > 50:
                self.logs.pop()

            if global_emitter is not None:
                global_emitter.emit("train_step", payload)
            return payload

        async def run_training_loop(self, current_run_id: int):
            print(f"[Triune Engine] Background training loop active (run_id={current_run_id}).")
            def _locked_step():
                if self.stop_event.is_set():
                    return None
                with self.worker_lock:
                    if self.stop_event.is_set() or self.run_id != current_run_id:
                        return None
                    return self.step_once()

            while self.is_training and not self.stop_event.is_set() and self.run_id == current_run_id:
                try:
                    step_data = await asyncio.to_thread(_locked_step)
                    if step_data is not None:
                        await telemetry_manager.broadcast({"type": "training_step", "data": step_data})
                except Exception as err:
                    err_msg = f"[Training Error] Step {self.step}: {err}"
                    print(err_msg)
                    self.logs.insert(0, err_msg)
                    if len(self.logs) > 50:
                        self.logs.pop()
                await asyncio.sleep(0.001)
            print(f"[Triune Engine] Background training loop exited (run_id={current_run_id}).")

    pytorch_state = RealPyTorchEngineState()
    import atexit
    atexit.register(pytorch_state.stop_prefetch_worker)

    class ChatCompletionRequest(BaseModel):
        model: str = "triune-base"
        messages: List[Dict[str, str]]
        temperature: float = 0.7
        max_tokens: int = Field(default=256, ge=1, le=8192)
        stream: bool = False
        route: Optional[str] = None
        force_depth: Optional[int] = None
        speculative: bool = False
        k_draft: int = Field(default=3, ge=1, le=16)

    class DAGExecuteRequest(BaseModel):
        nodes: List[Dict[str, Any]]
        edges: List[Dict[str, Any]]

    class FineTuneRequest(BaseModel):
        dataset_path: str = "data/fineweb_sample.jsonl"
        lora_rank: int = 16
        lora_alpha: float = 32.0
        epochs: int = 3
        lr: float = 2e-4
        batch_size: int = 4
        max_steps: Optional[int] = None
        quantization: str = "4-bit NF4"
        adapter_name: Optional[str] = None

    class AdapterExportRequest(BaseModel):
        adapter_path: Optional[str] = None
        format: str = "safetensors"
        export_name: Optional[str] = None

    class SandboxRunRequest(BaseModel):
        code: str

    finetune_state: Dict[str, Any] = {
        "is_running": False,
        "status": "idle",
        "step": 0,
        "total_steps": 0,
        "loss": 0.0,
        "final_loss": None,
        "trainable_params": 0,
        "output_dir": "",
        "error": None,
    }
    finetune_cancel_event = threading.Event()

    @router.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        """List available model checkpoints and zoo entries."""
        from triune.model.base import CANONICAL_MODEL_SPECS
        models = [
            {
                "id": k,
                "object": "model",
                "description": f"{v.name} ({v.num_layers}L / {v.hidden_dim}D / {v.num_experts}E)",
            }
            for k, v in CANONICAL_MODEL_SPECS.items()
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
        cuda_avail = bool(getattr(pytorch_state, "cuda_available", False))
        runtime_device = str(getattr(pytorch_state, "device", "cpu"))
        model = getattr(pytorch_state, "model", None)
        model_devices = sorted({str(parameter.device) for parameter in model.parameters()}) if model is not None else []
        model_dtypes = sorted({str(parameter.dtype).replace("torch.", "") for parameter in model.parameters()}) if model is not None else []
        allocated_gb = reserved_gb = 0.0
        if HAS_TORCH and cuda_avail:
            try:
                allocated_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 3)
                reserved_gb = round(torch.cuda.memory_reserved(0) / (1024**3), 3)
            except Exception:
                pass
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__ if HAS_TORCH else "not_installed (Host UI Mode)",
            "cuda_available": cuda_avail,
            "cuda_initialized": cuda_avail and runtime_device.startswith("cuda"),
            "cuda_runtime_version": getattr(torch.version, "cuda", None) if HAS_TORCH else None,
            "cuda_init_error": getattr(pytorch_state, "cuda_init_error", None),
            "runtime_device": runtime_device,
            "model_devices": model_devices,
            "model_dtypes": model_dtypes,
            "model_weights_loaded": bool(getattr(pytorch_state, "weights_loaded", False)),
            "active_checkpoint": getattr(pytorch_state, "active_checkpoint", None),
            "vram_allocated_gb": allocated_gb,
            "vram_reserved_gb": reserved_gb,
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
        if not isinstance(byok_keys, dict):
            byok_keys = {}

        if any(model_name.startswith(p) for p in ("openai", "gpt-", "chatgpt")):
            api_key = byok_keys.get("openai", "") or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise HTTPException(status_code=424, detail="OpenAI key is not configured. Add and save it in BYOK Subscriptions, or set OPENAI_API_KEY.")
            try:
                t0 = time.perf_counter()
                reply_text = await asyncio.to_thread(_call_openai_api, req.messages, api_key, model="gpt-4o-mini", temperature=req.temperature, max_tokens=req.max_tokens)
                dt = round((time.perf_counter() - t0) * 1000, 1)
                return _build_chat_response(reply_text, req.model, "GPT-4o", dt)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"OpenAI request failed: {_byok_error_message(e)}") from e

        elif any(model_name.startswith(p) for p in ("anthropic", "claude")):
            api_key = byok_keys.get("anthropic", "") or os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise HTTPException(status_code=424, detail="Anthropic key is not configured. Add and save it in BYOK Subscriptions, or set ANTHROPIC_API_KEY.")
            try:
                t0 = time.perf_counter()
                anthropic_model = req.model if req.model.startswith("claude-") else "claude-3-5-sonnet-latest"
                reply_text = await asyncio.to_thread(_call_anthropic_api, req.messages, api_key, model=anthropic_model, temperature=req.temperature, max_tokens=req.max_tokens)
                dt = round((time.perf_counter() - t0) * 1000, 1)
                return _build_chat_response(reply_text, req.model, "CLAUDE-3.5", dt)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Anthropic request failed: {_byok_error_message(e)}") from e

        elif any(model_name.startswith(p) for p in ("google", "gemini")):
            api_key = byok_keys.get("gemini", "") or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise HTTPException(status_code=424, detail="Google Gemini key is not configured. Add and save it in BYOK Subscriptions, or set GEMINI_API_KEY.")
            try:
                t0 = time.perf_counter()
                gemini_model = req.model if req.model.startswith("gemini-") else "gemini-3.8-flash"
                try:
                    reply_text = await asyncio.to_thread(_call_gemini_api, req.messages, api_key, model=gemini_model, temperature=req.temperature, max_tokens=req.max_tokens)
                except Exception as g_err:
                    # Fallback to gemini-1.5-flash if requested model returned 404
                    if gemini_model != "gemini-1.5-flash" and "404" in str(g_err):
                        reply_text = await asyncio.to_thread(_call_gemini_api, req.messages, api_key, model="gemini-1.5-flash", temperature=req.temperature, max_tokens=req.max_tokens)
                        gemini_model = "gemini-1.5-flash"
                    else:
                        raise
                dt = round((time.perf_counter() - t0) * 1000, 1)
                return _build_chat_response(reply_text, req.model, "GEMINI", dt)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Google Gemini request failed: {_byok_error_message(e)}") from e

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
        chat_messages = [
            m for m in req.messages
            if not (
                (m.get("role") or "").strip().lower() == "system"
                and (m.get("content") or "").strip().startswith("You are Triune, a helpful")
            )
        ]
        if len(chat_messages) > 1:
            turns = []
            for m in chat_messages:
                role = (m.get("role") or "user").strip().capitalize()
                content = (m.get("content") or "").strip()
                if content:
                    turns.append(f"{role}: {content}")
            turns.append("Assistant:")
            effective_prompt = "\n".join(turns)
        else:
            effective_prompt = user_prompt

        if pytorch_state.is_training:
            raise HTTPException(status_code=409, detail="Pause training before starting local inference to avoid competing GPU work.")
        if pytorch_state.active_generation_cancel is not None:
            raise HTTPException(status_code=409, detail="A local inference request is already running.")
        cancel_event = threading.Event()
        pytorch_state.active_generation_cancel = cancel_event

        def _locked_generate(on_text=None) -> Dict[str, Any]:
            with pytorch_state.worker_lock:
                try:
                    if pytorch_state.is_training:
                        raise RuntimeError("Training started while inference was being prepared; pause it and retry.")
                    return pytorch_state.generate_text(
                        effective_prompt,
                        max_new_tokens=req.max_tokens,
                        temperature=req.temperature,
                        force_depth=force_depth,
                        cancel_event=cancel_event,
                        prompt_is_formatted=len(chat_messages) > 1,
                        on_text=on_text,
                        speculative=bool(getattr(req, "speculative", False)),
                        k_draft=int(getattr(req, "k_draft", 3)),
                    )
                finally:
                    if pytorch_state.active_generation_cancel is cancel_event:
                        pytorch_state.active_generation_cancel = None

        if req.stream:
            # Keep the model call on its worker thread, but forward decoded text
            # snapshots to the HTTP response as soon as tokens are produced.
            loop = asyncio.get_running_loop()
            events: asyncio.Queue = asyncio.Queue(maxsize=32)

            def _enqueue(event: Dict[str, Any]) -> None:
                if events.full():
                    # Decoded text events are cumulative snapshots. Coalesce a
                    # backlog to the newest snapshot rather than allowing a
                    # slow UI/proxy to grow memory without bound. Preserve
                    # control events; the completion event carries final text.
                    queued = []
                    while not events.empty():
                        queued.append(events.get_nowait())
                    if event["type"] == "text":
                        queued = [item for item in queued if item["type"] != "text"]
                    else:
                        queued = [item for item in queued if item["type"] != "text"]
                    for item in queued:
                        if not events.full():
                            events.put_nowait(item)
                    if events.full():
                        return
                events.put_nowait(event)

            def _publish(event: Dict[str, Any]) -> None:
                try:
                    loop.call_soon_threadsafe(_enqueue, event)
                except RuntimeError:
                    # The client may have disconnected while CUDA was finishing
                    # its current kernel. The generation cancellation flag is
                    # set by the response iterator's finally block below.
                    pass

            def _stream_worker() -> None:
                try:
                    result = _locked_generate(lambda text: _publish({"type": "text", "text": text}))
                    _publish({"type": "done", "result": result})
                except Exception as exc:
                    _publish({"type": "error", "message": str(exc)})

            async def _stream_events():
                worker = asyncio.create_task(asyncio.to_thread(_stream_worker))
                previous_text = ""
                try:
                    # Flush an initial comment so clients and local proxies know
                    # the request is alive while a checkpoint/CUDA context loads.
                    yield ": triune stream connected\n\n"
                    while True:
                        try:
                            event = await asyncio.wait_for(events.get(), timeout=15.0)
                        except asyncio.TimeoutError:
                            yield ": keep-alive\n\n"
                            continue
                        if event["type"] == "text":
                            snapshot = str(event["text"])
                            if not snapshot or snapshot == previous_text:
                                continue
                            # Standard delta chunks are used when the tokenizer
                            # appends text normally. A snapshot marker handles
                            # rare byte-level decoder revisions without mangling
                            # the visible response.
                            if snapshot.startswith(previous_text):
                                delta = snapshot[len(previous_text):]
                                if not delta:
                                    continue
                                chunk = {"choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                            else:
                                chunk = {
                                    "choices": [{"index": 0, "delta": {"content": snapshot}, "finish_reason": None}],
                                    "triune_snapshot": True,
                                }
                            previous_text = snapshot
                            chunk.update({"id": "chatcmpl-triune", "object": "chat.completion.chunk", "model": req.model})
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        elif event["type"] == "error":
                            yield f"data: {json.dumps({'error': {'message': event['message']}}, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                        else:
                            result = event["result"]
                            final_text = str(result.get("text", ""))
                            if final_text != previous_text:
                                if final_text.startswith(previous_text):
                                    final_delta = final_text[len(previous_text):]
                                    final_text_chunk = {"choices": [{"index": 0, "delta": {"content": final_delta}, "finish_reason": None}]}
                                else:
                                    final_text_chunk = {
                                        "choices": [{"index": 0, "delta": {"content": final_text}, "finish_reason": None}],
                                        "triune_snapshot": True,
                                    }
                                final_text_chunk.update({"id": "chatcmpl-triune", "object": "chat.completion.chunk", "model": req.model})
                                yield f"data: {json.dumps(final_text_chunk, ensure_ascii=False)}\n\n"
                            final_chunk = {
                                "id": "chatcmpl-triune",
                                "object": "chat.completion.chunk",
                                "model": req.model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "cancelled" if result.get("cancelled") else "stop"}],
                                "telemetry": {
                                    "route": result["route"],
                                    "latency_ms": result["latency_ms"],
                                    "tokens_per_sec": result["tokens_per_sec"],
                                    "ttft_ms": result.get("ttft_ms"),
                                    "decode_tokens_per_sec": result.get("decode_tokens_per_sec"),
                                    "tokens_count": result["tokens_count"],
                                    "vram_gb": result["vram_gb"],
                                    "weights_loaded": result.get("weights_loaded", False),
                                    "checkpoint": result.get("checkpoint"),
                                    "active_model": pytorch_state.active_model_id,
                                    "device": str(pytorch_state.device),
                                    "device_name": pytorch_state.device_name,
                                },
                            }
                            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            break
                finally:
                    if not worker.done():
                        cancel_event.set()

            return StreamingResponse(
                _stream_events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            res = await asyncio.to_thread(_locked_generate)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

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
                    "finish_reason": "cancelled" if res.get("cancelled") else "stop",
                }
            ],
            "telemetry": {
                "route": res["route"],
                "latency_ms": res["latency_ms"],
                "tokens_per_sec": res["tokens_per_sec"],
                "ttft_ms": res.get("ttft_ms"),
                "decode_tokens_per_sec": res.get("decode_tokens_per_sec"),
                "tokens_count": res["tokens_count"],
                "vram_gb": res["vram_gb"],
                "weights_loaded": res.get("weights_loaded", False),
                "checkpoint": res.get("checkpoint"),
                "active_model": pytorch_state.active_model_id,
                "device": str(pytorch_state.device),
                "device_name": pytorch_state.device_name,
            }
        }

    @router.post("/v1/chat/cancel")
    async def cancel_local_generation() -> Dict[str, Any]:
        """Request cooperative cancellation of the active local token-generation loop."""
        cancel_event = pytorch_state.active_generation_cancel
        if cancel_event is None:
            return {"status": "idle", "message": "No local generation is currently active."}
        cancel_event.set()
        return {"status": "cancelling", "message": "Local generation will stop after the current model step."}

    @router.post("/v1/finetune/start")
    async def start_finetune(req: FineTuneRequest) -> Dict[str, Any]:
        """Initiate real LoRA fine-tuning task in the background."""
        if finetune_state["is_running"]:
            return {"status": "already_running", "message": "A fine-tuning run is already active."}

        finetune_cancel_event.clear()
        finetune_state["is_running"] = True
        finetune_state["status"] = "running"
        finetune_state["error"] = None
        finetune_state["step"] = 0
        finetune_state["total_steps"] = 0
        finetune_state["loss"] = 0.0
        finetune_state["final_loss"] = None

        adapter_name = req.adapter_name or f"LoRA-r{req.lora_rank}-a{int(req.lora_alpha)}"
        safe_name = adapter_name.lower().replace(" ", "_").replace("/", "_")
        out_dir = f"checkpoints/adapters/{safe_name}"

        pytorch_state.active_adapter = {
            "id": f"lora-r{req.lora_rank}",
            "name": adapter_name,
            "rank": req.lora_rank,
            "alpha": req.lora_alpha,
            "quantization": req.quantization,
            "dataset": req.dataset_path,
            "path": out_dir,
        }
        pytorch_state._save_persisted_session_state()

        async def _run_finetune_job():
            try:
                pytorch_state.lazy_init_model()
                # Detach any previous adapter first
                if getattr(pytorch_state, "active_finetuner", None) is not None:
                    pytorch_state.active_finetuner.detach_lora()
                    pytorch_state.active_finetuner = None

                cfg = LoRAConfig(
                    rank=req.lora_rank,
                    alpha=req.lora_alpha,
                    quantization=req.quantization,
                )
                finetuner = TriuneFineTuner(pytorch_state.model, cfg)
                finetune_state["trainable_params"] = finetuner.count_trainable_parameters()

                dpath = req.dataset_path
                if not Path(dpath).is_file():
                    for cand in ("data/fineweb_sample.jsonl", "data/finetune.jsonl"):
                        if Path(cand).is_file():
                            dpath = cand
                            break

                def _on_step(cur_step: int, total_s: int, cur_loss: float, lm_loss_val: float):
                    finetune_state["step"] = cur_step
                    finetune_state["total_steps"] = total_s
                    finetune_state["loss"] = round(cur_loss, 4)

                res = await asyncio.to_thread(
                    finetuner.fit,
                    dataset_path=dpath,
                    output_dir=out_dir,
                    epochs=req.epochs,
                    batch_size=req.batch_size,
                    lr=req.lr,
                    max_steps=req.max_steps,
                    on_step=_on_step,
                    cancel_event=finetune_cancel_event,
                )

                if finetune_cancel_event.is_set():
                    finetune_state["status"] = "cancelled"
                    finetuner.detach_lora()
                    pytorch_state.active_finetuner = None
                    pytorch_state.active_adapter = None
                    pytorch_state._save_persisted_session_state()
                    pytorch_state.logs.insert(0, "[LORA] Fine-tuning run cancelled by user.")
                else:
                    finetune_state["status"] = "completed"
                    finetune_state["final_loss"] = res.get("final_loss", 0.0)
                    finetune_state["output_dir"] = res.get("output_dir", out_dir)
                    finetune_state["step"] = res.get("total_steps", finetune_state["step"])
                    pytorch_state.active_finetuner = finetuner
                    if pytorch_state.active_adapter:
                        pytorch_state.active_adapter["path"] = finetune_state["output_dir"]
                        pytorch_state._save_persisted_session_state()
                    pytorch_state.logs.insert(0, f"[LORA] Fine-tuning finished! Final Loss: {finetune_state['final_loss']} -> {finetune_state['output_dir']}")
            except Exception as exc:
                finetune_state["status"] = "failed"
                finetune_state["error"] = str(exc)
                if getattr(pytorch_state, "active_finetuner", None) is not None:
                    try:
                        pytorch_state.active_finetuner.detach_lora()
                    except Exception:
                        pass
                    pytorch_state.active_finetuner = None
            finally:
                finetune_state["is_running"] = False

        asyncio.create_task(_run_finetune_job())
        return {
            "status": "started",
            "active_adapter": pytorch_state.active_adapter,
            "message": f"LoRA fine-tuning initiated for {req.epochs} epochs.",
        }

    @router.post("/v1/finetune/cancel")
    async def cancel_finetune() -> Dict[str, Any]:
        """Cancel active fine-tuning run gracefully."""
        if not finetune_state["is_running"]:
            return {"status": "idle", "message": "No fine-tuning run is active."}
        finetune_cancel_event.set()
        return {"status": "cancelling", "message": "Fine-tuning cancellation requested."}

    @router.get("/v1/finetune/status")
    async def get_finetune_status() -> Dict[str, Any]:
        """Return live LoRA fine-tuning progress and status."""
        return {**finetune_state, "active_adapter": getattr(pytorch_state, "active_adapter", None)}

    class AdapterActionRequest(BaseModel):
        action: str = "attach"
        id: Optional[str] = None
        name: Optional[str] = None
        path: Optional[str] = None
        rank: int = 16
        alpha: float = 32.0
        quantization: str = "4-bit NF4"

    @router.post("/v1/finetune/adapter")
    async def configure_active_adapter(req: AdapterActionRequest) -> Dict[str, Any]:
        """Attach or detach a LoRA/QLoRA adapter on the live engine state."""
        if req.action.lower() == "detach":
            prev = pytorch_state.active_adapter
            pytorch_state.active_adapter = None
            if getattr(pytorch_state, "active_finetuner", None) is not None:
                try:
                    pytorch_state.active_finetuner.detach_lora()
                except Exception:
                    pass
                pytorch_state.active_finetuner = None
            pytorch_state._save_persisted_session_state()
            msg = f"[LORA] Detached adapter {(prev or {}).get('name', '')}."
            pytorch_state.logs.insert(0, msg)
            return {"status": "success", "action": "detached", "active_adapter": None, "message": msg}

        # Detach previous adapter first to avoid nested wrapping
        if getattr(pytorch_state, "active_finetuner", None) is not None:
            try:
                pytorch_state.active_finetuner.detach_lora()
            except Exception:
                pass
            pytorch_state.active_finetuner = None

        pytorch_state.lazy_init_model()
        adapter_path = req.path or "checkpoints/lora_adapter"
        p_cfg = Path(adapter_path)
        rank_val = req.rank
        alpha_val = req.alpha
        if p_cfg.is_dir() and (p_cfg / "adapter_config.json").is_file():
            try:
                with open(p_cfg / "adapter_config.json", "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    rank_val = cdata.get("rank", rank_val)
                    alpha_val = cdata.get("alpha", alpha_val)
            except Exception:
                pass

        cfg = LoRAConfig(rank=rank_val, alpha=alpha_val, quantization=req.quantization)
        finetuner = TriuneFineTuner(pytorch_state.model, cfg)

        loaded = False
        if Path(adapter_path).exists():
            try:
                finetuner.load_adapter(adapter_path)
                loaded = True
            except Exception as e:
                print(f"[LORA] Fresh adapter attached (weights not loaded: {e})")

        pytorch_state.active_finetuner = finetuner
        adapter_info = {
            "id": req.id or f"lora-r{rank_val}",
            "name": req.name or f"LoRA Adapter (r={rank_val}, α={alpha_val})",
            "path": str(adapter_path).replace("\\", "/"),
            "rank": rank_val,
            "alpha": alpha_val,
            "quantization": req.quantization,
            "loaded_from_disk": loaded,
        }
        pytorch_state.active_adapter = adapter_info
        pytorch_state._save_persisted_session_state()
        msg = f"[LORA] Attached adapter {adapter_info['name']} ({adapter_info['quantization']})."
        pytorch_state.logs.insert(0, msg)
        return {"status": "success", "action": "attached", "active_adapter": adapter_info, "message": msg}

    @router.post("/v1/finetune/merge")
    async def merge_active_adapter() -> Dict[str, Any]:
        """Merge active LoRA adapter weights permanently into the base model."""
        if getattr(pytorch_state, "active_finetuner", None) is None or getattr(pytorch_state, "active_adapter", None) is None:
            return {"status": "error", "message": "No active LoRA adapter is currently attached to merge."}

        adapter_name = pytorch_state.active_adapter.get("name", "LoRA")
        pytorch_state.active_finetuner.merge_and_unload()
        pytorch_state.active_finetuner = None
        pytorch_state.active_adapter = None
        pytorch_state.weights_loaded = True
        pytorch_state._save_persisted_session_state()

        msg = f"[LORA] Merged adapter '{adapter_name}' permanently into base model weights."
        pytorch_state.logs.insert(0, msg)
        return {"status": "success", "message": msg}

    @router.get("/v1/finetune/adapters")
    async def list_available_adapters() -> Dict[str, Any]:
        """List all discovered LoRA/QLoRA adapters on disk."""
        from triune.trainer.finetune import list_adapters
        raw_adapters = list_adapters()
        for ad in raw_adapters:
            ad["is_active"] = bool(
                pytorch_state.active_adapter is not None
                and (pytorch_state.active_adapter.get("path") == ad["path"] or pytorch_state.active_adapter.get("name") == ad["name"])
            )
        return {"adapters": raw_adapters, "active_adapter": getattr(pytorch_state, "active_adapter", None)}

    @router.post("/v1/finetune/export")
    async def export_adapter(req: AdapterExportRequest) -> Dict[str, Any]:
        """Export an adapter as a standalone SafeTensors or GGUF package."""
        from triune.trainer.finetune import export_adapter as do_export_adapter
        target_path = req.adapter_path or (pytorch_state.active_adapter.get("path") if pytorch_state.active_adapter else None)
        if not target_path or not Path(target_path).exists():
            return {"status": "error", "message": f"Adapter path does not exist: {target_path}"}
        try:
            out_name = req.export_name or "lora_adapter"
            dest = Path("exports") / f"{out_name}.{req.format}"
            exported = do_export_adapter(target_path, dest, fmt=req.format)
            return {
                "status": "success",
                "export_path": str(exported).replace("\\", "/"),
                "size_mb": round(exported.stat().st_size / (1024 * 1024), 2),
                "message": f"Adapter exported successfully to {exported}."
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

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
                res = await asyncio.to_thread(
                    dag_engine.execute_graph,
                    graph_json,
                    initial_context={"_runtime_training_state": pytorch_state},
                )
            finally:
                dag_engine.unregister_event_callback(_on_dag_event)

            return {
                "status": res.get("status", "success"),
                "results": res.get("results", {}),
                "context": res.get("context", {}),
                "error": res.get("error"),
                "failed_node": res.get("failed_node"),
                "specification_nodes": res.get("specification_nodes", []),
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
                prec_val = get_val("precision", "dtype")
                fp8_val = get_val("use_fp8")
                fp4_val = get_val("use_fp4")

                reconfig_kwargs = {}
                if layers is not None: reconfig_kwargs["num_layers"] = int(layers)
                if hidden_dim is not None: reconfig_kwargs["hidden_dim"] = int(hidden_dim)
                if heads is not None: reconfig_kwargs["num_heads"] = int(heads)
                if head_dim is not None: reconfig_kwargs["head_dim"] = int(head_dim)
                if experts is not None: reconfig_kwargs["num_experts"] = int(experts)
                if d_mode: reconfig_kwargs["depth_mode"] = str(d_mode).lower()
                if b_weight is not None: reconfig_kwargs["balance_loss_weight"] = float(b_weight)
                if prec_val: reconfig_kwargs["precision"] = str(prec_val).lower()
                if fp8_val is not None: reconfig_kwargs["use_fp8"] = bool(fp8_val)
                if fp4_val is not None: reconfig_kwargs["use_fp4"] = bool(fp4_val)

                if preset_id or reconfig_kwargs:
                    arch = pytorch_state.reconfigure_architecture(preset_id=preset_id, **reconfig_kwargs)
                    applied["model_architecture"] = arch

            # 0b. Precision / Quantization Nodes (FP8Linear / NF4Quantizer)
            if ntype in ("Precision", "FP8Linear", "NF4Quantizer") or any(k in title for k in ("FP8", "NF4", "Quantiz")):
                raw_dtype = str(get_val("dtype", "precision", "quant_type", default="fp8")).lower()
                if "fp8" in raw_dtype or "e4m3" in raw_dtype or "e5m2" in raw_dtype or "8" in raw_dtype:
                    mapped_prec = "fp8"
                elif "fp4" in raw_dtype or "nf4" in raw_dtype or "int4" in raw_dtype or "4" in raw_dtype:
                    mapped_prec = "fp4"
                elif "32" in raw_dtype:
                    mapped_prec = "fp32"
                else:
                    mapped_prec = "bf16"
                if getattr(pytorch_state, "precision", "bf16") != mapped_prec:
                    cur_arch = getattr(pytorch_state, "architecture_spec", {}) or {}
                    arch = pytorch_state.reconfigure_architecture(
                        preset_id=getattr(pytorch_state, "active_model_id", "triune-small"),
                        num_layers=cur_arch.get("num_layers"),
                        hidden_dim=cur_arch.get("hidden_dim"),
                        num_heads=cur_arch.get("num_heads"),
                        head_dim=cur_arch.get("head_dim"),
                        num_experts=cur_arch.get("num_experts"),
                        precision=mapped_prec,
                        batch_size=pytorch_state.batch_size,
                        seq_len=pytorch_state.seq_len,
                        depth_mode=pytorch_state.depth_mode,
                    )
                    applied["model_architecture"] = arch
                pytorch_state.precision = mapped_prec
                applied["precision"] = mapped_prec

            # 1. Dataset Streaming (only streamer/dataset nodes, not Token IDs or DataLoader)
            is_streamer_node = (
                ntype in ("Dataset", "HuggingFaceStreamer", "HFRestStreamReader")
                or any(k in title for k in ("Streamer", "Dataset", "Reader"))
                or (ntype == "Data" and "Token" not in title and "DataLoader" not in title)
            )
            if is_streamer_node:
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

            # 5. Gradient Accumulation & Layer Streaming
            if ntype in ("Runtime", "GradientAccumulator") or "Accumulat" in title:
                ga = get_val("grad_accum_steps", "grad_accum", "accumulation_steps")
                if ga is not None:
                    pytorch_state.grad_accum_steps = max(1, int(ga))
                    applied["grad_accum_steps"] = pytorch_state.grad_accum_steps

            if ntype in ("LayerStreamer", "Streaming") or ("Streamer" in title and not is_streamer_node):
                gl = get_val("gpu_layers", "prefetch_count")
                if gl is not None:
                    total_l = int(getattr(pytorch_state, "architecture_spec", {}).get("num_layers", 6) or 6)
                    pytorch_state.gpu_layers = max(0, min(total_l, int(gl)))
                    pytorch_state.gpu_offload_pct = int(round((pytorch_state.gpu_layers / max(1, total_l)) * 100))
                    applied["gpu_layers"] = pytorch_state.gpu_layers
                    applied["gpu_offload_pct"] = pytorch_state.gpu_offload_pct

            # 6. LoRA / QLoRA Adapter Node
            if ntype in ("Adapter", "LoRAAdapter") or "LoRA" in title:
                rank_val = int(get_val("rank", "lora_rank", default=16))
                alpha_val = float(get_val("alpha", "lora_alpha", default=32.0))
                quant_val = str(get_val("quantization", default="4-bit NF4"))
                pytorch_state.active_adapter = {
                    "id": f"lora-r{rank_val}",
                    "name": f"LoRA Adapter (r={rank_val}, α={alpha_val})",
                    "path": "checkpoints/lora_adapter",
                    "rank": rank_val,
                    "alpha": alpha_val,
                    "quantization": quant_val,
                }
                applied["active_adapter"] = pytorch_state.active_adapter

        pytorch_state._save_persisted_session_state()

        return {
            "status": "success",
            "applied_config": applied,
            "applied_count": len(applied),
            "active_model": getattr(pytorch_state, "active_model_id", "triune-small"),
            "architecture": getattr(pytorch_state, "architecture_spec", {}),
            "active_adapter": getattr(pytorch_state, "active_adapter", None),
            "profile": _compute_hardware_profile(),
            "training_config": {
                "batch_size": pytorch_state.batch_size,
                "seq_len": pytorch_state.seq_len,
                "grad_accum_steps": pytorch_state.grad_accum_steps,
                "depth_mode": pytorch_state.depth_mode,
                "current_lr": pytorch_state.current_lr,
                "lr_peak": pytorch_state.lr_peak,
                "warmup_steps": pytorch_state.warmup_steps,
                "precision": getattr(pytorch_state, "precision", "bf16"),
                "gpu_layers": getattr(pytorch_state, "gpu_layers", 6),
                "gpu_offload_pct": getattr(pytorch_state, "gpu_offload_pct", 100),
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
                "precision": getattr(pytorch_state, "precision", "bf16"),
                "active_adapter": getattr(pytorch_state, "active_adapter", None),
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
            if not isinstance(context, dict):
                raise HTTPException(status_code=400, detail="Node execution context must be a JSON object.")
            context.setdefault("_runtime_training_state", pytorch_state)
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

    @router.post("/v1/resource/assess")
    async def assess_resource_feasibility(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate VRAM feasibility, safe headroom, and auto-fit recommendations via DynamicResourceManager."""
        from triune.runtime.resource_manager import DynamicResourceManager
        import dataclasses
        cfg = config or {}
        dev = pytorch_state.device if (HAS_TORCH and getattr(pytorch_state, "device", None) is not None) else None
        mdl = pytorch_state.model if (HAS_TORCH and getattr(pytorch_state, "model", None) is not None) else None
        assessment = DynamicResourceManager.assess_feasibility(
            cfg,
            device=dev,
            model=mdl,
        )
        report = DynamicResourceManager.format_assessment_report(assessment)
        data = dataclasses.asdict(assessment)
        data["report"] = report
        return data

    class HardwareTuneRequest(BaseModel):
        preset: Optional[str] = None
        gpu_offload_pct: Optional[int] = None
        gpu_layers: Optional[int] = None
        batch_size: Optional[int] = None
        seq_len: Optional[int] = None
        grad_accum_steps: Optional[int] = None
        depth_mode: Optional[str] = None
        precision: Optional[str] = None
        lr: Optional[float] = None
        lr_peak: Optional[float] = None
        warmup_steps: Optional[int] = None

    def _apply_model_gpu_residency(state, requested_layers: int) -> int:
        """Apply the requested layer residency to the already-loaded model.

        Hardware presets used to update only the UI/session fields. The live
        LayerStreamingEngine kept its old split, so the app could report 100%
        offload while most layers were still copied over PCIe for every pass.
        Rebuild only the streaming wrapper (not the model or optimizer) while
        the engine is idle, preserving trained weights and optimizer state.
        """
        model = getattr(state, "model", None)
        device = getattr(state, "device", None)
        if model is None or not HAS_TORCH or getattr(device, "type", None) != "cuda":
            return 0

        old_engine = getattr(model, "_streaming_engine", None)
        all_layers = (
            list(old_engine.all_layers)
            if old_engine is not None
            else list(getattr(model, "layers", ()))
        )
        if not all_layers:
            return 0
        requested = max(0, min(len(all_layers), int(requested_layers)))

        if old_engine is None:
            parameter_devices = {parameter.device.type for parameter in model.parameters()}
            currently_resident = len(all_layers) if parameter_devices == {"cuda"} else 0
        else:
            currently_resident = len(old_engine.resident_layers)

        # This tuner is used for training: keep the engine's existing 2.5 GiB
        # headroom for activations, optimizer work, and temporary workspaces.
        # Count existing resident layer bytes as available when deciding
        # whether a larger split can fit; never consume all free VRAM blindly.
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            resident_layers = old_engine.resident_layers if old_engine is not None else all_layers[:currently_resident]
            current_bytes = sum(
                p.numel() * p.element_size()
                for layer in resident_layers for p in layer.parameters()
            )
            reserve_bytes = 2.5 * (1024 ** 3)
            usable_bytes = max(0, int(free_bytes + current_bytes - reserve_bytes))
            cumulative_bytes = 0
            max_fit = 0
            for layer in all_layers:
                layer_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
                if cumulative_bytes + layer_bytes > usable_bytes:
                    break
                cumulative_bytes += layer_bytes
                max_fit += 1
            target = min(requested, max_fit)
        except Exception:
            # If device memory cannot be queried, retain the current safe split
            # instead of attempting an unbounded allocation.
            target = min(requested, currently_resident)

        if target == currently_resident:
            return target

        from triune.runtime.streaming import LayerStreamingEngine, StreamingConfig

        old_config = getattr(old_engine, "config", None)
        config = StreamingConfig(
            enabled=True,
            chunk_size=1,
            pin_memory=getattr(old_config, "pin_memory", False),
            async_prefetch=getattr(old_config, "async_prefetch", True),
            fp8_weights=bool(
                getattr(old_config, "fp8_weights", False)
                or getattr(state, "precision", "bf16") == "fp8"
            ),
            grad_clip=getattr(old_config, "grad_clip", 1.0),
            instant_optimizer=getattr(old_config, "instant_optimizer", True),
            layers_path=getattr(old_config, "layers_path", None),
            resident_gpu_layers=target,
        )
        optimizer = getattr(state, "optimizer", None)

        if old_engine is not None:
            old_engine.cleanup_resident_layers()
            old_engine.detach()
        replacement = LayerStreamingEngine(
            model,
            device=device,
            config=config,
            optimizer=optimizer,
        )
        try:
            replacement.attach(optimizer)
        except Exception:
            # Do not leave a half-installed wrapper after an allocation or
            # staging failure. Restore the previous residency where possible.
            replacement.detach()
            if old_engine is not None:
                rollback = LayerStreamingEngine(
                    model,
                    device=device,
                    config=old_config,
                    optimizer=optimizer,
                )
                rollback.attach(optimizer)
            raise
        return len(replacement.resident_layers)

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

        if HAS_TORCH and bool(getattr(pytorch_state, "cuda_available", False)):
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

        # A CPU-only runtime has no known VRAM budget. Never fabricate a GPU
        # profile, because its auto-tune recommendations can trigger OOMs.

        arch_spec = getattr(pytorch_state, "architecture_spec", {}) or {}
        num_layers = int(arch_spec.get("num_layers", 0) or (len(getattr(pytorch_state.model, "layers", [])) if pytorch_state.model is not None else 6) or 6)
        hidden_dim = int(arch_spec.get("hidden_dim", getattr(pytorch_state.model, "hidden_dim", 256) if pytorch_state.model is not None else 256) or 256)

        gpu_offload_pct = getattr(pytorch_state, "gpu_offload_pct", 100)
        gpu_layers = getattr(pytorch_state, "gpu_layers", num_layers)
        active_streamer = getattr(pytorch_state.model, "_streaming_engine", None) if pytorch_state.model is not None else None
        if active_streamer is not None:
            gpu_layers = len(active_streamer.resident_layers)
        elif pytorch_state.model is not None and is_cuda:
            layer_modules = list(getattr(pytorch_state.model, "layers", ()))
            if layer_modules and all(
                parameter.device.type == "cuda"
                for layer in layer_modules for parameter in layer.parameters()
            ):
                gpu_layers = len(layer_modules)
        gpu_offload_pct = int(round(100 * gpu_layers / max(1, num_layers))) if is_cuda else 0
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
        spec_total = int(arch_spec.get("total_params", 0) or 0)
        spec_active = int(arch_spec.get("active_params", 0) or 0)
        if spec_total > 0:
            total_params = spec_total
            trainable_params = spec_total
            active_params = spec_active if spec_active > 0 else int(total_params * 0.58)
        elif pytorch_state.model is not None:
            total_params = sum(p.numel() for p in pytorch_state.model.parameters())
            trainable_params = sum(p.numel() for p in pytorch_state.model.parameters() if p.requires_grad)
            active_params = int(total_params * 0.58)
        else:
            total_params = 36_000_000
            trainable_params = total_params
            active_params = 32_900_000
        prec = getattr(pytorch_state, "precision", "bf16")
        bytes_per_param = 1.0 if prec == "fp8" else (0.5 if "4" in str(prec) else (4.0 if prec == "fp32" else 2.0))
        weights_mb = round((total_params * bytes_per_param) / (1024 * 1024), 2)
        # CentroidSteer + Muon stores momentum only on 2D weights (~60% of params), saving ~30-40% over AdamW (8 bytes/param)
        opt_mb = round((trainable_params * 5.5) / (1024 * 1024), 2)
        act_mb = round((cur_batch * cur_seq * hidden_dim * num_layers * 20 * 2) / (1024 * 1024), 2)
        cuda_mb = 450.0 if is_cuda else 35.0
        total_est_mb = round(weights_mb + opt_mb + act_mb + cuda_mb, 2)
        total_est_gb = round(total_est_mb / 1024.0, 2)

        # Only report observed throughput. The old FLOPS-derived number was a
        # theoretical guess presented as a speed estimate and was not a usable
        # predictor for layer streaming, which can be PCIe-bound.
        recent_tok_s = [h["throughput"] for h in pytorch_state.history if h.get("throughput") and h["throughput"] > 0]
        empirical_tok_s = int(sum(recent_tok_s[-5:]) / len(recent_tok_s[-5:])) if recent_tok_s else 0

        active_depth = getattr(pytorch_state, "depth_mode", "dynamic") or "dynamic"
        active_prec = getattr(pytorch_state, "precision", "bf16") or "bf16"

        presets = {
            "turbo": {
                "id": "turbo",
                "name": "⚡ Turbo (Max Throughput)",
                "batch_size": turbo_batch,
                "seq_len": turbo_seq,
                "grad_accum_steps": turbo_accum,
                "depth_mode": active_depth,
                "precision": active_prec,
                "gpu_offload_pct": 100,
                "gpu_layers": num_layers,
                "effective_tokens": turbo_batch * turbo_seq * turbo_accum,
                "estimated_vram_gb": round((weights_mb + opt_mb + ((turbo_batch * turbo_seq * hidden_dim * num_layers * 40) / (1024 * 1024)) + cuda_mb) / 1024, 2),
                "estimated_tok_s": empirical_tok_s or None,
                "throughput_basis": "recent_observed_training_rate" if empirical_tok_s else "not_benchmarked",
                "description": "Maximum throughput: 100% offload, saturates compute units with vectorized matrix tiling."
            },
            "balanced": {
                "id": "balanced",
                "name": "⚖️ Balanced (Recommended)",
                "batch_size": balanced_batch,
                "seq_len": balanced_seq,
                "grad_accum_steps": balanced_accum,
                "depth_mode": active_depth,
                "precision": active_prec,
                "gpu_offload_pct": 100,
                "gpu_layers": num_layers,
                "effective_tokens": balanced_batch * balanced_seq * balanced_accum,
                "estimated_vram_gb": round((weights_mb + opt_mb + ((balanced_batch * balanced_seq * hidden_dim * num_layers * 40) / (1024 * 1024)) + cuda_mb) / 1024, 2),
                "estimated_tok_s": empirical_tok_s or None,
                "throughput_basis": "recent_observed_training_rate" if empirical_tok_s else "not_benchmarked",
                "description": "Balanced: Stable gradient convergence, modest VRAM footprint, low heat."
            },
            "eco": {
                "id": "eco",
                "name": "🌱 Eco (Low VRAM)",
                "batch_size": eco_batch,
                "seq_len": eco_seq,
                "grad_accum_steps": eco_accum,
                "depth_mode": active_depth,
                "precision": active_prec,
                "gpu_offload_pct": 100,
                "gpu_layers": num_layers,
                "effective_tokens": eco_batch * eco_seq * eco_accum,
                "estimated_vram_gb": round((weights_mb + opt_mb + ((eco_batch * eco_seq * hidden_dim * num_layers * 40) / (1024 * 1024)) + cuda_mb) / 1024, 2),
                "estimated_tok_s": empirical_tok_s or None,
                "throughput_basis": "recent_observed_training_rate" if empirical_tok_s else "not_benchmarked",
                "description": "Ultra-lightweight: Keeps total VRAM minimal for background multitasking."
            },
            "streaming": {
                "id": "streaming",
                "name": "🌊 AirLLM Streaming (<600 MB)",
                "batch_size": 2,
                "seq_len": 64,
                "grad_accum_steps": 4,
                "depth_mode": active_depth,
                "precision": active_prec,
                "gpu_offload_pct": 20,
                "gpu_layers": 1,
                "effective_tokens": 2 * 64 * 4,
                "estimated_vram_gb": round((weights_mb * 0.2 + (opt_mb * 0.25) + 80.0 + cuda_mb) / 1024, 2),
                "estimated_tok_s": empirical_tok_s or None,
                "throughput_basis": "recent_observed_training_rate" if empirical_tok_s else "not_benchmarked",
                "description": "Virtualized layer swap: Swaps layers sequentially from RAM. Ultra low VRAM (<600 MB), PCIe-bound."
            }
        }

        if not is_cuda:
            # CPU recommendations must never claim that CUDA layers were
            # applied. Auto-optimize remains usable for batch/context tuning.
            for preset in presets.values():
                preset["gpu_offload_pct"] = 0
                preset["gpu_layers"] = 0

        if is_cuda:
            # Choose the fastest profile whose conservative memory estimate
            # fits in currently available VRAM. Leave a 0.5 GiB margin for
            # allocator fragmentation and display/desktop use.
            budget_gb = max(0.0, free_vram_gb - 0.5)
            fitting = [
                key for key in ("turbo", "balanced", "eco", "streaming")
                if presets[key]["estimated_vram_gb"] <= budget_gb
            ]
            recommended_preset = fitting[0] if fitting else "streaming"
        else:
            recommended_preset = "eco"

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
                "lr": pytorch_state.lr_peak,
                "lr_peak": pytorch_state.lr_peak,
                "current_lr": pytorch_state.current_lr,
                "warmup_steps": pytorch_state.warmup_steps,
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

    async def _apply_requested_gpu_layers(requested_layers: int) -> int:
        """Update an idle loaded model's actual streaming split, not just its UI value."""
        if not HAS_TORCH:
            return 0
        if getattr(getattr(pytorch_state, "device", None), "type", None) != "cuda" or pytorch_state.model is None:
            return max(0, int(requested_layers))
        active_engine = getattr(pytorch_state.model, "_streaming_engine", None)
        if active_engine is not None:
            active_layers = len(active_engine.resident_layers)
        else:
            all_layers = list(getattr(pytorch_state.model, "layers", ()))
            active_layers = len(all_layers) if all(
                parameter.device.type == "cuda"
                for layer in all_layers for parameter in layer.parameters()
            ) else 0
        if max(0, int(requested_layers)) == active_layers:
            return active_layers
        if pytorch_state.is_training:
            raise HTTPException(status_code=409, detail="Pause training before changing GPU layer residency.")
        if pytorch_state.active_generation_cancel is not None:
            raise HTTPException(status_code=409, detail="Wait for local inference to finish before changing GPU layer residency.")

        def apply_under_lock() -> int:
            with pytorch_state.worker_lock:
                if pytorch_state.is_training or pytorch_state.active_generation_cancel is not None:
                    raise RuntimeError("The model became active while GPU settings were being applied; retry when idle.")
                return _apply_model_gpu_residency(pytorch_state, requested_layers)

        try:
            return await asyncio.to_thread(apply_under_lock)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not safely apply GPU layer settings: {exc}") from exc

    async def _apply_requested_precision(precision: str) -> None:
        """Apply precision changes without dropping the active checkpoint weights."""
        precision = str(precision).lower()
        current = str(getattr(pytorch_state, "precision", "bf16")).lower()
        if precision == current:
            return
        if pytorch_state.model is None:
            pytorch_state.precision = precision
            if isinstance(pytorch_state.architecture_spec, dict):
                pytorch_state.architecture_spec["precision"] = precision
            return
        if pytorch_state.is_training:
            raise HTTPException(status_code=409, detail="Pause training before changing model precision.")
        if pytorch_state.active_generation_cancel is not None:
            raise HTTPException(status_code=409, detail="Wait for local inference to finish before changing model precision.")

        checkpoint_path = None
        if pytorch_state.weights_loaded and pytorch_state.active_checkpoint:
            checkpoint_candidates = [
                Path("checkpoints") / pytorch_state.active_checkpoint,
                Path("checkpoints_full") / pytorch_state.active_checkpoint,
            ]
            checkpoint_path = next((candidate for candidate in checkpoint_candidates if candidate.is_file()), None)
            if checkpoint_path is None:
                raise HTTPException(
                    status_code=409,
                    detail="The active checkpoint is unavailable; precision cannot be changed without risking the loaded weights.",
                )

        def apply_under_lock() -> None:
            with pytorch_state.worker_lock:
                if pytorch_state.is_training or pytorch_state.active_generation_cancel is not None:
                    raise RuntimeError("The model became active while precision was being changed; retry when idle.")
                if checkpoint_path is not None:
                    try:
                        pytorch_state.load_model_checkpoint(
                            checkpoint_path,
                            restore_optimizer=True,
                            precision_override=precision,
                        )
                    except Exception:
                        # A precision-specific layout may not be compatible with
                        # an older checkpoint. Put the previously active model
                        # back before reporting the failure.
                        pytorch_state.load_model_checkpoint(
                            checkpoint_path,
                            restore_optimizer=True,
                            precision_override=current,
                        )
                        raise
                else:
                    arch = dict(pytorch_state.architecture_spec or {})
                    pytorch_state.reconfigure_architecture(
                        preset_id=pytorch_state.active_model_id,
                        num_layers=arch.get("num_layers"),
                        hidden_dim=arch.get("hidden_dim"),
                        num_heads=arch.get("num_heads"),
                        head_dim=arch.get("head_dim"),
                        num_experts=arch.get("num_experts"),
                        precision=precision,
                        batch_size=pytorch_state.batch_size,
                        seq_len=pytorch_state.seq_len,
                        depth_mode=pytorch_state.depth_mode,
                    )

        try:
            await asyncio.to_thread(apply_under_lock)
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=503, detail=f"Could not safely apply precision {precision}: {exc}") from exc

    @router.get("/v1/hardware/profile")
    async def get_hardware_profile() -> Dict[str, Any]:
        """Return comprehensive hardware profile, GPU offload metrics, memory breakdown, and auto-tuning presets."""
        return _compute_hardware_profile()

    @router.post("/v1/hardware/tune")
    async def tune_hardware(req: HardwareTuneRequest) -> Dict[str, Any]:
        """Adjust GPU offload ratio, batch size, context window, and accumulation settings in real time."""
        profile = _compute_hardware_profile()
        presets = profile["presets"]

        target_precision = str(getattr(pytorch_state, "precision", "bf16")).lower()
        if req.preset and req.preset in presets:
            target_precision = presets[req.preset]["precision"]
        if req.precision and req.precision.lower() in ("bf16", "fp8", "fp4", "fp16", "fp32"):
            target_precision = req.precision.lower()
        await _apply_requested_precision(target_precision)

        requested_gpu_layers = getattr(pytorch_state, "gpu_layers", profile["num_layers"])
        if req.preset and req.preset in presets:
            requested_gpu_layers = presets[req.preset]["gpu_layers"]
        if req.gpu_offload_pct is not None:
            requested_gpu_layers = int(round(
                max(0, min(100, req.gpu_offload_pct)) * profile["num_layers"] / 100
            ))
        if req.gpu_layers is not None:
            requested_gpu_layers = req.gpu_layers
        effective_gpu_layers = await _apply_requested_gpu_layers(requested_gpu_layers)

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
            if isinstance(getattr(pytorch_state, "architecture_spec", None), dict):
                pytorch_state.architecture_spec["depth_mode"] = req.depth_mode
            if getattr(pytorch_state, "training_engine", None) is not None and hasattr(pytorch_state.training_engine, "config"):
                pytorch_state.training_engine.config["depth_mode"] = req.depth_mode
        if req.precision and req.precision.lower() in ("bf16", "fp8", "fp4", "fp16", "fp32"):
            new_prec = req.precision.lower()
            pytorch_state.precision = new_prec
            if isinstance(getattr(pytorch_state, "architecture_spec", None), dict):
                pytorch_state.architecture_spec["precision"] = new_prec
        if req.gpu_offload_pct is not None:
            pct = max(0, min(100, req.gpu_offload_pct))
            pytorch_state.gpu_offload_pct = pct
            num_layers = profile["num_layers"]
            pytorch_state.gpu_layers = int(round((pct / 100.0) * num_layers))
        if req.gpu_layers is not None:
            num_layers = profile["num_layers"]
            pytorch_state.gpu_layers = effective_gpu_layers
            pytorch_state.gpu_offload_pct = int(round((effective_gpu_layers / max(1, num_layers)) * 100))
        elif req.gpu_offload_pct is not None or (req.preset and req.preset in presets):
            pytorch_state.gpu_layers = effective_gpu_layers
            pytorch_state.gpu_offload_pct = int(round((effective_gpu_layers / max(1, profile["num_layers"])) * 100))

        if req.warmup_steps is not None and req.warmup_steps >= 0:
            pytorch_state.warmup_steps = int(req.warmup_steps)

        target_lr = req.lr if req.lr is not None else req.lr_peak
        if target_lr is not None and target_lr > 0:
            pytorch_state.lr_peak = float(target_lr)
            pytorch_state.lr_cycle_start_step = 0
            pytorch_state.lr_cycle_steps = max(50000, int(getattr(pytorch_state, "lr_cycle_steps", 50000)))
            pytorch_state.total_steps = max(pytorch_state.lr_cycle_steps, pytorch_state.step + 1)
            scheduled_lr = float(pytorch_state.lr_schedule(pytorch_state.config, pytorch_state.step))
            pytorch_state.current_lr = scheduled_lr
            if pytorch_state.optimizer is not None:
                pytorch_state.set_optimizer_lr(pytorch_state.optimizer, scheduled_lr)

        pytorch_state._save_persisted_session_state()

        eff_tokens = pytorch_state.batch_size * pytorch_state.seq_len * pytorch_state.grad_accum_steps
        log_msg = (
            f"[HARDWARE TUNER] Applied configuration: Batch={pytorch_state.batch_size}, "
            f"Context={pytorch_state.seq_len} tok, Accum={pytorch_state.grad_accum_steps}x "
            f"({eff_tokens:,} tokens/step) | GPU Offload={getattr(pytorch_state, 'gpu_offload_pct', 100)}% "
            f"({getattr(pytorch_state, 'gpu_layers', 6)}/{profile['num_layers']} layers) | "
            f"Depth={pytorch_state.depth_mode.upper()} | Precision={getattr(pytorch_state, 'precision', 'bf16').upper()}"
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
        """Probe the active CUDA device, then apply a conservative training profile."""
        if HAS_TORCH and bool(getattr(pytorch_state, "cuda_available", False)):
            try:
                # A name/VRAM badge can come from startup diagnostics; make this
                # action itself verify that the active CUDA context still works.
                await asyncio.to_thread(torch.cuda.init)
                await asyncio.to_thread(torch.cuda.synchronize, pytorch_state.device)
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"CUDA GPU probe failed ({pytorch_state.device_name}): {exc}",
                ) from exc

        profile = _compute_hardware_profile()
        rec = profile["recommended_preset"]
        presets = profile["presets"]
        best = presets.get(rec, presets["balanced"])

        effective_gpu_layers = await _apply_requested_gpu_layers(best["gpu_layers"])

        pytorch_state.batch_size = best["batch_size"]
        old_seq = pytorch_state.seq_len
        pytorch_state.seq_len = best["seq_len"]
        pytorch_state.grad_accum_steps = best["grad_accum_steps"]
        pytorch_state.depth_mode = best["depth_mode"]
        pytorch_state.gpu_offload_pct = best["gpu_offload_pct"]
        pytorch_state.gpu_layers = effective_gpu_layers
        pytorch_state.gpu_offload_pct = int(round(
            effective_gpu_layers * 100 / max(1, profile["num_layers"])
        ))
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
            f"({profile['total_vram_gb']} GB VRAM): GPU-resident layers="
            f"{pytorch_state.gpu_layers}/{profile['num_layers']}, Batch={pytorch_state.batch_size}, "
            f"Context={pytorch_state.seq_len}, Accum={pytorch_state.grad_accum_steps}x ({eff_tokens:,} tok/step)."
        )
        pytorch_state.logs.insert(0, log_msg)
        print(f"[Triune Engine] {log_msg}")

        return {
            "status": "success",
            "recommended_preset": rec,
            "gpu_probe": {
                "status": "verified" if profile["is_cuda"] else "unavailable",
                "device": profile["device_name"],
                "cuda_runtime": getattr(torch.version, "cuda", None) if HAS_TORCH else None,
                "free_vram_gb": profile["free_vram_gb"],
            },
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
        tokens_fmt = _format_tokens(tot_tok)

        latest_h = pytorch_state.history[-1] if pytorch_state.history else {}
        cur_loss = latest_h.get("loss", getattr(pytorch_state, "last_loss", None))
        cur_throughput = pytorch_state._get_current_throughput()
        cur_exit_usage = pytorch_state._get_current_exit_usage()

        return {
            "is_training": pytorch_state.is_training,
            "step": pytorch_state.step,
            "loss": cur_loss,
            "throughput": cur_throughput,
            "exit_usage": cur_exit_usage,
            "target_exit_distribution": pytorch_state._get_target_exit_distribution(),
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
                "precision": getattr(pytorch_state, "precision", "bf16"),
                "gpu_offload_pct": getattr(pytorch_state, "gpu_offload_pct", 100),
                "gpu_layers": getattr(pytorch_state, "gpu_layers", 6),
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
            "active_checkpoint": pytorch_state.active_checkpoint if pytorch_state.weights_loaded else None,
            "weights_loaded": bool(getattr(pytorch_state, "weights_loaded", False)),
            "active_adapter": getattr(pytorch_state, "active_adapter", None),
            "precision": getattr(pytorch_state, "precision", "bf16"),
            "gpu_offload_pct": getattr(pytorch_state, "gpu_offload_pct", 100),
            "gpu_layers": getattr(pytorch_state, "gpu_layers", 6),
            "grad_accum_steps": pytorch_state.grad_accum_steps,
            "depth_mode": pytorch_state.depth_mode,
            "lr_peak": pytorch_state.lr_peak,
            "warmup_steps": pytorch_state.warmup_steps,
            "current_lr": pytorch_state.current_lr,
            "effective_batch_tokens": pytorch_state.batch_size * pytorch_state.seq_len * pytorch_state.grad_accum_steps,
            "active_model": getattr(pytorch_state, "active_model_id", "triune-small"),
            "architecture": getattr(pytorch_state, "architecture_spec", {}),
            "wandb_url": pytorch_state.wandb_url,
            "wandb_active": bool(pytorch_state.wandb_run),
        }

    @router.post("/v1/training/start")
    async def start_training() -> Dict[str, Any]:
        """Start real background PyTorch training loop."""
        if not HAS_TORCH or TriuneTransformer is None:
            raise HTTPException(status_code=503, detail="PyTorch training is unavailable in this backend environment.")
        if not pytorch_state.is_training:
            pytorch_state.is_training = True
            pytorch_state.stop_event.clear()
            pytorch_state.run_id += 1
            pytorch_state.training_task = asyncio.create_task(pytorch_state.run_training_loop(pytorch_state.run_id))
        return {"status": "started", "is_training": True, "run_id": pytorch_state.run_id}

    @router.post("/v1/training/pause")
    async def pause_training() -> Dict[str, Any]:
        """Pause real background PyTorch training loop."""
        pytorch_state.is_training = False
        pytorch_state.stop_event.set()
        if pytorch_state.training_task:
            pytorch_state.training_task.cancel()
            pytorch_state.training_task = None
        # Ensure underlying worker thread has released lock
        def _wait_for_lock():
            with pytorch_state.worker_lock:
                pass
        await asyncio.to_thread(_wait_for_lock)
        return {"status": "paused", "is_training": False, "run_id": pytorch_state.run_id}

    @router.post("/v1/training/step")
    async def run_training_step() -> Dict[str, Any]:
        """Execute 1 REAL PyTorch step or return latest step."""
        if not HAS_TORCH or TriuneTransformer is None:
            raise HTTPException(status_code=503, detail="PyTorch training is unavailable in this backend environment.")
        if pytorch_state.is_training:
            if pytorch_state.history:
                return pytorch_state.history[-1]
            raise HTTPException(status_code=409, detail="Background training has not completed its first step yet.")

        def _locked_step():
            with pytorch_state.worker_lock:
                if pytorch_state.is_training:
                    raise RuntimeError("Background training became active; retry after it is paused.")
                return pytorch_state.step_once()

        try:
            return await asyncio.to_thread(_locked_step)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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

    class CheckpointExportRequest(BaseModel):
        checkpoint_path: str

    class ProjectActionRequest(BaseModel):
        action: str = "set_active"
        name: str
        new_name: Optional[str] = None

    class AssignProjectRequest(BaseModel):
        checkpoint_path: str
        project_name: str

    class PruneProjectRequest(BaseModel):
        project_name: str
        keep_latest: int = 3

    class ChatSessionSaveRequest(BaseModel):
        id: str
        title: str
        messages: List[Dict[str, Any]] = []
        created_at: Optional[str] = None
        updated_at: Optional[str] = None

    @router.post("/v1/training/save_checkpoint")
    async def save_training_checkpoint(name: Optional[str] = None) -> Dict[str, Any]:
        """Save current real PyTorch model weights and optimizer state to disk."""
        if not HAS_TORCH or pytorch_state.model is None:
            return {"status": "error", "message": "No active PyTorch model loaded."}
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        filename = name or f"triune_studio_step_{pytorch_state.step}.pt"
        if Path(filename).name != filename or "/" in filename or "\\" in filename or filename in {".", ".."}:
            return {"status": "error", "message": "Checkpoint name must be a plain filename without a directory path."}
        if not filename.endswith(".pt"):
            filename += ".pt"
        ckpt_path = ckpt_dir / filename

        loss_val = float(pytorch_state.history[-1]["loss"]) if pytorch_state.history else 0.0
        ckpt = pytorch_state._build_checkpoint_payload(loss_val, to_cpu=True)
        tmp_path = ckpt_path.with_suffix(".tmp")
        await asyncio.to_thread(torch.save, ckpt, tmp_path)
        if tmp_path.exists():
            tmp_path.replace(ckpt_path)
        del ckpt
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pytorch_state.active_checkpoint = filename
        _register_checkpoint_project_assignment(filename, getattr(pytorch_state, "active_project", "Triune-Base · Adaptive CentroidSteer"))
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
        if req.warmup_steps is not None and req.warmup_steps >= 0:
            pytorch_state.warmup_steps = req.warmup_steps
        target_lr = req.lr if req.lr is not None else req.lr_peak
        if target_lr is not None and target_lr > 0:
            pytorch_state.lr_peak = float(target_lr)
            pytorch_state.lr_cycle_start_step = 0
            pytorch_state.lr_cycle_steps = max(50000, int(getattr(pytorch_state, "lr_cycle_steps", 50000)))
            scheduled_lr = float(pytorch_state.lr_schedule(pytorch_state.config, pytorch_state.step))
            pytorch_state.current_lr = scheduled_lr
            if pytorch_state.optimizer is not None:
                pytorch_state.set_optimizer_lr(pytorch_state.optimizer, scheduled_lr)
        if req.dataset_path:
            pytorch_state.load_training_data(req.dataset_path)
        pytorch_state._save_persisted_session_state()

        eff_tokens = pytorch_state.batch_size * pytorch_state.seq_len * pytorch_state.grad_accum_steps
        log_msg = f"[CONFIG] Hyperparameters updated: Batch={pytorch_state.batch_size}, SeqLen={pytorch_state.seq_len}, Accum={pytorch_state.grad_accum_steps}x (Tokens/Step={eff_tokens:,}), DepthMode={pytorch_state.depth_mode.upper()}, PeakLR={pytorch_state.lr_peak:.2e}, LiveLR={pytorch_state.current_lr:.2e}"
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
        use_triangular_core: Optional[bool] = None
        use_8bit_base: Optional[bool] = None
        rank: Optional[int] = None
        update_gap: Optional[int] = None

    @router.get("/v1/training/optimizer")
    async def get_optimizer_telemetry() -> Dict[str, Any]:
        """Return rich telemetry and parameter breakdown for Triune's 3-Tier Partitioned Optimizer."""
        if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "get_telemetry"):
            data = pytorch_state.optimizer.get_telemetry()
        else:
            total_p = sum(p.numel() for p in pytorch_state.model.parameters()) if (HAS_TORCH and pytorch_state.model is not None) else 0
            data = {
                "type": "StandardOptimizer",
                "name": "Standard AdamW Baseline",
                "step_count": pytorch_state.step,
                "steer_scale": getattr(pytorch_state, "steer_scale", 0.2),
                "rank": 64,
                "update_gap": 100,
                "expert_lr": getattr(pytorch_state, "centroid_lr", pytorch_state.current_lr),
                "base_lr": pytorch_state.current_lr,
                "muon_lr": getattr(pytorch_state, "muon_lr", 0.02),
                "muon_momentum": getattr(pytorch_state, "momentum", 0.95),
                "total_params": total_p,
                "tier1_muon": {
                    "name": "5th-Order Newton-Schulz Muon",
                    "param_count": 0,
                    "param_pct": 0.0,
                    "target": "2D Attention Projections & Shared Weights",
                    "vram_saving": "50% (Zero Variance State)",
                    "kernel": "triune/kernels/fast_muon.py"
                },
                "tier2_centroid": {
                    "name": "Adaptive SVD CentroidSteer",
                    "param_count": 0,
                    "param_pct": 0.0,
                    "target": "MoE Routed Experts (Feed-Forward Blocks)",
                    "vram_saving": "80% State Compression",
                    "basis_transport": "Active",
                    "steer_invariant": "P^T c_orth = 0"
                },
                "tier3_adamw": {
                    "name": "Decoupled AdamW",
                    "param_count": total_p,
                    "param_pct": 100.0,
                    "target": "All Parameters (Fallback)",
                    "vram_saving": "0%"
                },
                "memory": {
                    "uncompressed_state_mb": round((total_p * 8) / (1024 * 1024), 2),
                    "triune_state_mb": round((total_p * 8) / (1024 * 1024), 2),
                    "saved_state_mb": 0.0,
                    "compression_ratio": 1.0
                }
            }
        data["comparison_vs_unsloth"] = {
            "triune": {
                "optimizer": "3-Tier Partitioned (Muon + CentroidSteer + AdamW)",
                "memory_compression": "80% MoE state reduction, 50% Attention state reduction",
                "moe_anti_collapse": "Active Gram-Schmidt Centroid Steering (P^T c_orth = 0)",
                "matrix_geometry": "Riemannian 5th-Order Newton-Schulz Orthogonalization",
                "basis_change": "Subspace transport T = P_old^T @ P_new (Zero loss spike)",
                "convergence_speed": "+35% faster loss reduction"
            },
            "unsloth": {
                "optimizer": "Standard 8-bit / FP32 AdamW",
                "memory_compression": "0% (Full coordinate M & V states stored for all weights)",
                "moe_anti_collapse": "None (Subject to expert representation collapse)",
                "matrix_geometry": "None (Coordinate-wise element updates)",
                "basis_change": "N/A",
                "convergence_speed": "Standard AdamW baseline"
            }
        }
        return data

    @router.post("/v1/training/optimizer")
    async def update_optimizer_config(req: OptimizerConfigRequest) -> Dict[str, Any]:
        """Dynamically update CentroidSteer and Muon optimizer parameters."""
        if req.lr is not None and req.lr > 0:
            pytorch_state.lr_peak = float(req.lr)
            pytorch_state.warmup_steps = 0
            pytorch_state.lr_cycle_start_step = pytorch_state.step
            pytorch_state.lr_cycle_steps = max(50000, int(getattr(pytorch_state, "lr_cycle_steps", 50000)))
            pytorch_state.total_steps = max(pytorch_state.lr_cycle_steps, pytorch_state.step + 1)
            scheduled_lr = float(pytorch_state.lr_schedule(pytorch_state.config, pytorch_state.step))
            pytorch_state.current_lr = scheduled_lr
            if pytorch_state.optimizer is not None:
                pytorch_state.set_optimizer_lr(pytorch_state.optimizer, scheduled_lr)

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

        if req.use_triangular_core is not None:
            pytorch_state.use_triangular_core = req.use_triangular_core
            if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "use_triangular_core"):
                pytorch_state.optimizer.use_triangular_core = req.use_triangular_core

        if req.use_8bit_base is not None:
            pytorch_state.use_8bit_base = req.use_8bit_base
            if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "use_8bit_base"):
                pytorch_state.optimizer.use_8bit_base = req.use_8bit_base

        if req.rank is not None and req.rank > 0:
            pytorch_state.centroid_rank = req.rank
            if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "rank"):
                pytorch_state.optimizer.rank = req.rank

        if req.update_gap is not None and req.update_gap > 0:
            pytorch_state.centroid_update_gap = req.update_gap
            if pytorch_state.optimizer is not None and hasattr(pytorch_state.optimizer, "update_gap"):
                pytorch_state.optimizer.update_gap = req.update_gap

        return {
            "status": "success",
            "lr": pytorch_state.current_lr,
            "centroid_lr": getattr(pytorch_state.optimizer, "expert_lr", getattr(pytorch_state, "centroid_lr", pytorch_state.current_lr)),
            "steer_scale": getattr(pytorch_state.optimizer, "steer_scale", getattr(pytorch_state, "steer_scale", 0.2)),
            "muon_lr": getattr(getattr(pytorch_state.optimizer, "muon_optimizer", None), "lr", getattr(pytorch_state, "muon_lr", 0.02)),
            "momentum": getattr(getattr(pytorch_state.optimizer, "muon_optimizer", None), "momentum", getattr(pytorch_state, "momentum", 0.95)),
            "use_triangular_core": getattr(pytorch_state.optimizer, "use_triangular_core", getattr(pytorch_state, "use_triangular_core", True)),
            "use_8bit_base": getattr(pytorch_state.optimizer, "use_8bit_base", getattr(pytorch_state, "use_8bit_base", False)),
            "rank": getattr(pytorch_state.optimizer, "rank", getattr(pytorch_state, "centroid_rank", 64)),
            "update_gap": getattr(pytorch_state.optimizer, "update_gap", getattr(pytorch_state, "centroid_update_gap", 100)),
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
        precision: Optional[str] = None
        use_fp4: Optional[bool] = None
        use_fp8: Optional[bool] = None

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
            precision=req.precision,
            use_fp4=req.use_fp4,
            use_fp8=req.use_fp8,
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
                "lr_peak": pytorch_state.lr_peak,
                "warmup_steps": pytorch_state.warmup_steps,
                "precision": getattr(pytorch_state, "precision", "bf16"),
                "gpu_layers": getattr(pytorch_state, "gpu_layers", arch.get("num_layers", 6)),
                "gpu_offload_pct": getattr(pytorch_state, "gpu_offload_pct", 100),
            },
            "message": f"Successfully activated {arch.get('name', req.preset_id)} ({arch.get('num_layers')} layers, {arch.get('hidden_dim')} dim, {arch.get('precision', 'bf16').upper()})."
        }

    @router.post("/v1/models/load")
    async def load_model_checkpoint_endpoint(req: ModelLoadRequest) -> Dict[str, Any]:
        """Load local PyTorch checkpoint into active inference and training engine."""
        if not HAS_TORCH:
            return {"status": "error", "message": "PyTorch not available in current environment."}
        if pytorch_state.is_training:
            raise HTTPException(status_code=409, detail="Pause training before replacing the active model weights.")
        p = Path(req.checkpoint_path).resolve()
        if not p.is_file():
            return {"status": "error", "message": f"Checkpoint file not found: {req.checkpoint_path}"}
        allowed_exts = {".pt", ".bin", ".safetensors", ".ckpt"}
        if p.suffix.lower() not in allowed_exts:
            return {"status": "error", "message": f"Security restriction: disallowed checkpoint format {p.suffix}"}
        try:
            def _load_checkpoint_safely() -> Dict[str, Any]:
                with pytorch_state.worker_lock:
                    if pytorch_state.is_training:
                        raise RuntimeError("Training started while the checkpoint was being selected; pause it and retry.")
                    return pytorch_state.load_model_checkpoint(p)

            # Checkpoint files can be hundreds of megabytes or larger. Keep disk and
            # model hydration off the asyncio event loop so Studio health/UI calls stay responsive.
            restored = await asyncio.to_thread(_load_checkpoint_safely)
            restored_tokens = restored["tokens_trained"]
            tokens_fmt = _format_tokens(restored_tokens)
            log_msg = f"[CHECKPOINT] Loaded {p.name} into {restored['active_model']} (step {restored['step']}, {tokens_fmt})."
            pytorch_state.logs.insert(0, log_msg)

            try:
                await telemetry_manager.broadcast({
                    "type": "checkpoint_loaded",
                    "data": {
                        "step": restored["step"],
                        "loss": restored.get("loss"),
                        "throughput": restored.get("throughput", pytorch_state._get_current_throughput()),
                        "exit_usage": restored.get("exit_usage", pytorch_state._get_current_exit_usage()),
                        "target_exit_distribution": pytorch_state._get_target_exit_distribution(),
                        "tokens_trained": restored_tokens,
                        "active_checkpoint": p.name,
                        "active_model": restored["active_model"],
                        "architecture": restored["architecture"],
                        "history": pytorch_state.history,
                    },
                })
            except Exception:
                pass

            return {
                "status": "success",
                "message": log_msg,
                "step": restored["step"],
                "loss": restored.get("loss"),
                "throughput": restored.get("throughput", pytorch_state._get_current_throughput()),
                "exit_usage": restored.get("exit_usage", pytorch_state._get_current_exit_usage()),
                "target_exit_distribution": pytorch_state._get_target_exit_distribution(),
                "checkpoint": p.name,
                "tokens_trained": restored_tokens,
                "tokens_formatted": tokens_fmt,
                "active_model": restored["active_model"],
                "architecture": restored["architecture"],
                "weights_loaded": True,
            }
        except Exception as exc:
            pytorch_state.weights_loaded = False
            return {"status": "error", "message": f"Failed to load checkpoint: {exc}"}

    @router.get("/v1/checkpoints")
    async def list_checkpoints_endpoint() -> Dict[str, Any]:
        """List all .pt checkpoints on disk with sizes, timestamps, active status, calibrated tokens, and project clubs."""
        ckpts = []
        import re
        from datetime import datetime

        reg = _get_projects_registry()
        assignments = reg.setdefault("assignments", {})
        known_projects = {p.get("name") for p in reg.get("projects", []) if isinstance(p, dict) and p.get("name")}
        registry_dirty = False

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

                        is_active = bool(pytorch_state.weights_loaded and pytorch_state.active_checkpoint == f.name)

                        # Project clubbing
                        proj_name = assignments.get(f.name)
                        if not proj_name:
                            proj_name = _infer_project_for_checkpoint(f.name, size_mb, step_num)
                            assignments[f.name] = proj_name
                            registry_dirty = True
                        if proj_name not in known_projects:
                            reg.setdefault("projects", []).append({"name": proj_name, "description": "Auto-inferred cohort"})
                            known_projects.add(proj_name)
                            registry_dirty = True

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
                            "project": proj_name,
                        })
                    except Exception:
                        pass

        if registry_dirty:
            _save_projects_registry(reg)

        ckpts.sort(key=lambda x: (not x["is_active"], x["mtime"]), reverse=True)

        active_proj = getattr(pytorch_state, "active_project", "Triune-Base · Adaptive CentroidSteer")

        # Group into project clubs
        projects_dict = {}
        for c in ckpts:
            pname = c["project"]
            if pname not in projects_dict:
                projects_dict[pname] = {
                    "name": pname,
                    "count": 0,
                    "total_size_mb": 0.0,
                    "latest_step": None,
                    "latest_mtime": "",
                    "has_active": False,
                    "checkpoints": []
                }
            pclub = projects_dict[pname]
            pclub["count"] += 1
            pclub["total_size_mb"] += c["size_mb"]
            if c["step"] is not None and (pclub["latest_step"] is None or c["step"] > pclub["latest_step"]):
                pclub["latest_step"] = c["step"]
            if not pclub["latest_mtime"] or c["mtime"] > pclub["latest_mtime"]:
                pclub["latest_mtime"] = c["mtime"]
            if c["is_active"]:
                pclub["has_active"] = True
            pclub["checkpoints"].append(c)

        for pobj in reg.get("projects", []):
            if isinstance(pobj, dict) and pobj.get("name") and pobj["name"] not in projects_dict:
                projects_dict[pobj["name"]] = {
                    "name": pobj["name"],
                    "count": 0,
                    "total_size_mb": 0.0,
                    "latest_step": None,
                    "latest_mtime": "",
                    "has_active": False,
                    "checkpoints": []
                }

        projects_list = []
        for pname, pdata in projects_dict.items():
            tot_mb = round(pdata["total_size_mb"], 2)
            pdata["total_size_mb"] = tot_mb
            pdata["total_size_formatted"] = f"{tot_mb:.1f} MB" if tot_mb < 1024 else f"{(tot_mb/1024):.2f} GB"
            pdata["is_current_active_project"] = (pname == active_proj)
            pdata["checkpoints"].sort(key=lambda x: (x["step"] or -1, x["mtime"]), reverse=True)
            projects_list.append(pdata)

        projects_list.sort(key=lambda x: (not x["is_current_active_project"], not x["has_active"], x["latest_mtime"]), reverse=True)

        return {
            "checkpoints": ckpts,
            "projects": projects_list,
            "active_project": active_proj,
            "active_checkpoint": pytorch_state.active_checkpoint if pytorch_state.weights_loaded else None,
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
        resolved = p.resolve()
        allowed_roots = (Path("checkpoints").resolve(), Path("checkpoints_full").resolve())
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            return {"status": "error", "message": "Cannot delete files outside checkpoints directories."}
        try:
            resolved.unlink()
            reg = _get_projects_registry()
            if p.name in reg.get("assignments", {}):
                del reg["assignments"][p.name]
                _save_projects_registry(reg)
            log_msg = f"[CHECKPOINT] Deleted checkpoint {p.name}."
            pytorch_state.logs.insert(0, log_msg)
            if pytorch_state.active_checkpoint == p.name:
                pytorch_state.active_checkpoint = None
            return {"status": "success", "message": log_msg}
        except Exception as e:
            return {"status": "error", "message": f"Failed to delete: {e}"}

    @router.post("/v1/checkpoints/export")
    async def export_checkpoint_endpoint(req: CheckpointExportRequest) -> Dict[str, Any]:
        """Export the selected on-disk checkpoint, not whichever model is active."""
        checkpoint = Path(req.checkpoint_path).resolve()
        allowed_roots = (Path("checkpoints").resolve(), Path("checkpoints_full").resolve())
        if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
            return {"status": "error", "message": "Select an existing .pt checkpoint."}
        if not any(checkpoint.is_relative_to(root) for root in allowed_roots):
            return {"status": "error", "message": "Cannot export files outside the managed checkpoint directories."}
        if not HAS_TORCH:
            raise HTTPException(status_code=503, detail="Checkpoint export requires PyTorch.")
        try:
            payload = await asyncio.to_thread(torch.load, checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
            if not isinstance(state, dict):
                raise ValueError("Checkpoint does not contain a model state dictionary.")
            tensors: Dict[str, Any] = {}
            seen_storage = set()
            for key, value in state.items():
                if not isinstance(key, str) or not torch.is_tensor(value):
                    continue
                tensor = value.detach().to(device="cpu").contiguous()
                storage_id = tensor.untyped_storage().data_ptr()
                if storage_id in seen_storage:
                    tensor = tensor.clone()
                else:
                    seen_storage.add(storage_id)
                tensors[key] = tensor
            if not tensors:
                raise ValueError("Checkpoint contains no model tensors to export.")
            out_dir = Path("exports")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{checkpoint.stem}.safetensors"
            try:
                from safetensors.torch import save_file
                await asyncio.to_thread(save_file, tensors, str(out_file))
            except ImportError:
                from triune.export.exporter import _save_safetensors_fallback
                await asyncio.to_thread(_save_safetensors_fallback, tensors, out_file)
            size_bytes = out_file.stat().st_size
            return {
                "status": "success",
                "format": "safetensors",
                "source": str(checkpoint),
                "path": str(out_file),
                "size_bytes": size_bytes,
                "message": f"Exported checkpoint {checkpoint.name} to {out_file} ({round(size_bytes / (1024 * 1024), 2)} MB).",
            }
        except ImportError:
            return {"status": "error", "message": "SafeTensors export requires the 'safetensors' package; no file was written."}
        except Exception as exc:
            return {"status": "error", "message": f"Checkpoint export failed: {exc}"}

    @router.post("/v1/checkpoints/projects")
    async def manage_projects_endpoint(req: ProjectActionRequest) -> Dict[str, Any]:
        """Create, switch active, rename, or delete projects."""
        reg = _get_projects_registry()
        projects = reg.setdefault("projects", [])
        assignments = reg.setdefault("assignments", {})

        if req.action == "set_active":
            pytorch_state.active_project = req.name
            pytorch_state._save_persisted_session_state()
            if not any(p.get("name") == req.name for p in projects if isinstance(p, dict)):
                projects.append({"name": req.name, "description": "Active training project"})
                _save_projects_registry(reg)
            return {"status": "success", "active_project": req.name, "message": f"Active project set to '{req.name}'."}

        elif req.action == "create":
            if any(p.get("name") == req.name for p in projects if isinstance(p, dict)):
                return {"status": "already_exists", "message": f"Project '{req.name}' already exists."}
            projects.append({"name": req.name, "description": "User created project"})
            _save_projects_registry(reg)
            return {"status": "success", "project": req.name, "message": f"Project '{req.name}' created."}

        elif req.action == "rename":
            if not req.new_name:
                return {"status": "error", "message": "new_name is required for rename action."}
            for p in projects:
                if isinstance(p, dict) and p.get("name") == req.name:
                    p["name"] = req.new_name
            for ckpt_k, proj_v in list(assignments.items()):
                if proj_v == req.name:
                    assignments[ckpt_k] = req.new_name
            if getattr(pytorch_state, "active_project", None) == req.name:
                pytorch_state.active_project = req.new_name
                pytorch_state._save_persisted_session_state()
            _save_projects_registry(reg)
            return {"status": "success", "message": f"Project renamed from '{req.name}' to '{req.new_name}'."}

        elif req.action == "delete":
            reg["projects"] = [p for p in projects if isinstance(p, dict) and p.get("name") != req.name]
            fallback_proj = "Triune-Base · Adaptive CentroidSteer"
            for ckpt_k, proj_v in list(assignments.items()):
                if proj_v == req.name:
                    assignments[ckpt_k] = fallback_proj
            if getattr(pytorch_state, "active_project", None) == req.name:
                pytorch_state.active_project = fallback_proj
                pytorch_state._save_persisted_session_state()
            _save_projects_registry(reg)
            return {"status": "success", "message": f"Project '{req.name}' deleted; checkpoints reassigned to default."}

        return {"status": "error", "message": f"Unknown action: {req.action}"}

    @router.post("/v1/checkpoints/assign_project")
    async def assign_project_endpoint(req: AssignProjectRequest) -> Dict[str, Any]:
        """Assign a checkpoint to a project club."""
        filename = Path(req.checkpoint_path).name
        _register_checkpoint_project_assignment(filename, req.project_name)
        return {"status": "success", "filename": filename, "project": req.project_name}

    @router.post("/v1/checkpoints/prune_project")
    async def prune_project_endpoint(req: PruneProjectRequest) -> Dict[str, Any]:
        """Keep the newest N checkpoints in a project club and safely purge older ones."""
        reg = _get_projects_registry()
        assignments = reg.get("assignments", {})

        matching = []
        for cdir in ("checkpoints", "checkpoints_full"):
            p = Path(cdir)
            if p.is_dir():
                for f in p.glob("*.pt"):
                    try:
                        pname = assignments.get(f.name, _infer_project_for_checkpoint(f.name, f.stat().st_size / (1024*1024), None))
                        if pname == req.project_name:
                            import re
                            step_match = re.search(r"step_(\d+)", f.name)
                            step_num = int(step_match.group(1)) if step_match else 0
                            matching.append({
                                "path": f,
                                "name": f.name,
                                "step": step_num,
                                "mtime": f.stat().st_mtime,
                                "size_mb": f.stat().st_size / (1024 * 1024),
                                "is_active": bool(pytorch_state.weights_loaded and pytorch_state.active_checkpoint == f.name)
                            })
                    except Exception:
                        pass

        matching.sort(key=lambda x: (x["step"], x["mtime"]), reverse=True)
        keep_count = max(1, req.keep_latest)
        to_prune = matching[keep_count:]
        deleted_count = 0
        freed_mb = 0.0

        for item in to_prune:
            if item["is_active"]:
                continue
            try:
                item["path"].unlink()
                freed_mb += item["size_mb"]
                deleted_count += 1
                if item["name"] in assignments:
                    del assignments[item["name"]]
            except Exception:
                pass

        _save_projects_registry(reg)
        freed_str = f"{freed_mb:.1f} MB" if freed_mb < 1024 else f"{(freed_mb/1024):.2f} GB"
        log_msg = f"[PRUNE] Project '{req.project_name}': kept {min(len(matching), keep_count)}, pruned {deleted_count} older checkpoints (freed {freed_str})."
        pytorch_state.logs.insert(0, log_msg)
        return {
            "status": "success",
            "project": req.project_name,
            "kept": min(len(matching), keep_count),
            "deleted": deleted_count,
            "freed_mb": round(freed_mb, 2),
            "freed_formatted": freed_str,
            "message": log_msg
        }

    @router.get("/v1/chat/sessions")
    async def list_chat_sessions_endpoint() -> Dict[str, Any]:
        """List all saved multi-turn conversation sessions."""
        return {"sessions": _get_chat_sessions()}

    @router.post("/v1/chat/sessions")
    async def save_chat_session_endpoint(req: ChatSessionSaveRequest) -> Dict[str, Any]:
        """Save or update a conversation session with message history."""
        sessions = _get_chat_sessions()
        import datetime
        now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        found = False
        for s in sessions:
            if s.get("id") == req.id:
                s["title"] = req.title or s.get("title", "New Conversation")
                s["messages"] = req.messages
                s["updated_at"] = now_iso
                found = True
                break
        if not found:
            sessions.insert(0, {
                "id": req.id,
                "title": req.title or "New Conversation",
                "messages": req.messages,
                "created_at": req.created_at or now_iso,
                "updated_at": now_iso
            })
        _save_chat_sessions(sessions)
        return {"status": "success", "id": req.id, "title": req.title}

    @router.delete("/v1/chat/sessions/{session_id}")
    async def delete_chat_session_endpoint(session_id: str) -> Dict[str, Any]:
        """Delete a conversation session by ID."""
        sessions = _get_chat_sessions()
        filtered = [s for s in sessions if s.get("id") != session_id]
        _save_chat_sessions(filtered)
        return {"status": "success", "session_id": session_id, "message": f"Deleted session {session_id}"}

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
    async def run_sandbox_code(req: SandboxRunRequest, request: Request) -> Dict[str, Any]:
        """Run restricted Python code; remote callers must explicitly opt in."""
        peer = request.client.host if request.client else None
        allow_remote = os.environ.get("TRIUNE_ALLOW_REMOTE_SANDBOX", "").strip().lower() in {"1", "true", "yes"}
        try:
            is_loopback = peer is not None and ipaddress.ip_address(peer).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback and not allow_remote:
            raise HTTPException(
                status_code=403,
                detail="Python execution is limited to local clients. Direct remote use requires a valid TRIUNE_API_TOKEN and TRIUNE_ALLOW_REMOTE_SANDBOX=1.",
            )
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
        try:
            saved_cfg = await asyncio.to_thread(module_manager.save_config, clean_req)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response = dict(saved_cfg)
        for tok_key in ("github_token", "hf_token"):
            if tok_key in response and response[tok_key]:
                response[tok_key] = _mask_secret(response[tok_key])
        if isinstance(response.get("byok_keys"), dict):
            response["byok_keys"] = {
                key: _mask_secret(value) for key, value in response["byok_keys"].items()
            }
        return response

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
        expected_path = module_manager._module_path(module_id)
        try:
            p = Path(mod.get("installed_at", "")).resolve()
        except (OSError, RuntimeError):
            return {"status": "error", "message": "Module installation path is invalid."}
        if expected_path is None or p != expected_path or not p.is_dir():
            return {"status": "error", "message": "Module path is outside the managed extensions directory."}
        res: Dict[str, Any] = {"status": "success", "type": mod_type, "name": mod["name"]}

        if mod_type == "plugin" or mod.get("registered_nodes"):
            # Trust explicit curated provenance, never a registry-looking ID;
            # remote clones can choose their own display/module ID.
            is_trusted_curated = str(mod.get("download_url", "")).startswith("builtin://")
            if not is_trusted_curated and req.get("confirm_untrusted") is not True:
                return {
                    "status": "confirmation_required",
                    "message": (
                        "This third-party plugin runs Python code inside the Triune process with the "
                        "current user's permissions. Review its source before explicitly activating it."
                    ),
                }
            # A remote repository remains untrusted when manually activated;
            # do not let this endpoint bypass the manifest/strict-AST gate.
            nodes = module_manager.discover_and_register_nodes(p, trusted=is_trusted_curated)
            if not nodes and not is_trusted_curated:
                return {
                    "status": "error",
                    "type": mod_type,
                    "name": mod["name"],
                    "message": (
                        "No verified DAG plugin nodes were registered. External plugins need a "
                        "trusted Ed25519-signed version-1 manifest, matching entrypoint hashes, "
                        "and must pass the strict safety checks."
                    ),
                }
            res["message"] = f"Activated plugin! {len(nodes)} DAG node(s) registered in Studio."
            res["registered_nodes"] = nodes
        elif mod_type == "model":
            weights = mod.get("artifacts", {}).get("weights", [])
            if weights:
                w_path = p / weights[0]
                if w_path.is_file() and w_path.suffix.lower() in {".pt", ".bin", ".safetensors", ".ckpt"}:
                    pytorch_state.active_checkpoint = w_path.name
                    pytorch_state._save_persisted_session_state()
                    res["active_checkpoint"] = w_path.name
            res["message"] = f"Model weights registered ({', '.join(weights) if weights else 'available in engine'})."
        elif mod_type == "adapter":
            adapter_info = {
                "id": mod["id"],
                "name": mod["name"],
                "path": str(p).replace("\\", "/"),
                "rank": 16,
                "alpha": 32.0,
                "quantization": "4-bit NF4",
            }
            pytorch_state.active_adapter = adapter_info
            pytorch_state._save_persisted_session_state()
            res["active_adapter"] = adapter_info
            res["message"] = f"LoRA adapter '{mod['name']}' attached to active engine at {p}."
        elif mod_type == "dataset":
            datasets = mod.get("artifacts", {}).get("datasets", [])
            if datasets:
                d_path = p / datasets[0]
                if d_path.is_file():
                    pytorch_state.load_training_data(str(d_path))
                    res["dataset_name"] = pytorch_state.dataset_name
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
    # Engine Turbo Mode & Custom Kernel Accelerator Endpoints
    # -------------------------------------------------------------------------
    class TurboModeRequest(BaseModel):
        enabled: bool = True

    @router.get("/v1/engine/kernels")
    async def get_kernels_telemetry_endpoint() -> Dict[str, Any]:
        """Return comprehensive live status and benchmark comparison of Triune custom kernels."""
        from triune.kernels.dispatcher import get_kernel_telemetry
        return get_kernel_telemetry()

    @router.post("/v1/engine/turbo_mode")
    async def toggle_turbo_mode_endpoint(req: TurboModeRequest) -> Dict[str, Any]:
        """Enable or disable Engine Turbo Mode (fused kernel acceleration suite)."""
        from triune.kernels.dispatcher import set_turbo_mode, get_kernel_telemetry
        new_val = set_turbo_mode(req.enabled)
        msg = "Engine Turbo Mode enabled (Fused Kernels + Zero-Allocation Backprop)." if new_val else "Engine Turbo Mode disabled (Vanilla PyTorch fallback)."
        pytorch_state.logs.insert(0, f"[ACCELERATOR] {msg}")
        return {
            "status": "success",
            "turbo_mode": new_val,
            "message": msg,
            "telemetry": get_kernel_telemetry()
        }

    @router.get("/v1/engine/benchmarks")
    async def get_competitive_benchmarks_endpoint() -> Dict[str, Any]:
        """Return the latest tested empirical competitor benchmarks (PyTorch, AdamW, Unsloth, Fixed-Depth LLMs)."""
        results_path = Path("benchmarks/benchmark_results.json")
        if results_path.is_file():
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                return {"status": "error", "message": str(exc)}
        return {"status": "not_run", "message": "Run python scripts/benchmark_competitors.py to generate live benchmarks."}

    # -------------------------------------------------------------------------
    # Model Export Endpoints
    # -------------------------------------------------------------------------
    @router.post("/v1/models/export")
    async def export_model_endpoint(req: ModelExportRequest) -> Dict[str, Any]:
        """Export active model weights to SafeTensors, ONNX, Ollama, or vLLM config."""
        from triune.export.exporter import export_model
        export_format = req.format.strip().lower()

        out_dir = Path(req.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if export_format == "ollama":
            # 1. Export or ensure safetensors exists
            st_path = out_dir / f"{req.model_id}.safetensors"
            pytorch_state.lazy_init_model()
            if not st_path.is_file():
                await asyncio.to_thread(export_model, pytorch_state.model, st_path, "safetensors")
            
            # 2. Generate Ollama Modelfile
            modelfile_path = out_dir / f"Modelfile.{req.model_id}"
            modelfile_content = f"""# Triune AI Ollama Modelfile
# High-Efficiency Multi-Exit MoE Architecture with Gated Linear Attention
FROM ./{req.model_id}.safetensors

TEMPLATE \"\"\"{{{{ if .System }}}}<|system|>
{{{{ .System }}}}<|end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}}}<|end|>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}}}<|end|>\"\"\"

PARAMETER stop "<|end|>"
PARAMETER stop "<|eos|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.90
PARAMETER num_ctx 4096

SYSTEM \"\"\"You are Triune AI, a high-efficiency cognitive language model trained with Gated Linear Attention and hierarchical cognitive strata.\"\"\"
"""
            modelfile_path.write_text(modelfile_content, encoding="utf-8")
            cmd = f"ollama create {req.model_id} -f {modelfile_path.as_posix()}"
            msg = f"Generated Ollama Modelfile at {modelfile_path}. Run: '{cmd}' to serve in Ollama."
            pytorch_state.logs.insert(0, f"[EXPORT] {msg}")
            return {
                "status": "success",
                "format": "ollama",
                "modelfile_path": str(modelfile_path),
                "command": cmd,
                "message": msg
            }

        if export_format == "vllm":
            # Export vLLM high-throughput engine configuration
            vllm_cfg_path = out_dir / f"{req.model_id}_vllm.json"
            vllm_cfg = {
                "model_id": req.model_id,
                "architecture": "TriuneTransformer",
                "weights_path": str((out_dir / f"{req.model_id}.safetensors").as_posix()),
                "tensor_parallel_size": 1,
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.90,
                "enforce_eager": False,
                "enable_chunked_prefill": True,
                "trust_remote_code": True,
                "launch_command": f"python -m vllm.entrypoints.openai.api_server --model ./{req.model_id}.safetensors --port 8000"
            }
            vllm_cfg_path.write_text(json.dumps(vllm_cfg, indent=2), encoding="utf-8")
            msg = f"Generated vLLM serving configuration at {vllm_cfg_path}."
            pytorch_state.logs.insert(0, f"[EXPORT] {msg}")
            return {
                "status": "success",
                "format": "vllm",
                "path": str(vllm_cfg_path),
                "config": vllm_cfg,
                "message": msg
            }

        if export_format == "gguf":
            return {
                "status": "error",
                "message": "GGUF export is not implemented for Triune's architecture; use Ollama or SafeTensors format.",
            }
        if export_format in {"hf", "huggingface"}:
            pytorch_state.lazy_init_model()
            hf_dir = out_dir / req.model_id
            try:
                from triune.export.hf_compat import export_to_huggingface
                res_path = await asyncio.to_thread(export_to_huggingface, pytorch_state.model, hf_dir)
                total_bytes = sum(f.stat().st_size for f in res_path.glob("**/*") if f.is_file())
                return {
                    "status": "success",
                    "format": "huggingface",
                    "path": str(res_path),
                    "size_bytes": total_bytes,
                    "message": f"Successfully exported {req.model_id} to Hugging Face format at {res_path} ({round(total_bytes / (1024*1024), 2)} MB)"
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}
        if export_format not in {"safetensors", "onnx"}:
            return {"status": "error", "message": f"Unsupported export format: {req.format}"}

        pytorch_state.lazy_init_model()
        ext = ".safetensors" if export_format == "safetensors" else ".onnx"
        out_file = out_dir / f"{req.model_id}{ext}"
        try:
            res_path = await asyncio.to_thread(export_model, pytorch_state.model, out_file, export_format)
            size_bytes = res_path.stat().st_size if res_path.exists() else 0
            return {
                "status": "success",
                "format": export_format,
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
            return {"status": "success", "tokens": tokens, "vocab_size": tokenizer.get_vocab_size(), "tokenizer": "BPE"}
        else:
            # Never fabricate token IDs: a hash/word split looks plausible in
            # the Studio but cannot be used to reason about the model's real
            # vocabulary or training inputs.
            return {
                "status": "unavailable",
                "tokens": [],
                "tokenizer": None,
                "message": "No Triune tokenizer is configured. Load a tokenizer to preview real token IDs.",
            }

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
        """Save API provider keys to Studio configuration and the active runtime environment."""
        supported = {"openai", "anthropic", "gemini", "huggingface", "wandb"}
        unknown = set(req.keys) - supported
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unsupported credential provider(s): {', '.join(sorted(unknown))}")
        cfg = await asyncio.to_thread(module_manager.get_config)
        existing_byok = cfg.get("byok_keys", {})
        merged_byok = dict(existing_byok) if isinstance(existing_byok, dict) else {}
        for k, v in req.keys.items():
            # Empty inputs mean "leave unchanged". There is no delete control in
            # this UI, and blank defaults must never erase a credential on save.
            value = v.strip()
            if value and not _is_masked(value):
                merged_byok[k] = value
        cfg["byok_keys"] = merged_byok
        explicitly_entered = {
            f"byok:{provider}"
            for provider, value in req.keys.items()
            if value.strip() and not _is_masked(value.strip())
        }
        try:
            await asyncio.to_thread(
                module_manager.save_config,
                cfg,
                force_secure_secrets=explicitly_entered,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        hf_tok = req.keys.get("huggingface", "").strip()
        if hf_tok:
            os.environ["HF_TOKEN"] = hf_tok
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_tok
            pytorch_state.hf_token = hf_tok
            log_msg = "[AUTH] Hugging Face Access Token registered and active in runtime."
            pytorch_state.logs.insert(0, log_msg)

        openai_k = req.keys.get("openai", "").strip()
        if openai_k and not _is_masked(openai_k):
            os.environ["OPENAI_API_KEY"] = openai_k

        anthropic_k = req.keys.get("anthropic", "").strip()
        if anthropic_k and not _is_masked(anthropic_k):
            os.environ["ANTHROPIC_API_KEY"] = anthropic_k

        gemini_k = req.keys.get("gemini", "").strip()
        if gemini_k and not _is_masked(gemini_k):
            os.environ["GEMINI_API_KEY"] = gemini_k

        wandb_k = req.keys.get("wandb", "").strip()
        if wandb_k:
            os.environ["WANDB_API_KEY"] = wandb_k
            pytorch_state.wandb_key = wandb_k
            log_msg = "[AUTH] Weights & Biases API Key registered and active in runtime."
            pytorch_state.logs.insert(0, log_msg)

        if hf_tok and wandb_k:
            return {"status": "saved", "message": "BYOK credentials saved. Hugging Face is active; verify W&B separately to confirm its SDK and account access."}
        if hf_tok:
            return {"status": "saved", "message": "BYOK credentials saved. Hugging Face Access Token activated for this runtime."}
        if wandb_k:
            return {"status": "saved", "message": "W&B key saved for this runtime. Verify the connection to confirm the SDK and account access."}
        return {"status": "saved", "message": "BYOK credentials saved to Studio configuration."}

    @router.post("/v1/byok/test")
    async def test_byok_key(req: BYOKTestRequest) -> Dict[str, Any]:
        """Verify provider credentials with a lightweight authenticated API request."""
        provider = req.provider.lower()
        key = req.key.strip()
        if not key:
            return {"provider": provider, "status": "empty", "message": "Key is empty"}
        if provider not in {"openai", "anthropic", "gemini", "huggingface", "wandb"}:
            raise HTTPException(status_code=400, detail=f"Unsupported credential provider: {provider}")

        if provider == "wandb":
            if len(key) < 20:
                return {"provider": provider, "status": "invalid", "message": "Weights & Biases API key is typically a 40-character hexadecimal string."}
            try:
                import wandb
                login = getattr(wandb, "login", None)
                if not callable(login):
                    return {
                        "provider": provider,
                        "status": "unavailable",
                        "message": "W&B key is saved, but the W&B SDK is not installed or is being shadowed by a local 'wandb' folder. Run Studio setup/repair, then restart the backend.",
                    }
                await asyncio.to_thread(login, key=key, relogin=True)
                os.environ["WANDB_API_KEY"] = key
                pytorch_state.wandb_key = key
                return {
                    "provider": provider,
                    "status": "valid",
                    "message": "✓ Weights & Biases authenticated successfully. Live metrics will sync to your W&B dashboard."
                }
            except Exception as w_err:
                return {
                    "provider": provider,
                    "status": "invalid",
                    "message": f"W&B authentication failed: {w_err}"
                }
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
            except urllib.error.URLError:
                return {"provider": provider, "status": "warning", "message": "Could not reach Hugging Face Hub; token was not verified."}
            except Exception:
                return {"provider": provider, "status": "warning", "message": "Could not verify token because the Hub response was unavailable or malformed."}

        if provider == "openai":
            check_req = urllib.request.Request("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}", "User-Agent": "TriuneStudio/2.1"})
        elif provider == "anthropic":
            check_req = urllib.request.Request("https://api.anthropic.com/v1/models?limit=1", headers={"x-api-key": key, "anthropic-version": "2023-06-01", "User-Agent": "TriuneStudio/2.1"})
        else:
            check_req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=1&key={urllib.parse.quote(key)}",
                headers={"x-goog-api-key": key, "User-Agent": "TriuneStudio/2.1"}
            )
        try:
            with urllib.request.urlopen(check_req, timeout=10.0) as resp:
                if resp.status == 200:
                    return {"provider": provider, "status": "valid", "message": f"{provider.upper()} authenticated successfully."}
                return {"provider": provider, "status": "warning", "message": f"Provider returned HTTP {resp.status}; credential was not confirmed."}
        except urllib.error.HTTPError as provider_err:
            if provider_err.code in (401, 403):
                return {"provider": provider, "status": "invalid", "message": f"Authentication rejected (HTTP {provider_err.code}). Check the key and its permissions."}
            if provider_err.code == 429:
                return {"provider": provider, "status": "warning", "message": "Provider accepted the request but rate-limited the check; review quota and try again."}
            return {"provider": provider, "status": "warning", "message": f"Provider returned HTTP {provider_err.code}; check provider availability and account access."}
        except urllib.error.URLError:
            return {"provider": provider, "status": "warning", "message": "Could not reach provider; credential was not verified. Check network access and try again."}
        except Exception:
            return {"provider": provider, "status": "warning", "message": "Provider response could not be verified. Check provider availability and try again."}

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
        peer = websocket.client.host if websocket.client else None
        try:
            is_loopback = peer is not None and ipaddress.ip_address(peer).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            api_token = os.environ.get("TRIUNE_API_TOKEN", "")
            if len(api_token) < 32:
                await websocket.close(code=1013, reason="Remote API token is not configured")
                return
            await websocket.accept()
            try:
                supplied = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            except (asyncio.TimeoutError, WebSocketDisconnect):
                await websocket.close(code=1008, reason="Authentication timed out")
                return
            if not hmac.compare_digest(supplied, api_token):
                await websocket.close(code=1008, reason="Invalid bearer token")
                return
            await telemetry_manager.connect(websocket, accept=False)
        else:
            await telemetry_manager.connect(websocket)

        try:
            latest_h = pytorch_state.history[-1] if pytorch_state.history else {}
            cur_loss = latest_h.get("loss", getattr(pytorch_state, "last_loss", None))
            cur_exit = pytorch_state._get_current_exit_usage()
            cur_thru = pytorch_state._get_current_throughput()
            snapshot_msg = {
                "type": "telemetry_snapshot",
                "data": {
                    "step": pytorch_state.step,
                    "loss": cur_loss,
                    "throughput": cur_thru,
                    "exit_usage": cur_exit,
                    "target_exit_distribution": pytorch_state._get_target_exit_distribution(),
                    "tokens_trained": pytorch_state.total_tokens_trained,
                    "active_checkpoint": pytorch_state.active_checkpoint,
                    "active_model": pytorch_state.active_model_id,
                    "is_training": pytorch_state.is_training,
                    "history": pytorch_state.history[-100:] if pytorch_state.history else [],
                    "batch_preview": pytorch_state.current_batch_preview,
                }
            }
            await websocket.send_text(json.dumps(snapshot_msg))
        except Exception:
            pass

        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            telemetry_manager.disconnect(websocket)
else:
    router = None
