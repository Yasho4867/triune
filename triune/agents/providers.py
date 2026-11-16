"""Multi-Provider LLM & Local Runtime Adapter for the Triune Agent Harness."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProviderConfig:
    provider: str  # "openai", "anthropic", "gemini", "huggingface", "openrouter", "ollama", "triune"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None


class ProviderManager:
    """Manages local (Ollama / Triune) and BYOK cloud providers for agent execution."""

    def __init__(self) -> None:
        self.providers: Dict[str, ProviderConfig] = {}
        self._discover_env_keys()
        self._discover_studio_keys()
        self._discover_local_ollama()

    def _discover_env_keys(self) -> None:
        """Automatically detect API keys present in environment variables."""
        if os.getenv("OPENAI_API_KEY"):
            self.providers["openai"] = ProviderConfig("openai", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("ANTHROPIC_API_KEY"):
            self.providers["anthropic"] = ProviderConfig("anthropic", api_key=os.getenv("ANTHROPIC_API_KEY"))
        if os.getenv("GEMINI_API_KEY"):
            self.providers["gemini"] = ProviderConfig("gemini", api_key=os.getenv("GEMINI_API_KEY"))
        if os.getenv("HF_TOKEN"):
            self.providers["huggingface"] = ProviderConfig("huggingface", api_key=os.getenv("HF_TOKEN"))
        if os.getenv("OPENROUTER_API_KEY"):
            self.providers["openrouter"] = ProviderConfig("openrouter", api_key=os.getenv("OPENROUTER_API_KEY"))

    def _discover_studio_keys(self) -> None:
        """Load any saved BYOK credentials from Triune Studio ModuleManager cleanly."""
        try:
            from triune.modules.manager import ModuleManager

            byok = ModuleManager().get_config().get("byok_keys") or {}
            if isinstance(byok, dict):
                for name, key in byok.items():
                    if key and str(name).lower() not in self.providers:
                        self.providers[str(name).lower()] = ProviderConfig(str(name).lower(), api_key=str(key))
        except Exception:
            pass

    def _discover_local_ollama(self) -> None:
        """Register local Ollama automatically when daemon responds on 127.0.0.1:11434."""
        try:
            from triune.runtime.model_scanner import query_ollama_running_tags, resolve_ollama_host

            tags = sorted(t for t in query_ollama_running_tags() if ":" in t)
            if tags:
                preferred = "gemma3:4b" if "gemma3:4b" in tags else tags[0]
                self.providers["ollama"] = ProviderConfig(
                    "ollama",
                    base_url=resolve_ollama_host(),
                    model_name=preferred,
                )
        except Exception:
            pass

    def register_provider(
        self, provider: str, api_key: str = "", base_url: str | None = None, model_name: str | None = None
    ) -> None:
        self.providers[provider.lower()] = ProviderConfig(
            provider=provider.lower(),
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )

    def get_provider(self, provider: str) -> Optional[ProviderConfig]:
        return self.providers.get(provider.lower())

    def list_active_providers(self) -> Dict[str, bool]:
        return {p: bool(cfg.api_key or p in {"ollama", "triune"}) for p, cfg in self.providers.items()}

    def pick_default_model(self) -> str | None:
        """Select the highest-priority ready local or BYOK model reference."""
        ol = self.providers.get("ollama")
        if ol and ol.model_name:
            return f"ollama:{ol.model_name}"
        if self.providers.get("openai"):
            return self.providers["openai"].model_name or "gpt-4o-mini"
        if self.providers.get("anthropic"):
            return self.providers["anthropic"].model_name or "claude-3-5-sonnet-latest"
        if self.providers.get("gemini"):
            return self.providers["gemini"].model_name or "gemini-2.5-flash"
        if self.providers.get("openrouter"):
            return self.providers["openrouter"].model_name or "openrouter/auto"
        return None

    def complete_chat(
        self,
        model_ref: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.35,
        max_tokens: int = 1024,
        force_depth: str | None = None,
    ) -> Dict[str, Any]:
        """Run one non-streaming turn across Ollama, OpenAI/OpenRouter, Anthropic, Gemini, or local PyTorch."""
        ref = str(model_ref or "").strip()
        low = ref.lower()
        if low.startswith("ollama:"):
            return self._call_ollama_chat(
                ref[len("ollama:") :],
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if low.startswith(("gpt-", "o1-", "o3-", "openai/")):
            return self._call_openai_compat(
                "openai",
                ref.removeprefix("openai/"),
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if low.startswith("openrouter/"):
            return self._call_openai_compat(
                "openrouter",
                ref.removeprefix("openrouter/"),
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if low.startswith(("claude-", "anthropic/")):
            return self._call_anthropic(
                ref.removeprefix("anthropic/"),
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if low.startswith(("gemini-", "gemini/")):
            return self._call_gemini(
                ref.removeprefix("gemini/"),
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise ValueError(
            f"Unsupported agent model reference '{model_ref}'. "
            "Use 'ollama:<tag>', an OpenAI/Anthropic/Gemini model ID, or pass a local completion callback."
        )

    @staticmethod
    def _post_json(
        url: str,
        payload: Dict[str, Any],
        *,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        req_headers = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=req_headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Provider returned non-JSON payload: {raw_text[:200]}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Provider returned unexpected non-object JSON payload.")
        return parsed

    @staticmethod
    def _normalize_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
        tool_calls: List[Dict[str, Any]] = []
        for tc in raw_calls or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(raw_args, dict):
                args = dict(raw_args)
            else:
                args = {}
            tool_calls.append({"name": str(fn["name"]), "arguments": args})
        return tool_calls

    def _call_ollama_chat(
        self,
        tag: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.35,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        from triune.runtime.model_scanner import resolve_ollama_host

        cfg = self.providers.get("ollama")
        url = f"{resolve_ollama_host(cfg.base_url if cfg else None)}/api/chat"
        body: Dict[str, Any] = {
            "model": tag,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
        }
        # Only pass native tools if present; some local GGUFs reject `tools` HTTP 400 -> retry without cleanly
        if tools:
            body["tools"] = tools

        try:
            data = self._post_json(url, body, timeout=180)
        except Exception as exc:
            if tools and "400" in str(exc):
                body.pop("tools", None)
                data = self._post_json(url, body, timeout=180)
            else:
                raise

        msg = data.get("message") or {}
        tool_calls = self._normalize_tool_calls(msg.get("tool_calls"))
        return {"content": str(msg.get("content") or ""), "tool_calls": tool_calls, "raw": data}

    def _call_openai_compat(
        self,
        provider_name: str,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.35,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        cfg = self.providers.get(provider_name)
        api_key = cfg.api_key if cfg else None
        if not api_key:
            raise RuntimeError(f"{provider_name} API key is not configured.")
        default_base = "https://openrouter.ai/api/v1" if provider_name == "openrouter" else "https://api.openai.com/v1"
        base = (cfg.base_url or default_base).rstrip("/")
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if tools:
            payload["tools"] = tools
        data = self._post_json(
            f"{base}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        tool_calls = self._normalize_tool_calls(msg.get("tool_calls"))
        return {"content": str(msg.get("content") or ""), "tool_calls": tool_calls, "raw": data}

    def _call_anthropic(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.35,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        cfg = self.providers.get("anthropic")
        if not cfg or not cfg.api_key:
            raise RuntimeError("Anthropic API key is not configured.")
        sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
        chat_msgs = [
            {"role": ("user" if m.get("role") == "tool" else m.get("role")), "content": m.get("content", "")}
            for m in messages
            if m.get("role") != "system"
        ]
        payload = {
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "system": "\n\n".join(sys_parts),
            "messages": chat_msgs,
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": cfg.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        text_blocks = [b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text"]
        return {"content": "\n".join(text_blocks), "tool_calls": [], "raw": data}

    def _call_gemini(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.35,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        cfg = self.providers.get("gemini")
        if not cfg or not cfg.api_key:
            raise RuntimeError("Gemini API key is not configured.")
        prompt_text = "\n\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={cfg.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_tokens)},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        parts = (((data.get("candidates") or [{}])[0]).get("content") or {}).get("parts") or []
        return {"content": "\n".join(p.get("text", "") for p in parts), "tool_calls": [], "raw": data}
