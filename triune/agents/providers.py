"""Multi-Provider Subscription & API Key Adapter.

Enables users to bring their own API keys / subscriptions (OpenAI, Anthropic, Google Gemini / Antigravity,
Hugging Face, OpenRouter, Ollama) into Triune Framework & Studio for agentic tasks, synthetic data generation,
and hybrid local/cloud workflows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ProviderConfig:
    provider: str  # "openai", "anthropic", "gemini", "huggingface", "openrouter", "ollama"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None


class ProviderManager:
    """Manages bring-your-own-key (BYOK) provider credentials."""

    def __init__(self) -> None:
        self.providers: Dict[str, ProviderConfig] = {}
        self._discover_env_keys()

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

    def register_provider(
        self, provider: str, api_key: str, base_url: str | None = None, model_name: str | None = None
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
        return {p: bool(cfg.api_key or p == "ollama") for p, cfg in self.providers.items()}
