"""GitHub and Hugging Face Hub API Integration for Module & Repository Discovery."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, List, Optional


class RepoSearchClient:
    """Unified client for searching GitHub and Hugging Face repositories."""

    GITHUB_API_URL = "https://api.github.com"
    HF_API_URL = "https://huggingface.co/api"

    def __init__(self, github_token: Optional[str] = None, hf_token: Optional[str] = None) -> None:
        self.github_token = github_token
        self.hf_token = hf_token

    def _get_github_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Triune-Studio-App/2.0",
        }
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        return headers

    def _get_hf_headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "Triune-Studio-App/2.0",
        }
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers

    def search_github_repositories(self, query: str, sort: str = "stars", per_page: int = 12) -> List[dict[str, Any]]:
        """Search GitHub for AI/ML model, extension, or DAG plugin repositories."""
        if not query:
            query = "pytorch transformer deep-learning"

        encoded_q = urllib.parse.quote(query)
        url = f"{self.GITHUB_API_URL}/search/repositories?q={encoded_q}&sort={sort}&order=desc&per_page={per_page}"
        req = urllib.request.Request(url, headers=self._get_github_headers())

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("items", [])
                results = []
                for item in items:
                    name_lower = item["name"].lower()
                    desc_lower = (item.get("description") or "").lower()

                    if any(k in name_lower or k in desc_lower for k in ("plugin", "node", "dag", "custom-node")):
                        mod_type = "plugin"
                    elif any(k in name_lower or k in desc_lower for k in ("lora", "adapter", "peft")):
                        mod_type = "adapter"
                    elif any(k in name_lower or k in desc_lower for k in ("dataset", "corpus")):
                        mod_type = "dataset"
                    else:
                        mod_type = "model"

                    results.append({
                        "id": f"gh-{item['id']}",
                        "name": item["name"],
                        "author": item["owner"]["login"],
                        "source": "github",
                        "type": mod_type,
                        "version": "latest",
                        "description": item.get("description") or "No description provided.",
                        "repo_url": item["html_url"],
                        "clone_url": item.get("clone_url") or f"{item['html_url']}.git",
                        "download_url": f"{item['html_url']}/archive/refs/heads/{item.get('default_branch', 'main')}.zip",
                        "stars": item.get("stargazers_count", 0),
                        "forks": item.get("forks_count", 0),
                        "default_branch": item.get("default_branch", "main"),
                        "updated_at": item.get("updated_at", ""),
                        "tags": [item.get("language") or "Python", f"★ {item.get('stargazers_count', 0)}"],
                        "requires_cuda": False
                    })
                return results
        except Exception as err:
            print(f"[GitHub API] Search fallback: {err}")
            return []

    def search_huggingface_models(self, query: str, limit: int = 12) -> List[dict[str, Any]]:
        """Search Hugging Face Hub for pretrained models and checkpoints."""
        encoded_q = urllib.parse.quote(query) if query else "triune"
        url = f"{self.HF_API_URL}/models?search={encoded_q}&limit={limit}&full=false"
        req = urllib.request.Request(url, headers=self._get_hf_headers())

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = []
                for item in data:
                    model_id = item.get("id", "")
                    author = model_id.split("/")[0] if "/" in model_id else "Community"
                    name = model_id.split("/")[-1] if "/" in model_id else model_id
                    likes = item.get("likes", 0)
                    downloads = item.get("downloads", 0)
                    pipeline = item.get("pipeline_tag") or "text-generation"

                    tags = [pipeline]
                    if downloads:
                        tags.append(f"⬇ {downloads:,}")
                    if likes:
                        tags.append(f"♥ {likes}")

                    results.append({
                        "id": f"hf-m-{model_id.replace('/', '--')}",
                        "name": model_id,
                        "display_name": name,
                        "author": author,
                        "source": "huggingface",
                        "type": "model",
                        "version": "main",
                        "description": f"Hugging Face Model: {model_id} ({pipeline})",
                        "repo_url": f"https://huggingface.co/{model_id}",
                        "clone_url": f"https://huggingface.co/{model_id}",
                        "download_url": f"https://huggingface.co/{model_id}/resolve/main/model.safetensors",
                        "stars": likes,
                        "downloads": downloads,
                        "pipeline_tag": pipeline,
                        "tags": tags,
                        "requires_cuda": False
                    })
                return results
        except Exception as err:
            print(f"[HuggingFace Models API] Search fallback: {err}")
            return []

    def search_huggingface_datasets(self, query: str, limit: int = 12) -> List[dict[str, Any]]:
        """Search Hugging Face Hub for datasets and training corpora."""
        encoded_q = urllib.parse.quote(query) if query else "stories"
        url = f"{self.HF_API_URL}/datasets?search={encoded_q}&limit={limit}&full=false"
        req = urllib.request.Request(url, headers=self._get_hf_headers())

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = []
                for item in data:
                    ds_id = item.get("id", "")
                    author = ds_id.split("/")[0] if "/" in ds_id else "Community"
                    name = ds_id.split("/")[-1] if "/" in ds_id else ds_id
                    likes = item.get("likes", 0)
                    downloads = item.get("downloads", 0)

                    tags = ["Dataset"]
                    if downloads:
                        tags.append(f"⬇ {downloads:,}")
                    if likes:
                        tags.append(f"♥ {likes}")

                    results.append({
                        "id": f"hf-d-{ds_id.replace('/', '--')}",
                        "name": ds_id,
                        "display_name": name,
                        "author": author,
                        "source": "huggingface",
                        "type": "dataset",
                        "version": "main",
                        "description": f"Hugging Face Dataset: {ds_id}",
                        "repo_url": f"https://huggingface.co/datasets/{ds_id}",
                        "clone_url": f"https://huggingface.co/datasets/{ds_id}",
                        "download_url": f"https://huggingface.co/datasets/{ds_id}",
                        "stars": likes,
                        "downloads": downloads,
                        "tags": tags,
                        "requires_cuda": False
                    })
                return results
        except Exception as err:
            print(f"[HuggingFace Datasets API] Search fallback: {err}")
            return []


# Backward-compatible alias
GitHubClient = RepoSearchClient
