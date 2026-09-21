"""GitHub API Integration for Module Repository Search and Download."""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from typing import Any


class GitHubClient:
    """Search and fetch repository details from GitHub API without requiring credentials."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Triune-Studio-App/2.0",
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    def search_repositories(self, query: str, sort: str = "stars", per_page: int = 12) -> list[dict[str, Any]]:
        """Search GitHub for AI/ML model or extension repositories."""
        if not query:
            query = "machine learning model pytorch"
        
        encoded_q = urllib.parse.quote(query)
        url = f"{self.BASE_URL}/search/repositories?q={encoded_q}&sort={sort}&order=desc&per_page={per_page}"
        req = urllib.request.Request(url, headers=self._get_headers())
        
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("items", [])
                results = []
                for item in items:
                    results.append({
                        "id": f"gh-{item['id']}",
                        "name": item["name"],
                        "author": item["owner"]["login"],
                        "type": "plugin" if "plugin" in item["name"].lower() else "model",
                        "version": "latest",
                        "description": item.get("description") or "No description provided.",
                        "repo_url": item["html_url"],
                        "download_url": f"{item['html_url']}/archive/refs/heads/{item.get('default_branch', 'main')}.zip",
                        "stars": item.get("stargazers_count", 0),
                        "updated_at": item.get("updated_at", ""),
                        "tags": [item.get("language") or "Python", f"★ {item.get('stargazers_count', 0)}"],
                        "requires_cuda": False
                    })
                return results
        except Exception as err:
            print(f"[GitHub API Note] Search fallback: {err}")
            return []

    def get_latest_release(self, owner: str, repo: str) -> dict[str, Any] | None:
        """Fetch latest release tag for version control tracking."""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/releases/latest"
        req = urllib.request.Request(url, headers=self._get_headers())
        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "tag_name": data.get("tag_name", "v1.0.0"),
                    "name": data.get("name"),
                    "published_at": data.get("published_at"),
                    "assets": [
                        {"name": a["name"], "download_url": a["browser_download_url"], "size": a["size"]}
                        for a in data.get("assets", [])
                    ]
                }
        except Exception:
            return None
