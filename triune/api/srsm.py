from __future__ import annotations

import collections
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch


class SessionRecurrentStateManager:
    def __init__(
        self,
        max_active_sessions: int = 256,
        session_ttl_seconds: float = 3600.0,
        pin_memory: bool = True,
    ):
        self.max_active_sessions = max_active_sessions
        self.session_ttl_seconds = session_ttl_seconds
        self.pin_memory = pin_memory
        self._lock = threading.Lock()
        self._sessions: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()

    def get_session_cache(
        self, session_id: str, device: torch.device
    ) -> Optional[Tuple[List[Any], Optional[int]]]:
        if not session_id:
            return None

        with self._lock:
            self._purge_expired_locked()
            if session_id not in self._sessions:
                return None

            entry = self._sessions[session_id]
            entry["last_active"] = time.time()
            self._sessions.move_to_end(session_id)
            raw_cache = entry["cache"]
            depth = entry.get("depth")

        staged_cache = []
        for item in raw_cache:
            if item is None:
                staged_cache.append((None, 0))
            elif isinstance(item, tuple):
                state, off = item
                if state is not None and torch.is_tensor(state):
                    staged_cache.append((state.to(device=device, non_blocking=True), off))
                else:
                    staged_cache.append((None, off))
            else:
                staged_cache.append(item)

        return staged_cache, depth

    def update_session_cache(
        self, session_id: str, cache: List[Any], depth: Optional[int] = None
    ) -> None:
        if not session_id or not cache:
            return

        host_cache = []
        for item in cache:
            if item is None:
                host_cache.append((None, 0))
            elif isinstance(item, tuple):
                state, off = item
                if state is not None and torch.is_tensor(state):
                    cpu_state = state.detach().cpu()
                    if self.pin_memory and torch.cuda.is_available():
                        try:
                            cpu_state = cpu_state.pin_memory()
                        except Exception:
                            pass
                    host_cache.append((cpu_state, off))
                else:
                    host_cache.append((None, off))
            else:
                host_cache.append(item)

        with self._lock:
            self._purge_expired_locked()
            self._sessions[session_id] = {
                "cache": host_cache,
                "last_active": time.time(),
                "depth": depth,
            }
            self._sessions.move_to_end(session_id)

            while len(self._sessions) > self.max_active_sessions:
                self._sessions.popitem(last=False)

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, entry in self._sessions.items()
            if now - entry["last_active"] > self.session_ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_bytes = 0
            for entry in self._sessions.values():
                for item in entry["cache"]:
                    if isinstance(item, tuple) and item[0] is not None and torch.is_tensor(item[0]):
                        total_bytes += item[0].numel() * item[0].element_size()
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self.max_active_sessions,
                "total_recurrent_memory_mb": round(total_bytes / (1024 * 1024), 2),
            }


class ServingRequestQueue:
    def __init__(self, max_concurrency: int = 16):
        self.max_concurrency = max_concurrency
        self.active_requests: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrency)

    def register_request(self, request_id: str) -> threading.Event:
        cancel_event = threading.Event()
        with self._lock:
            self.active_requests[request_id] = cancel_event
        return cancel_event

    def unregister_request(self, request_id: str) -> None:
        with self._lock:
            self.active_requests.pop(request_id, None)

    def cancel_request(self, request_id: str) -> bool:
        with self._lock:
            event = self.active_requests.get(request_id)
            if event:
                event.set()
                return True
            return False

    def acquire(self, timeout: float = 30.0) -> bool:
        return self._semaphore.acquire(timeout=timeout)

    def release(self) -> None:
        self._semaphore.release()
