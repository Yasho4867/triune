"""Atomic storage helpers, schema validation, and TF-IDF semantic matching for agent harness."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Any

_WRITE_LOCK = threading.Lock()

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "please",
        "make",
        "sure",
        "have",
        "what",
        "when",
        "where",
        "how",
        "are",
        "was",
        "were",
        "will",
        "can",
        "use",
    }
)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text via sibling temporary file + os.replace."""
    resolved = path.resolve()
    with _WRITE_LOCK:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(resolved.parent), prefix=f".{resolved.name}.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, resolved)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically serialize and write JSON payload to disk."""
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def load_validated_json_dict(path: Path) -> dict[str, Any]:
    """Read a JSON object safely; return empty dict if missing, corrupt, or non-dict."""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def tokenize_terms(text: str, *, include_char_ngrams: bool = True) -> list[str]:
    """Extract informative lexical words plus optional 3-char subword n-grams."""
    words = [w for w in re.findall(r"[a-z0-9_]{2,}", (text or "").lower()) if w not in _STOPWORDS]
    if not include_char_ngrams:
        return words
    terms: list[str] = list(words)
    for w in words:
        if len(w) >= 4:
            for i in range(len(w) - 2):
                terms.append(f"#{w[i : i + 3]}")
    return terms


def compute_tfidf_cosine_scores(query: str, docs: list[str]) -> list[float]:
    """Compute normalized TF-IDF + char-ngram cosine similarity between query and documents."""
    if not docs:
        return []
    q_terms = tokenize_terms(query)
    if not q_terms:
        return [0.0 for _ in docs]

    doc_counters = [Counter(tokenize_terms(d)) for d in docs]
    q_counter = Counter(q_terms)

    n_docs = len(docs) + 1
    df: Counter[str] = Counter()
    for tc in doc_counters:
        for term in tc:
            df[term] += 1
    for term in q_counter:
        df[term] += 1

    idf = {term: math.log((1.0 + n_docs) / (1.0 + freq)) + 1.0 for term, freq in df.items()}

    q_vec = {term: (1.0 + math.log(cnt)) * idf[term] for term, cnt in q_counter.items()}
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    scores: list[float] = []
    for d_counter in doc_counters:
        if not d_counter:
            scores.append(0.0)
            continue
        dot = 0.0
        d_sq = 0.0
        for term, cnt in d_counter.items():
            w = (1.0 + math.log(cnt)) * idf.get(term, 1.0)
            d_sq += w * w
            if term in q_vec:
                dot += w * q_vec[term]
        d_norm = math.sqrt(d_sq) or 1.0
        scores.append(max(0.0, min(1.0, dot / (q_norm * d_norm))))
    return scores
