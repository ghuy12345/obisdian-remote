"""Embeddings via any OpenAI-compatible /embeddings endpoint (OpenRouter, OpenAI, ...)."""
from __future__ import annotations

import logging
import time

import httpx
import numpy as np

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(self, base_url: str, api_key: str, model: str, batch_size: int = 64, timeout: float = 60):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.batch_size = batch_size
        self.client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return out

    def _embed_batch(self, texts: list[str], attempts: int = 4) -> list[np.ndarray]:
        # Keep inputs well under typical 8k-token limits.
        texts = [t[:24000] for t in texts]
        delay = 2.0
        last = None
        for _ in range(attempts):
            try:
                r = self.client.post(self.url, json={"model": self.model, "input": texts})
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(delay)
                    delay *= 2
                    continue
                if r.status_code >= 400:
                    raise EmbeddingError(f"HTTP {r.status_code}: {r.text[:300]}")
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                if len(data) != len(texts):
                    raise EmbeddingError(f"expected {len(texts)} vectors, got {len(data)}")
                return [np.asarray(d["embedding"], dtype=np.float32) for d in data]
            except (httpx.TransportError, KeyError, ValueError) as e:
                last = repr(e)
                time.sleep(delay)
                delay *= 2
        raise EmbeddingError(f"embedding request failed after {attempts} attempts: {last}")
