from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import urllib.request


class BCPClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        with urllib.request.urlopen(url, timeout=15) as resp:  # nosec B310
            return json.loads(resp.read().decode("utf-8"))

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    def ingest_maes(self, filename: str) -> Dict[str, Any]:
        return self._post("/v1/ingest/maes", {"filename": filename})

    def embed(self, signal: List[float], delay_ms: int = 20, sample_rate_hz: float = 1000.0, embed_dim: int = 3) -> Dict[str, Any]:
        return self._post(
            "/v1/attractor/embed",
            {
                "signal": signal,
                "delay_ms": delay_ms,
                "sample_rate_hz": sample_rate_hz,
                "embed_dim": embed_dim,
            },
        )

    def metrics_summary(self, signal: List[float]) -> Dict[str, Any]:
        return self._post("/v1/metrics/summary", {"signal": signal})

    def chaos_window(
        self,
        signal: List[float],
        sample_rate_hz: float,
        modality: str = "generic",
        embed_dim: int | None = None,
        delay_ms: int | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "signal": signal,
            "sample_rate_hz": sample_rate_hz,
            "modality": modality,
        }
        if embed_dim is not None:
            payload["embed_dim"] = int(embed_dim)
        if delay_ms is not None:
            payload["delay_ms"] = int(delay_ms)
        return self._post("/v1/chaos/window", payload)
