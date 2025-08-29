from __future__ import annotations

from typing import Dict, List

import numpy as np


def delay_embedding(signal: List[float], delay_ms: int, sample_rate_hz: float, embed_dim: int = 3) -> Dict[str, List[float]]:
    """Simple delay embedding of a 1D signal.

    Returns dict with keys: x, y, z (up to embed_dim).
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    delay_samples = max(int((delay_ms / 1000.0) * sample_rate_hz), 1)
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError("signal must be 1D")
    n = arr.shape[0]
    if n < embed_dim * delay_samples + 1:
        raise ValueError("signal too short for requested embedding")

    embedded = np.array([
        arr[i : i + embed_dim * delay_samples : delay_samples] for i in range(n - embed_dim * delay_samples)
    ])

    out: Dict[str, List[float]] = {}
    keys = ["x", "y", "z", "w", "v"]
    for d in range(embed_dim):
        out[keys[d]] = embedded[:, d].astype(float).tolist()
    return out

