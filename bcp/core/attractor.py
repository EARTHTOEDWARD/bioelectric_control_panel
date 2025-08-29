from __future__ import annotations

from typing import Dict, List

import numpy as np


def delay_embedding(signal: List[float], delay_ms: int, sample_rate_hz: float, embed_dim: int = 3) -> Dict[str, List[float]]:
    """Simple delay embedding of a 1D signal.

    Returns dict with keys: x, y, z (up to embed_dim).
    Degrades gracefully for short signals by reducing the effective delay.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    arr = np.asarray(signal, dtype=float)
    if arr.ndim != 1:
        raise ValueError("signal must be 1D")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("signal too short for requested embedding")
    # Convert delay from ms to samples (at least 1)
    delay_samples = max(int((delay_ms / 1000.0) * sample_rate_hz), 1)
    # Ensure at least one embedded vector by reducing the effective delay if needed
    if embed_dim > 1:
        max_tau = max(1, (n - 1) // (embed_dim - 1))
        eff_tau = min(delay_samples, max_tau)
    else:
        eff_tau = 1

    # Build embedded matrix: shape (N, embed_dim)
    span = (embed_dim - 1) * eff_tau
    N = max(0, n - span)
    if N == 0:
        # Should not happen due to eff_tau adjustment, but guard regardless
        return {"x": []}
    embedded = np.array([
        arr[i : i + embed_dim * eff_tau : eff_tau] for i in range(N)
    ])

    out: Dict[str, List[float]] = {}
    keys = ["x", "y", "z", "w", "v"]
    for d in range(min(embed_dim, embedded.shape[1])):
        out[keys[d]] = embedded[:, d].astype(float).tolist()
    return out
