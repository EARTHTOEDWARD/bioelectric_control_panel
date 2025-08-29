from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .chaos import compute_window_metrics


@dataclass
class StreamConfig:
    window_s: float
    step_s: float
    modality: str = "generic"


def streaming_windows(
    signal: np.ndarray,
    fs: float,
    window_s: float = 5.0,
    step_s: float = 1.0,
    modality: str = "generic",
) -> Iterable[Tuple[float, dict]]:
    """Yield (t_start_seconds, metrics_dict) over rolling windows of the signal."""
    x = np.asarray(signal, dtype=float)
    win = max(int(window_s * fs), 1)
    step = max(int(step_s * fs), 1)
    n = x.size
    for start in range(0, max(0, n - win + 1), step):
        seg = x[start : start + win]
        t0 = start / fs
        m = compute_window_metrics(seg, fs=fs, modality=modality)
        yield t0, m

