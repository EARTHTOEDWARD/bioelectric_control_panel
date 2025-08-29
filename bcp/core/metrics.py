from __future__ import annotations

from typing import Dict, List

import numpy as np


def summary_metrics(signal: List[float]) -> Dict[str, float]:
    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        raise ValueError("signal is empty")
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "rms": float(np.sqrt(np.mean(arr ** 2))),
    }

