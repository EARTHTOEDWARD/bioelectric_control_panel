from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class StationarityReport:
    ok: bool
    mean_drift_ratio: float
    var_ratio_last_first: float
    acf1_abs: float
    notes: List[str]


def stationarity_checks(x: np.ndarray, fs: float) -> StationarityReport:
    x = np.asarray(x, dtype=float)
    n = x.size
    notes: List[str] = []
    ok = True

    # Mean drift ratio over window
    t = np.arange(n) / max(fs, 1e-9)
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _ = np.linalg.lstsq(A, x, rcond=None)[0]
    amp = float(np.ptp(x)) + 1e-12
    mean_drift_ratio = float(abs(slope) * (n / max(fs, 1e-9)) / amp)
    if mean_drift_ratio > 0.1:
        ok = False
        notes.append("High mean drift (>10% of amplitude).")

    # Variance stability
    one_third = max(1, n // 3)
    var_first = float(np.var(x[:one_third]))
    var_last = float(np.var(x[-one_third:]))
    var_ratio = (var_last + 1e-12) / (var_first + 1e-12)
    if not (0.5 <= var_ratio <= 2.0):
        ok = False
        notes.append("Variance change >2x across window.")

    # ACF of differenced signal (lag-1)
    dx = np.diff(x)
    if dx.size >= 2 and np.std(dx[:-1]) * np.std(dx[1:]) > 0:
        r = float(np.corrcoef(dx[:-1], dx[1:])[0, 1])
        acf1_abs = float(abs(r))
        if acf1_abs > 0.95:
            ok = False
            notes.append("High autocorrelation in differences (|acf1|>0.95).")
    else:
        acf1_abs = float("nan")
        notes.append("Short window for acf check.")

    if ok:
        notes.append("Stationarity checks passed.")
    return StationarityReport(
        ok=bool(ok),
        mean_drift_ratio=mean_drift_ratio,
        var_ratio_last_first=float(var_ratio),
        acf1_abs=acf1_abs,
        notes=notes,
    )


def quality_gate(
    lam1_ci: Tuple[float, float], D2_ci: Tuple[float, float], stationarity_ok: bool
) -> Tuple[bool, List[str]]:
    ok = True
    msgs: List[str] = []
    l_lo, l_hi = lam1_ci
    d_lo, d_hi = D2_ci
    if not stationarity_ok:
        ok = False
        msgs.append("Stationarity failed.")
    if not (np.isfinite(l_lo) and np.isfinite(l_hi)) or (l_lo <= 0 <= l_hi):
        ok = False
        msgs.append("λ1 CI overlaps 0 or invalid.")
    if not (np.isfinite(d_lo) and np.isfinite(d_hi)) or (d_hi - d_lo > 1.0):
        msgs.append("Wide D2 CI (>1.0).")
    return bool(ok), msgs


def classify_cardiac(lam1: float, D2: float) -> Tuple[str, float, List[str]]:
    reasons: List[str] = []
    if lam1 <= 0.02 and D2 <= 1.2:
        reasons.append("λ1≤0.02 & D2≤1.2")
        return "nsr", 0.95, reasons
    if lam1 >= 0.3 and D2 >= 3.0:
        reasons.append("λ1≥0.3 & D2≥3.0")
        return "vf", 0.90, reasons
    reasons.append("intermediate λ1/D2")
    return "vt_borderline", 0.6, reasons

