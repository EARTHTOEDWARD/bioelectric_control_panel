from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import numpy as np


def _autocorr_first_zero(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    if n <= 1:
        return 1
    f = np.fft.fft(x, n * 2)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    if acf[0] != 0:
        acf /= acf[0]
    for k in range(1, min(n, 2000)):
        if acf[k] <= 0:
            return k
    return max(1, n // 10)


def _takens_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    N = x.size - (m - 1) * tau
    if N <= 0:
        raise ValueError("signal too short for requested embedding")
    Y = np.zeros((N, m), dtype=float)
    for i in range(m):
        Y[:, i] = x[i * tau : i * tau + N]
    return Y


def lyap_rosenstein(x: np.ndarray, m: int, tau: int, fs: float, theiler: int | None = None,
                    max_t: float = 0.4, max_points: int = 4000) -> float:
    """
    Largest Lyapunov exponent via Rosenstein et al.
    Returns slope (1/seconds). If insufficient data, returns NaN.
    """
    if theiler is None:
        theiler = max(1, int(1.5 * tau))
    Y = _takens_embedding(x, m, tau)
    N = Y.shape[0]
    # downsample points to cap O(N^2)
    idx = np.arange(N)
    if N > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(N, size=max_points, replace=False))
    YS = Y[idx]
    D = np.linalg.norm(YS[:, None, :] - YS[None, :, :], axis=2)
    # crude Theiler: exclude a diagonal band
    for i in range(YS.shape[0]):
        lo = max(0, i - theiler)
        hi = min(YS.shape[0], i + theiler + 1)
        D[i, lo:hi] = np.inf
        D[i, i] = np.inf
    nn = np.argmin(D, axis=1)
    max_k = min(int(max_t * fs), YS.shape[0] - 1)
    if max_k <= 1:
        return float("nan")
    valid_counts = np.zeros(max_k)
    mean_log = np.zeros(max_k)
    for k in range(1, max_k):
        vals = []
        for i in range(YS.shape[0] - k):
            j = nn[i]
            if j + k >= YS.shape[0] or i + k >= YS.shape[0]:
                continue
            d0 = np.linalg.norm(YS[i] - YS[j])
            d1 = np.linalg.norm(YS[i + k] - YS[j + k])
            if d0 > 0 and d1 > 0:
                vals.append(np.log(d1 / d0))
        if len(vals) >= 5:
            mean_log[k] = np.mean(vals)
            valid_counts[k] = len(vals)
        else:
            mean_log[k] = np.nan
    ks = np.arange(max_k)
    mask = np.isfinite(mean_log) & (valid_counts > 10)
    if np.sum(mask) < 5:
        return float("nan")
    # fit initial ~30% of valid portion
    sel = np.where(mask)[0]
    end = int(0.3 * sel.size) + 2
    sel = sel[:end]
    t_axis = ks[sel] / fs
    slope, _ = np.polyfit(t_axis, mean_log[sel], 1)
    return float(slope)


def correlation_dimension_gp(x: np.ndarray, m: int, tau: int, n_r: int = 24,
                             max_points: int = 5000) -> float:
    """Grassberger–Procaccia correlation dimension (single-slope fit)."""
    Y = _takens_embedding(x, m, tau)
    N = Y.shape[0]
    idx = np.arange(N)
    cap = int(math.sqrt(max_points)) * 2
    if N > cap:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(N, size=cap, replace=False))
    YS = Y[idx]
    D = np.linalg.norm(YS[:, None, :] - YS[None, :, :], axis=2)
    tri = np.triu_indices_from(D, k=1)
    dists = D[tri]
    dists = dists[dists > 0]
    if dists.size < 10:
        return float("nan")
    dists.sort()
    rmin = np.percentile(dists, 5)
    rmax = np.percentile(dists, 95)
    if rmin <= 0 or rmax <= 0 or not np.isfinite(rmin) or not np.isfinite(rmax):
        return float("nan")
    rs = np.logspace(np.log10(rmin), np.log10(rmax), n_r)
    C = np.array([np.mean(dists < r) for r in rs])
    slope, _ = np.polyfit(np.log(rs), np.log(C + 1e-12), 1)
    return float(slope)


def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    x = np.asarray(x, dtype=float)
    N = x.size
    if r is None:
        r = 0.15 * np.std(x)
    if N < (m + 2) * 4:
        return float("nan")
    def _phi(mm: int) -> float:
        X = np.array([x[i : i + mm] for i in range(N - mm + 1)])
        if X.shape[0] < 2:
            return float("nan")
        count = 0
        for i in range(X.shape[0]):
            d = np.max(np.abs(X - X[i]), axis=1)
            count += np.sum(d <= r) - 1
        denom = (N - mm + 1) * (N - mm)
        return (count / denom) if denom > 0 else float("nan")
    A = _phi(m + 1)
    B = _phi(m)
    if A <= 0 or B <= 0 or not np.isfinite(A) or not np.isfinite(B):
        return float("nan")
    return float(-np.log(A / B))


def mse_auc(x: np.ndarray, max_scale: int = 20, m: int = 2, r: float | None = None) -> float:
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.15 * np.std(x)
    vals = []
    for s in range(1, max_scale + 1):
        n = x.size // s
        if n < (m + 2) * 4:
            vals.append(np.nan)
            continue
        xs = x[: n * s].reshape(n, s).mean(axis=1)
        vals.append(sample_entropy(xs, m=m, r=r))
    return float(np.nansum(vals))


def rqa_metrics(x: np.ndarray, m: int, tau: int, recurrence_rate: float = 0.03,
                lmin: int = 2, vmin: int = 2) -> Dict[str, float]:
    Y = _takens_embedding(x, m, tau)
    D = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    tri = np.triu_indices_from(D, k=1)
    d = D[tri]
    if d.size == 0:
        return {"recurrence_rate": float("nan"), "determinism": float("nan"), "Lmax": float("nan"), "trapping_time": float("nan")}
    thr = np.quantile(d, recurrence_rate)
    R = (D <= thr).astype(np.uint8)
    np.fill_diagonal(R, 0)
    N = R.shape[0]
    rec_points = int(np.sum(R))
    rec_rate = rec_points / max(1, (N * N - N))
    det_points = 0
    Lmax = 0
    for k in range(-(N - 1), N):
        diag = np.diag(R, k=k)
        if diag.size < lmin:
            continue
        cnt = 0
        for v in diag:
            if v == 1:
                cnt += 1
            else:
                if cnt >= lmin:
                    det_points += cnt
                    Lmax = max(Lmax, cnt)
                cnt = 0
        if cnt >= lmin:
            det_points += cnt
            Lmax = max(Lmax, cnt)
    det = det_points / rec_points if rec_points > 0 else float("nan")
    # Trapping time
    traps = []
    for j in range(N):
        col = R[:, j]
        cnt = 0
        for i in range(N):
            if col[i] == 1:
                cnt += 1
            else:
                if cnt >= vmin:
                    traps.append(cnt)
                cnt = 0
        if cnt >= vmin:
            traps.append(cnt)
    trapping_time = float(np.mean(traps)) if traps else float("nan")
    return {
        "recurrence_rate": float(rec_rate),
        "determinism": float(det),
        "Lmax": float(Lmax),
        "trapping_time": trapping_time,
    }


@dataclass
class ChaosConfig:
    m: int
    tau: int


def _defaults_for_modality(modality: str, fs: float) -> ChaosConfig:
    mod = (modality or "generic").lower()
    if mod == "cardiac":
        tau = max(1, int(0.05 * fs))  # ~50 ms
        m = 3
    elif mod == "vagus":
        tau = max(1, int(0.002 * fs))  # ~2 ms
        m = 5
    elif mod == "eeg":
        tau = max(1, int(0.015 * fs))  # ~15 ms
        m = 5
    else:
        # Generic: let compute_window_metrics pick tau from the actual signal; keep m moderate
        # Set a conservative placeholder tau; will be overridden for 'generic'
        tau = max(8, int(0.01 * fs))
        m = 5
    return ChaosConfig(m=m, tau=tau)


def compute_window_metrics(signal: np.ndarray, fs: float, modality: str = "generic",
                           override_m: int | None = None, override_tau_samples: int | None = None) -> Dict[str, float]:
    if fs <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 100:
        raise ValueError("signal must be 1D and sufficiently long")
    cfg = _defaults_for_modality(modality, fs)
    m = override_m or cfg.m
    if override_tau_samples is not None:
        tau = override_tau_samples
    else:
        if (modality or "").lower() == "generic":
            # Choose tau from signal autocorr zero-crossing for generic modality
            tau = _autocorr_first_zero(xc)
        else:
            tau = cfg.tau
    # center/scale lightly for robustness
    xc = x - np.mean(x)
    s = np.std(xc) + 1e-12
    xc = xc / s
    lam1 = lyap_rosenstein(xc, m=m, tau=tau, fs=fs)
    D2 = correlation_dimension_gp(xc, m=m, tau=tau)
    mse_auc_val = mse_auc(xc, max_scale=20, m=2)
    rq = rqa_metrics(xc, m=min(m, 3), tau=tau, recurrence_rate=0.03, lmin=2, vmin=2)
    horizon = (1.0 / lam1) if (lam1 is not None and np.isfinite(lam1) and lam1 > 0) else float("nan")
    notes = []
    quality = "ok"
    if not np.isfinite(lam1) or not np.isfinite(D2):
        quality = "insufficient"
        notes.append("metric failure (NaN)")
    return {
        "lambda1": float(lam1) if lam1 is not None else float("nan"),
        "predictability_horizon_s": float(horizon) if np.isfinite(horizon) else float("nan"),
        "D2": float(D2) if D2 is not None else float("nan"),
        "mse_auc_1_20": float(mse_auc_val),
        "rqa_det": float(rq.get("determinism", float("nan"))),
        "rqa_Lmax": float(rq.get("Lmax", float("nan"))),
        "rqa_trap": float(rq.get("trapping_time", float("nan"))),
        "quality": quality,
        "notes": "; ".join(notes) if notes else "",
        "m": int(m),
        "tau_samples": int(tau),
    }
