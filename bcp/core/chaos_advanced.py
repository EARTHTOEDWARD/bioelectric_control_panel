from __future__ import annotations

from typing import Dict, Tuple, List

import math
import numpy as np


def _takens_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    N = x.size - (m - 1) * tau
    if N <= 0:
        raise ValueError("signal too short for requested embedding")
    Y = np.zeros((N, m), dtype=float)
    for i in range(m):
        Y[:, i] = x[i * tau : i * tau + N]
    return Y


def lyap_rosenstein_ci(
    x: np.ndarray,
    m: int,
    tau: int,
    fs: float,
    theiler: int | None = None,
    max_t: float = 0.4,
    max_points: int = 4000,
    n_boot: int = 100,
) -> Tuple[float, Tuple[float, float], np.ndarray, np.ndarray]:
    """Largest Lyapunov exponent (Rosenstein) with bootstrap CI.

    Returns: (slope, (lo, hi), time_axis, mean_log_curve)
    """
    if theiler is None:
        theiler = max(1, int(1.5 * tau))
    Y = _takens_embedding(x, m, tau)
    N = Y.shape[0]
    # downsample to cap O(N^2)
    idx = np.arange(N)
    if N > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(N, size=max_points, replace=False))
    YS = Y[idx]
    D = np.linalg.norm(YS[:, None, :] - YS[None, :, :], axis=2)
    # crude Theiler band
    for i in range(YS.shape[0]):
        lo = max(0, i - theiler)
        hi = min(YS.shape[0], i + theiler + 1)
        D[i, lo:hi] = np.inf
        D[i, i] = np.inf
    nn = np.argmin(D, axis=1)
    max_k = min(int(max_t * fs), YS.shape[0] - 1)
    ks = np.arange(max_k)
    valid_counts = np.zeros(max_k)
    mean_log = np.zeros(max_k)
    for k in range(1, max_k):
        vals: List[float] = []
        for i in range(YS.shape[0] - k):
            j = nn[i]
            if j + k >= YS.shape[0] or i + k >= YS.shape[0]:
                continue
            d0 = np.linalg.norm(YS[i] - YS[j])
            d1 = np.linalg.norm(YS[i + k] - YS[j + k])
            if d0 > 0 and d1 > 0:
                vals.append(float(np.log(d1 / d0)))
        if len(vals) >= 5:
            mean_log[k] = float(np.mean(vals))
            valid_counts[k] = len(vals)
        else:
            mean_log[k] = np.nan
    mask = np.isfinite(mean_log) & (valid_counts > 10)
    if np.sum(mask) < 5:
        return float("nan"), (float("nan"), float("nan")), ks / fs, mean_log
    sel = np.where(mask)[0]
    end = max(3, int(0.3 * sel.size))
    seg = sel[:end]
    slope = float(np.polyfit(ks[seg] / fs, mean_log[seg], 1)[0])
    # bootstrap CI
    lo, hi = float("nan"), float("nan")
    if seg.size >= 5 and n_boot > 2:
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(n_boot):
            bsel = rng.choice(seg, size=seg.size, replace=True)
            boots.append(float(np.polyfit(ks[bsel] / fs, mean_log[bsel], 1)[0]))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return slope, (lo, hi), ks / fs, mean_log


def lyap_rosenstein_ci_ex(
    x: np.ndarray,
    m: int,
    tau: int,
    fs: float,
    theiler: int | None = None,
    max_t: float = 0.4,
    max_points: int = 4000,
    n_boot: int = 100,
) -> Tuple[float, Tuple[float, float], np.ndarray, np.ndarray, int]:
    """As lyap_rosenstein_ci, but also returns the fitted segment length.

    Returns: (slope, (lo, hi), time_axis, mean_log_curve, seg_len)
    """
    if theiler is None:
        theiler = max(1, int(1.5 * tau))
    Y = _takens_embedding(x, m, tau)
    N = Y.shape[0]
    idx = np.arange(N)
    if N > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(N, size=max_points, replace=False))
    YS = Y[idx]
    D = np.linalg.norm(YS[:, None, :] - YS[None, :, :], axis=2)
    for i in range(YS.shape[0]):
        lo = max(0, i - theiler)
        hi = min(YS.shape[0], i + theiler + 1)
        D[i, lo:hi] = np.inf
        D[i, i] = np.inf
    nn = np.argmin(D, axis=1)
    max_k = min(int(max_t * fs), YS.shape[0] - 1)
    ks = np.arange(max_k)
    valid_counts = np.zeros(max_k)
    mean_log = np.zeros(max_k)
    for k in range(1, max_k):
        vals: List[float] = []
        for i in range(YS.shape[0] - k):
            j = nn[i]
            if j + k >= YS.shape[0] or i + k >= YS.shape[0]:
                continue
            d0 = np.linalg.norm(YS[i] - YS[j])
            d1 = np.linalg.norm(YS[i + k] - YS[j + k])
            if d0 > 0 and d1 > 0:
                vals.append(float(np.log(d1 / d0)))
        if len(vals) >= 5:
            mean_log[k] = float(np.mean(vals))
            valid_counts[k] = len(vals)
        else:
            mean_log[k] = np.nan
    mask = np.isfinite(mean_log) & (valid_counts > 10)
    if np.sum(mask) < 5:
        return float("nan"), (float("nan"), float("nan")), ks / fs, mean_log, 0
    sel = np.where(mask)[0]
    end = max(3, int(0.3 * sel.size))
    seg = sel[:end]
    slope = float(np.polyfit(ks[seg] / fs, mean_log[seg], 1)[0])
    lo, hi = float("nan"), float("nan")
    if seg.size >= 5 and n_boot > 2:
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(n_boot):
            bsel = rng.choice(seg, size=seg.size, replace=True)
            boots.append(float(np.polyfit(ks[bsel] / fs, mean_log[bsel], 1)[0]))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return slope, (lo, hi), ks / fs, mean_log, int(seg.size)


def gp_dimension_scaling_ci(
    x: np.ndarray,
    m: int,
    tau: int,
    n_r: int = 28,
    max_points: int = 4000,
    n_boot: int = 100,
) -> Tuple[float, Tuple[float, float], np.ndarray, np.ndarray, Tuple[int, int]]:
    """Grassberger–Procaccia correlation dimension with auto scaling-region + CI.

    Returns: (slope, (lo, hi), radii, C(r), (i0, i1) band)
    """
    Y = _takens_embedding(x, m, tau)
    N = Y.shape[0]
    idx = np.arange(N)
    if N > max_points:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(N, size=max_points, replace=False))
    YS = Y[idx]
    D = np.linalg.norm(YS[:, None, :] - YS[None, :, :], axis=2)
    tri = np.triu_indices_from(D, k=1)
    d = D[tri]
    d = d[d > 0]
    d.sort()
    if d.size < 100:
        return float("nan"), (float("nan"), float("nan")), np.array([]), np.array([]), (0, 0)
    rmin = float(np.percentile(d, 2))
    rmax = float(np.percentile(d, 98))
    if rmin <= 0 or rmax <= 0 or not np.isfinite(rmin) or not np.isfinite(rmax):
        return float("nan"), (float("nan"), float("nan")), np.array([]), np.array([]), (0, 0)
    rs = np.logspace(np.log10(rmin), np.log10(rmax), n_r)
    C = np.array([np.mean(d < r) for r in rs])
    eps = 1e-12
    local = np.diff(np.log(C + eps)) / np.diff(np.log(rs + eps))
    if local.size < 6:
        slope = float(np.polyfit(np.log(rs), np.log(C + eps), 1)[0])
        return slope, (float("nan"), float("nan")), rs, C, (0, rs.size)
    MA = np.convolve(local, np.ones(3) / 3.0, mode="valid")
    best_len, best_i, best_var = 0, None, np.inf
    for L in range(5, MA.size + 1):
        for i in range(0, MA.size - L + 1):
            v = float(np.var(MA[i : i + L]))
            if v < best_var:
                best_var = v
                best_len = L
                best_i = i
    if best_i is None:
        slope = float(np.polyfit(np.log(rs), np.log(C + eps), 1)[0])
        return slope, (float("nan"), float("nan")), rs, C, (0, rs.size)
    i0 = best_i + 1  # convert from local-slope indices to rs indices
    i1 = i0 + best_len
    X = np.log(rs[i0:i1])
    Yv = np.log(C[i0:i1] + eps)
    slope = float(np.polyfit(X, Yv, 1)[0])
    # bootstrap CI within [i0, i1)
    rng = np.random.default_rng(0)
    boots: List[float] = []
    for _ in range(n_boot):
        sel = rng.integers(i0, i1, size=(i1 - i0))
        bX = X[sel - i0]
        bY = Yv[sel - i0]
        boots.append(float(np.polyfit(bX, bY, 1)[0]))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return slope, (lo, hi), rs, C, (i0, i1)


def rqa_sweep(
    Y: np.ndarray, rec_rates: List[float] | None = None, lmin: int = 2, vmin: int = 2
) -> Dict[str, Dict[str, float]]:
    """Sweep RQA over multiple recurrence rates and summarize stability.

    Returns: {"determinism": {mean, std, rel_std}, "Lmax": {...}, "trapping_time": {...}}
    """
    if rec_rates is None:
        rec_rates = [0.02, 0.03, 0.04]
    dets: List[float] = []
    lmaxs: List[float] = []
    traps: List[float] = []
    for rr in rec_rates:
        D = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
        tri = np.triu_indices_from(D, k=1)
        d = D[tri]
        thr = float(np.quantile(d, rr))
        R = (D <= thr).astype(np.uint8)
        np.fill_diagonal(R, 0)
        N = R.shape[0]
        rec_pts = int(np.sum(R))
        # Diagonal determinism and Lmax
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
        det = det_points / max(1, rec_pts)
        # Trapping time (vertical lines)
        trap_lengths: List[int] = []
        for j in range(N):
            col = R[:, j]
            cnt = 0
            for i in range(N):
                if col[i] == 1:
                    cnt += 1
                else:
                    if cnt >= vmin:
                        trap_lengths.append(cnt)
                    cnt = 0
            if cnt >= vmin:
                trap_lengths.append(cnt)
        trap = float(np.mean(trap_lengths)) if trap_lengths else float("nan")
        dets.append(float(det))
        lmaxs.append(float(Lmax))
        traps.append(float(trap))
    def _stats(vals: List[float]) -> Dict[str, float]:
        a = np.asarray(vals, dtype=float)
        m = float(np.nanmean(a))
        s = float(np.nanstd(a))
        return {"mean": m, "std": s, "rel_std": float(s / (abs(m) + 1e-9))}
    return {
        "determinism": _stats(dets),
        "Lmax": _stats(lmaxs),
        "trapping_time": _stats(traps),
    }
