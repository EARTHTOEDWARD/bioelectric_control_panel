from typing import Tuple, Optional, Dict, List
import numpy as np

# --------- Window helpers ---------

def _hanning(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def _detrend(x: np.ndarray, mode: str) -> np.ndarray:
    if mode is None or mode == "none":
        return x
    if mode == "mean":
        return x - np.nanmean(x)
    if mode == "linear":
        n = x.size
        t = np.arange(n, dtype=float)
        # least squares fit
        A = np.vstack([t, np.ones(n)]).T
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        trend = A @ coef
        return x - trend
    return x


# --------- Welch PSD/CSD ---------

def _segment_indices(n: int, seg_len: int, overlap: float) -> List[Tuple[int, int]]:
    if seg_len <= 0 or seg_len > n:
        seg_len = n
    step = max(1, int(seg_len * (1.0 - overlap)))
    idxs = []
    start = 0
    while start + seg_len <= n:
        idxs.append((start, start + seg_len))
        start += step
    if not idxs:
        idxs = [(0, n)]
    return idxs


def welch_psd(x: np.ndarray, fs: float, seg_len_s: float, overlap: float, detrend: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid fs for PSD")
    n = x.size
    seg_len = int(max(8, round(seg_len_s * fs)))
    idxs = _segment_indices(n, seg_len, overlap)
    win = _hanning(seg_len)
    U = (win**2).sum() / seg_len
    psd = None
    for (i0, i1) in idxs:
        seg = x[i0:i1]
        if detrend:
            seg = _detrend(seg, detrend)
        seg = seg * win
        # rfft
        X = np.fft.rfft(seg)
        Pxx = (np.abs(X)**2) / (fs * seg_len * U)
        if psd is None:
            psd = Pxx
        else:
            psd = psd + Pxx
    psd = psd / len(idxs)
    f = np.fft.rfftfreq(seg_len, d=1.0/fs)
    return f, psd


def welch_csd(x: np.ndarray, y: np.ndarray, fs: float, seg_len_s: float, overlap: float, detrend: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size != x.size:
        n = min(x.size, y.size)
        x = x[:n]; y = y[:n]
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid fs for CSD")
    n = x.size
    seg_len = int(max(8, round(seg_len_s * fs)))
    idxs = _segment_indices(n, seg_len, overlap)
    win = _hanning(seg_len)
    U = (win**2).sum() / seg_len
    Pxx = None; Pyy = None; Pxy = None
    for (i0, i1) in idxs:
        sx = x[i0:i1]; sy = y[i0:i1]
        if detrend:
            sx = _detrend(sx, detrend)
            sy = _detrend(sy, detrend)
        sx = sx * win; sy = sy * win
        X = np.fft.rfft(sx); Y = np.fft.rfft(sy)
        cur_Pxx = (np.abs(X)**2) / (fs * seg_len * U)
        cur_Pyy = (np.abs(Y)**2) / (fs * seg_len * U)
        cur_Pxy = (X * np.conj(Y)) / (fs * seg_len * U)
        if Pxx is None:
            Pxx = cur_Pxx; Pyy = cur_Pyy; Pxy = cur_Pxy
        else:
            Pxx += cur_Pxx; Pyy += cur_Pyy; Pxy += cur_Pxy
    Pxx /= len(idxs); Pyy /= len(idxs); Pxy /= len(idxs)
    f = np.fft.rfftfreq(seg_len, d=1.0/fs)
    return f, Pxx, Pyy, Pxy


def band_mask(f: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    fmin, fmax = band
    return (f >= fmin) & (f <= fmax)


def magnitude_squared_coherence(x: np.ndarray, y: np.ndarray, fs: float, seg_len_s: float, overlap: float, detrend: str) -> Tuple[np.ndarray, np.ndarray, float]:
    f, Pxx, Pyy, Pxy = welch_csd(x, y, fs, seg_len_s, overlap, detrend)
    eps = 1e-12
    coh_f = (np.abs(Pxy)**2) / (np.maximum(Pxx, eps) * np.maximum(Pyy, eps))
    # Average coherence across band (caller chooses band)
    return f, coh_f, float('nan')


# --------- Robust beta (1/f slope) ---------

def robust_beta_loglog(f: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> Tuple[float, float]:
    """
    Fit log10(PSD) ~ a + b*log10(f) in band. Return beta = -b and R^2.
    Robust IRLS (Huber) for up to 20 iters.
    """
    mask = band_mask(f, band)
    f_fit = f[mask]
    p_fit = psd[mask]
    ok = (f_fit > 0) & np.isfinite(p_fit) & (p_fit > 0)
    f_fit = f_fit[ok]
    p_fit = p_fit[ok]
    if f_fit.size < 8:
        return float('nan'), float('nan')
    X = np.log10(f_fit)
    Y = np.log10(p_fit)
    A = np.vstack([X, np.ones_like(X)]).T
    w = np.ones_like(Y)
    for _ in range(20):
        # weighted least squares
        Aw = A * w[:, None]
        Yw = Y * w
        coef, *_ = np.linalg.lstsq(Aw, Yw, rcond=None)
        resid = Y - (A @ coef)
        s = 1.345 * np.median(np.abs(resid - np.median(resid)) + 1e-12)
        if s <= 0: s = np.std(resid) + 1e-12
        r = resid / (s + 1e-12)
        w_new = 1.0 / np.maximum(1.0, np.abs(r))  # Huber
        if np.allclose(w_new, w, atol=1e-3):
            w = w_new
            break
        w = w_new
    slope = coef[0]
    beta = -float(slope)
    yhat = A @ coef
    ss_res = float(np.sum((Y - yhat)**2))
    ss_tot = float(np.sum((Y - np.mean(Y))**2) + 1e-12)
    r2 = 1.0 - ss_res/ss_tot
    return beta, r2


# --------- Step metrics ---------

def compute_step_metrics(t: np.ndarray, x: np.ndarray, event_t: float, pre_win: float, pre_gap: float, post_start: float, post_win: float, search_win: float) -> Tuple[float, float, float, float]:
    t = np.asarray(t, dtype=float); x = np.asarray(x, dtype=float)
    if not np.isfinite(event_t):
        return float('nan'), float('nan'), float('nan'), float('nan')
    # pre window
    pre_start = event_t - pre_gap - pre_win
    pre_end = event_t - pre_gap
    pre_mask = (t >= pre_start) & (t < pre_end)
    if pre_mask.sum() < 8:
        idx_pre = np.where(t < (event_t - pre_gap))[0]
        if idx_pre.size >= 8:
            K = min(max(8, int(0.2*len(t))), idx_pre.size)
            sel = idx_pre[-K:]
            pre_mask = np.zeros_like(t, dtype=bool)
            pre_mask[sel] = True
        else:
            return float('nan'), float('nan'), float('nan'), float('nan')
    # post window
    post_start_abs = event_t + post_start
    post_end_abs = post_start_abs + post_win
    post_mask = (t >= post_start_abs) & (t < post_end_abs)
    if post_mask.sum() < 8:
        idx_post = np.where(t >= event_t)[0]
        if idx_post.size >= 8:
            K = min(max(8, int(0.2*len(t))), idx_post.size)
            sel = idx_post[:K]
            post_mask = np.zeros_like(t, dtype=bool)
            post_mask[sel] = True
        else:
            return float('nan'), float('nan'), float('nan'), float('nan')
    pre_mean = float(np.nanmean(x[pre_mask]))
    post_mean = float(np.nanmean(x[post_mask]))
    amp = post_mean - pre_mean
    tau63 = float('nan')
    after_mask = (t >= event_t) & (t <= event_t + search_win)
    if after_mask.any() and np.isfinite(amp):
        target = pre_mean + 0.632 * amp
        w_t = t[after_mask]; w_x = x[after_mask]
        if amp >= 0:
            hits = np.where(w_x >= target)[0]
        else:
            hits = np.where(w_x <= target)[0]
        if hits.size:
            tau63 = float(w_t[hits[0]] - event_t)
    return float(amp), float(tau63), pre_mean, post_mean


# --------- Windowed ---------

def sliding_windows(n: int, win: int, step: int) -> List[Tuple[int, int]]:
    out = []
    i = 0
    while i + win <= n:
        out.append((i, i + win))
        i += step
    if not out and n >= win:
        out = [(0, win)]
    return out


# --------- Fractal dimensions ---------

def higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float('nan')
    Lk = []
    k_vals = np.arange(1, kmax+1)
    for k in k_vals:
        Lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            diffs = np.abs(np.diff(x[idx]))
            scale = (n - 1) / ( (idx.size - 1) * k )
            Lm.append(scale * diffs.sum())
        if len(Lm) == 0:
            Lk.append(np.nan)
        else:
            Lk.append(np.mean(Lm))
    Lk = np.array(Lk)
    ok = np.isfinite(Lk) & (Lk > 0)
    if ok.sum() < 2:
        return float('nan')
    logk = np.log(1.0 / k_vals[ok])
    logL = np.log(Lk[ok])
    A = np.vstack([logk, np.ones_like(logk)]).T
    coef, *_ = np.linalg.lstsq(A, logL, rcond=None)
    return float(coef[0])


def katz_fd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float('nan')
    L = np.sum(np.abs(np.diff(x)))
    d = np.max(np.abs(x - x[0]))
    if d == 0:
        return 1.0
    return float(np.log(n) / (np.log(n) + np.log(L/d)))


# --------- RQA ---------

def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size - (m-1)*tau
    if n <= 0:
        return np.empty((0, m))
    out = np.zeros((n, m), dtype=float)
    for i in range(m):
        out[:, i] = x[i*tau:i*tau+n]
    return out


def recurrence_plot(X: np.ndarray, eps: float) -> np.ndarray:
    if X.size == 0:
        return np.zeros((0, 0), dtype=bool)
    # pairwise distances
    d2 = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    R = d2 <= (eps**2)
    return R


def rqa_metrics(x: np.ndarray, m: int, tau: int, thr_q: float, diag_min: int, vert_min: int) -> Tuple[float, float, float]:
    X = embed(x, m, tau)
    if X.shape[0] < 8:
        return float('nan'), float('nan'), float('nan')
    # choose epsilon as quantile of distances
    d2 = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    if d2.size == 0:
        return float('nan'), float('nan'), float('nan')
    tri = d2[np.triu_indices_from(d2, k=1)]
    if tri.size == 0:
        return float('nan'), float('nan'), float('nan')
    eps = float(np.sqrt(np.quantile(tri, thr_q)))
    R = d2 <= (eps**2)
    # remove line of identity
    np.fill_diagonal(R, False)
    # determinism: ratio of points in diag lines >= diag_min to total recurrence points
    total = int(R.sum())
    if total == 0:
        return float('nan'), float('nan'), float('nan')
    det_points = 0
    lam_points = 0
    vert_lengths: List[int] = []
    # diagonal lines
    for offset in range(-R.shape[0]+1, R.shape[0]):
        diag = np.diag(R, k=offset)
        if diag.size == 0:
            continue
        # count runs of True
        count = 0
        for v in diag:
            if v:
                count += 1
            elif count > 0:
                if count >= diag_min:
                    det_points += count
                count = 0
        if count >= diag_min:
            det_points += count
    # vertical lines
    for j in range(R.shape[1]):
        col = R[:, j]
        count = 0
        for v in col:
            if v:
                count += 1
            elif count > 0:
                if count >= vert_min:
                    lam_points += count
                    vert_lengths.append(count)
                count = 0
        if count >= vert_min:
            lam_points += count
            vert_lengths.append(count)
    DET = det_points / total if total else float('nan')
    LAM = lam_points / total if total else float('nan')
    TT = float(np.mean(vert_lengths)) if vert_lengths else float('nan')
    return float(DET), float(LAM), TT


# --------- Directed metrics ---------

def phase_slope_index(x: np.ndarray, y: np.ndarray, fs: float, seg_len_s: float, overlap: float, detrend: str, band: Tuple[float, float]) -> float:
    f, Pxx, Pyy, Pxy = welch_csd(x, y, fs, seg_len_s, overlap, detrend)
    # coherency
    eps = 1e-12
    Cxy = Pxy / np.sqrt(np.maximum(Pxx, eps) * np.maximum(Pyy, eps))
    mask = band_mask(f, band)
    idx = np.where(mask)[0]
    if idx.size < 3:
        return float('nan')
    # PSI estimate: sum imag(conj(C(f)) * C(f+1)) across band (Nolte 2008 inspired)
    psi_sum = 0.0
    for i in range(idx[0], idx[-1]):
        psi_sum += np.imag(np.conj(Cxy[i]) * Cxy[i+1])
    return float(psi_sum)


def lagged_xcorr_peak(x: np.ndarray, y: np.ndarray, max_lag: int) -> Tuple[int, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    max_lag = min(max_lag, n-1)
    best_lag = 0; best_val = -1e9
    denom = (np.std(x) * np.std(y) + 1e-12)
    for lag in range(-max_lag, max_lag+1):
        if lag >= 0:
            a = x[lag:]; b = y[:n-lag]
        else:
            a = x[:n+lag]; b = y[-lag:]
        if a.size < 8:
            continue
        val = float(np.dot(a, b) / (a.size * denom))
        if val > best_val:
            best_val = val; best_lag = lag
    return best_lag, best_val
