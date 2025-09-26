from typing import List, Tuple, Dict
import numpy as np

# --------- Mann-Whitney U (normal approx with tie correction) ---------

def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n1 = x.size; n2 = y.size
    if n1 == 0 or n2 == 0:
        return float('nan'), float('nan')
    all_vals = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(all_vals)) + 1  # dense ranks (average ties handled below)
    # Handle average ranks for ties
    order = np.argsort(all_vals)
    sorted_vals = all_vals[order]
    rank_vals = np.empty_like(sorted_vals, dtype=float)
    i = 0
    while i < sorted_vals.size:
        j = i + 1
        while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_vals[i:j] = avg_rank
        i = j
    # Place back
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    rank_series = rank_vals[inv]
    R1 = np.sum(rank_series[:n1])
    U1 = R1 - n1*(n1+1)/2.0
    U2 = n1*n2 - U1
    U = min(U1, U2)
    mu = n1*n2/2.0
    # tie correction
    _, counts = np.unique(sorted_vals, return_counts=True)
    tie_term = np.sum(counts*(counts**2 - 1))
    sigma = np.sqrt(n1*n2*(n1+n2+1 - tie_term/((n1+n2)*(n1+n2-1))) / 12.0)
    if sigma == 0:
        return U, float('nan')
    z = (U - mu + 0.5) / sigma  # continuity correction
    # two-sided p
    p = 2.0 * 0.5 * (1 - erf(abs(z)/np.sqrt(2)))
    return float(U), float(p)


def erf(x: float) -> float:
    # numerical approximation via numpy
    return float(np.math.erf(x))


# --------- Cliff's delta ---------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n1 = x.size; n2 = y.size
    if n1 == 0 or n2 == 0:
        return float('nan')
    # Efficient delta via sorting
    x_sorted = np.sort(x); y_sorted = np.sort(y)
    i = j = 0
    n_gt = n_lt = 0
    while i < n1 and j < n2:
        if x_sorted[i] > y_sorted[j]:
            n_gt += n1 - i
            j += 1
        elif x_sorted[i] < y_sorted[j]:
            n_lt += n2 - j
            i += 1
        else:
            # ties: advance both while equal
            v = x_sorted[i]
            ci = 0; cj = 0
            while i < n1 and x_sorted[i] == v:
                i += 1; ci += 1
            while j < n2 and y_sorted[j] == v:
                j += 1; cj += 1
            # ties do not contribute to gt/lt
    delta = (n_gt - n_lt) / (n1 * n2)
    return float(delta)


# --------- Bootstrap CI (percentile or BCa) ---------

def bootstrap_ci_diff_median(x: np.ndarray, y: np.ndarray, iters: int = 10000, method: str = "percentile", seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float('nan'), float('nan')
    b = []
    n1, n2 = x.size, y.size
    for _ in range(iters):
        xb = x[rng.integers(0, n1, n1)]
        yb = y[rng.integers(0, n2, n2)]
        b.append(np.median(yb) - np.median(xb))
    b = np.array(b)
    if method == "percentile":
        lo, hi = np.quantile(b, [0.025, 0.975])
        return float(lo), float(hi)
    # BCa: compute acceleration via jackknife
    theta_hat = np.median(y) - np.median(x)
    # jackknife
    jack = []
    for i in range(n1):
        jx = np.delete(x, i)
        jack.append(np.median(y) - np.median(jx))
    for j in range(n2):
        jy = np.delete(y, j)
        jack.append(np.median(jy) - np.median(x))
    jack = np.array(jack)
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack)**3)
    den = 6.0 * (np.sum((jack_mean - jack)**2) ** 1.5 + 1e-12)
    a = num / (den + 1e-12)
    # bias-correction
    z0 = np.sqrt(2) * inv_erf(2 * (b < theta_hat).mean() - 1)
    def z(q):
        return np.sqrt(2) * inv_erf(2*q - 1)
    def Phi(zv):
        return 0.5 * (1 + erf(zv/np.sqrt(2)))
    alphas = [0.025, 0.975]
    zs = []
    for a0 in alphas:
        adj = z0 + (z0 + z(a0)) / (1 - a * (z0 + z(a0)) + 1e-12)
        zs.append(adj)
    qs = [Phi(zs[0]), Phi(zs[1])]
    lo, hi = np.quantile(b, qs)
    return float(lo), float(hi)


def inv_erf(y: float) -> float:
    # Inverse error function via numpy
    return float(np.math.erfinv(y))


# --------- FDR (Benjamini-Hochberg) ---------

def fdr_bh(pvals: List[float]) -> List[float]:
    p = np.array([0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in pvals], dtype=float)
    n = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n+1)
    q = p * n / ranks
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    qvals = np.empty_like(q)
    qvals[order] = np.clip(q_sorted, 0, 1)
    return [float(v) for v in qvals]
