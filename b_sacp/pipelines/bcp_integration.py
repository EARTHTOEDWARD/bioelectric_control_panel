"""
Thin glue for the Bioelectric Control Panel:
- Accepts time series (N x C)
- Embeds channels (Takens)
- Computes LLE (Rosenstein), RQA, and invariant measure on 2D projections
"""
import numpy as np
from typing import Dict, Tuple
from ..analysis.rqa import recurrence_matrix, rqa_metrics
from ..analysis.invariant_measure import invariant_histogram

def takens_embed(x: np.ndarray, m=3, tau=5) -> np.ndarray:
    N = x.shape[0] - (m-1)*tau
    if N <= 0: raise ValueError("Time series too short for embedding.")
    Y = np.stack([x[i:i+N] for i in range(0, m*tau, tau)], axis=1)
    return Y

def rosenstein_lle(x: np.ndarray, m=6, tau=4, k=10, fit_range=(5,50)) -> float:
    Y = takens_embed(x, m=m, tau=tau)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(Y)
    dists, idxs = nbrs.kneighbors(Y)
    nn = idxs[:,1]  # nearest excluding self
    # divergence curve
    L = min(Y.shape[0] - nn.shape[0], 200)
    d = []
    for j in range(L):
        valid = (nn + j) < Y.shape[0]
        dj = np.linalg.norm(Y[j: j+valid.sum()] - Y[nn[:valid.sum()] + j], axis=1)
        d.append(np.log(dj + 1e-12).mean())
    d = np.array(d)
    a, b = np.polyfit(np.arange(fit_range[0], min(fit_range[1], len(d))), d[fit_range[0]:min(fit_range[1], len(d))], 1)
    return float(a)

def analyze_bioelectric_timeseries(ts: np.ndarray, bins=128, eps_quant=0.1) -> Dict:
    """
    ts: array (N, C)
    Returns invariant histogram (coarse), RQA on first PC, and Rosenstein LLE.
    """
    X = ts
    # invariant measure on first 2 channels (coarse proxy)
    H, edges = invariant_histogram(X[:,:min(3, X.shape[1])], bins=bins)
    # RQA on PC1
    Xc = X - X.mean(0, keepdims=True)
    u, s, vh = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Xc @ vh.T[:,0]
    D = np.abs(pc1[:,None] - pc1[None,:])
    thr = np.quantile(D, eps_quant)
    R = (D <= thr).astype(np.uint8)
    rq = rqa_metrics(R)
    lle = rosenstein_lle(pc1)
    return {"invariant_H": H, "RQA": rq, "LLE": lle}

