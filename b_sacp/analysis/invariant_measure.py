import numpy as np
from .poincare import section_crossings

def invariant_histogram(X: np.ndarray, bins=100, ranges=None, normalize=True):
    """
    Full-state histogram as a crude invariant measure estimator.
    """
    d = X.shape[1]
    if isinstance(bins, int): bins = [bins]*d
    if ranges is None:
        ranges = [(X[:,j].min(), X[:,j].max()) for j in range(d)]
    H, edges = np.histogramdd(X, bins=bins, range=ranges)
    if normalize: H = H / (H.sum() + 1e-15)
    return H, edges

def invariant_on_section(X: np.ndarray, i:int, c:float, dims=(0,2), bins=200, ranges=None):
    """
    Invariant measure on Poincaré section x_i=c projected to dims (e.g., (x1,x3)).
    """
    P = section_crossings(X, i=i, c=c, direction=+1)
    Y = P[:, list(dims)]
    if ranges is None:
        ranges = [(Y[:,k].min(), Y[:,k].max()) for k in range(Y.shape[1])]
    H, xe, ye = np.histogram2d(Y[:,0], Y[:,1], bins=bins, range=ranges, density=True)
    return H, (xe, ye)

def convergence_trace(counts: np.ndarray):
    """
    Track ||hist(t)-hist_avg||_1 vs time to check 1/t-like decay.
    counts: sequence of cumulative hist counts (N_t x bins...)
    """
    flat = counts.reshape(counts.shape[0], -1)
    avg = flat.mean(axis=0)
    diffs = np.abs(flat - avg).sum(axis=1)
    return diffs / (avg.sum() + 1e-15)

