import numpy as np

def recurrence_matrix(x: np.ndarray, eps: float):
    D = np.linalg.norm(x[:,None,:] - x[None,:,:], axis=2)
    return (D <= eps).astype(np.uint8)

def rqa_metrics(R: np.ndarray, lmin=2):
    N = R.shape[0]
    # Recurrence rate
    RR = R.sum() / (N*N)
    # Diagonal line lengths
    Ls = []
    for k in range(-N+1, N):
        diag = np.diag(R, k=k)
        c = 0
        for v in diag:
            if v: c += 1
            elif c>0:
                if c>=lmin: Ls.append(c)
                c = 0
        if c>=lmin: Ls.append(c)
    DET = (sum(Ls)/ (R.sum()+1e-15)) if R.sum()>0 else 0.0
    return {"RR": RR, "DET": DET}

