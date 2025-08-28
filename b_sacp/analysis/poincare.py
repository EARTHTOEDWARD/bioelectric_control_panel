import numpy as np

def section_crossings(X: np.ndarray, i:int, c:float, direction:int=+1):
    """
    Return intersection points of trajectory X with plane x_i = c.
    Linear interpolation between samples; direction=+1 for upward crossings,
    -1 for downward, 0 for both.
    """
    xi = X[:, i]
    sgn = np.sign(xi - c)
    idx = np.where((sgn[:-1] < 0) & (sgn[1:] > 0))[0] if direction>=0 else \
          np.where((sgn[:-1] > 0) & (sgn[1:] < 0))[0]
    P = []
    for k in idx:
        a = (c - xi[k])/(xi[k+1]-xi[k] + 1e-15)
        p = X[k] + a*(X[k+1]-X[k])
        P.append(p)
    return np.array(P)

