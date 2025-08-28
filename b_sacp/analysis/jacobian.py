import numpy as np
from typing import Dict, Optional, Tuple
from scipy.optimize import root
from ..models.itik_banks import IB3DParams, ib3d_rhs


def jacobian_ib3d(x: np.ndarray, p: IB3DParams) -> np.ndarray:
    x1, x2, x3 = x
    J = np.zeros((3,3), dtype=float)
    # d(dx1)/dx*
    J[0,0] = 1.0 - 2.0*x1 - p.a12*x2 - p.a13*x3
    J[0,1] = -p.a12*x1
    J[0,2] = -p.a13*x1
    # d(dx2)/dx*
    J[1,0] = -p.a21*x2
    J[1,1] = p.r2 - 2.0*p.r2*x2 - p.a21*x1
    J[1,2] = 0.0
    # d(dx3)/dx*
    denom = (x1 + p.k3)
    J[2,0] = (p.r3 * x3 * p.k3) / (denom*denom) - p.a31*x3
    J[2,1] = 0.0
    J[2,2] = (p.r3 * x1) / denom - p.a31*x1 - p.d3
    return J


def classify_eigs(eigs: np.ndarray, tol: float = 1e-6) -> str:
    r = np.real(eigs)
    pos = np.sum(r > tol)
    neg = np.sum(r < -tol)
    zer = eigs.size - pos - neg
    if pos == 0 and neg == eigs.size:
        return "stable"
    if pos == eigs.size:
        return "unstable"
    if pos == 1 and neg == eigs.size-1:
        return "saddle-1"
    if pos == 2 and neg == 1:
        return "saddle-2"
    if zer >= 1 and pos == 0 and neg >= 1:
        return "saddle-degenerate"
    return "mixed"


def heuristics_zero_hopf(eigs: np.ndarray, tol_zero: float = 5e-4, tol_real: float = 5e-3, min_imag: float = 1e-3) -> bool:
    """
    Zero–Hopf heuristic: one eigenvalue near zero; the other two form a complex pair with
    small real parts and sufficiently large imaginary parts.
    """
    # Identify near-zero eigenvalue
    mag = np.abs(eigs)
    idx0 = np.argmin(mag)
    lam0 = eigs[idx0]
    if np.abs(lam0) > tol_zero:
        return False
    # Remaining two
    others = np.delete(eigs, idx0)
    # Should be complex conjugates with small real part
    if np.abs(others[0] - np.conj(others[1])) > 1e-6:
        # numeric noise tolerance
        pass
    if (np.abs(np.real(others)) < tol_real).all() and (np.abs(np.imag(others)) > min_imag).all():
        return True
    return False


def heuristics_bt(eigs: np.ndarray, tol_zero: float = 5e-4) -> bool:
    """Bogdanov–Takens heuristic: two eigenvalues near zero (by magnitude)."""
    mag = np.sort(np.abs(eigs))
    return (mag[0] < tol_zero) and (mag[1] < tol_zero)


def find_equilibrium_from_guess(p: IB3DParams, guess: np.ndarray) -> Optional[np.ndarray]:
    f = lambda z: ib3d_rhs(0.0, z, p)
    sol = root(f, np.asarray(guess, dtype=float), method="hybr")
    if sol.success:
        return sol.x
    return None


def analyze_equilibrium_near_traj(X: np.ndarray, p: IB3DParams,
                                  guess: Optional[np.ndarray] = None,
                                  tol_zero: float = 5e-4,
                                  tol_real: float = 5e-3,
                                  min_imag: float = 1e-3) -> Dict:
    """
    Try to locate an equilibrium near the trajectory (use mean as default guess),
    compute Jacobian eigenvalues and heuristic flags.
    """
    if guess is None:
        guess = np.median(X, axis=0)
    eq = find_equilibrium_from_guess(p, guess)
    if eq is None:
        return {"found": False}
    J = jacobian_ib3d(eq, p)
    eigs = np.linalg.eigvals(J)
    return {
        "found": True,
        "equilibrium": eq,
        "eigs": eigs,
        "class": classify_eigs(eigs),
        "zero_hopf": heuristics_zero_hopf(eigs, tol_zero, tol_real, min_imag),
        "bt": heuristics_bt(eigs, tol_zero),
    }

