"""
Quick smoke test of IB3D Jacobian/eigen heuristics without SciPy.
Implements:
- Analytic Jacobian for Itik–Banks 3D model
- Simple RK4 integrator (optional usage)
- Damped Newton equilibrium solver
- Zero–Hopf / BT heuristics using eigenvalue patterns
"""
import numpy as np


# -------------------- Model & Jacobian --------------------

class IB3DParams:
    def __init__(self, a12=1.0, a13=2.5, a21=1.6, a31=2.4, r2=0.6, r3=6.0, d3=0.5, k3=1.0):
        self.a12 = a12
        self.a13 = a13
        self.a21 = a21
        self.a31 = a31
        self.r2 = r2
        self.r3 = r3
        self.d3 = d3
        self.k3 = k3


def ib3d_rhs(x, p: IB3DParams):
    x1, x2, x3 = x
    dx1 = x1*(1.0 - x1) - p.a12*x1*x2 - p.a13*x1*x3
    dx2 = p.r2*x2*(1.0 - x2) - p.a21*x1*x2
    dx3 = (p.r3*x1*x3)/(x1 + p.k3) - p.a31*x1*x3 - p.d3*x3
    return np.array([dx1, dx2, dx3], dtype=float)


def jacobian_ib3d(x, p: IB3DParams):
    x1, x2, x3 = x
    a12, a13, a21, a31, r2, r3, d3, k3 = p.a12, p.a13, p.a21, p.a31, p.r2, p.r3, p.d3, p.k3
    # dx1 partials
    d11 = (1.0 - 2.0*x1) - a12*x2 - a13*x3
    d12 = -a12 * x1
    d13 = -a13 * x1
    # dx2 partials
    d21 = -a21 * x2
    d22 = r2*(1.0 - 2.0*x2) - a21*x1
    d23 = 0.0
    # dx3 partials
    denom = (x1 + k3)
    if denom == 0:
        denom = 1e-12
    g = x1 / denom
    dg_dx1 = k3 / (denom*denom)
    d31 = r3 * (dg_dx1 * x3) - a31 * x3
    d32 = 0.0
    d33 = r3 * g - a31 * x1 - d3
    J = np.array([[d11, d12, d13],
                  [d21, d22, d23],
                  [d31, d32, d33]], dtype=float)
    return J


# -------------------- Newton equilibrium finder --------------------

def newton_equilibrium(p: IB3DParams, x0, max_iter=100, tol=1e-12):
    x = np.array(x0, dtype=float)
    last_norm = None
    for it in range(max_iter):
        F = ib3d_rhs(x, p)
        nrm = float(np.linalg.norm(F))
        if nrm < tol:
            return x, True, it, nrm
        J = jacobian_ib3d(x, p)
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(J, -F, rcond=None)
        # backtracking line search
        alpha = 1.0
        for _ in range(10):
            x_new = x + alpha*step
            nrm_new = float(np.linalg.norm(ib3d_rhs(x_new, p)))
            if nrm_new < nrm:
                x = x_new
                break
            alpha *= 0.5
        else:
            x = x + step
        last_norm = nrm
    return x, False, max_iter, last_norm if last_norm is not None else float("inf")


# -------------------- Heuristics --------------------

def classify_eigs(eigs, tol_zero=5e-4):
    re = np.real(eigs)
    im = np.imag(eigs)
    near_zero = np.sum(np.abs(re) < tol_zero)
    pos = np.sum(re > tol_zero)
    neg = np.sum(re < -tol_zero)
    has_complex_pair = np.sum(np.abs(im) > 1e-8) >= 2
    if pos == 0 and neg == 3:
        label = "stable_node_or_focus"
    elif pos == 1 and neg == 2:
        label = "saddle_12"
    elif pos == 2 and neg == 1:
        label = "saddle_21"
    elif pos == 3:
        label = "unstable_node_or_focus"
    else:
        label = "degenerate_or_borderline"
    return {"label": label, "pos": int(pos), "neg": int(neg), "near_zero": int(near_zero), "has_complex_pair": bool(has_complex_pair)}


def heuristics_zero_hopf(eigs, tol_zero=5e-4, tol_real=5e-3, min_imag=1e-3):
    eigs = np.array(eigs)
    re = np.real(eigs)
    im = np.imag(eigs)
    complex_small_real = np.where((np.abs(im) >= min_imag) & (np.abs(re) <= tol_real))[0]
    near_zero = np.sum(np.abs(re) < tol_zero)
    return (len(complex_small_real) >= 2) and (near_zero >= 1)


def heuristics_bt(eigs, tol_zero=5e-4):
    re = np.real(eigs)
    near_zero = np.sum(np.abs(re) < tol_zero)
    return near_zero >= 2


# -------------------- Optional RK4 helper --------------------

def rk4_step(x, p: IB3DParams, dt=0.01):
    k1 = ib3d_rhs(x, p)
    k2 = ib3d_rhs(x + 0.5*dt*k1, p)
    k3 = ib3d_rhs(x + 0.5*dt*k2, p)
    k4 = ib3d_rhs(x + dt*k3, p)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def run_smoke(a21=1.6, a31=2.4):
    p = IB3DParams(a21=a21, a31=a31, r2=0.6, r3=6.0, d3=0.5, k3=1.0)
    guesses = [
        (0.2, 0.25, 0.5),
        (0.3, 0.3, 0.4),
        (0.4, 0.2, 0.3),
        (0.15, 0.2, 0.6),
    ]
    eq = None
    for g in guesses:
        x_star, ok, iters, nrm = newton_equilibrium(p, g, max_iter=60, tol=1e-14)
        if ok and np.all(x_star > 0):
            eq = x_star
            break
    if eq is None:
        return {"eq_found": False, "message": "Equilibrium not found with provided guesses."}
    J = jacobian_ib3d(eq, p)
    eigs = np.linalg.eigvals(J)
    cls = classify_eigs(eigs)
    zh = heuristics_zero_hopf(eigs)
    bt = heuristics_bt(eigs)
    return {
        "eq_found": True,
        "eq": eq,
        "jacobian": J,
        "eigs": eigs,
        "classification": cls,
        "zero_hopf": zh,
        "bt": bt,
    }

