# Itik–Banks 3D cancer model (dimensionless form used in Karatetskaia et al.)
# References: Regular and Chaotic Dynamics (2024): Eqs. (1.1), params (1.2).  # see paper
from dataclasses import dataclass
import numpy as np
from typing import Callable, Dict, Tuple
from scipy.integrate import solve_ivp

@dataclass
class IB3DParams:
    a12: float = 1.0
    a13: float = 2.5
    a21: float = 1.6
    a31: float = 2.4
    r2: float = 0.6
    r3: float = 6.0
    d3: float = 0.5
    k3: float = 1.0

def ib3d_rhs(t, x, p: IB3DParams):
    x1, x2, x3 = x
    dx1 = x1*(1.0 - x1) - p.a12*x1*x2 - p.a13*x1*x3
    dx2 = p.r2*x2*(1.0 - x2) - p.a21*x1*x2
    dx3 = (p.r3*x1*x3)/(x1 + p.k3) - p.a31*x1*x3 - p.d3*x3
    return np.array([dx1, dx2, dx3])

def integrate_ib3d(p: IB3DParams,
                   x0=(0.3, 0.3, 0.3),
                   t_span=(0.0, 2.0e5),
                   dt=0.1,
                   transient=5.0e4,
                   rtol=1e-9, atol=1e-12) -> Dict[str, np.ndarray]:
    """Long integration suitable for Lyapunov & invariant-measure sampling."""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lambda t, x: ib3d_rhs(t, x, p),
                    t_span, x0, t_eval=t_eval, rtol=rtol, atol=atol, method="RK45")
    t = sol.t
    X = sol.y.T
    keep = t >= transient
    return {"t": t[keep], "X": X[keep]}

