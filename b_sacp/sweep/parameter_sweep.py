import numpy as np, pandas as pd
from typing import Iterable, Tuple
from ..models.itik_banks import IB3DParams, ib3d_rhs, integrate_ib3d
from ..analysis.invariant_measure import invariant_on_section
from ..analysis.lyapunov import wolf_lle, make_ib3d_flow
from ..analysis.jacobian import analyze_equilibrium_near_traj

def sweep_a31_line(a21=1.6, a31_min=3.0, a31_max=1.86, steps=40,
                   x0=(0.31,0.22,0.18), dt=0.05, add_equilibrium_flags: bool = True):
    """Karatetskaia 'route to Shilnikov' line sweep along a31 with fixed a21."""
    recs = []
    for a31 in np.linspace(a31_min, a31_max, steps):
        p = IB3DParams(a21=a21, a31=a31)
        # short warm-up then sampling
        out = integrate_ib3d(p, x0=x0, t_span=(0, 2.0e5), dt=0.05, transient=5.0e4)
        X = out["X"]
        # LLE (discrete flow wrapper)
        flow = make_ib3d_flow(ib3d_rhs, p)
        lle = wolf_lle(flow, np.array(x0, dtype=float), dt=dt, steps=100000, p=p, renorm=200)
        # invariant measure on Poincaré x2=c plane (c near median)
        c = float(np.median(X[:,1]))
        H, (xe, ye) = invariant_on_section(X, i=1, c=c, dims=(0,2), bins=128)
        mass_peak = float(H.max())
        row = {"a21":a21, "a31":a31, "LLE":lle, "sec_mass_peak":mass_peak, "x2_sec":c}
        if add_equilibrium_flags:
            eq = analyze_equilibrium_near_traj(X, p)
            if eq.get("found"):
                dmin = float(np.min(np.linalg.norm(X - eq["equilibrium"], axis=1)))
                row.update({
                    "eq_found": True,
                    "eq_x1": float(eq["equilibrium"][0]),
                    "eq_x2": float(eq["equilibrium"][1]),
                    "eq_x3": float(eq["equilibrium"][2]),
                    "eig_class": eq["class"],
                    "zero_hopf": bool(eq["zero_hopf"]),
                    "bt": bool(eq["bt"]),
                    "dmin_eq": dmin,
                })
            else:
                row.update({"eq_found": False})
        recs.append(row)
    return pd.DataFrame.from_records(recs)

def sweep_grid_a21_a31(a21_grid: Iterable[float], a31_grid: Iterable[float],
                       x0=(0.31,0.22,0.18), dt=0.05, add_equilibrium_flags: bool = True):
    """2D grid for coarse Lyapunov and section-peak maps (homO1 proxies)."""
    rows = []
    for a21 in a21_grid:
        for a31 in a31_grid:
            p = IB3DParams(a21=a21, a31=a31)
            out = integrate_ib3d(p, x0=x0, t_span=(0, 1.2e5), dt=0.05, transient=3.0e4)
            X = out["X"]
            flow = make_ib3d_flow(ib3d_rhs, p)
            lle = wolf_lle(flow, np.array(x0, dtype=float), dt=dt, steps=60000, p=p, renorm=200)
            c = float(np.median(X[:,1]))
            H, _ = invariant_on_section(X, i=1, c=c, dims=(0,2), bins=64)
            row = {"a21":a21, "a31":a31, "LLE":lle, "sec_mass_peak":float(H.max())}
            if add_equilibrium_flags:
                eq = analyze_equilibrium_near_traj(X, p)
                if eq.get("found"):
                    dmin = float(np.min(np.linalg.norm(X - eq["equilibrium"], axis=1)))
                    row.update({
                        "eq_found": True,
                        "eig_class": eq["class"],
                        "zero_hopf": bool(eq["zero_hopf"]),
                        "bt": bool(eq["bt"]),
                        "dmin_eq": dmin,
                    })
                else:
                    row.update({"eq_found": False})
            rows.append(row)
    return pd.DataFrame(rows)
