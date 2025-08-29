#!/usr/bin/env python3
"""
Run a SciPy-free smoke test of the IB3D Jacobian/eigen heuristics.
Prints a JSON-like dictionary of results.
"""
import argparse
import json
import numpy as np
import sys
from pathlib import Path
try:
    from b_sacp.analysis.smoke_ib3d_no_scipy import run_smoke
except ModuleNotFoundError:
    # Add repo root to sys.path for local execution
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from b_sacp.analysis.smoke_ib3d_no_scipy import run_smoke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a21", type=float, default=1.6)
    ap.add_argument("--a31", type=float, default=2.4)
    args = ap.parse_args()
    res = run_smoke(a21=args.a21, a31=args.a31)
    # Sanitize to JSON-friendly types
    out = dict(res)
    # Top-level flags and numbers to native types
    if "eq_found" in out:
        out["eq_found"] = bool(out["eq_found"])
    if "zero_hopf" in out:
        out["zero_hopf"] = bool(out["zero_hopf"])
    if "bt" in out:
        out["bt"] = bool(out["bt"])
    if out.get("eq_found") and "eq" in out:
        out["eq"] = [float(v) for v in np.asarray(out["eq"]).ravel()]
    if "jacobian" in out and isinstance(out["jacobian"], np.ndarray):
        out["jacobian"] = [[float(v) for v in row] for row in out["jacobian"]]
    if "eigs" in out:
        eigs = np.asarray(out["eigs"]).ravel()
        out["eigs"] = [{"re": float(np.real(l)), "im": float(np.imag(l))} for l in eigs]
    if "classification" in out and isinstance(out["classification"], dict):
        c = out["classification"]
        if "pos" in c: c["pos"] = int(c["pos"])
        if "neg" in c: c["neg"] = int(c["neg"])
        if "near_zero" in c: c["near_zero"] = int(c["near_zero"])
        if "has_complex_pair" in c: c["has_complex_pair"] = bool(c["has_complex_pair"])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
