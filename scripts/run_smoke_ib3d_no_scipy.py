#!/usr/bin/env python3
"""
Run a SciPy-free smoke test of the IB3D Jacobian/eigen heuristics.
Prints a JSON-like dictionary of results.
"""
import argparse
import json
import numpy as np
from b_sacp.analysis.smoke_ib3d_no_scipy import run_smoke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a21", type=float, default=1.6)
    ap.add_argument("--a31", type=float, default=2.4)
    args = ap.parse_args()
    res = run_smoke(a21=args.a21, a31=args.a31)
    # Convert arrays to lists for JSON
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    print(json.dumps(res, default=conv, indent=2))


if __name__ == "__main__":
    main()

