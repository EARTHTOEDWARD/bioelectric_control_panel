import argparse
import numpy as np
import pandas as pd
from b_sacp.sweep.parameter_sweep import sweep_grid_a21_a31


def main():
    ap = argparse.ArgumentParser(description="Run (a21, a31) grid sweep for IB-3D with heuristics")
    ap.add_argument("--a21_min", type=float, default=1.0)
    ap.add_argument("--a21_max", type=float, default=2.2)
    ap.add_argument("--a21_steps", type=int, default=25)
    ap.add_argument("--a31_min", type=float, default=1.6)
    ap.add_argument("--a31_max", type=float, default=3.2)
    ap.add_argument("--a31_steps", type=int, default=25)
    ap.add_argument("--out", type=str, default="grid_sweep.csv")
    args = ap.parse_args()

    a21_grid = np.linspace(args.a21_min, args.a21_max, args.a21_steps)
    a31_grid = np.linspace(args.a31_min, args.a31_max, args.a31_steps)
    df = sweep_grid_a21_a31(a21_grid, a31_grid)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(df)} rows")


if __name__ == "__main__":
    main()

