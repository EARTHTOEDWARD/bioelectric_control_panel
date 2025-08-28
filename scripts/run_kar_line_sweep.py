import argparse, pandas as pd
from b_sacp.ui.control_panel import get_ib3d_line_sweep

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a21", type=float, default=1.6)
    ap.add_argument("--a31_min", type=float, default=3.0)
    ap.add_argument("--a31_max", type=float, default=1.86)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", type=str, default="kar_line_sweep.csv")
    args = ap.parse_args()
    df = get_ib3d_line_sweep(a21=args.a21, a31_min=args.a31_min, a31_max=args.a31_max, steps=args.steps)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

