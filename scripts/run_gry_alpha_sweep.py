import argparse, numpy as np, pandas as pd
from b_sacp.models.grytsay_cell import GryParams, integrate_gry
from b_sacp.analysis.invariant_measure import invariant_histogram

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha_min", type=float, default=0.03217)
    ap.add_argument("--alpha_max", type=float, default=0.03255)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--out", type=str, default="gry_alpha_sweep.csv")
    args = ap.parse_args()
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.steps)
    recs = []
    for a in alphas:
        p = GryParams(alpha=a)
        out = integrate_gry(p, t_span=(0,2.0e6), dt=10.0, transient=1.0e6)
        X = out["X"]
        # invariant measure roughness proxy: peak density in (G,E1,B) 3D histogram
        H, _ = invariant_histogram(X[:,[0,3,2]], bins=64)
        recs.append({"alpha": a, "hist_peak": float(H.max())})
    pd.DataFrame(recs).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

