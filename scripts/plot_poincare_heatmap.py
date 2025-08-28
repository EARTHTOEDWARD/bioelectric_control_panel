import argparse
import numpy as np
import matplotlib.pyplot as plt
from b_sacp.models.itik_banks import IB3DParams, integrate_ib3d
from b_sacp.analysis.invariant_measure import invariant_on_section


def main():
    ap = argparse.ArgumentParser(description="Plot Poincaré invariant-measure heatmap for IB-3D")
    ap.add_argument("--a21", type=float, default=1.6)
    ap.add_argument("--a31", type=float, default=2.4)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--t_end", type=float, default=2.0e5)
    ap.add_argument("--t_transient", type=float, default=5.0e4)
    ap.add_argument("--out", type=str, default="poincare_heatmap.png")
    args = ap.parse_args()

    p = IB3DParams(a21=args.a21, a31=args.a31)
    out = integrate_ib3d(p, t_span=(0, args.t_end), dt=args.dt, transient=args.t_transient)
    X = out["X"]
    c = float(np.median(X[:,1]))
    H, (xe, ye) = invariant_on_section(X, i=1, c=c, dims=(0,2), bins=args.bins)

    fig, ax = plt.subplots(figsize=(6,5))
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    im = ax.imshow(H.T, origin='lower', aspect='auto', extent=extent, cmap='magma')
    ax.set_xlabel('x1')
    ax.set_ylabel('x3')
    ax.set_title(f'Poincaré IM: a21={args.a21}, a31={args.a31}, x2={c:.3f}')
    fig.colorbar(im, ax=ax, label='density')
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

