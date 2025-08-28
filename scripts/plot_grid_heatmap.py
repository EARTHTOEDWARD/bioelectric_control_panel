import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pivot_grid(df: pd.DataFrame, value: str):
    piv = df.pivot(index='a21', columns='a31', values=value)
    a21_vals = np.array(piv.index)
    a31_vals = np.array(piv.columns)
    Z = piv.values
    return a21_vals, a31_vals, Z


def main():
    ap = argparse.ArgumentParser(description="Plot grid heatmaps from sweep CSV")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, default="grid_heatmaps.png")
    ap.add_argument("--fields", type=str, nargs='+', default=["LLE", "sec_mass_peak"])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    n = len(args.fields)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), squeeze=False)

    for i, field in enumerate(args.fields):
        ax = axes[0, i]
        a21_vals, a31_vals, Z = pivot_grid(df, field)
        im = ax.imshow(Z, origin='lower', aspect='auto',
                       extent=[a31_vals.min(), a31_vals.max(), a21_vals.min(), a21_vals.max()],
                       cmap='viridis')
        ax.set_xlabel('a31')
        ax.set_ylabel('a21')
        ax.set_title(field)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

