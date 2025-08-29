from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from ..core.stream import streaming_windows
from ..core.chaos import _autocorr_first_zero, _takens_embedding


def _load_signal(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".npy"}:
        return np.load(path)
    if path.suffix.lower() in {".csv"}:
        # read first column of numeric CSV quickly
        data = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    v = float(line.split(",")[0])
                except Exception:
                    # skip header or bad lines
                    continue
                data.append(v)
        if not data:
            raise ValueError("no numeric data in CSV")
        return np.asarray(data, dtype=float)
    # JSON array
    arr = json.loads(path.read_text())
    return np.asarray(arr, dtype=float)


def _save_attractor_png(signal: np.ndarray, fs: float, delay_ms: int | None, embed_dim: int, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        m = max(2, int(embed_dim))
        if delay_ms is None:
            tau = _autocorr_first_zero(signal)
        else:
            tau = max(1, int((delay_ms / 1000.0) * fs))
        Y = _takens_embedding(signal, m=min(3, m), tau=tau)
        fig = plt.figure(figsize=(6, 4))
        if Y.shape[1] >= 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], linewidth=0.6)
        else:
            ax = fig.add_subplot(111)
            ax.plot(Y[:, 0], Y[:, 1], linewidth=0.6)
        ax.set_title("Attractor View")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    except Exception as e:  # pragma: no cover - plotting optional
        print(f"[warn] attractor plot skipped: {e}")


def main():  # pragma: no cover - exercised via CLI
    ap = argparse.ArgumentParser(description="BCP streaming chaos metrics over rolling windows")
    ap.add_argument("--input", type=str, required=False, help="Path to CSV/JSON/NPY signal (1D)")
    ap.add_argument("--fs", type=float, default=1000.0, help="Sample rate Hz")
    ap.add_argument("--modality", type=str, default="generic", choices=["generic", "cardiac", "vagus", "eeg"])
    ap.add_argument("--window", type=float, default=5.0, help="Window length (s)")
    ap.add_argument("--step", type=float, default=1.0, help="Step length (s)")
    ap.add_argument("--embed_dim", type=int, default=3)
    ap.add_argument("--delay_ms", type=int, default=None)
    ap.add_argument("--out_csv", type=str, default="outputs/stream_metrics.csv")
    ap.add_argument("--out_png", type=str, default="outputs/attractor.png")
    ap.add_argument("--demo", action="store_true", help="Run on built-in demo signals and write a small report")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []

    def run_once(name: str, sig: np.ndarray):
        for t0, m in streaming_windows(sig, fs=args.fs, window_s=args.window, step_s=args.step, modality=args.modality):
            rec = {"t0": float(t0), "signal": name, **m}
            records.append(rec)
        # Save a representative attractor from the first window
        _save_attractor_png(sig[: int(args.window * args.fs)], fs=args.fs, delay_ms=args.delay_ms, embed_dim=args.embed_dim, out_path=out_png)

    if args.demo:
        # Sine wave (limit cycle)
        t = np.arange(0, 8.0, 1.0 / args.fs)
        x = np.sin(2 * np.pi * 3.5 * t)
        run_once("sine", x)
        # Lorenz x(t) simple integration
        def lorenz(T=8.0, dt=1.0 / 200.0, sigma=10.0, beta=8 / 3, rho=28.0):
            n = int(T / dt)
            xyz = np.zeros((n, 3))
            x, y, z = 1.0, 1.0, 1.0
            for i in range(n):
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z
                x += dx * dt
                y += dy * dt
                z += dz * dt
                xyz[i] = (x, y, z)
            return xyz[:, 0]
        lx = lorenz(T=8.0, dt=1.0 / args.fs)
        lx = (lx - lx.mean()) / (lx.std() + 1e-12)
        run_once("lorenz_x", lx)
        # 1/f noise
        n = int(8.0 * args.fs)
        freqs = np.fft.rfftfreq(n, d=1.0 / args.fs)
        phases = np.exp(1j * 2 * np.pi * np.random.rand(freqs.size))
        amp = np.ones_like(freqs)
        amp[1:] = 1.0 / np.power(freqs[1:], 0.5)
        x = np.fft.irfft(phases * amp, n=n)
        x = (x - x.mean()) / (x.std() + 1e-12)
        run_once("noise_1_f", x)
    else:
        if not args.input:
            ap.error("--input required unless --demo is set")
        sig = _load_signal(Path(args.input))
        run_once(Path(args.input).stem, sig)

    # Write CSV
    import csv

    if records:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        print(f"wrote {out_csv}")
    if out_png.exists():
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()

