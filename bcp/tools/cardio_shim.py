from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..core.chaos_advanced import (
    lyap_rosenstein_ci_ex,
    gp_dimension_scaling_ci,
    rqa_sweep,
)
from ..core.quality import stationarity_checks, quality_gate, classify_cardiac
from ..core.guardrails import guardrail_action_cardiac


def _takens_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    N = x.size - (m - 1) * tau
    if N <= 0:
        raise ValueError("signal too short for requested embedding")
    Y = np.zeros((N, m), dtype=float)
    for i in range(m):
        Y[:, i] = x[i * tau : i * tau + N]
    return Y


def _gen_sine(fs: float, duration: float, freq: float = 2.0, noise_std: float = 0.02) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / fs)
    x = np.sin(2 * np.pi * freq * t) + np.random.normal(0.0, noise_std, size=t.size)
    return x.astype(float)


def _gen_vf_like(fs: float, duration: float) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / fs)
    freqs = [5.3, 7.7, 12.1]
    x = np.zeros_like(t)
    for f in freqs:
        phase = 2 * np.pi * np.random.rand()
        x += np.sin(2 * np.pi * f * t + phase)
    # phase jitter increases unpredictability
    jitter = np.cumsum(np.random.normal(0, 0.02, size=t.size))
    x += 0.5 * np.sin(2 * np.pi * 8.0 * t + jitter)
    x += np.random.normal(0, 0.3, size=t.size)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    return x.astype(float)


def run_shim(
    out_events: Path, out_csv: Path, fs: float = 800.0, window_s: float = 5.0, n_windows: int = 18
) -> None:
    m = 3
    tau = max(1, int(0.05 * fs))
    rows: List[Dict[str, float | int | str | bool]] = []

    out_events.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_events.open("w") as f:
        for w in range(n_windows):
            # Simulated regime switch: NSR → VF
            if w < 8:
                x = _gen_sine(fs, window_s, freq=2.0, noise_std=0.02)
            else:
                x = _gen_vf_like(fs, window_s)

            # Stationarity
            st = stationarity_checks(x, fs)

            # Metrics with CI
            lam1, (l_lo, l_hi), _, _, seg_len = lyap_rosenstein_ci_ex(x, m, tau, fs=fs, n_boot=100)
            D2, (d_lo, d_hi), _, _, band = gp_dimension_scaling_ci(x, m, tau, n_boot=100)

            # RQA sweep (robust summary)
            Y = _takens_embedding(x, min(3, m), tau)
            rqa = rqa_sweep(Y, rec_rates=[0.02, 0.03, 0.04])

            # Quality gate
            q_ok, q_msgs = quality_gate((l_lo, l_hi), (d_lo, d_hi), st.ok)

            # Explicit quality tags
            l_good = np.isfinite(l_lo) and np.isfinite(l_hi)
            d_good = np.isfinite(d_lo) and np.isfinite(d_hi)
            d_len = 0
            try:
                d_len = int(band[1] - band[0])
            except Exception:
                d_len = 0
            st_flags: list[str] = []
            if st.mean_drift_ratio > 0.1:
                st_flags.append("high_mean_drift")
            if not (0.5 <= st.var_ratio_last_first <= 2.0):
                st_flags.append("variance_flip")
            if np.isfinite(st.acf1_abs) and st.acf1_abs > 0.95:
                st_flags.append("high_diff_acf1")
            qtags = {
                "stationarity_pass": bool(st.ok),
                "stationarity_flags": st_flags,
                "lambda1_ci_overlaps_zero": bool(l_good and (l_lo <= 0.0 <= l_hi)),
                "lambda1_ci_width": float((l_hi - l_lo) if l_good else np.nan),
                "lambda1_seg_len": int(seg_len),
                "d2_ci_width": float((d_hi - d_lo) if d_good else np.nan),
                "d2_ci_wide": bool(((d_hi - d_lo) > 1.0) if d_good else True),
                "d2_scaling_region_len": int(d_len),
                "d2_indeterminate": bool((not d_good) or (d_len < 5)),
                "rqa_det_rel_std": float(rqa["determinism"]["rel_std"]),
                "rqa_unstable_thresholding": bool(rqa["determinism"]["rel_std"] > 0.5),
            }

            # Classify state
            state, conf, reasons = classify_cardiac(lam1, D2)

            # Propose action
            if state == "vf":
                proposed = {"type": "shock", "energy_J": 150.0}
            elif state == "vt_borderline":
                proposed = {"type": "pace", "rate_bpm": 180.0, "amp_mA": 5.0, "pulse_width_ms": 2.0}
            else:
                proposed = {"type": "pace", "rate_bpm": 90.0, "amp_mA": 1.0, "pulse_width_ms": 0.5}

            guard = guardrail_action_cardiac(proposed, {"lambda1": lam1, "D2": D2}, q_ok)

            # Emit events
            ev_metrics = {
                "type": "metrics_update",
                "t_window": float(w * window_s),
                "modality": "cardiac",
                "embedding": {"m": int(m), "tau": int(tau)},
                "stationarity": {
                    "ok": bool(st.ok),
                    "mean_drift_ratio": float(st.mean_drift_ratio),
                    "var_ratio_last_first": float(st.var_ratio_last_first),
                    "acf1_abs": float(st.acf1_abs),
                    "notes": st.notes,
                },
                "metrics": {
                    "lambda1": float(lam1),
                    "lambda1_ci": [float(l_lo), float(l_hi)],
                    "D2": float(D2),
                    "D2_ci": [float(d_lo), float(d_hi)],
                    "rqa": {
                        "determinism_mean": float(rqa["determinism"]["mean"]),
                        "determinism_rel_std": float(rqa["determinism"]["rel_std"]),
                        "Lmax_mean": float(rqa["Lmax"]["mean"]),
                        "trap_mean": float(rqa["trapping_time"]["mean"]),
                    },
                },
                "quality_ok": bool(q_ok),
                "quality_notes": q_msgs,
                "quality_tags": qtags,
            }
            f.write(json.dumps(ev_metrics) + "\n")
            f.write(
                json.dumps(
                    {
                        "type": "state_update",
                        "t_window": float(w * window_s),
                        "state": state,
                        "confidence": float(conf),
                        "reasons": reasons,
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps({"type": "guardrail", "t_window": float(w * window_s), **guard})
                + "\n"
            )

            rows.append(
                {
                    "t_window": float(w * window_s),
                    "lambda1": float(lam1),
                    "lambda1_ci_lo": float(l_lo),
                    "lambda1_ci_hi": float(l_hi),
                    "D2": float(D2),
                    "D2_ci_lo": float(d_lo),
                    "D2_ci_hi": float(d_hi),
                    "det_mean": float(rqa["determinism"]["mean"]),
                    "det_rel_std": float(rqa["determinism"]["rel_std"]),
                    "state": state,
                    "quality_ok": bool(q_ok),
                }
            )

    # Write CSV without pandas
    with out_csv.open("w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Cardiac streaming shim: emits NDJSON events with CI and guardrails")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Output directory for events and trends")
    ap.add_argument("--fs", type=float, default=800.0, help="Sample rate (Hz)")
    ap.add_argument("--window", type=float, default=5.0, help="Window length (s)")
    ap.add_argument("--windows", type=int, default=18, help="Number of windows to simulate")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    events = out_dir / "events.ndjson"
    trend = out_dir / "trend_metrics.csv"
    run_shim(events, trend, fs=float(args.fs), window_s=float(args.window), n_windows=int(args.windows))
    print(f"wrote {events}")
    print(f"wrote {trend}")


if __name__ == "__main__":
    main()
