Bioelectric Control Panel — Real-Data Analysis Toolkit

Purpose

- Compute robust summaries and features from real, per-cell time series in a unified long-format CSV.
- Fix Δ‑coherence pre→post by recomputing magnitude‑squared coherence with Welch in a defined band.
- Provide robust β (1/f slope) estimators and bootstrapped CIs, effect sizes, and FDR.
- Add features: Higuchi/Katz fractal dimension, RQA metrics, windowed β/coherence, and directed metrics (PSI, lagged xcorr).
- Emit a QC sidecar recording band limits, tapers, detrending, and coherence band used.

Input schema (long format CSV)

- Required columns:
  - `source_id`: string identifier for file/record
  - `channel`: one of `ATP/ADP`, `NAD(P)H`, `dpsi` (Δψm) or other
  - `condition`: e.g., `baseline`, `FCCP`, `oligomycin`, `rotenone`
  - `cell_id`: per‑cell or ROI identifier
  - `t_s`: time in seconds (float)
  - `value`: measurement value (float)
  - `fs_Hz`: sampling rate in Hz (float) or empty; if empty, we infer from `t_s`

- Optional columns (used if present):
  - `drug`: name of perturbation (duplicate of `condition` if you prefer)
  - `unit`, `doi`, `file_origin`, `imaging_modality`

Pre/Post segmentation

- Provide event times via a JSON mapping to seconds: `{ "FCCP": 1800, "oligomycin": 900, ... }`
- Pass this file with `--event-times events.json`. If omitted, pre/post metrics are skipped.

CLI

- Run: `python -m dataset_pipeline build --input data/long.csv --out out/ --config dataset_pipeline/config_default.json --event-times events.json`

Outputs (written under `--out`)

- `qc_sidecar.json`: exact parameters (bands, Welch, detrend) used.
- `spectral_beta_per_cell.csv`: per‑cell β estimates with band and fit quality.
- `spectral_beta_group.csv`: group medians by channel×condition, Δ vs baseline, bootstrap CIs, MWU, Cliff’s δ, FDR.
- `coherence_pairs.csv`: per‑pair median coherence (per cell, channel pairs).
- `coherence_group.csv`: group medians by pair×condition and Δ vs baseline with stats.
- `delta_coherence_pre_post.csv`: pre/post coherence recomputed correctly with Welch; includes Δ.
- `step_metrics.csv`: amplitude and τ63 per cell/condition (if `--event-times` provided).
- `windowed_metrics.csv`: windowed β and coherence summaries per series.
- `features_fractal_rqa.csv`: Higuchi/Katz FD and RQA (DET, LAM, TT) per trace.
- `directed_metrics.csv`: phase slope index and lagged cross‑correlation per pair.

Configuration

- See `config_default.json`. Adjust spectral band, coherence band, Welch segmentation, detrend, and window lengths.

Assumptions & notes

- Coherence is magnitude‑squared coherence computed from Welch PSD/CSD with Hann window and stated overlap.
- β is the absolute slope of log10(PSD) vs log10(f) within the stated band, estimated via robust IRLS (Huber) with guardrails.
- Bootstrapped CIs use percentile method by default; BCa is implemented but can be slower.
- RQA uses embedding (m=2, τ=1 by default) and distance threshold at a percentile of pairwise distances (configurable).
- PSI uses coherency phase slope across the band; positive PSI suggests `x → y` direction.

Minimal example

- Prepare `long.csv` with rows for at least one channel under `baseline` with `t_s` and `value` per cell.
- Optional: add perturbation conditions and provide `events.json`.
- Run the CLI as above; inspect `out/` CSVs for summaries.
