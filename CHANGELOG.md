Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.1.2] - 2025-08-29
### Fixed
- chaos: compute_window_metrics used `xc` before assignment when auto-selecting `tau` for the `generic` modality; now center/scales prior to `tau` selection.
- chaos: tuned `lyap_rosenstein` fitting window (`max_t` 0.2) to stabilize λ1 on periodic signals.
- attractor: delay embedding now guards short signals and corrects indexing to ensure at least one vector when possible.

## [0.1.1] - 2025-08-29
### Added
- Advanced chaos metrics with confidence intervals:
  - Largest Lyapunov exponent (Rosenstein) with bootstrap 95% CI.
  - Correlation dimension (Grassberger–Procaccia) with auto scaling-region selection + bootstrap CI.
- RQA stability summary across recurrence rates (mean, std, rel_std).
- Stationarity checks and quality gating (mean drift, variance ratio, diff-ACF1; gate when λ1 CI overlaps 0).
- Cardiac guardrails for actions:
  - Shock gating: allow only when λ1≥0.3 and D2≥3.0; clamp shock energy.
  - Pacing parameter clamping to device-safe ranges.
- Streaming shim CLI `bcp-shim-cardiac` emitting NDJSON events (metrics/state/guardrail) and a trend CSV.
- Streaming windows CLI retained (`bcp-stream`) and documented.

### Changed
- Improved generic τ selection behavior within window metric computation.
- README updated with shim usage and guardrail policy.

### Tests
- Unit tests for quality gate and guardrails.
- Basic chaos metrics sanity checks.

## [0.1.0] - 2025-08-28
### Added
- Initial project structure with FastAPI skeleton, core attractor embedding, and summary metrics.
- Basic chaos metrics window endpoint (`/v1/chaos/window`).
- SDK client and simple unit tests.

[0.1.1]: https://github.com/EARTHTOEDWARD/bioelectric_control_panel/compare/v0.1.0...v0.1.1
