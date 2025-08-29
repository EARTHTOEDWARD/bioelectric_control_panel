# Bioelectric Control Panel (BCP)

A standalone service and SDK for bioelectric data ingestion, validation, and analysis (EEG, vagus ENG, optical mapping), designed for eventual clinical workflows and PHI isolation. Exposes a FastAPI service and a small Python SDK. Intended to integrate with SACP via HTTP while remaining independently deployable.

## Quick Start

- Create and activate a virtual environment (Python 3.10+)
- Install dev deps: `pip install -e ".[dev,test,bio]"`
- Run API: `uvicorn bcp.api.main:app --reload`
- Health check: `GET http://127.0.0.1:8000/health`

### UI (Frontend) — Local Dev

- `cd bcp_frontend_v1 && npm install && npm run dev`
- Open `http://127.0.0.1:5173` and set API/WS endpoints in Settings, or create `bcp_frontend_v1/.env` with:
  - `VITE_API_URL=http://127.0.0.1:8000`
  - `VITE_WS_URL=ws://127.0.0.1:8765/ws`

### UI Served by FastAPI

- Build the frontend: `cd bcp_frontend_v1 && VITE_BASE=/app/ npm run build`
- Start the API; the app auto-mounts UI from `bcp_frontend_v1/dist` (or set `BCP_UI_STATIC_DIR=/path/to/dist`).
- Open `http://127.0.0.1:8000/app/`

### One Container (API + Built UI)

- Build image: `docker build -t bcp .`
- Run: `docker run --rm -p 8000:8000 bcp`
- Open `http://127.0.0.1:8000/app/`
- Note: The WS bridge runs separately (see below). Point the UI to it in Settings.

### Docker Compose (API + WS bridge + demo events)

- Build and run everything: `docker compose up --build`
- Opens:
  - UI: `http://127.0.0.1:8000/app/`
  - WS bridge: `ws://127.0.0.1:8765/ws`
- The compose stack does:
  - Initializes `outputs/events.ndjson`
  - Generates demo events with `bcp-shim-cardiac`
  - Broadcasts events via WS bridge in tail mode
  - Serves the API and the built UI together

## Environment

Configure via env vars or `.env` in repo root (see `.env.example`). Key vars:

- `BCP_MAES_PROJECT_PATH`: path to MAES project root
- `BCP_CONTROL_PANEL_DATA_IMPORT`: incoming JSON dir
- `BCP_CONTROL_PANEL_DATA_EXPORT`: outgoing JSON dir
- `BCP_SHARED_FORMAT_VERSION`: shared format version string

## Endpoints (v1)

- `GET /health`: Service status
- `POST /v1/ingest/maes`: Import `.npz` from MAES; returns standardized JSON
- `POST /v1/attractor/embed`: Delay-embedding for a signal
- `POST /v1/metrics/summary`: Basic metrics over a signal
- `POST /v1/chaos/window`: Chaos metrics for a window (λ1, D2, MSE‑AUC, RQA)
- `POST /v1/validate/nwb`: Validate NWB (skips if deps unavailable)
- `POST /v1/validate/zarr`: Validate OME-Zarr (skips if deps unavailable)

### FastAPI example — Chaos metrics (λ1/D2/MSE/RQA)

Compute chaos/complexity metrics for a single window using per‑modality defaults (cardiac/vagus/EEG) or provide overrides.

Curl example (cardiac, 1 kHz, 5 Hz sine, m=3, τ≈50 ms):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chaos/window \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [0.0, 0.0314, 0.0628, 0.0941, 0.1253, 0.1564],
    "sample_rate_hz": 1000.0,
    "modality": "cardiac",
    "embed_dim": 3,
    "delay_ms": 50
  }'
```

Response shape:

```json
{
  "lambda1": 0.02,
  "predictability_horizon_s": 50.0,
  "D2": 1.1,
  "mse_auc_1_20": 4.4,
  "rqa_det": 0.93,
  "rqa_Lmax": 120,
  "rqa_trap": 3.2,
  "quality": "ok",
  "notes": "",
  "m": 3,
  "tau_samples": 50
}
```

Python (requests):

```python
import requests

payload = {
    "signal": [0.0, 0.0314, 0.0628, 0.0941, 0.1253, 0.1564],
    "sample_rate_hz": 1000.0,
    "modality": "cardiac",
    "embed_dim": 3,
    "delay_ms": 50,
}
r = requests.post("http://127.0.0.1:8000/v1/chaos/window", json=payload)
print(r.json())
```

### CLI — Streaming windows + Attractor snapshot

Compute metrics over rolling windows from a 1D signal (CSV/JSON/NPY) or use built‑in demos. Writes a CSV and a quick Attractor View PNG.

```bash
# Install viz extra for plotting once in your venv
pip install -e ".[viz]"

# Demo (sine, Lorenz x(t), 1/f noise)
bcp-stream --demo --fs 1000 --window 5 --step 1 \
  --modality cardiac \
  --out_csv outputs/stream_metrics.csv \
  --out_png outputs/attractor.png

# From a CSV (first column), sample rate 500 Hz
bcp-stream --input path/to/signal.csv --fs 500 --window 10 --step 2 \
  --modality eeg --out_csv outputs/eeg_metrics.csv --out_png outputs/eeg_attractor.png
```

CSV rows contain `t0` (window start seconds) plus the chaos metrics fields (`lambda1`, `predictability_horizon_s`, `D2`, `mse_auc_1_20`, `rqa_*`, `quality`, `m`, `tau_samples`).

### Shim — Cardiac NDJSON with Confidence Intervals & Guardrails (Sprint 1.1)

Simulate a cardiac stream transitioning NSR → VF. Emits NDJSON events (metrics_update, state_update, guardrail), plus a trend CSV for quick plots.

```bash
# Run the shim (outputs to ./outputs by default)
bcp-shim-cardiac --fs 800 --window 5 --windows 18 --out_dir outputs

# Inspect the first few lines
head -n 6 outputs/events.ndjson
```

Each `metrics_update` includes:
- embedding: `m`, `tau`
- stationarity: mean drift ratio, variance ratio, `acf1_abs`, and notes
- metrics with CI: `lambda1`, `lambda1_ci`, `D2`, `D2_ci`, `rqa` stability
- `quality_ok` + notes; actions should be gated when false

Guardrail policy (cardiac):
- Block actions when analysis quality is insufficient (stationarity failed or `λ1` CI overlaps 0).
- Allow shock only when `λ1≥0.3` and `D2≥3.0`; clamp shock energy and pacing params to device limits.

### Shim — Vagus ENG with CI, Guardrails, Entrainment Tags, and Viewer (Sprint 1.3)

Simulate vagus ENG windows with respiratory phase and a mid‑run stimulation episode. Emits NDJSON events enriched with entrainment/quality tags and simple vagus‑specific guardrails. Includes a dedicated viewer.

```bash
# Generate a demo session (events + trend CSV)
bcp-shim-vagus --fs 1000 --window 2.5 --windows 12 --out_dir outputs

# Broadcast over WebSocket with static demo clients
pip install -e ".[ws]"
bcp-ws-bridge --events outputs/vagus_events.ndjson --mode replay --static
# Open: http://127.0.0.1:8766/client_vagus.html
```

Event schema additions (vagus):
- `quality_tags.resp_phase` and `quality_tags.stim_on` to surface context
- `entrainment_tags` with `determinism_delta`, `entrainment_observed`, `lambda1_drop`, `d2_drop`
- Guardrail event includes `proposed`, `allowed`, `messages`, and optional `suggestion` with params

Guardrails (vagus):
- Only propose `stim_on` from `vagal_chaotic_rest` and when analysis `quality_ok`
- Only propose `stim_off` from `vagal_periodic_entrained`
- Duty‑cycle limiter (demo: 4 windows); always block on poor analysis quality

### Shim — Vagus from NWB/HDF5 (Sprint 1.4)

Read a real NWB/HDF5 ElectricalSeries via `h5py` (no `pynwb` required) and emit the same NDJSON schema for the Vagus viewer and guardrails.

```bash
# Emit NDJSON from NWB
bcp-shim-vagus-nwb --nwb /path/to/session.nwb --channel 0 \
  --window 2.0 --step 2.0 --out_dir outputs

# Stream to the UI
pip install -e ".[ws]"
bcp-ws-bridge --events outputs/vagus_events_from_nwb.ndjson --mode replay --static
# Open: http://127.0.0.1:8766/client_vagus.html
```

Discovery: finds an ElectricalSeries under `/acquisition/**/data`, uses `rate` attribute or infers fs from `timestamps`. Supports shapes `(T,)`, `(T,C)`, `(C,T)`.
Preprocessing: moving-average high‑pass + first difference + z‑score.
Metrics/tags/guardrails: same as Sprint 1.3 Vagus shim.

## SDK

```python
from bcp.sdk import BCPClient
client = BCPClient(base_url="http://127.0.0.1:8000")
status = client.health()
```

## Tests

- Integration tests require optional deps and datasets: `pytest -m "integration and not slow"`
- Unit tests run by default: `pytest -m "not integration"`

## Migration from Monorepo

See `docs/MIGRATION.md` for history-preserving options (`git subtree` or `git filter-repo`).

## Security

- No datasets in git; see `.gitignore`
- Secrets via env only; `.env` is git-ignored
- Add auth (JWT/OIDC) and audit logs before handling PHI
