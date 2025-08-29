# Bioelectric Control Panel (BCP)

A standalone service and SDK for bioelectric data ingestion, validation, and analysis (EEG, vagus ENG, optical mapping), designed for eventual clinical workflows and PHI isolation. Exposes a FastAPI service and a small Python SDK. Intended to integrate with SACP via HTTP while remaining independently deployable.

## Quick Start

- Create and activate a virtual environment (Python 3.10+)
- Install dev deps: `pip install -e ".[dev,test,bio]"`
- Run API: `uvicorn bcp.api.main:app --reload`
- Health check: `GET http://127.0.0.1:8000/health`

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
