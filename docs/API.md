# API Overview

Base: `/`

- `GET /health` → `{ status, version }`
- `POST /v1/ingest/maes` → `{ metadata, parameters, data }`
  - body: `{ "filename": "xenopus_preprocessed.npz" }`
- `POST /v1/attractor/embed` → `{ x, y, z }`
  - body: `{ signal: number[], delay_ms: number, sample_rate_hz: number, embed_dim: 3 }`
- `POST /v1/metrics/summary` → `{ mean, std, min, max, rms }`
- `POST /v1/validate/nwb` → `{ valid, details? }`
  - body: `{ path: string }`
- `POST /v1/validate/zarr` → `{ valid, details? }`
  - body: `{ path: string }`

