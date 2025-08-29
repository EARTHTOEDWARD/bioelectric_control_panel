# BCP Frontend (Vite + React + TypeScript)

A production-grade scaffold for the Bioelectric Control Panel UI. 
It plugs into your existing FastAPI + WS bridge and renders chaos metrics (λ₁, D₂, %DET) with CIs, state/quality/guardrail chips, and provides NWB ingest, NDJSON replay, and an on‑the‑fly Attractor view.

## Quickstart

```bash
cd bcp_frontend_v1
npm install
npm run dev
# open http://127.0.0.1:5173
```

Set endpoints via env or UI Settings:

```ini
# .env
VITE_API_URL=http://127.0.0.1:8000
VITE_WS_URL=ws://127.0.0.1:8765/ws
```

### Build & Serve via FastAPI

```bash
# Ensure asset paths are rooted at /app when served by FastAPI
VITE_BASE=/app/ npm run build
```

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
app = FastAPI()
app.mount("/app", StaticFiles(directory="bcp_frontend_v1/dist", html=True), name="bcp-ui")
# now open http://127.0.0.1:8000/app/
```

## Pages
- Live — real‑time monitor (WS), state badge, quality/guardrail chips, λ₁/D₂/%DET with CIs and sparklines.
- Ingest — upload NWB to your FastAPI (POST /v1/ingest/nwb-upload). Sends Authorization header if token is set in Settings.
- Explorer — replay NDJSON event logs (same store as live). Includes Attractor (Takens embedding) from a pasted raw channel (Task 2).
- Guardrails — textual explanation of current gating & suggestion.
- Settings — API/WS endpoints and a dev auth token stub (Task 4).

## Event Schema
Matches Sprint 1.3/1.4: metrics_update, state_update, guardrail with vagus‑specific quality/entrainment tags. Types and Zod schemas live in `src/api/types.ts`. Socket parsing uses Zod (Task 1).

## Stimulus Overlays (Task 3)
Sparklines shade spans when `quality_tags.stim_on` is true in the live history.

## Notes
- Minimal CSS for portability; swap in Tailwind/Chakra if desired.
- Charts use lightweight canvas sparklines; 3D attractor via three.js.
- Strong typing via TS & Zod; add more runtime validation as needed.
