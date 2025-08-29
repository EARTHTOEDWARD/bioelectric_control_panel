BCP WebSocket Bridge (Sprint 1.2)

This tiny bridge broadcasts BCP NDJSON events over WebSocket and includes a minimal static viewer.

Quick Start

- Install websockets extra:
  - `pip install -e ".[ws]"`
- Generate or point to NDJSON events (e.g., from the cardiac shim):
  - `bcp-shim-cardiac --fs 800 --window 5 --windows 18 --out_dir outputs`
- Run the bridge in replay mode and serve the demo page:
  - `bcp-ws-bridge --events outputs/events.ndjson --mode replay --static`
  - WebSocket: `ws://127.0.0.1:8765/ws`
  - Demo UI: `http://127.0.0.1:8766/client.html`

Modes

- `replay`: reads the file from the start and emits events in time order, honoring `t_window` spacing (clamped for snappy demos). Use `--speed` to accelerate.
- `tail`: follows the file for new lines (like `tail -f`).

Contract

- Expects newline-delimited JSON where each line is one event (`metrics_update`, `state_update`, `guardrail`) as emitted by the Sprint 1.1 shim.
- The minimal viewer renders λ₁/D₂ (with CI), the current state badge, a quality/guardrail chip, and sparklines.

Notes

- Replace `bcp/tools/public/client.html` with your production UI; keep the WebSocket path `/ws` and the event schema.
- This bridge is intentionally minimal and backend-agnostic; any NDJSON producer with the same schema will work.

