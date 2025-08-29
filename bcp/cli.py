from __future__ import annotations

import uvicorn


def run_api():  # pragma: no cover - simple runner
    uvicorn.run("bcp.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run_api()

