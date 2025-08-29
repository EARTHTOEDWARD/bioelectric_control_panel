from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import __version__
from .schemas import HealthResponse
from .routers import ingest, metrics, validate, attractor
from .routers import chaos


def create_app() -> FastAPI:
    app = FastAPI(title="Bioelectric Control Panel API", version=__version__)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:  # pragma: no cover - trivial
        return HealthResponse(status="ok", version=__version__)

    app.include_router(ingest.router)
    app.include_router(metrics.router)
    app.include_router(validate.router)
    app.include_router(attractor.router)
    app.include_router(chaos.router)

    return app


app = create_app()
