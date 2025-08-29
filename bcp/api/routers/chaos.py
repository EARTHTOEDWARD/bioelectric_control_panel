from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.chaos import compute_window_metrics
from ..schemas import ChaosRequest, ChaosResponse


router = APIRouter(prefix="/v1/chaos", tags=["chaos"])


@router.post("/window", response_model=ChaosResponse)
def window_metrics(req: ChaosRequest):
    try:
        override_m = req.embed_dim
        override_tau = None
        if req.delay_ms is not None:
            if req.sample_rate_hz <= 0:
                raise ValueError("sample_rate_hz must be > 0 when delay_ms provided")
            override_tau = max(int((req.delay_ms / 1000.0) * req.sample_rate_hz), 1)
        out = compute_window_metrics(
            signal=req.signal,
            fs=req.sample_rate_hz,
            modality=req.modality,
            override_m=override_m,
            override_tau_samples=override_tau,
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

