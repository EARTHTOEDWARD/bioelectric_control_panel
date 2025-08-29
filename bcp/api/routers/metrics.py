from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.metrics import summary_metrics
from ..schemas import MetricsRequest, MetricsResponse


router = APIRouter(prefix="/v1/metrics", tags=["metrics"])


@router.post("/summary", response_model=MetricsResponse)
def summary(req: MetricsRequest):
    try:
        return summary_metrics(req.signal)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

