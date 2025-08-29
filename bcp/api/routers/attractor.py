from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...core.attractor import delay_embedding
from ..schemas import EmbedRequest, EmbedResponse


router = APIRouter(prefix="/v1/attractor", tags=["attractor"])


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    try:
        result = delay_embedding(
            signal=req.signal,
            delay_ms=req.delay_ms,
            sample_rate_hz=req.sample_rate_hz,
            embed_dim=req.embed_dim,
        )
        # Coerce into EmbedResponse-compatible fields
        out = {k: v for k, v in result.items() if k in {"x", "y", "z"}}
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

