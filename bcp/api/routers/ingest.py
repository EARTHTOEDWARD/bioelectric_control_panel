from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...io.maes_interface import MaesInterface
from ..schemas import ImportMaesRequest, StandardizedResponse


router = APIRouter(prefix="/v1/ingest", tags=["ingest"])


@router.post("/maes", response_model=StandardizedResponse)
def ingest_maes(req: ImportMaesRequest):
    iface = MaesInterface()
    data = iface.import_maes_data(filename=req.filename)
    if data is None:
        raise HTTPException(status_code=404, detail="MAES file not found")
    return data

