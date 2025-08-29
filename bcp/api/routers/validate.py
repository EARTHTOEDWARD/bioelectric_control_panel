from __future__ import annotations

from fastapi import APIRouter

from ..schemas import ValidatePathRequest, ValidateResponse


router = APIRouter(prefix="/v1/validate", tags=["validate"])


@router.post("/nwb", response_model=ValidateResponse)
def validate_nwb(req: ValidatePathRequest):
    try:
        import pynwb  # noqa: F401
        from pynwb import NWBHDF5IO
    except Exception:
        return ValidateResponse(valid=False, details="pynwb not available; skipping")

    try:
        with NWBHDF5IO(req.path, "r") as io:  # type: ignore[name-defined]
            _ = io.read()
        return ValidateResponse(valid=True)
    except Exception as e:
        return ValidateResponse(valid=False, details=str(e))


@router.post("/zarr", response_model=ValidateResponse)
def validate_zarr(req: ValidatePathRequest):
    try:
        import zarr  # noqa: F401
        import xarray as xr  # noqa: F401
    except Exception:
        return ValidateResponse(valid=False, details="zarr/xarray not available; skipping")

    try:
        ds = xr.open_zarr(req.path)  # type: ignore[name-defined]
        _ = list(ds.data_vars)
        return ValidateResponse(valid=True)
    except Exception as e:
        return ValidateResponse(valid=False, details=str(e))

