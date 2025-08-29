from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ImportMaesRequest(BaseModel):
    filename: str = Field(default="xenopus_preprocessed.npz")


class StandardizedResponse(BaseModel):
    metadata: dict
    parameters: dict
    data: dict


class EmbedRequest(BaseModel):
    signal: List[float]
    delay_ms: int = 20
    sample_rate_hz: float = 1000.0
    embed_dim: int = 3


class EmbedResponse(BaseModel):
    x: List[float]
    y: Optional[List[float]] = None
    z: Optional[List[float]] = None


class MetricsRequest(BaseModel):
    signal: List[float]


class MetricsResponse(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    rms: float


class ChaosRequest(BaseModel):
    signal: List[float]
    sample_rate_hz: float = 1000.0
    modality: str = "generic"
    embed_dim: int | None = None
    delay_ms: int | None = None


class ChaosResponse(BaseModel):
    lambda1: float
    predictability_horizon_s: float | None
    D2: float
    mse_auc_1_20: float
    rqa_det: float
    rqa_Lmax: float
    rqa_trap: float
    quality: str
    notes: str | None = None
    m: int
    tau_samples: int


class ValidatePathRequest(BaseModel):
    path: str


class ValidateResponse(BaseModel):
    valid: bool
    details: Optional[str] = None
