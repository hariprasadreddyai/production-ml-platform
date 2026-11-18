"""Pydantic schemas for the serving API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")
    request_id: str | None = Field(
        default=None, description="Optional id for deterministic A/B routing"
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    canary_enabled: bool


class ModelInfoResponse(BaseModel):
    stable_version: str
    canary_version: str | None
    labels: list[str]
    ab_testing_enabled: bool
    stable_weight: float
    canary_weight: float


class ReloadRequest(BaseModel):
    slot: str = Field(..., pattern="^(stable|canary)$")
    model_path: str = Field(
        ..., description="Path to a saved model directory, or 'stub'"
    )
    version: str = Field(default="v1.0.0")
