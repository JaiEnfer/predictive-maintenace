from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    temperature_c: float = Field(..., ge=-40, le=200)
    vibration_mm_s: float = Field(..., ge=0, le=200)
    pressure_bar: float = Field(..., ge=0, le=500)
    runtime_hours: float = Field(..., ge=0, le=100_000)

class PredictResponse(BaseModel):
    maintenance_required: bool
    probability: float
    model_version: str
