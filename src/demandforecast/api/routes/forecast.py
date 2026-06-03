from datetime import date

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class ForecastRequest(BaseModel):
    pharmacy_id: str
    sku_ids: list[str]
    horizon_days: int = Field(default=30, ge=1, le=365)
    include_covariates: bool = True


class DailyPoint(BaseModel):
    date: date
    p10: float
    p50: float
    p90: float


class SkuForecast(BaseModel):
    sku_id: str
    drug_name: str
    daily_forecast: list[DailyPoint]
    trend: str
    seasonality_signal: str | None = None
    model_weights: dict[str, float]


class ForecastResponse(BaseModel):
    forecasts: list[SkuForecast]
    generated_at: str
    latency_ms: int


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest) -> ForecastResponse:
    # TODO: load ensemble + run inference
    return ForecastResponse(forecasts=[], generated_at="", latency_ms=0)
