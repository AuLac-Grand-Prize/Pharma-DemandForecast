from __future__ import annotations

import time
from datetime import UTC, date, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, model_validator

from demandforecast.services.forecast_service import (
    DeterministicForecastBackend,
    FakeForecastCache,
    ForecastBackend,
    ForecastCache,
    cache_key,
)

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

    @model_validator(mode="after")
    def _quantiles_monotone(self) -> DailyPoint:
        # Contract guard: a backend that emits out-of-order quantiles must not
        # be able to silently ship malformed points.
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError(
                f"quantiles must satisfy p10 <= p50 <= p90, got "
                f"p10={self.p10}, p50={self.p50}, p90={self.p90}"
            )
        return self


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


# ---------------------------------------------------------------------------
# Injectable dependencies (overridable via app.dependency_overrides in tests).
# A module-level cache instance gives the running app a process-wide cache
# without pulling in Redis; tests swap in their own FakeForecastCache.
# ---------------------------------------------------------------------------

_default_backend = DeterministicForecastBackend()
_default_cache = FakeForecastCache()


def get_forecast_backend() -> ForecastBackend:
    return _default_backend


def get_forecast_cache() -> ForecastCache:
    return _default_cache


def _to_sku_forecasts(backend_forecasts: list) -> list[SkuForecast]:
    return [
        SkuForecast(
            sku_id=f.sku_id,
            drug_name=f.drug_name,
            daily_forecast=[
                DailyPoint(date=p.date, p10=p.p10, p50=p.p50, p90=p.p90)
                for p in f.points
            ],
            trend=f.trend,
            seasonality_signal=f.seasonality_signal,
            model_weights=f.model_weights,
        )
        for f in backend_forecasts
    ]


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(
    req: ForecastRequest,
    backend: ForecastBackend = Depends(get_forecast_backend),
    cache: ForecastCache = Depends(get_forecast_cache),
) -> ForecastResponse:
    started = time.perf_counter()

    key = cache_key(
        pharmacy_id=req.pharmacy_id,
        sku_ids=req.sku_ids,
        horizon_days=req.horizon_days,
        include_covariates=req.include_covariates,
    )

    cached = cache.get(key)
    if cached is not None:
        # Cache hit: rebuild the validated response from stored data WITHOUT
        # re-invoking the backend.
        forecasts = [SkuForecast.model_validate(item) for item in cached]
    else:
        backend_forecasts = backend.predict(
            pharmacy_id=req.pharmacy_id,
            sku_ids=req.sku_ids,
            horizon_days=req.horizon_days,
            include_covariates=req.include_covariates,
        )
        forecasts = _to_sku_forecasts(backend_forecasts)
        cache.set(key, [f.model_dump(mode="json") for f in forecasts])

    latency_ms = int((time.perf_counter() - started) * 1000)
    return ForecastResponse(
        forecasts=forecasts,
        generated_at=datetime.now(UTC).isoformat(),
        latency_ms=max(latency_ms, 0),
    )
