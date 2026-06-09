"""Unit tests for the injectable forecast backend, cache, and DailyPoint guard."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest
from pydantic import ValidationError

from demandforecast.api.routes.forecast import DailyPoint
from demandforecast.services.forecast_service import (
    BackendDailyPoint,
    DeterministicForecastBackend,
    FakeForecastCache,
    ForecastBackend,
    ForecastCache,
    cache_key,
)

# --- Req 1: deterministic, reproducible, structurally correct ---------------


def test_backend_satisfies_protocol():
    assert isinstance(DeterministicForecastBackend(), ForecastBackend)


def test_backend_one_forecast_per_sku_with_horizon_points():
    backend = DeterministicForecastBackend()
    out = backend.predict("ph-1", ["a", "b", "c"], horizon_days=10)
    assert [f.sku_id for f in out] == ["a", "b", "c"]
    for f in out:
        assert len(f.points) == 10


def test_backend_reproducible_across_calls():
    b1 = DeterministicForecastBackend()
    b2 = DeterministicForecastBackend()
    o1 = b1.predict("ph-1", ["a", "b"], 14)
    o2 = b2.predict("ph-1", ["a", "b"], 14)
    assert o1 == o2


def test_backend_points_monotone():
    backend = DeterministicForecastBackend()
    out = backend.predict("ph-1", ["a", "b"], 30)
    for f in out:
        for p in f.points:
            assert p.p10 <= p.p50 <= p.p90
            assert p.p10 >= 0


def test_backend_weights_sum_to_one():
    backend = DeterministicForecastBackend()
    for f in backend.predict("ph-1", ["a", "b", "c"], 5):
        assert abs(sum(f.model_weights.values()) - 1.0) < 1e-9
        assert all(v >= 0 for v in f.model_weights.values())


def test_backend_covariates_toggle_seasonality():
    backend = DeterministicForecastBackend()
    with_cov = backend.predict("ph-1", ["a"], 5, include_covariates=True)[0]
    without_cov = backend.predict("ph-1", ["a"], 5, include_covariates=False)[0]
    assert without_cov.seasonality_signal is None
    assert with_cov.seasonality_signal is not None


# --- Req 3: DailyPoint validator rejects out-of-order quantiles -------------


def test_dailypoint_accepts_monotone():
    p = DailyPoint(date=date(2026, 1, 1), p10=1.0, p50=2.0, p90=3.0)
    assert p.p50 == 2.0


def test_dailypoint_rejects_p10_gt_p50():
    with pytest.raises(ValidationError):
        DailyPoint(date=date(2026, 1, 1), p10=5.0, p50=2.0, p90=9.0)


def test_dailypoint_rejects_p50_gt_p90():
    with pytest.raises(ValidationError):
        DailyPoint(date=date(2026, 1, 1), p10=1.0, p50=8.0, p90=3.0)


def test_dailypoint_allows_equal_quantiles():
    p = DailyPoint(date=date(2026, 1, 1), p10=2.0, p50=2.0, p90=2.0)
    assert p.p10 == p.p90


def test_backend_daily_point_is_frozen_dataclass():
    bp = BackendDailyPoint(date=date(2026, 1, 1), p10=1.0, p50=2.0, p90=3.0)
    with pytest.raises(FrozenInstanceError):
        bp.p50 = 9.0  # type: ignore[misc]


# --- Req 8: cache key + FakeForecastCache behavior --------------------------


def test_cache_key_stable_and_order_independent():
    k1 = cache_key("ph-1", ["a", "b"], 30, True)
    k2 = cache_key("ph-1", ["b", "a"], 30, True)
    assert k1 == k2


def test_cache_key_differs_on_pharmacy_horizon_covariates():
    base = cache_key("ph-1", ["a"], 30, True)
    assert cache_key("ph-2", ["a"], 30, True) != base
    assert cache_key("ph-1", ["a"], 31, True) != base
    assert cache_key("ph-1", ["a"], 30, False) != base
    assert cache_key("ph-1", ["a", "b"], 30, True) != base


def test_cache_key_is_injective_over_delimiter_components():
    # Regression: a delimiter (",") inside a sku_id must not let a structurally
    # different request collide onto the same key (cache poisoning).
    assert cache_key("ph", ["a,b"], 30, True) != cache_key("ph", ["a", "b"], 30, True)
    # A ":" in the pharmacy_id must not collide with the same char in a sku_id.
    assert cache_key("p:1", ["a"], 30, True) != cache_key("p", ["1:a"], 30, True)
    # Identical inputs still map to an identical (stable) key.
    assert cache_key("ph", ["a,b"], 30, True) == cache_key("ph", ["a,b"], 30, True)


def test_fake_cache_satisfies_protocol():
    assert isinstance(FakeForecastCache(), ForecastCache)


def test_fake_cache_hit_miss_counting():
    cache = FakeForecastCache()
    assert cache.get("k") is None
    assert cache.misses == 1
    assert cache.hits == 0
    payload = [{"sku_id": "a"}]
    cache.set("k", payload)
    assert cache.get("k") == payload
    assert cache.hits == 1
    assert cache.misses == 1


# --- ForecastService orchestrator delegates to its backend ------------------


async def test_forecast_service_delegates_to_backend():
    from demandforecast.services.forecast_service import ForecastService

    svc = ForecastService()  # defaults to the deterministic fake
    out = await svc.forecast("ph-1", ["a", "b"], horizon=7)
    assert [f.sku_id for f in out] == ["a", "b"]
    assert all(len(f.points) == 7 for f in out)


async def test_forecast_service_accepts_injected_backend():
    from demandforecast.services.forecast_service import ForecastService

    svc = ForecastService(backend=DeterministicForecastBackend())
    out = await svc.forecast("ph-1", ["a"], horizon=3, include_covariates=False)
    assert out[0].seasonality_signal is None
