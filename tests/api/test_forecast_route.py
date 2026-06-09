"""HTTP contract tests for POST /v1/forecast (offline, TestClient)."""

from __future__ import annotations

from datetime import datetime

import pytest

from demandforecast.api.routes.forecast import (
    get_forecast_backend,
    get_forecast_cache,
)
from demandforecast.services.forecast_service import (
    DeterministicForecastBackend,
    FakeForecastCache,
)


def _install_fakes(app, backend=None, cache=None):
    backend = backend or DeterministicForecastBackend()
    cache = cache or FakeForecastCache()
    app.dependency_overrides[get_forecast_backend] = lambda: backend
    app.dependency_overrides[get_forecast_cache] = lambda: cache
    return backend, cache


def _body(**overrides):
    body = {
        "pharmacy_id": "ph-001",
        "sku_ids": ["sku-001", "sku-002"],
        "horizon_days": 14,
        "include_covariates": True,
    }
    body.update(overrides)
    return body


# --- Req 2: populated, contract-valid response -----------------------------


def test_forecast_returns_one_skuforecast_per_sku(app, client):
    _install_fakes(app)
    resp = client.post("/v1/forecast", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["forecasts"]) == 2
    returned_ids = {f["sku_id"] for f in data["forecasts"]}
    assert returned_ids == {"sku-001", "sku-002"}
    for f in data["forecasts"]:
        assert len(f["daily_forecast"]) == 14
        assert f["drug_name"]  # non-empty
        assert f["trend"] in {"increasing", "decreasing", "flat"}


# --- Req 6 (response contract): model_weights sum to ~1 --------------------


def test_model_weights_nonneg_and_sum_to_one(app, client):
    _install_fakes(app)
    resp = client.post("/v1/forecast", json=_body(horizon_days=30))
    assert resp.status_code == 200
    for f in resp.json()["forecasts"]:
        weights = f["model_weights"]
        assert all(v >= 0 for v in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-6


# --- Req 3: quantile monotonicity p10 <= p50 <= p90 ------------------------


def test_quantiles_monotone_every_point(app, client):
    _install_fakes(app)
    resp = client.post(
        "/v1/forecast",
        json=_body(sku_ids=["sku-001", "sku-002", "sku-003"], horizon_days=30),
    )
    assert resp.status_code == 200
    for f in resp.json()["forecasts"]:
        for p in f["daily_forecast"]:
            assert p["p10"] <= p["p50"] <= p["p90"]


# --- Req 4: latency_ms non-negative int + generated_at ISO-8601 ------------


def test_latency_and_generated_at(app, client):
    _install_fakes(app)
    resp = client.post("/v1/forecast", json=_body())
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["latency_ms"], int)
    assert data["latency_ms"] >= 0
    # Parses as ISO-8601 (raises if not).
    datetime.fromisoformat(data["generated_at"])


# --- Req 1: determinism — identical request -> identical forecasts ---------


def test_two_identical_requests_byte_identical_forecasts(app, client):
    _install_fakes(app)
    r1 = client.post("/v1/forecast", json=_body())
    r2 = client.post("/v1/forecast", json=_body())
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["forecasts"] == r2.json()["forecasts"]


# --- Req 5: request validation (HTTP contract) -----------------------------


@pytest.mark.parametrize(
    "bad_body",
    [
        {"sku_ids": ["sku-001"]},  # missing pharmacy_id
        {"pharmacy_id": "ph-001"},  # missing sku_ids
        {"pharmacy_id": "ph-001", "sku_ids": ["sku-001"], "horizon_days": 0},
        {"pharmacy_id": "ph-001", "sku_ids": ["sku-001"], "horizon_days": 366},
        {"pharmacy_id": "ph-001", "sku_ids": "not-a-list"},  # wrong type
    ],
)
def test_invalid_requests_return_422(app, client, bad_body):
    _install_fakes(app)
    resp = client.post("/v1/forecast", json=bad_body)
    assert resp.status_code == 422


def test_valid_baseline_defaults_return_200(app, client):
    _install_fakes(app)
    # Default horizon_days (30) and include_covariates (true) omitted.
    resp = client.post(
        "/v1/forecast",
        json={"pharmacy_id": "ph-001", "sku_ids": ["sku-001"]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["forecasts"][0]["daily_forecast"]) == 30


@pytest.mark.parametrize("horizon", [1, 365])
def test_boundary_horizons_return_200(app, client, horizon):
    _install_fakes(app)
    resp = client.post("/v1/forecast", json=_body(horizon_days=horizon))
    assert resp.status_code == 200
    assert len(resp.json()["forecasts"][0]["daily_forecast"]) == horizon


# --- Req 8: caching — backend invoked once on identical calls ---------------


def test_cache_hit_backend_invoked_once(app, client, counting_backend, fake_cache):
    _install_fakes(app, backend=counting_backend, cache=fake_cache)
    r1 = client.post("/v1/forecast", json=_body())
    r2 = client.post("/v1/forecast", json=_body())
    assert r1.status_code == r2.status_code == 200
    # Backend invoked exactly once across two identical requests.
    assert counting_backend.calls == 1
    assert fake_cache.hits == 1
    assert fake_cache.misses == 1
    # Cache hit returns the same forecasts payload.
    assert r1.json()["forecasts"] == r2.json()["forecasts"]


def test_cache_miss_on_different_request_reinvokes(
    app, client, counting_backend, fake_cache
):
    _install_fakes(app, backend=counting_backend, cache=fake_cache)
    client.post("/v1/forecast", json=_body())
    # Different pharmacy_id -> miss.
    client.post("/v1/forecast", json=_body(pharmacy_id="ph-OTHER"))
    # Different sku_ids -> miss.
    client.post("/v1/forecast", json=_body(sku_ids=["sku-999"]))
    # Different horizon_days -> miss.
    client.post("/v1/forecast", json=_body(horizon_days=7))
    assert counting_backend.calls == 4


def test_cache_key_order_independent_for_skus(
    app, client, counting_backend, fake_cache
):
    _install_fakes(app, backend=counting_backend, cache=fake_cache)
    client.post("/v1/forecast", json=_body(sku_ids=["sku-001", "sku-002"]))
    # Reordered sku_ids hit the same logical cache entry.
    client.post("/v1/forecast", json=_body(sku_ids=["sku-002", "sku-001"]))
    assert counting_backend.calls == 1


# --- Default (un-overridden) dependency wiring works offline ----------------


def test_default_providers_serve_request_without_overrides(app, client):
    # No dependency_overrides installed: exercises the real module-level
    # DeterministicForecastBackend + FakeForecastCache providers. Proves the
    # app runs offline out of the box (no DB/Redis/MLflow).
    resp = client.post(
        "/v1/forecast",
        json={"pharmacy_id": "ph-default", "sku_ids": ["sku-xyz"], "horizon_days": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["forecasts"]) == 1
    assert len(data["forecasts"][0]["daily_forecast"]) == 5
