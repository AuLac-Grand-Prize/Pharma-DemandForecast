"""Shared test fixtures for the DemandForecast service.

Critically, this file populates the required-but-defaultless ``Settings`` env
vars (``postgres_dsn``, ``redis_url``, ``moh_*``, ``weather_*``,
``internal_api_token``) at *import time* — before any application module is
imported — so that even an accidental ``get_settings()`` / ``Settings()`` call
in the import path cannot raise ``pydantic.ValidationError`` offline.

The forecast/reorder routes themselves do NOT touch ``Settings`` (the backend
seam keeps config out of the test path), so the suite runs with no
TimescaleDB / Redis / MLflow / network.
"""

from __future__ import annotations

import os

# Set BEFORE importing anything from demandforecast.* so config validation is
# safe regardless of whether a module reads settings at import time.
_REQUIRED_ENV = {
    "POSTGRES_DSN": "postgresql+asyncpg://test:test@localhost:5432/test",
    "REDIS_URL": "redis://localhost:6379/0",
    "MOH_DISEASE_FEED_URL": "https://example.test/moh",
    "MOH_API_KEY": "test-moh-key",
    "WEATHER_API_URL": "https://example.test/weather",
    "WEATHER_API_KEY": "test-weather-key",
    "INTERNAL_API_TOKEN": "test-internal-token",
}
for _k, _v in _REQUIRED_ENV.items():
    os.environ.setdefault(_k, _v)

import importlib.util  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from demandforecast.api.main import app as fastapi_app  # noqa: E402

# The M5 research-pipeline tests (tests/test_naive.py, tests/test_features.py,
# tests/test_evaluate.py) import numpy/pandas and exercise src/ research code,
# which is explicitly out of scope for this offline service/API phase. When the
# heavy scientific stack is not installed (the offline test environment), skip
# collecting them so the in-scope forecast/reorder suite runs cleanly. In a
# full-deps environment numpy is present and these tests run unchanged.
if importlib.util.find_spec("numpy") is None:
    collect_ignore_glob = [
        "test_naive.py",
        "test_features.py",
        "test_evaluate.py",
    ]
from demandforecast.services.forecast_service import (  # noqa: E402
    BackendDailyPoint,
    BackendSkuForecast,
    DeterministicForecastBackend,
    FakeForecastCache,
)


@pytest.fixture
def app():
    """The FastAPI app with a clean dependency-override slate per test."""
    fastapi_app.dependency_overrides.clear()
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def deterministic_backend() -> DeterministicForecastBackend:
    return DeterministicForecastBackend()


@pytest.fixture
def fake_cache() -> FakeForecastCache:
    return FakeForecastCache()


class CountingBackend:
    """Wraps a real backend and counts ``predict`` invocations.

    Used to prove the cache prevents backend re-invocation on a hit.
    """

    def __init__(self, inner: DeterministicForecastBackend | None = None) -> None:
        self.inner = inner or DeterministicForecastBackend()
        self.calls = 0

    def predict(
        self,
        pharmacy_id: str,
        sku_ids: list[str],
        horizon_days: int,
        include_covariates: bool = True,
    ) -> list[BackendSkuForecast]:
        self.calls += 1
        return self.inner.predict(
            pharmacy_id, sku_ids, horizon_days, include_covariates
        )


@pytest.fixture
def counting_backend() -> CountingBackend:
    return CountingBackend()


# Re-export backend shapes for tests that build malformed/out-of-order points.
__all__ = [
    "BackendDailyPoint",
    "BackendSkuForecast",
    "CountingBackend",
]
