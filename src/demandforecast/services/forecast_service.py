"""Orchestrate sales history + covariates -> ensemble prediction.

This module defines the *injectable seam* the FastAPI routes depend on:

- ``ForecastBackend`` — a :class:`typing.Protocol` describing the one method the
  routes need (produce per-SKU daily quantile forecasts). The real production
  backend (TimescaleDB history + Prophet/LSTM/TFT + MLflow weights) is a later
  phase; the routes never import those heavy deps directly.
- ``DeterministicForecastBackend`` — a seeded, reproducible fake with **no I/O**
  and **no cross-run randomness**. Identical inputs -> byte-identical output.
- ``ForecastCache`` / ``FakeForecastCache`` — an injectable cache abstraction
  with an in-memory fake (no Redis) so the route's cache hit/miss path is
  testable offline.

Everything here is pure-stdlib (``hashlib``/``dataclasses``) so it imports and
runs without darts/prophet/torch/lightgbm/xgboost/asyncpg/redis/mlflow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol, runtime_checkable

from demandforecast.ensemble.stacker import EnsembleWeights

# ---------------------------------------------------------------------------
# Backend data shapes (decoupled from the HTTP response models so the seam does
# not depend on the route module — avoids an import cycle).
# ---------------------------------------------------------------------------

_TREND_CHOICES = ("increasing", "decreasing", "flat")
_SEASONALITY_CHOICES = ("none", "weekly", "monthly")


@dataclass(frozen=True)
class BackendDailyPoint:
    """One day of quantile forecast. Invariant: ``p10 <= p50 <= p90``."""

    date: date
    p10: float
    p50: float
    p90: float


@dataclass(frozen=True)
class BackendSkuForecast:
    """Per-SKU forecast produced by a :class:`ForecastBackend`."""

    sku_id: str
    drug_name: str
    points: list[BackendDailyPoint]
    trend: str
    model_weights: dict[str, float]
    seasonality_signal: str | None = None


@runtime_checkable
class ForecastBackend(Protocol):
    """Injectable forecasting backend the routes depend on.

    Implementations must return one :class:`BackendSkuForecast` per requested
    ``sku_id`` (same order), each with exactly ``horizon_days`` daily points
    satisfying ``p10 <= p50 <= p90``.
    """

    def predict(
        self,
        pharmacy_id: str,
        sku_ids: list[str],
        horizon_days: int,
        include_covariates: bool = True,
    ) -> list[BackendSkuForecast]:
        ...


# ---------------------------------------------------------------------------
# Deterministic fake backend
# ---------------------------------------------------------------------------


def _seed_int(*parts: object) -> int:
    """Stable, cross-run, cross-platform integer seed from arbitrary parts.

    Uses BLAKE2b over the joined string form — ``hash()`` is randomized per
    process (PYTHONHASHSEED) and would break determinism, so we never use it.
    """
    raw = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _unit(seed: int) -> float:
    """Map a seed to a reproducible float in ``[0, 1)``."""
    return (seed % 1_000_000) / 1_000_000.0


class DeterministicForecastBackend:
    """A reproducible, I/O-free :class:`ForecastBackend` for offline tests.

    The forecast for a given ``(sku_id, date)`` is derived purely from a
    BLAKE2b seed of those values plus ``pharmacy_id`` — so two calls with the
    same request produce identical output, with no randomness across runs or
    platforms and no database/Redis/MLflow access.

    Quantiles are constructed monotone by design: a non-negative ``p50`` base,
    then ``p10 = p50 * (1 - spread)`` and ``p90 = p50 * (1 + spread)`` with
    ``spread in [0, 0.5)`` — guaranteeing ``p10 <= p50 <= p90``.
    """

    #: Base demand floor so low-stock SKUs always need reordering.
    _BASE_DEMAND = 8.0
    #: Per-SKU demand amplitude (added on top of the floor).
    _DEMAND_AMPLITUDE = 16.0

    def __init__(self, start_date: date | None = None) -> None:
        # A fixed anchor keeps generated dates deterministic regardless of the
        # wall clock; tests assert *shape*, not the literal calendar day.
        self._start_date = start_date or date(2026, 1, 1)

    def _drug_name(self, sku_id: str) -> str:
        catalog = (
            "Paracetamol 500mg",
            "Amoxicillin 500mg",
            "Cetirizine 10mg",
            "Omeprazole 20mg",
            "Metformin 850mg",
            "Vitamin C 1000mg",
        )
        idx = _seed_int("drug", sku_id) % len(catalog)
        return catalog[idx]

    def _model_weights(self, sku_id: str) -> dict[str, float]:
        # Reuse the *real* EnsembleWeights normalization (source of truth for
        # the model_weights contract) rather than reimplementing the sum-to-1.
        raw = EnsembleWeights(
            prophet=0.2 + _unit(_seed_int("w_prophet", sku_id)),
            lstm=0.2 + _unit(_seed_int("w_lstm", sku_id)),
            tft=0.2 + _unit(_seed_int("w_tft", sku_id)),
        ).normalize()
        return {"prophet": raw.prophet, "lstm": raw.lstm, "tft": raw.tft}

    def _trend(self, sku_id: str) -> str:
        return _TREND_CHOICES[_seed_int("trend", sku_id) % len(_TREND_CHOICES)]

    def _seasonality(self, sku_id: str, include_covariates: bool) -> str | None:
        if not include_covariates:
            return None
        return _SEASONALITY_CHOICES[
            _seed_int("season", sku_id) % len(_SEASONALITY_CHOICES)
        ]

    def _point(self, pharmacy_id: str, sku_id: str, day: date) -> BackendDailyPoint:
        base_seed = _seed_int(pharmacy_id, sku_id, day.isoformat())
        # p50 in [BASE, BASE + AMPLITUDE), rounded to 2dp for stable equality.
        p50 = self._BASE_DEMAND + self._DEMAND_AMPLITUDE * _unit(base_seed)
        spread = 0.5 * _unit(_seed_int("spread", base_seed))  # [0, 0.5)
        p10 = p50 * (1.0 - spread)
        p90 = p50 * (1.0 + spread)
        return BackendDailyPoint(
            date=day,
            p10=round(p10, 4),
            p50=round(p50, 4),
            p90=round(p90, 4),
        )

    def predict(
        self,
        pharmacy_id: str,
        sku_ids: list[str],
        horizon_days: int,
        include_covariates: bool = True,
    ) -> list[BackendSkuForecast]:
        results: list[BackendSkuForecast] = []
        for sku_id in sku_ids:
            points = [
                self._point(pharmacy_id, sku_id, self._start_date + timedelta(days=d))
                for d in range(horizon_days)
            ]
            results.append(
                BackendSkuForecast(
                    sku_id=sku_id,
                    drug_name=self._drug_name(sku_id),
                    points=points,
                    trend=self._trend(sku_id),
                    model_weights=self._model_weights(sku_id),
                    seasonality_signal=self._seasonality(sku_id, include_covariates),
                )
            )
        return results


# ---------------------------------------------------------------------------
# Cache seam
# ---------------------------------------------------------------------------


def cache_key(
    pharmacy_id: str,
    sku_ids: list[str],
    horizon_days: int,
    include_covariates: bool,
) -> str:
    """Deterministic cache key for a forecast request.

    Order-independent over ``sku_ids`` so ``[a, b]`` and ``[b, a]`` collide on
    the same logical forecast; differing pharmacy/horizon/covariate flags do
    not collide.

    The key is **injective**: it is derived from a JSON-serialized payload of
    the structural request fields (not a raw delimiter-join), so components
    containing the old delimiters (``,`` or ``:``) cannot make structurally
    different requests collide. For example ``sku_ids=['a,b']`` no longer
    collides with ``['a', 'b']``, and ``pharmacy_id='p:1', sku_ids=['a']`` no
    longer collides with ``pharmacy_id='p', sku_ids=['1:a']``.
    """
    payload = json.dumps(
        {
            "p": pharmacy_id,
            "s": sorted(sku_ids),
            "h": horizon_days,
            "c": int(include_covariates),
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "forecast:" + hashlib.blake2b(
        payload.encode("utf-8"), digest_size=16
    ).hexdigest()


# The route stores the serialized forecasts list (each item a
# ``SkuForecast.model_dump(mode="json")`` dict) under one key.
CachedForecasts = list[dict[str, Any]]


@runtime_checkable
class ForecastCache(Protocol):
    """Injectable cache for serialized forecast responses (keyed by request)."""

    def get(self, key: str) -> CachedForecasts | None:
        ...

    def set(self, key: str, value: CachedForecasts) -> None:
        ...


class FakeForecastCache:
    """In-memory :class:`ForecastCache` for offline tests (no Redis).

    Stores the serialized forecasts payload (a list of dicts) and records
    hit/miss counts for assertions.
    """

    def __init__(self) -> None:
        self._store: dict[str, CachedForecasts] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> CachedForecasts | None:
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def set(self, key: str, value: CachedForecasts) -> None:
        self._store[key] = value


# ---------------------------------------------------------------------------
# Legacy orchestrator placeholder (real DB/model path is a later phase).
# ---------------------------------------------------------------------------


@dataclass
class ForecastService:
    """High-level orchestrator.

    The production implementation (TimescaleDB history + covariates + ensemble
    inference) is out of scope for the offline-seam phase. It delegates to an
    injected :class:`ForecastBackend`, defaulting to the deterministic fake so
    the object is usable without any external service.
    """

    backend: ForecastBackend = field(default_factory=DeterministicForecastBackend)

    async def forecast(
        self,
        pharmacy_id: str,
        sku_ids: list[str],
        horizon: int,
        include_covariates: bool = True,
    ) -> list[BackendSkuForecast]:
        return self.backend.predict(
            pharmacy_id=pharmacy_id,
            sku_ids=sku_ids,
            horizon_days=horizon,
            include_covariates=include_covariates,
        )
