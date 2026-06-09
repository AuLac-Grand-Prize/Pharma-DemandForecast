# Phase: DemandForecast — Offline Forecast Seam & API Contract Tests — Specification

**Created:** 2026-06-09
**Ambiguity score:** 0.12 (gate: ≤ 0.20)
**Requirements:** 8 locked

## Goal

Make `POST /v1/forecast` and `POST /v1/reorder-suggestions` return real, contract-valid responses driven by an **injectable forecasting backend** with a **deterministic fake**, so both endpoints (request validation, quantile monotonicity p10 ≤ p50 ≤ p90, `latency_ms`, caching) are fully HTTP-testable offline — with no TimescaleDB, Redis, or MLflow running.

## Background

Grounded in the current code of `Pharma-DemandForecast` (its own git repo on `main`):

- `src/demandforecast/api/routes/forecast.py:38-41` — the `/v1/forecast` handler is a stub: `# TODO: load ensemble + run inference` then `return ForecastResponse(forecasts=[], generated_at="", latency_ms=0)`. It never calls a service and has no dependency-injection seam. `generated_at=""` is not a valid timestamp and `latency_ms=0` is hard-coded.
- `src/demandforecast/api/routes/reorder.py:26-29` — the `/v1/reorder-suggestions` handler is a stub: `# TODO: compute reorder qty ...` then `return ReorderResponse(suggestions=[], expected_stockout_avoided=0)`. It never calls `ReorderService` despite that service being fully implemented and tested.
- `src/demandforecast/services/forecast_service.py` — `ForecastService.__init__` and `ForecastService.forecast(pharmacy_id, sku_ids, horizon)` are entirely TODO; `forecast()` returns `[]`. There is no Postgres/Redis client, no model loading, and no caching anywhere in the repo.
- `src/demandforecast/services/reorder_service.py:6-19` — `ReorderService.suggest_qty(...)` is **real and correct**: `target = avg(p50)*lead_time + max(p90)*safety_stock_days`, `qty = max(int(target - current_stock), 0)`, returns `0` when `forecast_p50` is empty. Covered by `tests/unit/test_reorder_service.py` (overstocked → 0; zero stock → positive). It is not yet wired to the route.
- `src/demandforecast/ensemble/stacker.py` — `EnsembleStacker` / `EnsembleWeights` are real and tested (`tests/unit/test_ensemble.py`); weights normalize and `combine()` is a weighted sum over prophet/lstm/tft. This is the source of truth for the `model_weights` field in the response.
- `src/demandforecast/models/{prophet_model,lstm_model,tft_model}.py` — all three model wrappers are stubs (`fit`/`predict` bodies are `pass`); there are **no trained artifacts** (`.gitignore` excludes `*.ckpt`, `*.pt`, `mlruns/`, `results/`). So the real inference path cannot run offline today.
- `src/demandforecast/core/config.py` — `Settings` declares `postgres_dsn`, `redis_url`, `moh_disease_feed_url`, `moh_api_key`, `weather_api_url`, `weather_api_key`, `internal_api_token` **with no defaults**. Instantiating `Settings()` (via `get_settings()`) without a populated `.env` raises `pydantic.ValidationError`. The routes don't import settings today, but any real backend that does would break offline import unless the seam keeps config out of the test path.
- `src/demandforecast/api/main.py` — mounts `health` (no prefix), `forecast` and `reorder` under `/v1`. `httpx>=0.27.0` is already a dependency and `pytest`/`pytest-asyncio` are dev deps (`pyproject.toml`), so `fastapi.testclient.TestClient` works with no extra install.
- Tests today: `tests/unit/test_ensemble.py`, `tests/unit/test_reorder_service.py`, plus research-pipeline tests `tests/test_evaluate.py`, `tests/test_features.py`, `tests/test_naive.py` (these exercise `src/` M5 research code, not the FastAPI app). There is **no `conftest.py`, no TestClient usage, and no test of `api/routes/forecast.py` or `api/routes/reorder.py`**. `pyproject.toml` enforces `--cov-fail-under=80` and `asyncio_mode = "auto"`.

The primary deliverable that does NOT exist yet: an injectable forecasting backend (Protocol + deterministic fake) so the two `/v1` routes produce monotone-quantile, latency-stamped, optionally-cached responses that can be asserted as HTTP contracts entirely offline.

## Requirements

1. **ForecastBackend seam (Protocol + deterministic fake)**: The forecast route depends on an injectable backend abstraction, not on Prophet/LSTM/TFT or any DB/Redis client.
   - Current: `forecast.py` has no service dependency; `ForecastService` is TODO and would require Postgres/Redis/models.
   - Target: Define a `ForecastBackend` Protocol (e.g. in `services/forecast_service.py`) with a method that, given `pharmacy_id`, `sku_ids`, `horizon_days`, `include_covariates`, returns per-SKU daily quantile forecasts. Provide a `DeterministicForecastBackend` fake that produces reproducible output from a seed derived from `(sku_id, date)` — no I/O, no randomness across runs. The route obtains its backend via a FastAPI dependency (`Depends`) that is overridable with `app.dependency_overrides`.
   - Acceptance: A test installs the deterministic fake via `dependency_overrides`, calls `/v1/forecast` twice with the same body, and gets byte-identical `forecasts` (excluding `latency_ms`/`generated_at`); no TimescaleDB/Redis/MLflow process is required for the test to pass.

2. **Forecast route returns a contract-valid populated response**: `/v1/forecast` returns one `SkuForecast` per requested `sku_id` with `horizon_days` daily points.
   - Current: returns `forecasts=[]`, `generated_at=""`, `latency_ms=0`.
   - Target: For each `sku_id` in the request, the response contains exactly one `SkuForecast` with `daily_forecast` of length `horizon_days`, a non-empty `drug_name`, a `trend` in a known set (e.g. `increasing`/`decreasing`/`flat`), and `model_weights` whose values are non-negative and sum to ~1.0 (consistent with `EnsembleWeights.normalize()`).
   - Acceptance: POST with `sku_ids=["sku-001","sku-002"]`, `horizon_days=14` → 200; `len(forecasts)==2`; each `daily_forecast` has 14 points; `abs(sum(model_weights.values()) - 1.0) < 1e-6`.

3. **Quantile monotonicity p10 ≤ p50 ≤ p90**: Every daily forecast point is quantile-ordered.
   - Current: `DailyPoint` (forecast.py:16-20) has `p10`/`p50`/`p90` floats with **no ordering validation**; the stub returns no points so the invariant is untested.
   - Target: The deterministic backend guarantees `p10 ≤ p50 ≤ p90` for every point, and a `DailyPoint` (or response) validator enforces it so a violating backend cannot silently ship malformed quantiles.
   - Acceptance: For a forecast over `horizon_days=30` across multiple SKUs, every point satisfies `p10 ≤ p50 ≤ p90`; a unit test feeding an out-of-order point (p10 > p50) raises a validation error.

4. **`latency_ms` is real, present, and non-negative**: The response reports measured handler latency.
   - Current: `latency_ms=0` hard-coded; `generated_at=""`.
   - Target: The handler measures wall-clock duration around backend invocation and sets `latency_ms` to a non-negative integer (milliseconds); `generated_at` is a valid ISO-8601 timestamp string.
   - Acceptance: Response includes `latency_ms` as an `int >= 0` and `generated_at` parses as ISO-8601; the field is present even when the fake backend returns instantly.

5. **Forecast request validation (HTTP contract)**: Malformed forecast requests are rejected with 422; valid ones honor `horizon_days` bounds.
   - Current: `ForecastRequest` exists with `horizon_days: int = Field(ge=1, le=365)` but no route test exercises validation.
   - Target: FastAPI/Pydantic validation is asserted as a contract: missing `pharmacy_id` or `sku_ids` → 422; `horizon_days=0` or `horizon_days=366` → 422; `sku_ids` of wrong type → 422. A valid body with default `horizon_days` (30) and `include_covariates=true` → 200.
   - Acceptance: Parametrized tests confirm 422 for each invalid case and 200 for the valid baseline; the boundary values `horizon_days=1` and `horizon_days=365` both return 200.

6. **Reorder route wired to `ReorderService` with forecast-driven quantities**: `/v1/reorder-suggestions` computes real suggestions through the same backend seam.
   - Current: `reorder.py` returns `suggestions=[]`, `expected_stockout_avoided=0`; never calls `ReorderService`.
   - Target: The reorder handler obtains forecasts via the injectable `ForecastBackend` (overridable) plus current-stock input, and computes `suggested_order_qty` per SKU using `ReorderService.suggest_qty(p50, p90, current_stock, lead_time_days, safety_stock_days)`. Each suggestion carries a non-empty `reason`. Quantities are clamped at ≥ 0 (per existing service logic).
   - Acceptance: With the deterministic fake installed, POST `/v1/reorder-suggestions` → 200 with one `ReorderItem` per SKU; an SKU whose `current_stock` exceeds projected demand yields `suggested_order_qty == 0`; a low-stock SKU yields `suggested_order_qty > 0`.

7. **Reorder formula edge cases (unit-level, falsifiable)**: The reorder math is pinned at its boundaries.
   - Current: `tests/unit/test_reorder_service.py` covers only overstocked→0 and zero-stock→positive; empty-forecast and exact-boundary cases are unverified.
   - Target: Add unit tests asserting: empty `forecast_p50` → `0`; empty `forecast_p90` falls back to `avg_daily` for the peak term; `current_stock` exactly equal to `target_stock` → `0`; result is always a non-negative `int`. These pin `ReorderService.suggest_qty` against regressions when it is wired into the route.
   - Acceptance: All four edge-case unit tests pass against the current `suggest_qty` implementation without modifying its formula.

8. **Caching behavior with a fake cache (offline)**: Identical forecast requests are served from an injectable cache; the backend is not re-invoked on a hit.
   - Current: No caching exists anywhere; `CACHE_TTL_SECONDS` (config.py / `.env.example`) is unused.
   - Target: Introduce a `ForecastCache` abstraction (get/set keyed by a deterministic function of the request) with an in-memory `FakeForecastCache` for tests (no Redis). The forecast route checks the cache before invoking the backend and stores the result after a miss. The cache is injectable/overridable.
   - Acceptance: With a counting fake backend + `FakeForecastCache`, two identical `/v1/forecast` calls invoke the backend exactly once (second call is a cache hit returning the same `forecasts`); a request with a different `pharmacy_id`/`sku_ids`/`horizon_days` is a cache miss and invokes the backend again.

## Boundaries

**In scope:**
- An injectable `ForecastBackend` Protocol + a `DeterministicForecastBackend` fake (seeded, reproducible, no I/O).
- Wiring `/v1/forecast` and `/v1/reorder-suggestions` to the seam so responses are populated and contract-valid.
- Enforcing `p10 ≤ p50 ≤ p90` and a real non-negative `latency_ms` + ISO-8601 `generated_at`.
- Wiring `/v1/reorder-suggestions` to the existing `ReorderService.suggest_qty`.
- An injectable `ForecastCache` + in-memory `FakeForecastCache`; cache hit/miss behavior.
- API contract tests (TestClient) for validation, monotonicity, latency, and caching; plus reorder edge-case unit tests.
- A `tests/conftest.py` (or equivalent) wiring `dependency_overrides` and an app fixture.

**Out of scope:**
- Training or shipping real Prophet/LSTM/TFT artifacts — models stay stubs; the real inference path is a later phase (no artifacts can run offline). — reason: no fitted models exist and `.gitignore` excludes them; out of scope keeps tests offline.
- Real TimescaleDB / asyncpg sales-history loading in `ForecastService` — the production backend adapter is a separate phase. — reason: requires a running database; this phase only builds the seam + fake.
- Real Redis cache adapter — only the `ForecastCache` interface + in-memory fake are built here. — reason: no Redis in the offline test environment.
- MLflow model-registry loading and per-pharmacy weight learning. — reason: requires MLflow server + trained runs.
- MoH disease feed / weather covariate HTTP integration — `include_covariates` may toggle fake behavior but does no network I/O. — reason: external APIs, no network in tests.
- Authentication/`INTERNAL_API_TOKEN` enforcement on these routes. — reason: cross-service auth is owned by the API Gateway SPEC/phase.
- The `src/` M5 research pipeline (`train.py`, `metrics/`, `data/features.py`) and its tests. — reason: already covered; this phase is the service/API layer only.

## Constraints

- The full forecast/reorder test suite must run with **no external services** — no TimescaleDB, no Redis, no MLflow, no network — via `dependency_overrides` + fakes.
- Tests must be **deterministic**: same request → same `forecasts` (excluding `latency_ms`/`generated_at`).
- Do not require a populated `.env`: the test path must not trigger `Settings()` (config.py) validation for the required-but-defaultless fields (`postgres_dsn`, `redis_url`, `moh_*`, `weather_*`, `internal_api_token`).
- Reuse the existing, tested `ReorderService` and `EnsembleStacker`/`EnsembleWeights` — do not reimplement their math.
- Stay within current dependencies in `pyproject.toml` (FastAPI, Pydantic v2, `httpx`, pytest, pytest-asyncio); add no new runtime deps.
- Coverage gate `--cov-fail-under=80` (pyproject.toml) must still pass; `ruff check` and `mypy src` stay clean.
- Preserve the existing HTTP contract field names/shapes in `forecast.py` / `reorder.py` response models (additive changes only).

## Acceptance Criteria

- [ ] `/v1/forecast` with the deterministic fake (via `dependency_overrides`) returns 200 with one `SkuForecast` per `sku_id` and `horizon_days` daily points each — no DB/Redis/MLflow needed.
- [ ] Two identical forecast requests return byte-identical `forecasts` (excluding `latency_ms`/`generated_at`).
- [ ] Every `DailyPoint` satisfies `p10 ≤ p50 ≤ p90`; an out-of-order point raises a validation error in a unit test.
- [ ] `latency_ms` is an `int >= 0` and `generated_at` parses as ISO-8601 on every forecast response.
- [ ] Forecast validation: missing `pharmacy_id`/`sku_ids` → 422; `horizon_days` 0 or 366 → 422; `horizon_days` 1 and 365 → 200.
- [ ] `model_weights` values are non-negative and sum to ~1.0 (`abs(sum - 1.0) < 1e-6`).
- [ ] `/v1/reorder-suggestions` returns one `ReorderItem` per SKU; overstocked SKU → `suggested_order_qty == 0`; low-stock SKU → `> 0`.
- [ ] Reorder edge-case unit tests pass: empty `p50` → 0; empty `p90` → fallback peak; `current_stock == target_stock` → 0; result is non-negative `int`.
- [ ] With a counting fake backend + `FakeForecastCache`, two identical forecast calls invoke the backend exactly once; a differing request invokes it again.
- [ ] `pytest` passes (including new tests) with `--cov-fail-under=80` satisfied; `ruff check src tests` and `mypy src` are clean.

## Ambiguity Report

| Dimension          | Score | Min  | Status | Notes |
|--------------------|-------|------|--------|-------|
| Goal Clarity       | 0.92  | 0.75 | ✓      | Injectable backend + fake → two routes HTTP-testable offline |
| Boundary Clarity   | 0.95  | 0.70 | ✓      | Explicit out-of-scope: real models, DB, Redis, MLflow, auth, M5 research |
| Constraint Clarity | 0.88  | 0.65 | ✓      | No external services; deterministic; no `.env` in test path; reuse ReorderService |
| Acceptance Criteria| 0.90  | 0.70 | ✓      | 10 pass/fail criteria tied to concrete HTTP/unit assertions |
| **Ambiguity**      | 0.12  | ≤0.20| ✓      | Below gate |

Status: ✓ = met minimum

## Interview Log

| Round | Perspective     | Question summary | Decision locked |
|-------|-----------------|------------------|-----------------|
| 1     | Researcher      | What does `/v1/forecast` do today and why can't it run offline? | Route is a TODO stub returning `[]`; `ForecastService` is empty; models are stubs with no artifacts; `Settings` requires Postgres/Redis/MoH/weather env → not offline-runnable. Seam + deterministic fake needed. |
| 2     | Simplifier      | Minimum to make both routes testable offline? | A `ForecastBackend` Protocol + `DeterministicForecastBackend` fake injected via `Depends`/`dependency_overrides`; reuse existing `ReorderService` + `EnsembleStacker`; in-memory `FakeForecastCache`. No real DB/Redis/MLflow/models. |
| 3     | Boundary Keeper | What is explicitly NOT in this phase? | Real Prophet/LSTM/TFT training + artifacts, TimescaleDB/asyncpg loader, real Redis adapter, MLflow registry, MoH/weather HTTP, auth enforcement, and the `src/` M5 research pipeline. |
| 4     | Failure Analyst | What contract violations must tests catch? | Non-monotone quantiles (p10>p50>p90), missing/zero `latency_ms`, empty `forecasts`, bad `horizon_days`, `model_weights` not summing to 1, reorder ignoring stock/lead-time, and cache not preventing backend re-invocation. |

---
*Phase: demandforecast-offline-tests*
*Spec created: 2026-06-09*
