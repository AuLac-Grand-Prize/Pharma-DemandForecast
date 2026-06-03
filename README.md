# DemandForecast AI — Drug demand-forecasting engine

> **Engine 4 / 4 of the PharmLink AI platform.**
> Time-series ensemble (statistical + gradient boosting + deep learning) → 40% less dead stock, 60% fewer sudden stockouts.

---

## 1. Problem it solves

Vietnamese pharmacies order on gut feel → 10–15% dead stock (expired drugs = an estimated 3,000–5,000 billion VND wasted per year industry-wide). When an outbreak hits (dengue, seasonal flu, COVID), some stores run out while others overstock — with no one coordinating.

DemandForecast AI forecasts demand per SKU per day, combines exogenous signals (epidemiology, weather, holidays), and converts them into **optimal reorder suggestions** (reorder quantity) for each pharmacy.

## 2. Core technology

- **Multi-tier model ensemble** (predictions combined via learned weights / a meta-learner):
  - **Statistical** — Prophet, ETS/Theta, AutoARIMA: capture seasonality + Vietnamese holidays (Tết, Mid-Autumn, flu season).
  - **Gradient Boosting** — LightGBM, XGBoost (recursive & direct multi-step): capture nonlinear relationships with covariates.
  - **Deep Learning** — LSTM, N-BEATS, N-HiTS, PatchTST, Temporal Fusion Transformer (TFT): multi-horizon, attention over long sequences.
- **Exogenous signals** (covariates):
  - Ministry of Health disease-surveillance data (dengue, flu, hand-foot-mouth).
  - Weather forecasts (National Hydrometeorological Service) — heat waves → more digestive medicine.
  - School calendars and public holidays — affect demand for cold/fever medicine.
- **Location-based personalization**: the model learns that a pharmacy near a cardiology hospital sells more heart medication, and one near a school spikes in cold medicine at the start of term.
- **Stack**: Darts (TFT/LSTM/N-HiTS), Prophet, PyTorch Lightning, scikit-learn, LightGBM/XGBoost, FastAPI, PostgreSQL + TimescaleDB, Redis, MLflow tracking.

## 3. Target KPIs

| Metric | Target |
|--------|--------|
| 30-day MAPE / WMAPE (top-100 SKUs) | ≤ 15% |
| Reduction in dead stock | −40% |
| Reduction in sudden stockouts | −60% |
| Forecast latency for 1,000 SKUs | ≤ 5 s |

## 4. Tech stack & versions

| Layer | Component | Version |
|-------|-----------|---------|
| Runtime | Python | ≥ 3.11 |
| API | FastAPI / Uvicorn / Pydantic | ≥ 0.115 / ≥ 0.32 / ≥ 2.9 |
| Time-series | Darts / Prophet | ≥ 0.31 / ≥ 1.1.5 |
| Deep learning | PyTorch / PyTorch Lightning | ≥ 2.4 / ≥ 2.4 |
| ML | scikit-learn / LightGBM / XGBoost | ≥ 1.5 / — / — |
| Data | pandas / numpy | ≥ 2.2 / ≥ 1.26 |
| Infrastructure | PostgreSQL+TimescaleDB / Redis | pg16 / ≥ 5.1 |
| Tracking | MLflow | ≥ 2.17 |
| Logging | structlog | ≥ 24.4 |
| Dev | pytest / pytest-cov / ruff / mypy | ≥ 8.3 / ≥ 5.0 / ≥ 0.7 / ≥ 1.13 |

## 5. Directory structure

```
pharma-demandforecast/
├── src/demandforecast/
│   ├── api/
│   │   ├── main.py            # FastAPI app bootstrap
│   │   └── routes/
│   │       ├── forecast.py    # POST /v1/forecast
│   │       ├── reorder.py     # POST /v1/reorder-suggestions
│   │       └── health.py      # GET /health
│   ├── core/
│   │   ├── config.py          # Pydantic-settings, loads .env
│   │   └── logging.py         # structlog bootstrap
│   ├── models/                # Per-model wrappers
│   │   ├── prophet_model.py   # Prophet (seasonality + holidays)
│   │   ├── lstm_model.py      # LSTM (PyTorch Lightning)
│   │   └── tft_model.py       # Temporal Fusion Transformer
│   ├── ensemble/
│   │   └── stacker.py         # EnsembleStacker + EnsembleWeights
│   ├── services/
│   │   ├── forecast_service.py# Orchestrate sales + covariates → ensemble
│   │   └── reorder_service.py # Forecast → reorder qty (lead time, safety stock)
│   ├── data/                  # Loaders (sales, weather, MoH disease feed)
│   └── training/              # Training pipeline (Lightning)
├── tests/
│   └── unit/
│       ├── test_ensemble.py
│       └── test_reorder_service.py
├── notebooks/                 # EDA, model comparison
├── scripts/                   # train.py, backtest.py, seed_sample.py
├── configs/                   # Per-model YAML configs (see §10)
├── data/                      # (gitignored) sales/weather/MoH cache
├── docker-compose.yml         # postgres+timescale, redis, mlflow
├── Dockerfile
├── Makefile
├── pyproject.toml
└── .env.example
```

## 6. Quick start

```bash
cp .env.example .env
make install             # create venv, pip install -e ".[dev]"
make services-up         # postgres+timescale, mlflow, redis (docker compose)
make seed-sample         # load 6 months of sample data
make dev                 # FastAPI at http://localhost:8004
```

Other Makefile targets: `make test`, `make lint`, `make format`, `make train`, `make backtest`, `make services-down`.

## 7. Configuration (.env)

| Variable | Purpose | Default |
|----------|---------|---------|
| `APP_ENV` | Environment (development/production) | development |
| `APP_PORT` | FastAPI port | 8004 |
| `LOG_LEVEL` | Log level | INFO |
| `POSTGRES_DSN` | Async TimescaleDB connection | postgresql+asyncpg://pharma:pharma@localhost:5432/demandforecast |
| `REDIS_URL` | Forecast cache | redis://localhost:6379/0 |
| `CACHE_TTL_SECONDS` | Forecast cache TTL | 600 |
| `MLFLOW_TRACKING_URI` | MLflow server | http://localhost:5000 |
| `MOH_DISEASE_FEED_URL` | Ministry of Health surveillance API | — |
| `MOH_API_KEY` | MoH auth token | — |
| `WEATHER_API_URL` | Weather API | — |
| `WEATHER_API_KEY` | Weather token | — |
| `INTERNAL_API_TOKEN` | Inter-service auth secret | — |
| `DEFAULT_HORIZON_DAYS` | Default forecast horizon | 30 |
| `DEFAULT_QUANTILES` | Quantile levels | 0.1,0.5,0.9 |

## 8. API contract

### `POST /v1/forecast`
Body:
```json
{
  "pharmacy_id": "uuid",
  "sku_ids": ["sku-001", "sku-002"],
  "horizon_days": 30,
  "include_covariates": true
}
```
Response:
```json
{
  "forecasts": [
    {
      "sku_id": "sku-001",
      "drug_name": "Paracetamol 500mg",
      "daily_forecast": [
        {"date": "2026-05-01", "p10": 12, "p50": 18, "p90": 25}
      ],
      "trend": "increasing",
      "seasonality_signal": "flu_season_start",
      "model_weights": {"prophet": 0.3, "lstm": 0.4, "tft": 0.3}
    }
  ],
  "generated_at": "2026-04-27T15:30:00+07:00",
  "latency_ms": 1820
}
```

### `POST /v1/reorder-suggestions`
Returns optimal reorder suggestions based on the forecast (p50) + safety stock + distributor lead time.

### `GET /health`
Liveness/readiness probe.

## 9. Testing

```bash
make test                # pytest + coverage (80% threshold)
make lint                # ruff check + mypy
make format              # ruff format + --fix
```

- `tests/unit/test_ensemble.py` — weight normalization & combination logic of `EnsembleStacker`.
- `tests/unit/test_reorder_service.py` — reorder-quantity formula.

## 10. ML training & research (merged from the `kaggle` branch)

The full model-benchmarking & training harness (previously on the `kaggle` branch) has been merged into `main`. This research pipeline runs on the **M5** forecasting benchmark (30,490 series × 1,913 days).

**Model catalog** (`src/models/`):

| Tier | Models | Files |
|------|--------|-------|
| Baseline | Naive (last-value, seasonal), Croston/SBA | `models/baseline/naive.py`, `models/baseline/croston.py` |
| Statistical | ETS, Theta, AutoARIMA | `models/statistical/ets.py` |
| Gradient Boosting | LightGBM, XGBoost (recursive + direct) | `models/ml/lgbm.py`, `models/ml/xgb.py`, `models/ml/_rollout.py` |
| Deep Learning | N-BEATS, N-HiTS, TFT, PatchTST | `models/deep/{nbeats,nhits,tft,patchtst}.py` |

**Pipeline** (`src/train.py`, config-driven via `configs/*.yaml`):

```bash
# Preprocess M5 data → long format + features
make research-preprocess

# Train by config (default configs/lgbm_recursive.yaml)
make research-train CONFIG=configs/lgbm_recursive.yaml
make research-train CONFIG=configs/nhits.yaml

# Re-evaluate from an existing run
make research-eval CONFIG=configs/ensemble_stack.yaml RUN_ID=<id>
```

- **Splits**: TRAIN_END=1857, VAL1=1858–1885, VAL2=1886–1913.
- **Features** (`src/data/features.py`): lags (7/14/28/35/42), rolling mean/std, day-of-week, month, holiday flags, price change.
- **Evaluation** (`src/metrics/evaluate.py`): **WMAPE** (volume-weighted MAPE) and **RMSSE** (M5 12-level hierarchy weights).
- **Ensemble** (`configs/ensemble_stack.yaml`): stacking with a Ridge meta-learner over OOF predictions.
- **Tracking**: MLflow logs hyperparams + metrics + git hash; results saved to `results/<run_id>/{metrics.json,predictions.parquet}`.

## 11. Exogenous integrations

| Source | Use | Frequency |
|--------|-----|-----------|
| Ministry of Health surveillance system | Province-level outbreak alerts | Daily |
| National Hydrometeorological Service | Temperature, humidity, season | Hourly |
| Vietnamese holiday calendar | Tết, traditional holidays | Annual (fixed) |
| Pharmacy sales | Sales history (federated) | Real-time |

## 12. Docker

```bash
docker compose up -d                 # postgres+timescale, redis, mlflow
docker build -t demandforecast .     # build API image
```

## 13. Roadmap

- **v0.1**: Prophet baseline, MAPE 25%.
- **v0.2**: Multi-model ensemble, MAPE 15%.
- **v1.0**: Federated learning across 1,000+ pharmacies.

---

## About PharmLink AI

This repository is **Engine 4 / 4** of [**PharmLink AI**](https://github.com/AuLac-Grand-Prize) — Vietnam's *Made-in-Vietnam* pharmaceutical AI platform serving 60,000+ pharmacies and up to 100 million citizens, in service of medication safety and national health-data sovereignty.

**The platform:**
- 💊 [VietDrug AI](https://github.com/AuLac-Grand-Prize/Pharma-VietDrugAI) — drug-interaction checks
- 📝 [PrescriptionVision](https://github.com/AuLac-Grand-Prize/Pharma-PrescriptionVision) — handwritten-prescription OCR
- 🤖 [PharmaGPT-VN](https://github.com/AuLac-Grand-Prize/PharmaGPT-VN) — Vietnamese pharma assistant
- 📈 **DemandForecast AI** — demand forecasting *(this repo)*
- 🖥️ [Pharma Portal](https://github.com/AuLac-Grand-Prize/Pharma-Portal) — the pharmacist workspace

**Technology ownership:** the forecasting models are trained entirely on Vietnamese market data, capturing local seasonality, outbreak patterns, and medication-usage habits — combined with real-time MoH epidemiology and weather signals.

> **Disclaimer:** PharmLink AI augments pharmacists — it does not replace them. Every clinical decision rests with a licensed pharmacist; outputs are validated by the Vietnamese Clinical Pharmacist Scientific Council.
