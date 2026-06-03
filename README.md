# DemandForecast AI — Engine dự báo nhu cầu thuốc

> **Engine 4 / 4 của nền tảng PharmLink AI.**
> Time-series ensemble (statistical + gradient boosting + deep learning) → giảm 40% tồn kho chết, giảm 60% hết hàng đột xuất.

---

## 1. Vấn đề giải quyết

Nhà thuốc Việt Nam đặt hàng theo cảm tính → tồn kho chết 10–15% (thuốc hết hạn = lãng phí 3.000–5.000 tỷ VNĐ/năm toàn ngành). Khi có dịch (sốt xuất huyết, cúm mùa, COVID), nơi thì hết thuốc nơi thì dư — không ai điều phối.

DemandForecast AI dự báo nhu cầu từng SKU theo ngày, kết hợp tín hiệu ngoại sinh (dịch tễ, thời tiết, lễ tết), rồi quy đổi sang **gợi ý đặt hàng tối ưu** (reorder quantity) cho từng nhà thuốc.

## 2. Công nghệ lõi

- **Ensemble nhiều tầng mô hình** (predictions kết hợp bằng learned weights / meta-learner):
  - **Statistical** — Prophet, ETS/Theta, AutoARIMA: bắt seasonality + holidays Việt Nam (Tết, Trung thu, mùa cảm cúm).
  - **Gradient Boosting** — LightGBM, XGBoost (recursive & direct multi-step): bắt quan hệ phi tuyến với covariates.
  - **Deep Learning** — LSTM, N-BEATS, N-HiTS, PatchTST, Temporal Fusion Transformer (TFT): multi-horizon, attention trên chuỗi dài.
- **Tín hiệu ngoại sinh** (covariates):
  - Dữ liệu giám sát dịch của Bộ Y tế (sốt xuất huyết, cúm, tay chân miệng).
  - Dự báo thời tiết (Tổng cục Khí tượng) — nắng nóng → tăng thuốc tiêu hóa.
  - Lịch học, lịch lễ tết — ảnh hưởng nhu cầu thuốc cảm/sốt.
- **Personalization theo location**: mô hình học rằng nhà thuốc gần BV tim mạch bán nhiều thuốc tim, nhà thuốc gần trường có spike thuốc cảm tựu trường.
- **Stack**: Darts (TFT/LSTM/N-HiTS), Prophet, PyTorch Lightning, scikit-learn, LightGBM/XGBoost, FastAPI, PostgreSQL + TimescaleDB, Redis, MLflow tracking.

## 3. KPIs mục tiêu

| Chỉ số | Mục tiêu |
|--------|----------|
| MAPE / WMAPE 30 ngày (top-100 SKU) | ≤ 15% |
| Giảm tồn kho chết | −40% |
| Giảm tỷ lệ hết hàng đột xuất | −60% |
| Latency forecast 1.000 SKU | ≤ 5 giây |

## 4. Tech stack & phiên bản

| Lớp | Thành phần | Phiên bản |
|-----|-----------|-----------|
| Runtime | Python | ≥ 3.11 |
| API | FastAPI / Uvicorn / Pydantic | ≥ 0.115 / ≥ 0.32 / ≥ 2.9 |
| Time-series | Darts / Prophet | ≥ 0.31 / ≥ 1.1.5 |
| Deep learning | PyTorch / PyTorch Lightning | ≥ 2.4 / ≥ 2.4 |
| ML | scikit-learn / LightGBM / XGBoost | ≥ 1.5 / — / — |
| Data | pandas / numpy | ≥ 2.2 / ≥ 1.26 |
| Hạ tầng | PostgreSQL+TimescaleDB / Redis | pg16 / ≥ 5.1 |
| Tracking | MLflow | ≥ 2.17 |
| Logging | structlog | ≥ 24.4 |
| Dev | pytest / pytest-cov / ruff / mypy | ≥ 8.3 / ≥ 5.0 / ≥ 0.7 / ≥ 1.13 |

## 5. Cấu trúc thư mục

```
pharma-demandforecast/
├── src/demandforecast/
│   ├── api/
│   │   ├── main.py            # Khởi tạo FastAPI app
│   │   └── routes/
│   │       ├── forecast.py    # POST /v1/forecast
│   │       ├── reorder.py     # POST /v1/reorder-suggestions
│   │       └── health.py      # GET /health
│   ├── core/
│   │   ├── config.py          # Pydantic-settings, nạp .env
│   │   └── logging.py         # Khởi tạo structlog
│   ├── models/                # Wrapper từng mô hình
│   │   ├── prophet_model.py   # Prophet (seasonality + holidays)
│   │   ├── lstm_model.py      # LSTM (PyTorch Lightning)
│   │   └── tft_model.py       # Temporal Fusion Transformer
│   ├── ensemble/
│   │   └── stacker.py         # EnsembleStacker + EnsembleWeights
│   ├── services/
│   │   ├── forecast_service.py# Orchestrate sales + covariates → ensemble
│   │   └── reorder_service.py # Forecast → reorder qty (lead time, safety stock)
│   ├── data/                  # Loaders (sales, weather, MoH disease feed)
│   └── training/              # Pipeline huấn luyện (Lightning)
├── tests/
│   └── unit/
│       ├── test_ensemble.py
│       └── test_reorder_service.py
├── notebooks/                 # EDA, so sánh mô hình
├── scripts/                   # train.py, backtest.py, seed_sample.py
├── configs/                   # YAML config từng mô hình (xem §10)
├── data/                      # (gitignored) sales/weather/MoH cache
├── docker-compose.yml         # postgres+timescale, redis, mlflow
├── Dockerfile
├── Makefile
├── pyproject.toml
└── .env.example
```

## 6. Khởi chạy nhanh

```bash
cp .env.example .env
make install             # tạo venv, pip install -e ".[dev]"
make services-up         # postgres+timescale, mlflow, redis (docker compose)
make seed-sample         # nạp 6 tháng dữ liệu mẫu
make dev                 # FastAPI tại http://localhost:8004
```

Các target Makefile khác: `make test`, `make lint`, `make format`, `make train`, `make backtest`, `make services-down`.

## 7. Cấu hình (.env)

| Biến | Mục đích | Mặc định |
|------|----------|----------|
| `APP_ENV` | Môi trường (development/production) | development |
| `APP_PORT` | Cổng FastAPI | 8004 |
| `LOG_LEVEL` | Mức log | INFO |
| `POSTGRES_DSN` | Kết nối async TimescaleDB | postgresql+asyncpg://pharma:pharma@localhost:5432/demandforecast |
| `REDIS_URL` | Cache dự báo | redis://localhost:6379/0 |
| `CACHE_TTL_SECONDS` | TTL cache forecast | 600 |
| `MLFLOW_TRACKING_URI` | Server MLflow | http://localhost:5000 |
| `MOH_DISEASE_FEED_URL` | API giám sát dịch Bộ Y tế | — |
| `MOH_API_KEY` | Token xác thực MoH | — |
| `WEATHER_API_URL` | API thời tiết | — |
| `WEATHER_API_KEY` | Token thời tiết | — |
| `INTERNAL_API_TOKEN` | Secret xác thực giữa các service | — |
| `DEFAULT_HORIZON_DAYS` | Horizon dự báo mặc định | 30 |
| `DEFAULT_QUANTILES` | Các mức quantile | 0.1,0.5,0.9 |

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
Trả về gợi ý đặt hàng tối ưu dựa trên forecast (p50) + safety stock + lead time của nhà phân phối.

### `GET /health`
Liveness/readiness probe.

## 9. Testing

```bash
make test                # pytest + coverage (ngưỡng 80%)
make lint                # ruff check + mypy
make format              # ruff format + --fix
```

- `tests/unit/test_ensemble.py` — chuẩn hóa trọng số & logic kết hợp của `EnsembleStacker`.
- `tests/unit/test_reorder_service.py` — công thức tính reorder quantity.

## 10. ML training & research (hợp nhất từ nhánh `kaggle`)

Toàn bộ harness benchmark & huấn luyện mô hình (trước đây ở nhánh `kaggle`) đã được hợp nhất vào `main`. Đây là pipeline nghiên cứu chạy trên benchmark forecasting **M5** (30.490 chuỗi × 1.913 ngày).

**Danh mục mô hình** (`src/models/`):

| Tầng | Mô hình | File |
|------|---------|------|
| Baseline | Naive (last-value, seasonal), Croston/SBA | `models/baseline/naive.py`, `models/baseline/croston.py` |
| Statistical | ETS, Theta, AutoARIMA | `models/statistical/ets.py` |
| Gradient Boosting | LightGBM, XGBoost (recursive + direct) | `models/ml/lgbm.py`, `models/ml/xgb.py`, `models/ml/_rollout.py` |
| Deep Learning | N-BEATS, N-HiTS, TFT, PatchTST | `models/deep/{nbeats,nhits,tft,patchtst}.py` |

**Pipeline** (`src/train.py`, config-driven qua `configs/*.yaml`):

```bash
# Tiền xử lý dữ liệu M5 → long format + features
make preprocess

# Huấn luyện theo config
python src/train.py --config configs/lgbm_recursive.yaml
python src/train.py --config configs/nhits.yaml

# Đánh giá lại từ run đã có
python src/train.py --config configs/ensemble_stack.yaml --eval-only --run-id <id>
```

- **Splits**: TRAIN_END=1857, VAL1=1858–1885, VAL2=1886–1913.
- **Features** (`src/data/features.py`): lag (7/14/28/35/42), rolling mean/std, day-of-week, month, holiday flags, price change.
- **Đánh giá** (`src/metrics/evaluate.py`): **WMAPE** (weighted MAPE theo sản lượng) và **RMSSE** (M5 12-level hierarchy weights).
- **Ensemble** (`configs/ensemble_stack.yaml`): stacking với meta-learner Ridge trên OOF predictions.
- **Tracking**: MLflow log hyperparams + metrics + git hash; kết quả lưu `results/<run_id>/{metrics.json,predictions.parquet}`.

## 11. Tích hợp ngoại sinh

| Nguồn | Sử dụng | Tần suất |
|-------|---------|----------|
| Hệ thống giám sát dịch Bộ Y tế | Cảnh báo dịch theo tỉnh | Hàng ngày |
| Tổng cục Khí tượng | Nhiệt độ, độ ẩm, mùa | Hàng giờ |
| Lịch lễ Việt Nam | Tết, lễ truyền thống | Hàng năm (cố định) |
| Bán hàng các nhà thuốc | Sales history (federated) | Realtime |

## 12. Docker

```bash
docker compose up -d                 # postgres+timescale, redis, mlflow
docker build -t demandforecast .     # build API image
```

## 13. Roadmap

- **v0.1**: Prophet baseline, MAPE 25%.
- **v0.2**: Ensemble nhiều mô hình, MAPE 15%.
- **v1.0**: Federated learning across 1.000+ pharmacies.
