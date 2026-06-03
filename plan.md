# Pharma Demand Forecast — Model Benchmark Plan

## Goal

Benchmark multiple forecasting model families on the M5 dataset and identify the best architecture(s) on two metrics:
- **WMAPE** (Weighted Mean Absolute Percentage Error)
- **RMSSE** (Root Mean Squared Scaled Error — the official M5 metric)

The repo must be clean, reproducible, and experiment-trackable.

---

## Dataset: M5 Forecasting Accuracy

Source: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

### Files

| File | Description |
|---|---|
| `sales_train_validation.csv` | Daily unit sales, 30,490 series × 1,913 days |
| `sales_train_evaluation.csv` | Same + 28 extra days (used for final eval) |
| `calendar.csv` | Date features, events, SNAP flags |
| `sell_prices.csv` | Weekly price per item × store |
| `sample_submission.csv` | Submission format reference |

### Key Properties

- **Series:** 30,490 (item × store), hierarchical across 12 levels
- **Horizon:** 28 days
- **Granularity:** Daily
- **Intermittency:** High (many zero-sales days, especially slow-moving items)
- **Covariates:** Prices, calendar events (holidays, SNAP), weekday

---

## Repository Structure

```
Pharma-DemandForecast/
├── data/
│   ├── raw/                  # Kaggle CSVs — never modified
│   └── processed/            # Featurized / pivoted outputs
├── configs/
│   └── <model_name>.yaml     # One config per experiment
├── src/
│   ├── data/
│   │   ├── download.py       # kaggle API download script
│   │   ├── preprocess.py     # Melt, merge calendar & prices
│   │   └── features.py       # Lag, rolling, calendar features
│   ├── metrics/
│   │   └── evaluate.py       # WMAPE + RMSSE implementations
│   ├── models/
│   │   ├── baseline/
│   │   │   ├── naive.py      # Last-value & seasonal naive
│   │   │   └── croston.py    # Croston / SBA for intermittent demand
│   │   ├── statistical/
│   │   │   └── ets.py        # ExponentialSmoothing (statsmodels)
│   │   ├── ml/
│   │   │   ├── lgbm.py       # LightGBM recursive / direct
│   │   │   └── xgb.py        # XGBoost
│   │   └── deep/
│   │       ├── nbeats.py     # N-BEATS (neuralforecast)
│   │       ├── nhits.py      # N-HiTS
│   │       ├── tft.py        # Temporal Fusion Transformer
│   │       └── patchtst.py   # PatchTST
│   └── train.py              # Unified train+eval entrypoint
├── notebooks/
│   ├── 00_eda.ipynb          # EDA: distributions, zeros, hierarchy
│   └── 01_results.ipynb      # Results comparison table & plots
├── results/
│   └── <run_id>/             # metrics.json + predictions.parquet per run
├── requirements.txt
├── Makefile                  # Shortcut targets
├── plan.md                   # This file
└── README.md
```

---

## Metrics Implementation

### WMAPE

```
WMAPE = sum(|y - ŷ|) / sum(|y|)
```

Computed at the **item-level** then aggregated by weighting each series by its total sales volume (same as M5 level-12 aggregation).

### RMSSE

```
RMSSE = sqrt( mean((y - ŷ)²) / h⁻¹ · mean(diff(y_train)²) )
```

The denominator uses the in-sample naive (random walk) scale. Computed per series then averaged with sales-volume weights across all 12 hierarchy levels. Reference: M5 competition guide.

Both are implemented in `src/metrics/evaluate.py` with unit tests.

---

## Models to Benchmark

### Tier 1 — Baselines (must beat these to be meaningful)

| Model | Strategy | Notes |
|---|---|---|
| Naive (last value) | Carry forward last observed value | Lower bound |
| Seasonal Naive | Repeat same weekday from last week | Captures weekly seasonality |
| Croston / SBA | For intermittent (zero-heavy) series | Common pharma baseline |

### Tier 2 — Statistical

| Model | Library | Notes |
|---|---|---|
| ETS (Error-Trend-Season) | `statsmodels` | Per-series, auto model selection |
| Theta | `statsforecast` | Strong M-competition performer |
| AutoARIMA | `statsforecast` | Fitted per series in parallel |

### Tier 3 — Gradient Boosting (Global Models)

| Model | Strategy | Notes |
|---|---|---|
| LightGBM | Recursive multi-step | Single global model across all series |
| LightGBM | Direct (28 separate models) | Avoids error accumulation |
| XGBoost | Recursive | Comparison point |

Features used: lags (7, 14, 28, 35, 42), rolling means/stds, day-of-week, month, event flags, snap flags, sell price & price change.

### Tier 4 — Deep Learning (Global Models)

| Model | Library | Notes |
|---|---|---|
| N-BEATS | `neuralforecast` | Interpretable blocks, strong on M4/M5 |
| N-HiTS | `neuralforecast` | Hierarchical interpolation, long-horizon |
| TFT | `pytorch-forecasting` | Attention + covariates |
| PatchTST | `neuralforecast` | Patch-based transformer, SOTA on many benchmarks |

### Tier 5 — Ensembles

| Strategy | Description |
|---|---|
| Simple average | Mean of top-3 Tier 3/4 predictions |
| Stacking | Linear regression meta-learner on OOF predictions |
| Reconciliation | Bottom-up / MinT reconciliation across hierarchy levels |

---

## Experimental Protocol

### Train / Validation Split

- **Training:** Days 1–1,857 (full history minus last 56 days)
- **Validation fold 1:** Days 1,858–1,885 (28 days)
- **Validation fold 2:** Days 1,886–1,913 (28 days) — held-out, evaluated last
- Final evaluation uses `sales_train_evaluation.csv` for Kaggle submission

This mirrors the M5 rolling-window evaluation and tests generalization across two periods.

### Reproducibility Requirements

- All random seeds set via config YAML (`seed: 42`)
- Data download script uses kaggle CLI (pinned credentials via env var)
- `requirements.txt` pins all library versions
- Each experiment logs: run_id, config path, git commit hash, WMAPE, RMSSE, runtime
- Results written to `results/<run_id>/metrics.json`

### Experiment Tracking

Use **MLflow** (local tracking server, no cloud needed):

```bash
mlflow ui --port 5000
```

Each `train.py` run auto-logs params, metrics, and artifacts.

---

## Implementation Phases

### Phase 0 — Scaffold (Day 1)
- [ ] Set up repo structure (dirs, `__init__.py`, `.gitignore`)
- [ ] `requirements.txt` with pinned versions
- [ ] `Makefile` with targets: `download`, `preprocess`, `train`, `eval`, `notebook`
- [ ] Implement and unit-test `evaluate.py` (WMAPE + RMSSE)

### Phase 1 — Data Pipeline (Day 1–2)
- [ ] `download.py`: pull M5 data via kaggle CLI
- [ ] `preprocess.py`: melt wide→long, merge calendar, merge prices, handle missing prices
- [ ] `features.py`: lag features, rolling stats, date features, event/SNAP dummies
- [ ] EDA notebook: zero-ratio by series, sales distributions, hierarchy counts, price variability

### Phase 2 — Baselines (Day 2)
- [ ] Naive, Seasonal Naive, Croston/SBA
- [ ] Evaluate on both validation folds
- [ ] Log to MLflow

### Phase 3 — Statistical Models (Day 3)
- [ ] ETS, Theta, AutoARIMA via `statsforecast` (vectorized, fast)
- [ ] Evaluate + log

### Phase 4 — Gradient Boosting (Day 4–5)
- [ ] Feature matrix construction (all series, all timesteps)
- [ ] LightGBM recursive (28-step rollout)
- [ ] LightGBM direct (28 separate models)
- [ ] Hyperparameter sweep via config YAML + Optuna (optional)
- [ ] Evaluate + log

### Phase 5 — Deep Learning (Day 6–9)
- [ ] N-BEATS and N-HiTS (faster to train, good starting point)
- [ ] TFT (add covariates: price, events)
- [ ] PatchTST
- [ ] Evaluate + log

### Phase 6 — Ensembles (Day 10)
- [ ] Simple average ensemble of top models
- [ ] Stacking meta-learner on OOF preds
- [ ] Bottom-up hierarchical reconciliation
- [ ] Evaluate + log

### Phase 7 — Analysis & Write-up (Day 11)
- [ ] `01_results.ipynb`: leaderboard table, error distributions per model family, WMAPE vs RMSSE scatter, best model deep-dive
- [ ] Update README with setup instructions and results table

---

## Configuration Format

Each model run is driven by a YAML config, e.g. `configs/lgbm_recursive.yaml`:

```yaml
model: lgbm
strategy: recursive
seed: 42
horizon: 28
features:
  lags: [7, 14, 28, 35, 42]
  rolling_windows: [7, 28]
  calendar: true
  price: true
lgbm_params:
  n_estimators: 1000
  learning_rate: 0.05
  num_leaves: 127
  subsample: 0.8
  colsample_bytree: 0.8
```

Run with:

```bash
python src/train.py --config configs/lgbm_recursive.yaml
```

---

## Key Dependencies

```
lightgbm>=4.3
xgboost>=2.0
statsforecast>=1.7
neuralforecast>=1.7
pytorch-forecasting>=1.1
mlflow>=2.12
pandas>=2.2
pyarrow>=15
scikit-learn>=1.4
optuna>=3.6
kaggle>=1.6
```

---

## Definition of Done

A model iteration is considered complete when:
1. WMAPE and RMSSE are computed on **both** validation folds
2. Results are logged to MLflow with the config and git hash
3. Predictions are saved as `results/<run_id>/predictions.parquet`
4. The run can be reproduced from scratch with `make train CONFIG=<path>`
