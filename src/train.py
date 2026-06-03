"""Unified train + evaluate entrypoint for all forecasting models.

Usage:
    python src/train.py --config configs/lgbm_recursive.yaml
    python src/train.py --config configs/naive_last.yaml --debug
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ensure repo root is on sys.path when running as `python src/train.py`
_repo_root = str(_Path(__file__).parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import argparse
import gc
import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path("results")

TRAIN_END = 1857
VAL1_START, VAL1_END = 1858, 1885
VAL2_START, VAL2_END = 1886, 1913


# ---------------------------------------------------------------------------
# CLI & config
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate a forecasting model.")
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate.")
    p.add_argument("--run-id", default=None, help="Existing run ID (for --eval-only).")
    p.add_argument("--debug", action="store_true", help="Use only 500 series for fast iteration.")
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = ["model", "horizon"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required key: '{key}'")
    cfg.setdefault("seed", 42)
    cfg.setdefault("strategy", "")
    return cfg


def _flatten_config(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested dict for MLflow param logging."""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_config(v, full_key))
        else:
            out[full_key] = v
    return out


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_or_preprocess(config: dict, debug: bool = False) -> pd.DataFrame:
    feat_path = PROCESSED_DIR / "long_with_features.parquet"
    if not feat_path.exists():
        long_path = PROCESSED_DIR / "long.parquet"
        if not long_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {long_path}. "
                "Run: make preprocess  (or make download && make preprocess)"
            )
        logger.info("Building feature matrix from %s …", long_path)
        from src.data.features import build_features
        build_features(in_path=long_path, out_path=feat_path, config=config)

    logger.info("Loading features from %s …", feat_path)
    df = pd.read_parquet(feat_path)

    if debug:
        ids = df["id"].unique()[:500]
        df = df[df["id"].isin(ids)].copy()
        logger.info("DEBUG mode: using %d series.", len(ids))

    return df


# ---------------------------------------------------------------------------
# Nixtla format helper
# ---------------------------------------------------------------------------

def to_nixtla_format(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convert long DataFrame to Nixtla (unique_id, ds, y[, covariates])."""
    feat_cfg = config.get("features", {})
    out = df[["id", "date", "sales"]].rename(columns={"id": "unique_id", "date": "ds", "sales": "y"})

    # Add historical covariates if requested
    hist_cols = config.get("hist_exog_list", [])
    for col in hist_cols:
        if col in df.columns:
            out = out.copy()
            out[col] = df[col].values

    # Add future covariates
    futr_cols = config.get("futr_exog_list", [])
    for col in futr_cols:
        if col in df.columns:
            out = out.copy()
            out[col] = df[col].values

    out["ds"] = pd.to_datetime(out["ds"])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(config: dict) -> Any:
    model_name = config["model"].lower()
    strategy = config.get("strategy", "").lower()

    if model_name == "naive":
        from src.models.baseline.naive import LastValueNaive, SeasonalNaive
        if strategy == "seasonal":
            m = SeasonalNaive(seasonal_period=config.get("seasonal_period", 7))
        else:
            m = LastValueNaive()
        return m

    if model_name == "croston":
        from src.models.baseline.croston import forecast_croston_all_series
        return _CrostonWrapper(config)

    if model_name in ("ets", "theta", "arima", "autoarima"):
        from src.models.statistical.ets import StatisticalForecaster
        model_key = "arima" if model_name == "autoarima" else model_name
        return StatisticalForecaster(
            models=[model_key],
            freq=config.get("freq", "D"),
            n_jobs=config.get("n_jobs", -1),
        )

    if model_name == "lgbm":
        from src.models.ml.lgbm import LightGBMForecaster
        return LightGBMForecaster(
            params=config.get("lgbm_params", {}),
            strategy=strategy or "recursive",
            horizon=config["horizon"],
            seed=config["seed"],
        )

    if model_name == "xgb":
        from src.models.ml.xgb import XGBoostForecaster
        return XGBoostForecaster(
            params=config.get("xgb_params", {}),
            horizon=config["horizon"],
            seed=config["seed"],
        )

    if model_name == "nbeats":
        from src.models.deep.nbeats import NBEATSForecaster
        return NBEATSForecaster(
            h=config["horizon"],
            input_size=config.get("input_size", 91),
            stack_types=config.get("stack_types", ["trend", "seasonality"]),
            n_harmonics=config.get("n_harmonics", 2),
            n_polynomials=config.get("n_polynomials", 2),
            max_steps=config.get("max_steps", 1000),
            batch_size=config.get("batch_size", 32),
            learning_rate=config.get("learning_rate", 1e-3),
            seed=config["seed"],
            accelerator=config.get("accelerator", "auto"),
            max_series=config.get("max_series"),
        )

    if model_name == "nhits":
        from src.models.deep.nhits import NHiTSForecaster
        return NHiTSForecaster(
            h=config["horizon"],
            input_size=config.get("input_size", 91),
            n_pool_kernel_size=config.get("n_pool_kernel_size", [2, 2, 1]),
            max_steps=config.get("max_steps", 1000),
            batch_size=config.get("batch_size", 32),
            learning_rate=config.get("learning_rate", 1e-3),
            seed=config["seed"],
            accelerator=config.get("accelerator", "auto"),
            max_series=config.get("max_series"),
        )

    if model_name == "tft":
        from src.models.deep.tft import TFTForecaster
        return TFTForecaster(
            h=config["horizon"],
            input_size=config.get("input_size", 91),
            hidden_size=config.get("hidden_size", 128),
            n_head=config.get("n_head", 4),
            dropout=config.get("dropout", 0.1),
            hist_exog_list=config.get("hist_exog_list", []),
            futr_exog_list=config.get("futr_exog_list", []),
            max_steps=config.get("max_steps", 500),
            batch_size=config.get("batch_size", 32),
            learning_rate=config.get("learning_rate", 1e-3),
            seed=config["seed"],
            accelerator=config.get("accelerator", "auto"),
            max_series=config.get("max_series"),
        )

    if model_name == "patchtst":
        from src.models.deep.patchtst import PatchTSTForecaster
        return PatchTSTForecaster(
            h=config["horizon"],
            input_size=config.get("input_size", 104),
            patch_len=config.get("patch_len", 16),
            stride=config.get("stride", 8),
            d_model=config.get("d_model", 128),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 256),
            max_steps=config.get("max_steps", 500),
            batch_size=config.get("batch_size", 32),
            learning_rate=config.get("learning_rate", 1e-3),
            seed=config["seed"],
            accelerator=config.get("accelerator", "auto"),
            max_series=config.get("max_series"),
        )

    if model_name == "ensemble":
        return _EnsembleWrapper(config)

    raise ValueError(
        f"Unknown model '{model_name}'. Valid options: naive, croston, ets, theta, arima, "
        "lgbm, xgb, nbeats, nhits, tft, patchtst, ensemble."
    )


# ---------------------------------------------------------------------------
# Model wrappers for non-standard interfaces
# ---------------------------------------------------------------------------

class _CrostonWrapper:
    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, train_wide: pd.DataFrame, h: int) -> pd.DataFrame:
        from src.models.baseline.croston import forecast_croston_all_series
        return forecast_croston_all_series(
            train_wide,
            h=h,
            variant=self.config.get("strategy", "sba"),
            alpha=self.config.get("alpha", 0.1),
        )


class _EnsembleWrapper:
    def __init__(self, config: dict) -> None:
        self.config = config

    def run(self, run_ids: list[str], val: str) -> pd.DataFrame:
        from sklearn.linear_model import Ridge
        strategy = self.config.get("strategy", "average")
        pred_dfs = []
        for rid in run_ids:
            p = RESULTS_DIR / rid / f"predictions_{val}.parquet"
            pred_dfs.append(pd.read_parquet(p))

        if strategy == "average":
            base = pred_dfs[0].copy()
            num_cols = [c for c in base.columns if c not in ("id",)]
            for df in pred_dfs[1:]:
                base[num_cols] = base[num_cols].values + df[num_cols].values
            base[num_cols] = base[num_cols] / len(pred_dfs)
            return base

        # Stacking: requires OOF predictions on val1 to train meta-learner
        raise NotImplementedError("Stacking ensemble: load val1 OOF preds and fit Ridge on them.")


# ---------------------------------------------------------------------------
# Prediction helpers per model family
# ---------------------------------------------------------------------------

def _predict_naive(model: Any, df: pd.DataFrame, config: dict, fold: str) -> pd.DataFrame:
    """Get naive predictions for a given fold."""
    lo, hi = (VAL1_START, VAL1_END) if fold == "val1" else (VAL2_START, VAL2_END)
    # Build wide training matrix (all data before the fold)
    train_df = df[df["d"] <= lo - 1]
    day_cols = sorted(train_df["d"].unique())
    wide = (
        train_df.pivot(index="id", columns="d", values="sales")
        .reset_index()
    )
    wide.columns = ["id"] + [f"d_{c}" for c in wide.columns[1:]]

    h = config["horizon"]
    if isinstance(model, LastValueNaive.__class__ if False else type(model)):
        pass  # handled below

    from src.models.baseline.naive import forecast_naive_all_series
    strategy = config.get("strategy", "last")
    preds_wide = forecast_naive_all_series(
        wide, h=h, model=strategy,
        seasonal_period=config.get("seasonal_period", 7),
    )
    return preds_wide


_WIDE_CACHE: dict[int, pd.DataFrame] = {}


def _make_train_wide(df: pd.DataFrame, max_d: int) -> pd.DataFrame:
    if max_d in _WIDE_CACHE:
        return _WIDE_CACHE[max_d]
    train_df = df[df["d"] <= max_d]
    wide = (
        train_df.pivot(index="id", columns="d", values="sales")
        .reset_index()
    )
    wide.columns = ["id"] + [f"d_{c}" for c in wide.columns[1:]]
    _WIDE_CACHE[max_d] = wide
    return wide


def _wide_to_long(wide_preds: pd.DataFrame, start_date: pd.Timestamp, h: int) -> pd.DataFrame:
    """Convert wide forecast to long format with (id, date, yhat) columns."""
    dates = pd.date_range(start=start_date, periods=h, freq="D")
    id_vals = wide_preds["id"].values
    pred_cols = [c for c in wide_preds.columns if c != "id"]
    long_records = []
    for i, uid in enumerate(id_vals):
        vals = wide_preds.iloc[i][pred_cols].values
        for j, v in enumerate(vals[:h]):
            long_records.append({"id": uid, "date": dates[j], "yhat": float(v)})
    return pd.DataFrame(long_records)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: dict,
    df: pd.DataFrame,
    run_id: str,
    config_path: Path,
) -> dict[str, float]:
    model_name = config["model"].lower()
    h = config["horizon"]
    seed = config["seed"]

    # --- Split data ---
    train_df = df[df["d"] <= TRAIN_END].copy()
    val1_df = df[(df["d"] >= VAL1_START) & (df["d"] <= VAL1_END)].copy()
    val2_df = df[(df["d"] >= VAL2_START) & (df["d"] <= VAL2_END)].copy()

    train_wide = _make_train_wide(df, TRAIN_END)

    # Calendar for date-mapping
    cal = df[["d", "date"]].drop_duplicates().sort_values("d")
    d_to_date = dict(zip(cal["d"], pd.to_datetime(cal["date"])))
    val1_start_date = d_to_date.get(VAL1_START, pd.Timestamp("2016-04-25"))
    val2_start_date = d_to_date.get(VAL2_START, pd.Timestamp("2016-05-23"))

    preds_val1: pd.DataFrame
    preds_val2: pd.DataFrame

    # -----------------------------------------------------------------------
    if model_name == "naive":
        strategy = config.get("strategy", "last")
        from src.models.baseline.naive import forecast_naive_all_series
        preds_v1_wide = forecast_naive_all_series(train_wide, h=h, model=strategy,
                                                  seasonal_period=config.get("seasonal_period", 7))
        preds_val1 = _wide_to_long(preds_v1_wide, val1_start_date, h)
        # For val2: train on train + val1
        wide_tv1 = _make_train_wide(df, VAL1_END)
        preds_v2_wide = forecast_naive_all_series(wide_tv1, h=h, model=strategy,
                                                  seasonal_period=config.get("seasonal_period", 7))
        preds_val2 = _wide_to_long(preds_v2_wide, val2_start_date, h)

    # -----------------------------------------------------------------------
    elif model_name == "croston":
        wrapper = _CrostonWrapper(config)
        preds_v1_wide = wrapper.run(train_wide, h)
        preds_val1 = _wide_to_long(preds_v1_wide, val1_start_date, h)
        wide_tv1 = _make_train_wide(df, VAL1_END)
        preds_v2_wide = wrapper.run(wide_tv1, h)
        preds_val2 = _wide_to_long(preds_v2_wide, val2_start_date, h)

    # -----------------------------------------------------------------------
    elif model_name in ("ets", "theta", "arima", "autoarima"):
        from src.models.statistical.ets import StatisticalForecaster
        model_key = "arima" if model_name == "autoarima" else model_name
        sf = StatisticalForecaster(
            models=[model_key],
            freq=config.get("freq", "D"),
            n_jobs=config.get("n_jobs", -1),
        )
        train_nixtla = to_nixtla_format(train_df, config)
        sf_preds_v1 = sf.fit_predict(train_nixtla, h=h)
        _SKIP = {"unique_id", "ds", "index"}
        pred_col = [c for c in sf_preds_v1.columns if c not in _SKIP][0]
        preds_val1 = sf_preds_v1[["unique_id", "ds", pred_col]].rename(
            columns={"unique_id": "id", "ds": "date", pred_col: "yhat"})

        train_val1_nixtla = to_nixtla_format(df[df["d"] <= VAL1_END], config)
        sf2 = StatisticalForecaster(models=[model_key], freq=config.get("freq", "D"),
                                    n_jobs=config.get("n_jobs", -1))
        sf_preds_v2 = sf2.fit_predict(train_val1_nixtla, h=h)
        preds_val2 = sf_preds_v2[["unique_id", "ds", pred_col]].rename(
            columns={"unique_id": "id", "ds": "date", pred_col: "yhat"})

    # -----------------------------------------------------------------------
    elif model_name in ("lgbm", "xgb"):
        from src.data.features import build_feature_matrix, add_lag_features, add_rolling_features
        from src.data.features import add_calendar_features, add_price_features, _add_snap_feature

        model = get_model(config)

        X_train, y_train = build_feature_matrix(df, config, split="train")
        X_val1, y_val1 = build_feature_matrix(df, config, split="val1")

        model.fit(X_train, y_train, X_val=X_val1, y_val=y_val1)
        del X_train, y_train
        gc.collect()

        # Seed features: last row of training data per series
        seed_rows = (
            df[df["d"] == TRAIN_END].set_index("id")
        )
        feature_cols = [c for c in X_val1.columns]
        id_order = df["id"].unique()

        def _get_seed(split_df: pd.DataFrame, end_d: int) -> pd.DataFrame:
            rows = df[df["d"] == end_d].copy()
            X_seed, _ = build_feature_matrix(
                df[(df["d"] >= end_d - 50) & (df["d"] <= end_d)],
                config, split="all",
            )
            # Keep only the last row per series
            last_d = df[(df["d"] >= end_d - 50) & (df["d"] <= end_d)]["d"].max()
            mask = df[(df["d"] >= end_d - 50) & (df["d"] <= end_d)]["d"] == last_d
            return X_seed[mask.values[:len(X_seed)]].copy()

        X_seed_v1 = _get_seed(df, TRAIN_END)
        raw_preds_v1 = model.predict(X_seed_v1)  # (n, h)
        ids_v1 = df[df["d"] == TRAIN_END]["id"].values
        preds_val1 = _preds_array_to_long(raw_preds_v1, ids_v1, val1_start_date, h)

        X_seed_v2 = _get_seed(df, VAL1_END)
        raw_preds_v2 = model.predict(X_seed_v2)
        ids_v2 = df[df["d"] == VAL1_END]["id"].values
        preds_val2 = _preds_array_to_long(raw_preds_v2, ids_v2, val2_start_date, h)

    # -----------------------------------------------------------------------
    elif model_name in ("nbeats", "nhits", "tft", "patchtst"):
        model = get_model(config)
        train_nixtla = to_nixtla_format(train_df, config)
        futr_df = None
        if model_name == "tft":
            futr_df = to_nixtla_format(df, config)

        if model_name == "tft":
            model.fit(train_nixtla, futr_df=futr_df)
            preds_v1_nixtla = model.predict()
        else:
            model.fit(train_nixtla)
            preds_v1_nixtla = model.predict()

        pred_col_deep = [c for c in preds_v1_nixtla.columns if c not in ("unique_id", "ds")][0]
        preds_val1 = preds_v1_nixtla[["unique_id", "ds", pred_col_deep]].rename(
            columns={"unique_id": "id", "ds": "date", pred_col_deep: "yhat"})

        # Retrain on train+val1 for val2
        train_val1_nixtla = to_nixtla_format(df[df["d"] <= VAL1_END], config)
        model2 = get_model(config)
        if model_name == "tft":
            model2.fit(train_val1_nixtla, futr_df=futr_df)
            preds_v2_nixtla = model2.predict()
        else:
            model2.fit(train_val1_nixtla)
            preds_v2_nixtla = model2.predict()
        preds_val2 = preds_v2_nixtla[["unique_id", "ds", pred_col_deep]].rename(
            columns={"unique_id": "id", "ds": "date", pred_col_deep: "yhat"})

    # -----------------------------------------------------------------------
    elif model_name == "ensemble":
        component_ids = config.get("component_run_ids", [])
        preds_val1 = _EnsembleWrapper(config).run(component_ids, "val1")
        preds_val2 = _EnsembleWrapper(config).run(component_ids, "val2")
        preds_val1 = preds_val1.rename(columns={c: "yhat" for c in preds_val1.columns
                                                 if c not in ("id", "date")}, errors="ignore")
        preds_val2 = preds_val2.rename(columns={c: "yhat" for c in preds_val2.columns
                                                 if c not in ("id", "date")}, errors="ignore")
    else:
        raise ValueError(f"No prediction logic for model '{model_name}'.")

    # -----------------------------------------------------------------------
    # Clip predictions to non-negative
    if "yhat" in preds_val1.columns:
        preds_val1["yhat"] = preds_val1["yhat"].clip(lower=0)
    if "yhat" in preds_val2.columns:
        preds_val2["yhat"] = preds_val2["yhat"].clip(lower=0)

    # -----------------------------------------------------------------------
    # Evaluate
    from src.metrics.evaluate import evaluate_forecasts

    actuals_v1 = val1_df[["id", "date", "sales"]].copy()
    actuals_v2 = val2_df[["id", "date", "sales"]].copy()
    actuals_v1["date"] = pd.to_datetime(actuals_v1["date"])
    actuals_v2["date"] = pd.to_datetime(actuals_v2["date"])
    preds_val1["date"] = pd.to_datetime(preds_val1["date"])
    preds_val2["date"] = pd.to_datetime(preds_val2["date"])

    metrics_v1 = evaluate_forecasts(preds_val1, actuals_v1, train_wide)
    metrics_v2 = evaluate_forecasts(preds_val2, actuals_v2, train_wide)

    metrics = {
        "wmape_val1": metrics_v1["wmape"],
        "rmsse_val1": metrics_v1["rmsse"],
        "wmape_val2": metrics_v2["wmape"],
        "rmsse_val2": metrics_v2["rmsse"],
    }
    logger.info("Val1 — WMAPE: %.4f  RMSSE: %.4f", metrics["wmape_val1"], metrics["rmsse_val1"])
    logger.info("Val2 — WMAPE: %.4f  RMSSE: %.4f", metrics["wmape_val2"], metrics["rmsse_val2"])

    return metrics, preds_val1, preds_val2


def _preds_array_to_long(
    preds: np.ndarray,
    ids: np.ndarray,
    start_date: pd.Timestamp,
    h: int,
) -> pd.DataFrame:
    """Convert (n_series, h) array + id list to long format DataFrame."""
    dates = pd.date_range(start=start_date, periods=h, freq="D")
    rows = []
    for i, uid in enumerate(ids):
        for j in range(h):
            rows.append({"id": uid, "date": dates[j], "yhat": float(preds[i, j]) if preds.ndim == 2 else float(preds[i])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(
    run_id: str,
    metrics: dict,
    preds_val1: pd.DataFrame,
    preds_val2: pd.DataFrame,
    config: dict,
    config_path: Path,
    git_hash: str,
    runtime: float,
) -> None:
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "config_path": str(config_path),
        "git_commit": git_hash,
        "model": config["model"],
        "strategy": config.get("strategy", ""),
        **metrics,
        "runtime_seconds": round(runtime, 1),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    preds_val1.to_parquet(out_dir / "predictions_val1.parquet", index=False)
    preds_val2.to_parquet(out_dir / "predictions_val2.parquet", index=False)

    logger.info("Results saved to %s", out_dir)


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def _log_to_mlflow(config: dict, config_path: Path, run_id: str, metrics: dict) -> None:
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed — skipping experiment tracking.")
        return

    mlflow.set_experiment(config["model"])
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(_flatten_config(config))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(config_path))
        results_dir = RESULTS_DIR / run_id
        if results_dir.exists():
            mlflow.log_artifact(str(results_dir / "metrics.json"))
    logger.info("Logged to MLflow experiment '%s'.", config["model"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    run_id = args.run_id or f"{config['model']}_{config.get('strategy', '')}_{uuid.uuid4().hex[:8]}"
    git_hash = _get_git_hash()

    logger.info("Run ID: %s", run_id)
    logger.info("Model: %s / Strategy: %s", config["model"], config.get("strategy", ""))

    t0 = time.time()
    df = load_or_preprocess(config, debug=args.debug)
    metrics, preds_val1, preds_val2 = run_experiment(config, df, run_id, config_path)
    runtime = time.time() - t0

    save_results(run_id, metrics, preds_val1, preds_val2, config, config_path, git_hash, runtime)
    _log_to_mlflow(config, config_path, run_id, metrics)

    logger.info("Done in %.1fs.", runtime)


if __name__ == "__main__":
    main()
