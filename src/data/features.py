from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

STATE_SNAP_MAP = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] = [7, 14, 28, 35, 42],
    target_col: str = "sales",
    group_col: str = "id",
) -> pd.DataFrame:
    """Add lag_{k} columns for each k in lags.

    Sorts by (id, d) once before shifting to avoid cross-series contamination.
    Uses float32 to save memory.
    """
    df = df.sort_values([group_col, "d"]).copy()
    for k in lags:
        logger.debug("  lag_%d", k)
        df[f"lag_{k}"] = (
            df.groupby(group_col, observed=True)[target_col]
            .shift(k)
            .astype(np.float32)
        )
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = [7, 28],
    lags: list[int] = [7, 28],
    target_col: str = "sales",
    group_col: str = "id",
) -> pd.DataFrame:
    """Add rolling mean and std columns.

    For each (window, lag) pair: shift(lag) then rolling(window).mean/std.
    This guarantees no future leakage.
    """
    for lag in lags:
        shifted = df.groupby(group_col, observed=True)[target_col].shift(lag)
        for window in windows:
            logger.debug("  rolling_mean_%d_lag%d", window, lag)
            df[f"rolling_mean_{window}_lag{lag}"] = (
                shifted.groupby(df[group_col], observed=True)
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
                .astype(np.float32)
            )
            df[f"rolling_std_{window}_lag{lag}"] = (
                shifted.groupby(df[group_col], observed=True)
                .transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
                .astype(np.float32)
            )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add date-based features from the 'date' column."""
    dt = pd.to_datetime(df["date"])
    df["dayofweek"] = dt.dt.dayofweek.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df["year"] = dt.dt.year.astype(np.int16)
    df["quarter"] = dt.dt.quarter.astype(np.int8)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.int8)
    df["event_flag_1"] = df["event_name_1"].notna().astype(np.int8)
    df["event_flag_2"] = df["event_name_2"].notna().astype(np.int8)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-derived features."""
    if "sell_price" not in df.columns:
        return df

    item_mean = df.groupby("item_id", observed=True)["sell_price"].transform("mean")
    df["price_norm"] = (df["sell_price"] / (item_mean + 1e-6)).astype(np.float32)

    price_lag1 = df.groupby("id", observed=True)["sell_price"].shift(1)
    df["price_change"] = (df["sell_price"] - price_lag1).astype(np.float32)
    df["price_change_pct"] = (
        df["price_change"] / (price_lag1 + 1e-6)
    ).astype(np.float32)

    return df


def _add_snap_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Map per-series SNAP flag using the series' state."""
    df["snap"] = np.int8(0)
    for state, col in STATE_SNAP_MAP.items():
        if col in df.columns:
            mask = df["state_id"].astype(str) == state
            df.loc[mask, "snap"] = df.loc[mask, col].astype(np.int8)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    config: dict[str, Any],
    split: str = "train",
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble ML feature matrix X and target y.

    Parameters
    ----------
    df:     Long-format DataFrame with lag/rolling/calendar/price features.
    config: Dict with keys: features.lags, features.rolling_windows,
            features.calendar, features.price.
    split:  "train" (d<=1857), "val1" (d=1858-1885), "val2" (d=1886-1913),
            or "all".

    Returns (X, y) where X.dtypes are all float32.
    """
    feat_cfg = config.get("features", {})
    lags = feat_cfg.get("lags", [7, 14, 28, 35, 42])

    split_ranges = {
        "train": (1, 1857),
        "val1": (1858, 1885),
        "val2": (1886, 1913),
        "all": (1, 9999),
    }
    lo, hi = split_ranges[split]
    sub = df[(df["d"] >= lo) & (df["d"] <= hi)].copy()

    # Drop rows where the largest lag is NaN (insufficient history)
    max_lag = max(lags) if lags else 0
    lag_col = f"lag_{max_lag}"
    if lag_col in sub.columns:
        sub = sub.dropna(subset=[lag_col])

    feature_cols: list[str] = []

    # Lag features
    if feat_cfg.get("lags"):
        feature_cols += [f"lag_{k}" for k in lags if f"lag_{k}" in sub.columns]

    # Rolling features
    if feat_cfg.get("rolling_windows"):
        windows = feat_cfg.get("rolling_windows", [7, 28])
        for lag in lags:
            for w in windows:
                for stat in ("mean", "std"):
                    col = f"rolling_{stat}_{w}_lag{lag}"
                    if col in sub.columns:
                        feature_cols.append(col)

    # Calendar features
    if feat_cfg.get("calendar", True):
        for col in ["dayofweek", "month", "day", "year", "quarter",
                    "is_weekend", "event_flag_1", "event_flag_2", "snap"]:
            if col in sub.columns:
                feature_cols.append(col)

    # Price features
    if feat_cfg.get("price", True):
        for col in ["sell_price", "price_norm", "price_change", "price_change_pct"]:
            if col in sub.columns:
                feature_cols.append(col)

    # Categorical ID columns (as codes)
    for cat_col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        if cat_col in sub.columns:
            sub[f"{cat_col}_code"] = sub[cat_col].cat.codes.astype(np.float32) \
                if hasattr(sub[cat_col], "cat") \
                else sub[cat_col].astype("category").cat.codes.astype(np.float32)
            feature_cols.append(f"{cat_col}_code")

    X = sub[feature_cols].astype(np.float32)
    y = sub["sales"].astype(np.float32)

    return X, y


def build_features(
    in_path: Path = PROCESSED_DIR / "long.parquet",
    out_path: Path = PROCESSED_DIR / "long_with_features.parquet",
    config: dict | None = None,
) -> pd.DataFrame:
    """Load preprocessed long Parquet, add all features, and save."""
    if config is None:
        config = {
            "features": {
                "lags": [7, 14, 28, 35, 42],
                "rolling_windows": [7, 28],
                "calendar": True,
                "price": True,
            }
        }

    logger.info("Loading %s …", in_path)
    df = pd.read_parquet(in_path)

    logger.info("Adding lag features …")
    df = add_lag_features(df, lags=config["features"].get("lags", [7, 14, 28, 35, 42]))
    gc.collect()

    logger.info("Adding rolling features …")
    df = add_rolling_features(
        df,
        windows=config["features"].get("rolling_windows", [7, 28]),
        lags=config["features"].get("lags", [7, 14, 28, 35, 42]),
    )
    gc.collect()

    logger.info("Adding calendar features …")
    df = add_calendar_features(df)

    logger.info("Adding price features …")
    df = add_price_features(df)

    logger.info("Adding SNAP feature …")
    df = _add_snap_feature(df)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving to %s …", out_path)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    logger.info("Done. Shape: %s", df.shape)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_features()
