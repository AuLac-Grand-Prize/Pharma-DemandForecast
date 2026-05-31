from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .naive import SeasonalNaive

logger = logging.getLogger(__name__)

ADI_THRESHOLD = 1.32  # Series with ADI > threshold are considered intermittent


def _adi(y: np.ndarray) -> float:
    """Average Demand Interval: proportion of periods with positive demand."""
    n_nonzero = np.count_nonzero(y)
    if n_nonzero == 0:
        return float("inf")
    return len(y) / n_nonzero


def croston(y: np.ndarray, alpha: float = 0.1, h: int = 28) -> np.ndarray:
    """Classic Croston (1972) intermittent demand forecast.

    Separately exponentially-smooths demand sizes and inter-demand intervals.
    Returns a constant h-step forecast = smoothed_size / smoothed_interval.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # Find first non-zero demand
    nonzero_idx = np.nonzero(y)[0]
    if len(nonzero_idx) == 0:
        return np.zeros(h, dtype=np.float32)

    # Initialise
    q = y[nonzero_idx[0]]   # smoothed demand size
    a = float(nonzero_idx[0] + 1)  # smoothed inter-demand interval
    last_demand_t = nonzero_idx[0]

    for t in range(nonzero_idx[0] + 1, n):
        if y[t] > 0:
            interval = t - last_demand_t
            q = alpha * y[t] + (1 - alpha) * q
            a = alpha * interval + (1 - alpha) * a
            last_demand_t = t

    forecast = q / a if a > 0 else 0.0
    return np.full(h, forecast, dtype=np.float32)


def sba(y: np.ndarray, alpha: float = 0.1, h: int = 28) -> np.ndarray:
    """Syntetos-Boylan Approximation: bias-corrected Croston forecast.

    Applies the (1 - alpha/2) correction factor to reduce Croston's upward bias.
    """
    raw = croston(y, alpha=alpha, h=h)
    correction = 1.0 - alpha / 2.0
    return (raw * correction).astype(np.float32)


def _forecast_single(
    y: np.ndarray,
    h: int,
    variant: str,
    alpha: float,
) -> np.ndarray:
    """Dispatch to Croston/SBA or fall back to SeasonalNaive for non-intermittent series."""
    if _adi(y) > ADI_THRESHOLD:
        if variant == "sba":
            return sba(y, alpha=alpha, h=h)
        return croston(y, alpha=alpha, h=h)
    # Non-intermittent: use seasonal naive as a better alternative
    return SeasonalNaive(seasonal_period=7).fit(y).predict(h)


def forecast_croston_all_series(
    train_df: pd.DataFrame,
    h: int = 28,
    variant: str = "sba",
    alpha: float = 0.1,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Apply Croston/SBA across all series in train_df using joblib parallelism.

    Parameters
    ----------
    train_df: Wide DataFrame with first column 'id', rest are ordered day columns.
    h:        Forecast horizon.
    variant:  "croston" or "sba".
    alpha:    Smoothing parameter.
    n_jobs:   Number of parallel workers (-1 = all CPUs).

    Returns DataFrame with columns: id, F1..Fh.
    """
    id_col = "id"
    day_cols = [c for c in train_df.columns if c != id_col]
    series_list = train_df[day_cols].values.astype(np.float32)

    logger.info("Running %s on %d series (n_jobs=%d) …", variant, len(series_list), n_jobs)

    preds = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_forecast_single)(series_list[i], h, variant, alpha)
        for i in range(len(series_list))
    )

    result = pd.DataFrame(
        np.stack(preds, axis=0),
        columns=[f"F{k+1}" for k in range(h)],
    )
    result.insert(0, id_col, train_df[id_col].values)
    return result
