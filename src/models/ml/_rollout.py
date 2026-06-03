"""Shared recursive multi-step rollout logic for global ML models."""
from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
import pandas as pd


def recursive_rollout(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X_seed: pd.DataFrame,
    lag_cols: dict[int, str],
    rolling_cols: list[tuple[int, int, str, str]],
    h: int,
) -> np.ndarray:
    """28-step recursive rollout for global ML forecasters.

    At each step k (1..h):
    1. Call predict_fn(X_current) → ŷ for all n_series.
    2. Shift the lag columns: if k >= lag_value, substitute ŷ_{k-lag} into lag_lag_value.
    3. Optionally update rolling feature columns.

    Parameters
    ----------
    predict_fn: A callable that takes a DataFrame X and returns 1D np.ndarray of length n_series.
    X_seed:     Feature matrix for the last training timestep (n_series × n_features).
                Must have columns matching lag_cols values and rolling_cols names.
    lag_cols:   dict mapping lag_value → column_name, e.g. {7: "lag_7", 14: "lag_14"}.
    rolling_cols: list of (window, lag, stat, col_name) tuples for rolling feature updates.
                  Can be empty [] to skip rolling updates.
    h:          Forecast horizon.

    Returns
    -------
    np.ndarray of shape (n_series, h) — forecasts for each series at each step.
    """
    n_series = len(X_seed)
    X_current = X_seed.copy()
    all_preds: list[np.ndarray] = []

    # Maintain a history buffer: for each lag value, keep a deque of the last
    # lag_value predictions so we can fill lag columns once enough steps are available.
    max_lag = max(lag_cols.keys()) if lag_cols else 0
    history: deque[np.ndarray] = deque(maxlen=max_lag)

    for step in range(1, h + 1):
        preds_step = predict_fn(X_current).astype(np.float32)  # (n_series,)
        all_preds.append(preds_step)
        history.append(preds_step)

        # Update lag columns: lag_k at time t+step is ŷ_{t+step-k}
        # Only update if we have enough predictions in the history buffer
        for lag_val, col_name in lag_cols.items():
            if col_name not in X_current.columns:
                continue
            if step >= lag_val:
                # The value at position -(lag_val) from current is at history[step - lag_val]
                hist_idx = step - lag_val  # 0-based index into all_preds
                X_current[col_name] = all_preds[hist_idx].copy()

        # Update rolling mean/std columns if specified
        for window, lag, stat, col_name in rolling_cols:
            if col_name not in X_current.columns:
                continue
            if step >= lag:
                # Gather the values that would be in the shifted window
                # shifted by lag: values from step-lag-window+1 .. step-lag
                start = step - lag - window
                end = step - lag
                vals: list[np.ndarray] = []
                for idx in range(start, end):
                    if 0 <= idx < len(all_preds):
                        vals.append(all_preds[idx])
                if vals:
                    stacked = np.stack(vals, axis=0)  # (n_vals, n_series)
                    if stat == "mean":
                        X_current[col_name] = stacked.mean(axis=0).astype(np.float32)
                    elif stat == "std":
                        X_current[col_name] = stacked.std(axis=0, ddof=0).astype(np.float32)

    return np.stack(all_preds, axis=1)  # (n_series, h)
