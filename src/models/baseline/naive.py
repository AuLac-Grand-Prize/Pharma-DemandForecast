from __future__ import annotations

import numpy as np
import pandas as pd


class LastValueNaive:
    """Forecasts by carrying forward the last observed value."""

    def __init__(self) -> None:
        self._last_value: float | None = None

    def fit(self, y_train: np.ndarray) -> "LastValueNaive":
        self._last_value = float(y_train[-1])
        return self

    def predict(self, h: int) -> np.ndarray:
        if self._last_value is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(h, self._last_value, dtype=np.float32)


class SeasonalNaive:
    """Forecasts by repeating the last seasonal_period observations cyclically."""

    def __init__(self, seasonal_period: int = 7) -> None:
        self.seasonal_period = seasonal_period
        self._season_window: np.ndarray | None = None

    def fit(self, y_train: np.ndarray, seasonal_period: int | None = None) -> "SeasonalNaive":
        sp = seasonal_period if seasonal_period is not None else self.seasonal_period
        self.seasonal_period = sp
        self._season_window = np.asarray(y_train[-sp:], dtype=np.float32)
        return self

    def predict(self, h: int) -> np.ndarray:
        if self._season_window is None:
            raise RuntimeError("Call fit() before predict().")
        reps = int(np.ceil(h / self.seasonal_period))
        return np.tile(self._season_window, reps)[:h].astype(np.float32)


def forecast_naive_all_series(
    train_df: pd.DataFrame,
    h: int = 28,
    model: str = "last",
    seasonal_period: int = 7,
) -> pd.DataFrame:
    """Vectorized naive forecasting across all series.

    Parameters
    ----------
    train_df: Wide DataFrame with first column 'id' and remaining columns
              being daily sales values (sorted chronologically left-to-right).
    h:        Forecast horizon.
    model:    "last" for LastValueNaive, "seasonal" for SeasonalNaive.

    Returns a DataFrame with columns: id, F1..Fh.
    """
    id_col = "id"
    day_cols = [c for c in train_df.columns if c != id_col]
    values = train_df[day_cols].values.astype(np.float32)  # shape (n_series, T)

    if model == "last":
        # Broadcast last observed value: (n_series, 1) → (n_series, h)
        preds = np.repeat(values[:, -1:], h, axis=1)

    elif model == "seasonal":
        # Tile last seasonal_period values to length h
        tail = values[:, -seasonal_period:]  # (n_series, seasonal_period)
        reps = int(np.ceil(h / seasonal_period))
        tiled = np.tile(tail, (1, reps))      # (n_series, reps*sp)
        preds = tiled[:, :h]                  # (n_series, h)

    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'last' or 'seasonal'.")

    result = pd.DataFrame(
        preds,
        columns=[f"F{k+1}" for k in range(h)],
    )
    result.insert(0, id_col, train_df[id_col].values)
    return result
