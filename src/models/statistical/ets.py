from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "ets": "AutoETS",
    "theta": "AutoTheta",
    "arima": "AutoARIMA",
}


class StatisticalForecaster:
    """Thin wrapper around statsforecast for ETS, Theta, and AutoARIMA.

    Expects input in Nixtla long format: unique_id (str), ds (Timestamp), y (float).
    """

    def __init__(
        self,
        models: list[str],
        freq: str = "D",
        n_jobs: int = -1,
    ) -> None:
        """
        Parameters
        ----------
        models: Subset of ["ets", "theta", "arima"].
        freq:   Pandas-style frequency string.
        n_jobs: Number of parallel workers.
        """
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
        except ImportError as e:
            raise ImportError("statsforecast is required. pip install statsforecast") from e

        self.freq = freq
        self.n_jobs = n_jobs
        self._model_names = models

        sf_models = []
        for name in models:
            if name == "ets":
                sf_models.append(AutoETS(model="ZZZ"))
            elif name == "theta":
                sf_models.append(AutoTheta(decomposition_type="additive"))
            elif name == "arima":
                sf_models.append(AutoARIMA())
            else:
                raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_MAP)}")

        self._sf = StatsForecast(models=sf_models, freq=freq, n_jobs=n_jobs)
        self._fitted = False

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        h: int = 28,
    ) -> pd.DataFrame:
        """Fit and forecast in one step.

        Parameters
        ----------
        train_df: Long-format DataFrame with columns unique_id, ds, y.
                  ds must be pd.Timestamp or datetime-castable.
        h:        Forecast horizon.

        Returns a DataFrame with columns: unique_id, ds, <model_name_1>, ...
        """
        train_df = train_df.copy()
        train_df["ds"] = pd.to_datetime(train_df["ds"])

        logger.info(
            "Fitting %s on %d series, h=%d …",
            self._model_names,
            train_df["unique_id"].nunique(),
            h,
        )
        preds = self._sf.forecast(df=train_df, h=h)
        self._fitted = True
        logger.info("Done. Predictions shape: %s", preds.shape)
        return preds.reset_index(drop=True)

    def to_wide(
        self,
        preds: pd.DataFrame,
        id_col: str = "unique_id",
        model_col: str | None = None,
    ) -> pd.DataFrame:
        """Convert Nixtla long-format predictions to wide matrix.

        Parameters
        ----------
        preds:     Output of fit_predict().
        id_col:    Series identifier column.
        model_col: Column containing predictions. If None, uses the first
                   non-id/ds column.

        Returns DataFrame with rows=series, cols = [id_col] + day columns.
        """
        if model_col is None:
            extra_cols = [c for c in preds.columns if c not in (id_col, "ds")]
            if not extra_cols:
                raise ValueError("No prediction columns found in preds DataFrame.")
            model_col = extra_cols[0]

        wide = preds.pivot(index=id_col, columns="ds", values=model_col).reset_index()
        wide.columns.name = None
        return wide

    @property
    def model_names(self) -> list[str]:
        return self._model_names
