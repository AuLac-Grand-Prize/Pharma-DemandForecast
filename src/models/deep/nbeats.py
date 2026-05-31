from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class NBEATSForecaster:
    """N-BEATS forecaster via neuralforecast."""

    def __init__(
        self,
        h: int = 28,
        input_size: int = 91,
        stack_types: list[str] | None = None,
        n_harmonics: int = 2,
        n_polynomials: int = 2,
        max_steps: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 42,
        freq: str = "D",
        accelerator: str = "auto",
        max_series: int | None = None,
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.stack_types = stack_types or ["trend", "seasonality"]
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.freq = freq
        self.accelerator = accelerator
        self.max_series = max_series
        self._nf: Any = None

    def _build(self) -> None:
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NBEATS
        except ImportError as e:
            raise ImportError("neuralforecast is required. pip install neuralforecast") from e

        model = NBEATS(
            h=self.h,
            input_size=self.input_size,
            stack_types=self.stack_types,
            n_harmonics=self.n_harmonics,
            n_polynomials=self.n_polynomials,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            random_seed=self.seed,
            accelerator=self.accelerator,
        )
        from neuralforecast import NeuralForecast
        self._nf = NeuralForecast(models=[model], freq=self.freq)

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit on Nixtla-format DataFrame (unique_id, ds, y).

        ds must be or be castable to pd.Timestamp.
        If max_series is set, a random sample of series is used.
        """
        self._build()
        df = train_df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        if self.max_series and df["unique_id"].nunique() > self.max_series:
            sampled_ids = (
                df["unique_id"].drop_duplicates().sample(self.max_series, random_state=self.seed)
            )
            df = df[df["unique_id"].isin(sampled_ids)]
            logger.info("Subsampled to %d series for N-BEATS training.", self.max_series)

        logger.info("Training N-BEATS on %d series …", df["unique_id"].nunique())
        self._nf.fit(df=df)
        logger.info("N-BEATS training complete.")

    def predict(self, h: int | None = None) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("Call fit() before predict().")
        preds = self._nf.predict()
        return preds.reset_index() if hasattr(preds, "reset_index") else preds

    def save(self, path: Path | str) -> None:
        if self._nf is None:
            raise RuntimeError("Nothing to save — model not fitted.")
        self._nf.save(str(path), overwrite=True)

    def load(self, path: Path | str) -> None:
        try:
            from neuralforecast import NeuralForecast
        except ImportError as e:
            raise ImportError("neuralforecast is required.") from e
        self._nf = NeuralForecast.load(str(path))
