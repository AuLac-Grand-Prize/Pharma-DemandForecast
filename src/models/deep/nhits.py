from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class NHiTSForecaster:
    """N-HiTS forecaster via neuralforecast."""

    def __init__(
        self,
        h: int = 28,
        input_size: int = 91,
        stack_types: list[str] | None = None,
        n_blocks: list[int] | None = None,
        mlp_units: list[list[int]] | None = None,
        n_pool_kernel_size: list[int] | None = None,
        interpolation_mode: str = "linear",
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
        self.stack_types = stack_types or ["identity", "identity", "identity"]
        self.n_blocks = n_blocks or [1, 1, 1]
        self.mlp_units = mlp_units or [[512, 512], [512, 512], [512, 512]]
        self.n_pool_kernel_size = n_pool_kernel_size or [2, 2, 1]
        self.interpolation_mode = interpolation_mode
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
            from neuralforecast.models import NHITS
        except ImportError as e:
            raise ImportError("neuralforecast is required. pip install neuralforecast") from e

        model = NHITS(
            h=self.h,
            input_size=self.input_size,
            stack_types=self.stack_types,
            n_blocks=self.n_blocks,
            mlp_units=self.mlp_units,
            n_pool_kernel_size=self.n_pool_kernel_size,
            interpolation_mode=self.interpolation_mode,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            random_seed=self.seed,
            accelerator=self.accelerator,
        )
        from neuralforecast import NeuralForecast
        self._nf = NeuralForecast(models=[model], freq=self.freq)

    def fit(self, train_df: pd.DataFrame) -> None:
        self._build()
        df = train_df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        if self.max_series and df["unique_id"].nunique() > self.max_series:
            sampled_ids = (
                df["unique_id"].drop_duplicates().sample(self.max_series, random_state=self.seed)
            )
            df = df[df["unique_id"].isin(sampled_ids)]
            logger.info("Subsampled to %d series for N-HiTS training.", self.max_series)

        logger.info("Training N-HiTS on %d series …", df["unique_id"].nunique())
        self._nf.fit(df=df)
        logger.info("N-HiTS training complete.")

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
