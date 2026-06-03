from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PatchTSTForecaster:
    """PatchTST forecaster via neuralforecast."""

    def __init__(
        self,
        h: int = 28,
        input_size: int = 104,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        max_steps: int = 500,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 42,
        freq: str = "D",
        accelerator: str = "auto",
        max_series: int | None = None,
    ) -> None:
        self.h = h
        self.input_size = input_size
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.attn_dropout = attn_dropout
        self.dropout = dropout
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
            from neuralforecast.models import PatchTST
        except ImportError as e:
            raise ImportError("neuralforecast is required. pip install neuralforecast") from e

        model = PatchTST(
            h=self.h,
            input_size=self.input_size,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            attn_dropout=self.attn_dropout,
            dropout=self.dropout,
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
            logger.info("Subsampled to %d series for PatchTST training.", self.max_series)

        logger.info("Training PatchTST on %d series …", df["unique_id"].nunique())
        self._nf.fit(df=df)
        logger.info("PatchTST training complete.")

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
