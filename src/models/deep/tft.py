from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class TFTForecaster:
    """Temporal Fusion Transformer via neuralforecast.

    Supports historical and future exogenous covariates.
    """

    def __init__(
        self,
        h: int = 28,
        input_size: int = 91,
        hidden_size: int = 128,
        n_head: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        hist_exog_list: list[str] | None = None,
        futr_exog_list: list[str] | None = None,
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
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.hist_exog_list = hist_exog_list or []
        self.futr_exog_list = futr_exog_list or []
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
            from neuralforecast.models import TFT
        except ImportError as e:
            raise ImportError("neuralforecast is required. pip install neuralforecast") from e

        kwargs: dict[str, Any] = dict(
            h=self.h,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            n_head=self.n_head,
            attn_dropout=self.attn_dropout,
            dropout=self.dropout,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            random_seed=self.seed,
            accelerator=self.accelerator,
        )
        if self.hist_exog_list:
            kwargs["hist_exog_list"] = self.hist_exog_list
        if self.futr_exog_list:
            kwargs["futr_exog_list"] = self.futr_exog_list

        model = TFT(**kwargs)
        from neuralforecast import NeuralForecast
        self._nf = NeuralForecast(models=[model], freq=self.freq)

    def fit(
        self,
        train_df: pd.DataFrame,
        futr_df: pd.DataFrame | None = None,
    ) -> None:
        """Fit TFT.

        Parameters
        ----------
        train_df: Nixtla format (unique_id, ds, y[, hist_exog_cols]).
        futr_df:  DataFrame with future-known covariate columns for the full
                  history + forecast window. Must have (unique_id, ds, futr_cols).
        """
        self._build()
        df = train_df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        if self.max_series and df["unique_id"].nunique() > self.max_series:
            sampled_ids = (
                df["unique_id"].drop_duplicates().sample(self.max_series, random_state=self.seed)
            )
            df = df[df["unique_id"].isin(sampled_ids)]
            if futr_df is not None:
                futr_df = futr_df[futr_df["unique_id"].isin(sampled_ids)].copy()
            logger.info("Subsampled to %d series for TFT training.", self.max_series)

        if futr_df is not None:
            futr_df = futr_df.copy()
            futr_df["ds"] = pd.to_datetime(futr_df["ds"])

        logger.info("Training TFT on %d series …", df["unique_id"].nunique())
        self._nf.fit(df=df, val_size=self.h)
        logger.info("TFT training complete.")
        self._futr_df = futr_df  # store for predict

    def predict(
        self,
        futr_df: pd.DataFrame | None = None,
        h: int | None = None,
    ) -> pd.DataFrame:
        if self._nf is None:
            raise RuntimeError("Call fit() before predict().")
        futr = futr_df if futr_df is not None else getattr(self, "_futr_df", None)
        if futr is not None:
            futr = futr.copy()
            futr["ds"] = pd.to_datetime(futr["ds"])
        preds = self._nf.predict(futr_df=futr)
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
