from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ._rollout import recursive_rollout

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
}


class XGBoostForecaster:
    """Global XGBoost recursive forecaster."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        horizon: int = 28,
        seed: int = 42,
    ) -> None:
        try:
            import xgboost as xgb
            self._xgb = xgb
        except ImportError as e:
            raise ImportError("xgboost is required. pip install xgboost") from e

        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.params["seed"] = seed
        self.horizon = horizon
        self.seed = seed
        self._booster = None
        self._feature_names: list[str] = []

    def _to_dmatrix(self, X: pd.DataFrame, y: pd.Series | None = None) -> Any:
        # XGBoost native categoricals: columns with dtype 'category'
        X_cat = X.copy()
        cat_cols = [c for c in X_cat.columns if "code" in c]
        for c in cat_cols:
            X_cat[c] = X_cat[c].astype("category")
        return self._xgb.DMatrix(X_cat, label=y, enable_categorical=True)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        self._feature_names = list(X_train.columns)
        n_est = self.params.pop("n_estimators", 1000)
        train_params = {k: v for k, v in self.params.items() if k not in ("n_estimators",)}

        logger.info("Training XGBoost on %d rows …", len(X_train))

        dtrain = self._to_dmatrix(X_train, y_train)
        evals: list[tuple[Any, str]] = [(dtrain, "train")]
        esr = None

        if X_val is not None:
            dval = self._to_dmatrix(X_val, y_val)
            evals.append((dval, "val"))
            esr = 50

        callbacks: list[Any] = []
        if esr:
            callbacks.append(self._xgb.callback.EarlyStopping(rounds=esr, save_best=True))

        self._booster = self._xgb.train(
            train_params,
            dtrain,
            num_boost_round=n_est,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=False,
        )
        # Restore
        self.params["n_estimators"] = n_est
        logger.info("XGBoost training complete.")

    def predict_recursive(
        self,
        X_seed: pd.DataFrame,
        horizon: int | None = None,
    ) -> np.ndarray:
        """28-step recursive rollout. Returns (n_samples, h)."""
        if self._booster is None:
            raise RuntimeError("Call fit() before predict_recursive().")
        h = horizon or self.horizon

        lag_cols = {
            int(col.replace("lag_", "")): col
            for col in X_seed.columns
            if col.startswith("lag_")
        }
        rolling_cols = []
        for col in X_seed.columns:
            parts = col.split("_")
            if parts[0] == "rolling" and len(parts) >= 4:
                stat = parts[1]
                window = int(parts[2])
                lag = int(parts[3].replace("lag", ""))
                rolling_cols.append((window, lag, stat, col))

        def _predict_fn(X: pd.DataFrame) -> np.ndarray:
            return self._booster.predict(self._to_dmatrix(X))

        return recursive_rollout(_predict_fn, X_seed, lag_cols, rolling_cols, h)

    def predict(self, X_seed: pd.DataFrame, horizon: int | None = None) -> np.ndarray:
        return self.predict_recursive(X_seed, horizon)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if self._booster is None:
            raise RuntimeError("Model not fitted yet.")
        scores = self._booster.get_score(importance_type=importance_type)
        return pd.Series(scores, name=importance_type).sort_values(ascending=False)
