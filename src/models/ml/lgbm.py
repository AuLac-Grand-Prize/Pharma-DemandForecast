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
    "num_leaves": 127,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
}

CAT_COLS = ["item_id_code", "dept_id_code", "cat_id_code", "store_id_code", "state_id_code"]


class LightGBMForecaster:
    """Global LightGBM model with recursive or direct multi-step strategy."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        strategy: str = "recursive",
        horizon: int = 28,
        seed: int = 42,
    ) -> None:
        try:
            import lightgbm as lgb
            self._lgb = lgb
        except ImportError as e:
            raise ImportError("lightgbm is required. pip install lightgbm") from e

        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.params["random_state"] = seed
        self.strategy = strategy
        self.horizon = horizon
        self.seed = seed

        self._boosters: list = []  # 1 booster (recursive) or horizon boosters (direct)
        self._feature_names: list[str] = []

    def _make_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        ref: Any = None,
    ) -> Any:
        cat_feat = [c for c in CAT_COLS if c in X.columns]
        ds = self._lgb.Dataset(X, label=y, categorical_feature=cat_feat or "auto", reference=ref)
        return ds

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        self._feature_names = list(X_train.columns)
        n_est = self.params.pop("n_estimators", 1000)
        train_params = {k: v for k, v in self.params.items() if k != "n_estimators"}

        callbacks = [self._lgb.early_stopping(50, verbose=False)] if X_val is not None else []
        valid_sets: list[Any]

        if self.strategy == "recursive":
            logger.info("Training LightGBM recursive (single model) on %d rows …", len(X_train))
            dtrain = self._make_dataset(X_train, y_train)
            if X_val is not None:
                dval = self._make_dataset(X_val, y_val, ref=dtrain)
                valid_sets = [dtrain, dval]
                valid_names = ["train", "val"]
            else:
                valid_sets = [dtrain]
                valid_names = ["train"]
            booster = self._lgb.train(
                train_params,
                dtrain,
                num_boost_round=n_est,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            self._boosters = [booster]

        elif self.strategy == "direct":
            logger.info("Training LightGBM direct (%d models) on %d rows …", self.horizon, len(X_train))
            self._boosters = []
            for step in range(1, self.horizon + 1):
                y_step = y_train.shift(-step) if hasattr(y_train, "shift") else pd.Series(y_train).shift(-step)
                valid_mask = y_step.notna()
                X_s = X_train[valid_mask]
                y_s = y_step[valid_mask]

                dtrain = self._make_dataset(X_s, y_s)
                if X_val is not None:
                    y_val_step = y_val.shift(-step) if hasattr(y_val, "shift") else pd.Series(y_val).shift(-step)
                    valid_mask_v = y_val_step.notna()
                    dval = self._make_dataset(X_val[valid_mask_v], y_val_step[valid_mask_v], ref=dtrain)
                    valid_sets = [dtrain, dval]
                    valid_names = ["train", "val"]
                else:
                    valid_sets = [dtrain]
                    valid_names = ["train"]

                b = self._lgb.train(
                    train_params,
                    dtrain,
                    num_boost_round=n_est,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks,
                )
                self._boosters.append(b)
                if step % 7 == 0:
                    logger.info("  Trained step %d/%d", step, self.horizon)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'. Choose 'recursive' or 'direct'.")

        # Restore n_estimators in params
        self.params["n_estimators"] = n_est

    def predict_direct(self, X: pd.DataFrame) -> np.ndarray:
        """Run each of the horizon step-specific models and return (n_samples, h)."""
        if len(self._boosters) != self.horizon:
            raise RuntimeError("Call fit() with strategy='direct' before predict_direct().")
        preds = np.stack([b.predict(X) for b in self._boosters], axis=1)
        return preds.astype(np.float32)

    def predict_recursive(
        self,
        X_seed: pd.DataFrame,
        horizon: int | None = None,
    ) -> np.ndarray:
        """28-step recursive rollout from X_seed. Returns (n_samples, h)."""
        if not self._boosters:
            raise RuntimeError("Call fit() before predict_recursive().")
        h = horizon or self.horizon
        booster = self._boosters[0]

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
            return booster.predict(X)

        return recursive_rollout(_predict_fn, X_seed, lag_cols, rolling_cols, h)

    def predict(self, X_seed: pd.DataFrame, horizon: int | None = None) -> np.ndarray:
        """Dispatch to the appropriate prediction method."""
        if self.strategy == "direct":
            return self.predict_direct(X_seed)
        return self.predict_recursive(X_seed, horizon)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        if not self._boosters:
            raise RuntimeError("Model not fitted yet.")
        imp = self._boosters[0].feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self._feature_names, name=importance_type).sort_values(ascending=False)
