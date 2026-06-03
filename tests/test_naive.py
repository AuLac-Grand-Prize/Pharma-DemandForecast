import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.models.baseline.naive import LastValueNaive, SeasonalNaive, forecast_naive_all_series


def test_last_value_constant():
    y_train = np.array([1.0, 2.0, 3.0, 7.5])
    model = LastValueNaive().fit(y_train)
    preds = model.predict(h=5)
    np.testing.assert_array_equal(preds, np.full(5, 7.5))


def test_seasonal_naive_repeats_correctly():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    model = SeasonalNaive(seasonal_period=7).fit(y_train)
    preds = model.predict(h=10)
    # Last 7 values: [2,3,4,5,6,7,8]; tiled to length 10 → [2,3,4,5,6,7,8,2,3,4]
    expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(preds, expected)


def test_vectorized_matches_per_series():
    np.random.seed(42)
    n_series = 10
    T = 30
    h = 7
    data = np.random.randint(0, 10, size=(n_series, T)).astype(float)
    ids = [f"s{i}" for i in range(n_series)]
    day_cols = [f"d_{d+1}" for d in range(T)]
    train_df = pd.DataFrame(data, columns=day_cols)
    train_df.insert(0, "id", ids)

    # Test "last" strategy
    vec_preds = forecast_naive_all_series(train_df, h=h, model="last")
    for i in range(n_series):
        expected = LastValueNaive().fit(data[i]).predict(h)
        np.testing.assert_array_almost_equal(
            vec_preds.iloc[i, 1:].values.astype(float), expected.astype(float)
        )

    # Test "seasonal" strategy
    vec_preds_s = forecast_naive_all_series(train_df, h=h, model="seasonal", seasonal_period=7)
    for i in range(n_series):
        expected = SeasonalNaive(7).fit(data[i]).predict(h)
        np.testing.assert_array_almost_equal(
            vec_preds_s.iloc[i, 1:].values.astype(float), expected.astype(float)
        )
