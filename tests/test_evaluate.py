import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.metrics.evaluate import wmape, rmsse_per_series, rmsse_weighted, evaluate_forecasts


def test_wmape_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert wmape(y, y) == 0.0


def test_wmape_weighted():
    y_true = np.array([2.0, 4.0])
    y_pred = np.array([1.0, 4.0])
    weights = np.array([1.0, 1.0])
    # |2-1|+|4-4| = 1; |2|+|4| = 6 → 1/6
    result = wmape(y_true, y_pred, weights=weights)
    np.testing.assert_allclose(result, 1 / 6, rtol=1e-6)


def test_wmape_zero_denominator():
    with pytest.raises(ValueError):
        wmape(np.zeros(3), np.ones(3))


def test_rmsse_per_series_perfect():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([6.0, 7.0])
    y_pred = y_true.copy()
    assert rmsse_per_series(y_true, y_pred, y_train) == 0.0


def test_rmsse_per_series_known_value():
    # y_train alternates 0, 1, 0, 1, ... → diff = ±1 always → mean(diff²) = 1
    y_train = np.array([float(i % 2) for i in range(20)])
    # constant forecast of 0.5 when true is 0 or 1 → MSE = 0.25
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    # RMSSE = sqrt(0.25 / 1.0) = 0.5
    result = rmsse_per_series(y_true, y_pred, y_train)
    np.testing.assert_allclose(result, 0.5, rtol=1e-6)


def test_rmsse_per_series_constant_train():
    y_train = np.ones(10)
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 1.0])
    # denominator = 0 → should return 0.0 without raising
    assert rmsse_per_series(y_true, y_pred, y_train) == 0.0


def test_evaluate_forecasts_round_trip():
    np.random.seed(0)
    n_series = 5
    n_train = 20
    n_test = 4

    ids = [f"series_{i}" for i in range(n_series)]
    train_records = []
    for uid in ids:
        for d in range(n_train):
            train_records.append({"id": uid, "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d), "sales": float(d)})
    train_df_long = pd.DataFrame(train_records)
    train_wide = train_df_long.pivot(index="id", columns="date", values="sales").reset_index()
    train_wide.columns = ["id"] + [f"d_{i}" for i in range(n_train)]

    test_records = []
    for uid in ids:
        for d in range(n_test):
            test_records.append({"id": uid, "date": pd.Timestamp("2020-01-21") + pd.Timedelta(days=d), "sales": float(n_train + d)})
    actuals = pd.DataFrame(test_records)
    predictions = actuals.copy()
    predictions["yhat"] = predictions["sales"]

    result = evaluate_forecasts(predictions, actuals, train_wide)
    assert result["wmape"] == 0.0
    assert result["rmsse"] == 0.0
