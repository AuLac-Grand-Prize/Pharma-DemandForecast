import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.data.features import (
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
)


def _make_two_series(n: int = 30) -> pd.DataFrame:
    """Synthetic long-format DataFrame with two series."""
    records = []
    for sid in ["A", "B"]:
        for d in range(1, n + 1):
            records.append({
                "id": sid,
                "item_id": "ITEM_1",
                "dept_id": "DEPT_1",
                "cat_id": "CAT_1",
                "store_id": "STORE_1",
                "state_id": "CA",
                "d": d,
                "sales": float(d) if sid == "A" else float(d * 2),
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=d - 1),
                "event_name_1": None,
                "event_name_2": None,
                "sell_price": 1.0,
                "snap_CA": 0,
                "snap_TX": 0,
                "snap_WI": 0,
            })
    df = pd.DataFrame(records)
    df["id"] = df["id"].astype("category")
    df["state_id"] = df["state_id"].astype("category")
    df["item_id"] = df["item_id"].astype("category")
    return df


def test_lag_no_cross_series_contamination():
    df = _make_two_series(30)
    df = add_lag_features(df, lags=[7])

    for sid in ["A", "B"]:
        sub = df[df["id"] == sid].sort_values("d")
        # lag_7 at d=8 should equal sales at d=1 for the same series
        row = sub[sub["d"] == 8].iloc[0]
        expected = sub[sub["d"] == 1]["sales"].iloc[0]
        np.testing.assert_allclose(row["lag_7"], expected, rtol=1e-5)


def test_rolling_no_future_leakage():
    df = _make_two_series(30)
    df = add_lag_features(df, lags=[7])
    df = add_rolling_features(df, windows=[7], lags=[7])

    # rolling_mean_7_lag7 at row d=14 should only use d=1..7 for series A
    sub_a = df[(df["id"] == "A")].sort_values("d")
    row14 = sub_a[sub_a["d"] == 14].iloc[0]
    # shift(7) at d=14 → d=7 → rolling over d=1..7 for series A → mean(1..7) = 4.0
    expected = np.mean([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    np.testing.assert_allclose(row14["rolling_mean_7_lag7"], expected, rtol=1e-4)


def test_build_feature_matrix_float32():
    df = _make_two_series(60)
    df = add_lag_features(df, lags=[7, 14])
    df = add_rolling_features(df, windows=[7], lags=[7])
    from src.data.features import add_calendar_features, add_price_features, _add_snap_feature
    df = add_calendar_features(df)
    df = add_price_features(df)
    df = _add_snap_feature(df)

    config = {
        "features": {
            "lags": [7, 14],
            "rolling_windows": [7],
            "calendar": True,
            "price": True,
        }
    }
    X, y = build_feature_matrix(df, config, split="all")
    assert X.dtypes.unique().tolist() == [np.float32], f"Expected float32, got {X.dtypes.unique().tolist()}"
    assert len(X) == len(y)
    assert len(X) > 0
