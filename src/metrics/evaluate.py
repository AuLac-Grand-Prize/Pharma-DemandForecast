from __future__ import annotations

import numpy as np
import pandas as pd


def wmape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted Mean Absolute Percentage Error.

    WMAPE = sum(w * |y - ŷ|) / sum(w * |y|)
    When weights is None, w=1 for all observations.

    Raises ValueError if the weighted sum of |y| is zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    abs_true = np.abs(y_true)

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        numerator = np.sum(w * abs_err)
        denominator = np.sum(w * abs_true)
    else:
        numerator = np.sum(abs_err)
        denominator = np.sum(abs_true)

    if denominator == 0.0:
        raise ValueError("sum(|y_true|) == 0; WMAPE is undefined.")
    return float(numerator / denominator)


def rmsse_per_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """Root Mean Squared Scaled Error for a single series.

    RMSSE = sqrt( MSE(y_true, y_pred) / (1/(T-1) * sum(diff(y_train)^2)) )

    Returns 0.0 for constant training series (denominator == 0), following
    the M5 competition convention.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    mse = np.mean((y_true - y_pred) ** 2)
    diffs = np.diff(y_train)
    denominator = np.mean(diffs ** 2) if len(diffs) > 0 else 0.0

    if denominator == 0.0:
        return 0.0
    return float(np.sqrt(mse / denominator))


def _compute_level_weights(train_df: pd.DataFrame) -> pd.DataFrame:
    """Derive M5 hierarchy level weights from training data.

    Weights = total sales in last 28 training days per series at each level,
    normalised within each level so they sum to 1.

    train_df must have columns: id, item_id, dept_id, cat_id, store_id, state_id,
    plus day columns d_1 .. d_N (wide format).

    Returns a DataFrame with columns: id, level, weight.
    """
    full_meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    available_cols = [c for c in full_meta_cols if c in train_df.columns]
    day_cols = [c for c in train_df.columns if c.startswith("d_")]
    last28 = day_cols[-28:]
    sales_last28 = train_df[last28].sum(axis=1).values

    has_full_meta = all(c in train_df.columns for c in full_meta_cols)

    if has_full_meta:
        level_keys = {
            "level_1": lambda r: "Total",
            "level_2": lambda r: r["state_id"],
            "level_3": lambda r: r["store_id"],
            "level_4": lambda r: r["cat_id"],
            "level_5": lambda r: r["dept_id"],
            "level_6": lambda r: f"{r['state_id']}_{r['cat_id']}",
            "level_7": lambda r: f"{r['state_id']}_{r['dept_id']}",
            "level_8": lambda r: f"{r['store_id']}_{r['cat_id']}",
            "level_9": lambda r: f"{r['store_id']}_{r['dept_id']}",
            "level_10": lambda r: r["item_id"],
            "level_11": lambda r: f"{r['state_id']}_{r['item_id']}",
            "level_12": lambda r: r["id"],
        }
    else:
        # Fallback: only bottom-level (level_12 = per-series), equal volume weighting
        level_keys = {"level_12": lambda r: r["id"]}

    meta = train_df[available_cols].copy()
    records = []
    for level_name, key_fn in level_keys.items():
        meta["_key"] = meta.apply(key_fn, axis=1)
        grp_sales = {}
        for idx, row in meta.iterrows():
            k = row["_key"]
            grp_sales[k] = grp_sales.get(k, 0.0) + sales_last28[idx]
        total = sum(grp_sales.values()) or 1.0
        for idx, row in meta.iterrows():
            k = row["_key"]
            records.append({
                "id": row["id"],
                "level": level_name,
                "weight": grp_sales[k] / total,
            })

    return pd.DataFrame(records)


def rmsse_weighted(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    y_train_df: pd.DataFrame,
    level_weights: pd.DataFrame | None = None,
) -> float:
    """Weighted RMSSE across all 12 M5 hierarchy levels.

    y_true_df, y_pred_df: wide format with first column 'id', rest are day cols.
    y_train_df: same structure but covering the training window.
    level_weights: DataFrame with columns id, level, weight (from _compute_level_weights).
                   If None, weights are computed from y_train_df.

    Returns the scalar WRMSSE.
    """
    if level_weights is None:
        level_weights = _compute_level_weights(y_train_df)

    id_col = "id"
    day_cols_true = [c for c in y_true_df.columns if c != id_col]
    day_cols_pred = [c for c in y_pred_df.columns if c != id_col]
    day_cols_train = [c for c in y_train_df.columns if c != id_col]

    true_vals = y_true_df.set_index(id_col)[day_cols_true]
    pred_vals = y_pred_df.set_index(id_col)[day_cols_pred]
    train_vals = y_train_df.set_index(id_col)[day_cols_train]

    rmsse_series = {}
    for uid in true_vals.index:
        yt = true_vals.loc[uid].values.astype(float)
        yp = pred_vals.loc[uid].values.astype(float)
        ytr = train_vals.loc[uid].values.astype(float)
        rmsse_series[uid] = rmsse_per_series(yt, yp, ytr)

    level_rmsse = []
    for level in level_weights["level"].unique():
        lw = level_weights[level_weights["level"] == level]
        # aggregate predictions to this level
        # (for bottom-level l12, ids are unique; higher levels share weight)
        seen_keys: dict[str, list] = {}
        for _, row in lw.iterrows():
            uid = row["id"]
            seen_keys.setdefault(uid, []).append(row["weight"])

        level_val = 0.0
        total_w = 0.0
        for uid, ws in seen_keys.items():
            w = ws[0]  # weight is same for all entries with same id in this level
            level_val += w * rmsse_series.get(uid, 0.0)
            total_w += w
        if total_w > 0:
            level_rmsse.append(level_val / total_w)

    return float(np.mean(level_rmsse)) if level_rmsse else 0.0


def evaluate_forecasts(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    train_data: pd.DataFrame,
    level_weights: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute WMAPE and RMSSE from long-format DataFrames.

    predictions / actuals: long format with columns (id, date, sales) where
        predictions also has a 'yhat' column, or actuals has 'sales' and
        predictions has 'sales' as the forecast column.

    Internally pivots to wide format for RMSSE computation.

    Returns {'wmape': float, 'rmsse': float}.
    """
    pred_col = "yhat" if "yhat" in predictions.columns else "sales"
    merged = actuals.merge(
        predictions[["id", "date", pred_col]],
        on=["id", "date"],
        suffixes=("_actual", "_pred"),
    )

    y_true = merged["sales_actual"].values if "sales_actual" in merged else merged["sales"].values
    y_pred = merged[pred_col + "_pred" if pred_col + "_pred" in merged else pred_col].values

    wmape_val = wmape(y_true, y_pred)

    # Pivot to wide for RMSSE
    date_col = "date"
    true_wide = (
        actuals.pivot(index="id", columns=date_col, values="sales")
        .reset_index()
        .rename(columns=lambda c: f"d_{c}" if c != "id" else c)
    )
    true_wide.columns.name = None

    pred_wide = (
        predictions.pivot(index="id", columns=date_col, values=pred_col)
        .reset_index()
        .rename(columns=lambda c: f"d_{c}" if c != "id" else c)
    )
    pred_wide.columns.name = None

    train_wide = train_data if "id" in train_data.columns else train_data.reset_index()

    rmsse_val = rmsse_weighted(true_wide, pred_wide, train_wide, level_weights)

    return {"wmape": wmape_val, "rmsse": rmsse_val}
