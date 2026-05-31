from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

META_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _read_sales_wide(sales_path: Path) -> pd.DataFrame:
    logger.info("Reading %s …", sales_path)
    dtype = {c: "category" for c in META_COLS}
    df = pd.read_csv(sales_path, dtype=dtype)
    logger.info("  Shape: %s", df.shape)
    return df


def merge_prices(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Left-join sell_prices on (store_id, item_id, wm_yr_wk).

    Fills NaN sell_price via forward-fill then backward-fill within each
    (store_id, item_id) group to handle pre-launch periods.
    """
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df = df.sort_values(["id", "d"])
    df["sell_price"] = (
        df.groupby("id", observed=True)["sell_price"]
        .transform(lambda s: s.ffill().bfill())
    )
    return df


def add_hierarchy_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Append M5 hierarchy aggregate key columns (level_1 .. level_12)."""
    df = df.copy()
    df["level_1"] = "Total"
    df["level_2"] = df["state_id"].astype(str)
    df["level_3"] = df["store_id"].astype(str)
    df["level_4"] = df["cat_id"].astype(str)
    df["level_5"] = df["dept_id"].astype(str)
    df["level_6"] = df["state_id"].astype(str) + "_" + df["cat_id"].astype(str)
    df["level_7"] = df["state_id"].astype(str) + "_" + df["dept_id"].astype(str)
    df["level_8"] = df["store_id"].astype(str) + "_" + df["cat_id"].astype(str)
    df["level_9"] = df["store_id"].astype(str) + "_" + df["dept_id"].astype(str)
    df["level_10"] = df["item_id"].astype(str)
    df["level_11"] = df["state_id"].astype(str) + "_" + df["item_id"].astype(str)
    df["level_12"] = df["id"].astype(str)
    return df


def melt_sales(
    sales_path: Path,
    calendar_path: Path,
    prices_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    """Melt wide-format sales CSV to long format and merge calendar + prices.

    Output columns:
        id, item_id, dept_id, cat_id, store_id, state_id,
        d (int16), date (datetime), wm_yr_wk (int),
        sales (float32), sell_price (float32),
        event_name_1, event_type_1, event_name_2, event_type_2,
        snap_CA (int8), snap_TX (int8), snap_WI (int8)

    Saves as Parquet (snappy) to out_path.
    """
    sales_wide = _read_sales_wide(sales_path)

    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    logger.info("Melting %d day columns …", len(day_cols))

    df_long = sales_wide.melt(
        id_vars=META_COLS,
        value_vars=day_cols,
        var_name="d_str",
        value_name="sales",
    )
    del sales_wide
    gc.collect()

    df_long["d"] = df_long["d_str"].str.replace("d_", "", regex=False).astype(np.int16)
    df_long.drop(columns=["d_str"], inplace=True)
    df_long["sales"] = df_long["sales"].astype(np.float32)

    logger.info("Reading calendar …")
    cal = pd.read_csv(
        calendar_path,
        dtype={"snap_CA": np.int8, "snap_TX": np.int8, "snap_WI": np.int8},
        parse_dates=["date"],
    )
    cal["d"] = cal["d"].str.replace("d_", "", regex=False).astype(np.int16)
    cal_keep = [
        "d", "date", "wm_yr_wk",
        "event_name_1", "event_type_1",
        "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    cal = cal[cal_keep]

    logger.info("Merging calendar …")
    df_long = df_long.merge(cal, on="d", how="left")
    del cal
    gc.collect()

    logger.info("Reading sell_prices …")
    prices = pd.read_csv(prices_path)
    prices["sell_price"] = prices["sell_price"].astype(np.float32)

    logger.info("Merging prices …")
    df_long = merge_prices(df_long, prices)
    del prices
    gc.collect()

    logger.info("Adding hierarchy level columns …")
    df_long = add_hierarchy_ids(df_long)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving to %s …", out_path)
    df_long.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    logger.info("Saved. Shape: %s", df_long.shape)

    return df_long


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    melt_sales(
        RAW_DIR / "sales_train_validation.csv",
        RAW_DIR / "calendar.csv",
        RAW_DIR / "sell_prices.csv",
        PROCESSED_DIR / "long.parquet",
    )
