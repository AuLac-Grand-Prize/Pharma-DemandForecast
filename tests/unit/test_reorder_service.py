from demandforecast.services.reorder_service import ReorderService


def test_suggest_qty_zero_when_overstocked() -> None:
    qty = ReorderService.suggest_qty(
        forecast_p50=[10] * 30,
        forecast_p90=[15] * 30,
        current_stock=10_000,
        lead_time_days=3,
        safety_stock_days=7,
    )
    assert qty == 0


def test_suggest_qty_positive_when_low_stock() -> None:
    qty = ReorderService.suggest_qty(
        forecast_p50=[10] * 30,
        forecast_p90=[15] * 30,
        current_stock=0,
        lead_time_days=3,
        safety_stock_days=7,
    )
    assert qty > 0


# --- Req 7: reorder formula edge cases (pin the math at its boundaries) ------


def test_suggest_qty_empty_p50_returns_zero() -> None:
    qty = ReorderService.suggest_qty(
        forecast_p50=[],
        forecast_p90=[15] * 30,
        current_stock=0,
        lead_time_days=3,
        safety_stock_days=7,
    )
    assert qty == 0


def test_suggest_qty_empty_p90_falls_back_to_avg_daily() -> None:
    # With empty p90, peak_daily falls back to avg_daily (= 10).
    # target = 10*3 + 10*7 = 100; current_stock 0 -> qty 100.
    qty = ReorderService.suggest_qty(
        forecast_p50=[10.0] * 30,
        forecast_p90=[],
        current_stock=0,
        lead_time_days=3,
        safety_stock_days=7,
    )
    assert qty == 100


def test_suggest_qty_zero_when_stock_equals_target() -> None:
    # avg_daily = 10, peak_daily = 15 -> target = 10*3 + 15*7 = 135.
    target = 10 * 3 + 15 * 7
    qty = ReorderService.suggest_qty(
        forecast_p50=[10.0] * 30,
        forecast_p90=[15.0] * 30,
        current_stock=target,
        lead_time_days=3,
        safety_stock_days=7,
    )
    assert qty == 0


def test_suggest_qty_result_is_nonnegative_int() -> None:
    qty = ReorderService.suggest_qty(
        forecast_p50=[3.7, 4.2, 5.9],
        forecast_p90=[6.1, 7.0, 8.3],
        current_stock=2,
        lead_time_days=2,
        safety_stock_days=5,
    )
    assert isinstance(qty, int)
    assert qty >= 0
