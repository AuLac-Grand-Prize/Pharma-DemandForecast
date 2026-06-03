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
