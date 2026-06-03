"""Reorder suggestion — Newsvendor-style optimization with safety stock."""


class ReorderService:
    @staticmethod
    def suggest_qty(
        forecast_p50: list[float],
        forecast_p90: list[float],
        current_stock: int,
        lead_time_days: int,
        safety_stock_days: int,
    ) -> int:
        if not forecast_p50:
            return 0
        avg_daily = sum(forecast_p50) / len(forecast_p50)
        peak_daily = max(forecast_p90) if forecast_p90 else avg_daily
        target_stock = avg_daily * lead_time_days + peak_daily * safety_stock_days
        qty = int(target_stock - current_stock)
        return max(qty, 0)
