from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from demandforecast.api.routes.forecast import get_forecast_backend
from demandforecast.services.forecast_service import ForecastBackend
from demandforecast.services.reorder_service import ReorderService

router = APIRouter()


class ReorderSkuInput(BaseModel):
    sku_id: str
    current_stock: int = Field(ge=0)


class ReorderRequest(BaseModel):
    pharmacy_id: str
    skus: list[ReorderSkuInput]
    horizon_days: int = Field(default=30, ge=1, le=365)
    distributor_lead_time_days: int = Field(default=3, ge=0)
    safety_stock_days: int = Field(default=7, ge=0)


class ReorderItem(BaseModel):
    sku_id: str
    drug_name: str
    current_stock: int
    suggested_order_qty: int
    reason: str


class ReorderResponse(BaseModel):
    suggestions: list[ReorderItem]
    expected_stockout_avoided: int


@router.post("/reorder-suggestions", response_model=ReorderResponse)
async def reorder_suggestions(
    req: ReorderRequest,
    backend: ForecastBackend = Depends(get_forecast_backend),
) -> ReorderResponse:
    sku_ids = [s.sku_id for s in req.skus]
    stock_by_sku = {s.sku_id: s.current_stock for s in req.skus}

    forecasts = backend.predict(
        pharmacy_id=req.pharmacy_id,
        sku_ids=sku_ids,
        horizon_days=req.horizon_days,
        include_covariates=True,
    )

    suggestions: list[ReorderItem] = []
    total_avoided = 0
    for fc in forecasts:
        p50 = [p.p50 for p in fc.points]
        p90 = [p.p90 for p in fc.points]
        current_stock = stock_by_sku[fc.sku_id]

        qty = ReorderService.suggest_qty(
            forecast_p50=p50,
            forecast_p90=p90,
            current_stock=current_stock,
            lead_time_days=req.distributor_lead_time_days,
            safety_stock_days=req.safety_stock_days,
        )

        if qty > 0:
            reason = (
                f"Projected demand over {req.distributor_lead_time_days}d lead time "
                f"+ {req.safety_stock_days}d safety stock exceeds current stock "
                f"({current_stock}); order {qty} units."
            )
        else:
            reason = (
                f"Current stock ({current_stock}) covers projected demand "
                f"plus safety stock; no reorder needed."
            )

        suggestions.append(
            ReorderItem(
                sku_id=fc.sku_id,
                drug_name=fc.drug_name,
                current_stock=current_stock,
                suggested_order_qty=qty,
                reason=reason,
            )
        )
        total_avoided += qty

    return ReorderResponse(
        suggestions=suggestions,
        expected_stockout_avoided=total_avoided,
    )
