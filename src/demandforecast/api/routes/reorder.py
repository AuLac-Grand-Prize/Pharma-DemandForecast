from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ReorderRequest(BaseModel):
    pharmacy_id: str
    distributor_lead_time_days: int = 3
    safety_stock_days: int = 7


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
async def reorder_suggestions(req: ReorderRequest) -> ReorderResponse:
    # TODO: compute reorder qty from forecast + lead time + safety stock
    return ReorderResponse(suggestions=[], expected_stockout_avoided=0)
