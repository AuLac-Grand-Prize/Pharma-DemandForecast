"""HTTP contract tests for POST /v1/reorder-suggestions (offline, TestClient)."""

from __future__ import annotations

from demandforecast.api.routes.forecast import get_forecast_backend
from demandforecast.services.forecast_service import DeterministicForecastBackend


def _install_backend(app, backend=None):
    backend = backend or DeterministicForecastBackend()
    app.dependency_overrides[get_forecast_backend] = lambda: backend
    return backend


def test_reorder_one_item_per_sku(app, client):
    _install_backend(app)
    resp = client.post(
        "/v1/reorder-suggestions",
        json={
            "pharmacy_id": "ph-001",
            "skus": [
                {"sku_id": "sku-001", "current_stock": 0},
                {"sku_id": "sku-002", "current_stock": 50},
            ],
            "horizon_days": 14,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["suggestions"]) == 2
    assert {s["sku_id"] for s in data["suggestions"]} == {"sku-001", "sku-002"}
    for s in data["suggestions"]:
        assert s["reason"]  # non-empty
        assert s["drug_name"]
        assert s["suggested_order_qty"] >= 0


def test_reorder_overstocked_sku_yields_zero(app, client):
    _install_backend(app)
    resp = client.post(
        "/v1/reorder-suggestions",
        json={
            "pharmacy_id": "ph-001",
            "skus": [{"sku_id": "sku-001", "current_stock": 1_000_000}],
            "horizon_days": 30,
        },
    )
    assert resp.status_code == 200
    item = resp.json()["suggestions"][0]
    assert item["suggested_order_qty"] == 0


def test_reorder_low_stock_sku_yields_positive(app, client):
    _install_backend(app)
    resp = client.post(
        "/v1/reorder-suggestions",
        json={
            "pharmacy_id": "ph-001",
            "skus": [{"sku_id": "sku-001", "current_stock": 0}],
            "horizon_days": 30,
        },
    )
    assert resp.status_code == 200
    item = resp.json()["suggestions"][0]
    assert item["suggested_order_qty"] > 0


def test_reorder_expected_stockout_avoided_is_sum(app, client):
    _install_backend(app)
    resp = client.post(
        "/v1/reorder-suggestions",
        json={
            "pharmacy_id": "ph-001",
            "skus": [
                {"sku_id": "sku-001", "current_stock": 0},
                {"sku_id": "sku-002", "current_stock": 1_000_000},
            ],
            "horizon_days": 14,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    total = sum(s["suggested_order_qty"] for s in data["suggestions"])
    assert data["expected_stockout_avoided"] == total
    assert data["expected_stockout_avoided"] >= 0


def test_reorder_determinism(app, client):
    _install_backend(app)
    body = {
        "pharmacy_id": "ph-001",
        "skus": [{"sku_id": "sku-001", "current_stock": 5}],
        "horizon_days": 14,
    }
    r1 = client.post("/v1/reorder-suggestions", json=body)
    r2 = client.post("/v1/reorder-suggestions", json=body)
    assert r1.json()["suggestions"] == r2.json()["suggestions"]


def test_reorder_validation_negative_stock_422(app, client):
    _install_backend(app)
    resp = client.post(
        "/v1/reorder-suggestions",
        json={
            "pharmacy_id": "ph-001",
            "skus": [{"sku_id": "sku-001", "current_stock": -5}],
        },
    )
    assert resp.status_code == 422


def test_reorder_validation_missing_skus_422(app, client):
    _install_backend(app)
    resp = client.post(
        "/v1/reorder-suggestions",
        json={"pharmacy_id": "ph-001"},
    )
    assert resp.status_code == 422
