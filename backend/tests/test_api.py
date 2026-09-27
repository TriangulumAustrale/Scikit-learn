"""Endpoint behaviour for /health and /predict."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "MedInc": 8.3,
    "HouseAge": 25,
    "AveRooms": 6.2,
    "AveBedrms": 1.1,
    "Population": 1500,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25,
}


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_expected_keys():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"predicted_value", "predicted_value_usd", "top_features"}


def test_predict_values_are_consistent():
    body = client.post("/predict", json=VALID_PAYLOAD).json()
    assert 0 < body["predicted_value"] <= 5.0
    assert body["predicted_value_usd"] == round(body["predicted_value"] * 100_000)


def test_predict_reports_three_features_by_descending_importance():
    top = client.post("/predict", json=VALID_PAYLOAD).json()["top_features"]
    assert len(top) == 3
    assert all(set(f) == {"feature", "importance"} for f in top)
    assert [f["importance"] for f in top] == sorted(
        (f["importance"] for f in top), reverse=True
    )


def test_predict_rejects_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Longitude"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "Longitude"]


@pytest.mark.parametrize(
    "field, value",
    [("Latitude", 99), ("Longitude", 0), ("MedInc", -1), ("AveOccup", 0)],
)
def test_predict_rejects_out_of_range_field(field, value):
    response = client.post("/predict", json={**VALID_PAYLOAD, field: value})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


def test_predict_rejects_non_numeric_field():
    response = client.post("/predict", json={**VALID_PAYLOAD, "MedInc": "abc"})
    assert response.status_code == 422
