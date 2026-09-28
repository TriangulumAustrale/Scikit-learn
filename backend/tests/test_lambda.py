"""The Mangum handler must answer API Gateway HTTP API (v2) events.

This exercises the Lambda entry point without deploying: if the payload
shape or the handler wiring is wrong, these fail the same way Lambda would.
"""

import json

from app.main import handler

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


def event(method, path, body=None):
    return {
        "version": "2.0",
        "routeKey": f"{method} {path}",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"content-type": "application/json"},
        "requestContext": {
            "http": {"method": method, "path": path, "sourceIp": "127.0.0.1"},
            "stage": "$default",
        },
        "body": json.dumps(body) if body is not None else None,
        "isBase64Encoded": False,
    }


def test_handler_serves_health():
    response = handler(event("GET", "/health"), None)
    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok"}


def test_handler_serves_predict():
    response = handler(event("POST", "/predict", VALID_PAYLOAD), None)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert set(body) == {"predicted_value", "predicted_value_usd", "top_features"}
    assert len(body["top_features"]) == 3


def test_handler_reports_validation_errors():
    bad = {**VALID_PAYLOAD, "Latitude": 99}
    response = handler(event("POST", "/predict", bad), None)
    assert response["statusCode"] == 422
