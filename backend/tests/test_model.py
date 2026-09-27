"""The artifacts train.py writes must load and behave as the API expects."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

EXPECTED_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

SAMPLE = {
    "MedInc": 8.3,
    "HouseAge": 25,
    "AveRooms": 6.2,
    "AveBedrms": 1.1,
    "Population": 1500,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25,
}


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_DIR / "model.joblib")


@pytest.fixture(scope="module")
def importances():
    with open(MODEL_DIR / "feature_importances.json") as f:
        return json.load(f)


def test_model_artifact_exists():
    assert (MODEL_DIR / "model.joblib").is_file()


def test_model_predicts_one_value_per_row(model):
    row = pd.DataFrame([SAMPLE], columns=EXPECTED_FEATURES)
    preds = model.predict(row)
    assert preds.shape == (1,)
    assert isinstance(float(preds[0]), float)


def test_model_prediction_is_in_target_range(model):
    # The California Housing target is median value in $100k, clipped at 5.0.
    row = pd.DataFrame([SAMPLE], columns=EXPECTED_FEATURES)
    assert 0 < float(model.predict(row)[0]) <= 5.0


def test_model_was_trained_on_expected_features(model):
    assert list(model.feature_names_in_) == EXPECTED_FEATURES


def test_importances_cover_all_eight_features(importances):
    assert sorted(importances) == sorted(EXPECTED_FEATURES)


def test_importances_are_floats_summing_to_one(importances):
    assert all(isinstance(v, float) for v in importances.values())
    assert sum(importances.values()) == pytest.approx(1.0)
