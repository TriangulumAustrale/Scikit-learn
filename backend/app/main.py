"""FastAPI service that predicts California median house values.

The model artifact and its feature importances are produced by train.py and
loaded once at import time, so predictions never touch disk.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

model = joblib.load(MODEL_DIR / "model.joblib")

with open(MODEL_DIR / "feature_importances.json") as f:
    FEATURE_IMPORTANCES = json.load(f)

# Column order the model was trained on; a DataFrame in this order keeps
# scikit-learn from warning about mismatched feature names.
FEATURE_NAMES = list(FEATURE_IMPORTANCES)

TOP_FEATURES = [
    {"feature": name, "importance": round(importance, 4)}
    for name, importance in sorted(
        FEATURE_IMPORTANCES.items(), key=lambda kv: kv[1], reverse=True
    )[:3]
]

app = FastAPI(title="House Price Predictor")


class HouseFeatures(BaseModel):
    """One California census block group. Ranges bracket the training data."""

    MedInc: float = Field(..., ge=0.1, le=20.0, description="Median income (10k USD)")
    HouseAge: float = Field(..., ge=1, le=60, description="Median house age (years)")
    AveRooms: float = Field(..., gt=0, le=150, description="Average rooms per household")
    AveBedrms: float = Field(..., gt=0, le=40, description="Average bedrooms per household")
    Population: float = Field(..., ge=1, le=40000, description="Block group population")
    AveOccup: float = Field(..., gt=0, le=1300, description="Average household occupancy")
    Latitude: float = Field(..., ge=32, le=42, description="Latitude (California)")
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude (California)")


class TopFeature(BaseModel):
    feature: str
    importance: float


class Prediction(BaseModel):
    predicted_value: float
    predicted_value_usd: int
    top_features: list[TopFeature]


@app.post("/predict", response_model=Prediction)
def predict(features: HouseFeatures):
    row = pd.DataFrame([features.model_dump()], columns=FEATURE_NAMES)
    # Round once, so the dollar figure is always derivable from the value shown.
    value = round(float(model.predict(row)[0]), 4)
    return Prediction(
        predicted_value=value,
        # The target is median house value in $100,000s.
        predicted_value_usd=round(value * 100_000),
        top_features=TOP_FEATURES,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
