"""Train the California Housing price model and save it for the API to load.

Run from backend/:  python train.py
"""

import json
from pathlib import Path

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent / "model"


def main():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("R2:", r2_score(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib", compress=3)

    importances = dict(zip(X.columns, model.feature_importances_.tolist()))
    with open(MODEL_DIR / "feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2)


if __name__ == "__main__":
    main()
