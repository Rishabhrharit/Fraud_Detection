"""FastAPI app for fraud scoring predictions and health checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.explain import explain_prediction
from api.schemas import PredictionOutput, TopFeature, TransactionInput

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
AMOUNT_STATS_PATH = MODEL_DIR / "amount_stats.json"

app = FastAPI(title="Credit Card Fraud Detection API")

app.state.model = None
app.state.scaler = None
app.state.threshold = None
app.state.feature_names = None
app.state.explainer = None
app.state.amount_stats = {"amount_mean": 0.0, "amount_std": 1.0}


def ensure_model_loaded() -> None:
    """Load model artifacts only when needed, including under unit tests or API startup-less environments."""
    if app.state.model is not None and app.state.scaler is not None and app.state.feature_names is not None:
        return

    model_path = MODEL_DIR / "xgb_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    threshold_path = MODEL_DIR / "threshold.json"
    feature_names_path = PROCESSED_DIR / "feature_names.json"

    if not all(path.exists() for path in [model_path, scaler_path, threshold_path, feature_names_path]):
        raise RuntimeError("Model assets are missing. Run the training pipeline before starting the API.")

    app.state.model = joblib.load(model_path)
    app.state.scaler = joblib.load(scaler_path)
    app.state.threshold = json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"]
    app.state.feature_names = json.loads(feature_names_path.read_text(encoding="utf-8"))
    app.state.amount_stats = json.loads(AMOUNT_STATS_PATH.read_text(encoding="utf-8")) if AMOUNT_STATS_PATH.exists() else {"amount_mean": 0.0, "amount_std": 1.0}
    app.state.explainer = None
    print(f"API startup: loaded model and scaler at {datetime.now(timezone.utc).isoformat()}")


@app.on_event("startup")
def startup_event() -> None:
    """Load the trained model, scaler, threshold, and feature names on startup."""
    try:
        ensure_model_loaded()
    except RuntimeError:
        app.state.model = None
        app.state.scaler = None
        app.state.threshold = None
        app.state.feature_names = None


@app.get("/health")
def health() -> dict[str, Any]:
    """Return the API health status and loaded model metadata."""
    try:
        ensure_model_loaded()
    except RuntimeError:
        pass
    return {
        "model_loaded": app.state.model is not None,
        "threshold": app.state.threshold,
        "feature_count": len(app.state.feature_names) if app.state.feature_names else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def engineer_features(payload: dict[str, float]) -> pd.DataFrame:
    """Apply the same feature engineering used during preprocessing.

    Args:
        payload: Transaction data represented as a flat dictionary.

    Returns:
        A DataFrame containing engineered feature columns in the trained order.
    """
    df = pd.DataFrame([payload])
    amount_mean = float(app.state.amount_stats.get("amount_mean", 0.0))
    amount_std = float(app.state.amount_stats.get("amount_std", 1.0))
    df["hour_of_day"] = (df["Time"] % 86400) // 3600
    df["amt_log"] = np.log1p(df["Amount"])
    df["amt_zscore"] = (df["Amount"] - amount_mean) / amount_std
    features = [f"V{i}" for i in range(1, 29)]
    engineered = df[features + ["hour_of_day", "amt_log", "amt_zscore"]].copy()
    engineered.columns = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "hour_of_day", "amt_log", "amt_zscore"]
    return engineered


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: TransactionInput) -> PredictionOutput:
    """Score a transaction and return fraud probability with SHAP-driven explanations.

    Args:
        payload: Transaction data including the anonymized PCA features.

    Returns:
        A prediction output containing probability, fraud flag, threshold, and top SHAP features.
    """
    try:
        ensure_model_loaded()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    payload_dict = payload.model_dump()
    engineered = engineer_features(payload_dict)
    feature_order = app.state.feature_names
    aligned = engineered[feature_order]
    scaled = app.state.scaler.transform(aligned)
    prob = app.state.model.predict_proba(scaled)[0, 1]
    threshold = float(app.state.threshold)
    is_fraud = bool(prob >= threshold)

    from src.explain import shap

    if app.state.explainer is None:
        app.state.explainer = shap.TreeExplainer(app.state.model)

    explanation = explain_prediction(
        {name: float(value) for name, value in zip(feature_order, scaled[0])},
        app.state.model,
        app.state.explainer,
        feature_order,
    )
    top_features = [
        TopFeature(feature=item["feature"], shap_value=float(item["shap_value"]), direction=item["direction"])
        for item in explanation["top_features"]
    ]

    return PredictionOutput(
        fraud_probability=float(prob),
        is_fraud=is_fraud,
        threshold_used=threshold,
        top_features=top_features,
    )
