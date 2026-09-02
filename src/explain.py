"""Generate SHAP explanations for model predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
SHAP_DIR = OUTPUT_DIR / "shap"


def load_explanation_inputs() -> tuple[object, pd.DataFrame, list[str]]:
    """Load the trained XGBoost model and the evaluation set.

    Args:
        None.

    Returns:
        The model, a sample of X_test, and the feature names list.
    """
    model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    with open(PROCESSED_DIR / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return model, X_test.sample(n=min(500, len(X_test)), random_state=42), feature_names


def explain_prediction(feature_dict: dict[str, float], model: object, explainer: shap.Explainer, feature_names: list[str]) -> dict[str, object]:
    """Explain a single prediction using the top SHAP feature contributions.

    Args:
        feature_dict: Dictionary of feature values in model input order.
        model: Trained model.
        explainer: SHAP explainer object.
        feature_names: Ordered feature names used during training.

    Returns:
        A dictionary describing the prediction probability, fraud flag, and top contributors.
    """
    ordered = {name: float(feature_dict[name]) for name in feature_names}
    row = pd.DataFrame([ordered], columns=feature_names)
    shap_values = explainer(row)
    base_value = float(shap_values.base_values[0]) if hasattr(shap_values, "base_values") else 0.0
    pred_prob = float(model.predict_proba(row)[0, 1])
    top_features = []
    for item in shap_values[0][:5]:
        feature_name = feature_names[int(item.feature)] if hasattr(item, "feature") else str(item)
        shap_value = float(item.values)
        top_features.append({
            "feature": feature_name,
            "shap_value": shap_value,
            "direction": "positive" if shap_value >= 0 else "negative",
        })
    return {
        "prediction_probability": pred_prob,
        "is_fraud": bool(pred_prob >= 0.5),
        "base_value": base_value,
        "top_features": top_features,
    }


def main() -> None:
    """Generate summary and per-instance SHAP plots for the XGBoost model."""
    model, X_test, feature_names = load_explanation_inputs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "summary_plot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap.Explanation(values=shap_values, data=X_test.values, feature_names=feature_names), max_display=10)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "bar_plot.png", dpi=200)
    plt.close()

    first_record = X_test.iloc[[0]]
    shap.waterfall_plot(explainer(first_record)[0], max_display=10)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "waterfall_plot.png", dpi=200)
    plt.close()

    print(f"Saved SHAP plots to {SHAP_DIR}")


if __name__ == "__main__":
    main()
