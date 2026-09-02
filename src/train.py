"""Train and track the fraud-detection models with MLflow."""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def load_split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the processed train/test data artifacts.

    Args:
        None.

    Returns:
        X_train, X_test, y_train, y_test arrays.
    """
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["Class"]
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["Class"]
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    """Compute core fraud-detection metrics for a model.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        A dictionary containing ROC-AUC, precision, recall, and F1 metrics.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def log_model_metrics(run_name: str, model_name: str, y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    """Log metrics for a trained model to MLflow.

    Args:
        run_name: MLflow run name.
        model_name: Human-readable model label.
        y_true: Target labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        A metrics dictionary.
    """
    metrics = compute_metrics(y_true, y_prob)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"model_name": model_name})
        mlflow.log_metrics({
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
        })
    return metrics


def main() -> None:
    """Train baseline, tree, and XGBoost models, then compare their performance."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlruns_dir = ROOT / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment("credit-card-fraud-detection")
    mlflow.xgboost.autolog()

    X_train, X_test, y_train, y_test = load_split_data()
    model_results: list[dict[str, object]] = []

    logistic_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    logistic_model.fit(X_train, y_train)
    logistic_prob = logistic_model.predict_proba(X_test)[:, 1]
    logistic_metrics = compute_metrics(y_test, logistic_prob)
    with mlflow.start_run(run_name="logistic-regression") as run:
        mlflow.log_params({
            "model_name": "logistic_regression",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        })
        mlflow.log_metrics({
            "roc_auc": logistic_metrics["roc_auc"],
            "precision": logistic_metrics["precision"],
            "recall": logistic_metrics["recall"],
            "f1": logistic_metrics["f1"],
        })
        mlflow.sklearn.log_model(logistic_model, artifact_path="model")
    model_results.append({
        "model": "Logistic Regression",
        "roc_auc": logistic_metrics["roc_auc"],
        "f1": logistic_metrics["f1"],
        "precision": logistic_metrics["precision"],
        "recall": logistic_metrics["recall"],
    })

    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = compute_metrics(y_test, rf_prob)
    with mlflow.start_run(run_name="random-forest") as run:
        mlflow.log_params({
            "model_name": "random_forest",
            "n_estimators": 200,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": 42,
        })
        mlflow.log_metrics({
            "roc_auc": rf_metrics["roc_auc"],
            "precision": rf_metrics["precision"],
            "recall": rf_metrics["recall"],
            "f1": rf_metrics["f1"],
        })
        mlflow.sklearn.log_model(rf_model, artifact_path="model")
    model_results.append({
        "model": "Random Forest",
        "roc_auc": rf_metrics["roc_auc"],
        "f1": rf_metrics["f1"],
        "precision": rf_metrics["precision"],
        "recall": rf_metrics["recall"],
    })

    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
        eval_metric="auc",
        random_state=42,
        objective="binary:logistic",
        use_label_encoder=False,
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False,
    )
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics(y_test, xgb_prob)
    with mlflow.start_run(run_name="xgboost") as run:
        mlflow.log_params({
            "model_name": "xgboost",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": float((y_train == 0).sum() / (y_train == 1).sum()),
            "eval_metric": "auc",
            "random_state": 42,
            "early_stopping_rounds": 50,
        })
        mlflow.log_metrics({
            "roc_auc": xgb_metrics["roc_auc"],
            "precision": xgb_metrics["precision"],
            "recall": xgb_metrics["recall"],
            "f1": xgb_metrics["f1"],
        })
        mlflow.xgboost.log_model(xgb_model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model = mlflow.register_model(model_uri=model_uri, name="fraud-detector-ulb")
        print(f"Registered model in MLflow registry: {registered_model.name} version {registered_model.version}")
    model_results.append({
        "model": "XGBoost",
        "roc_auc": xgb_metrics["roc_auc"],
        "f1": xgb_metrics["f1"],
        "precision": xgb_metrics["precision"],
        "recall": xgb_metrics["recall"],
    })

    joblib.dump(xgb_model, MODEL_DIR / "xgb_model.pkl")
    print("Model comparison table:")
    print("model_name | roc_auc | f1 | precision | recall")
    for result in model_results:
        print(
            f"{result['model']} | {result['roc_auc']:.4f} | {result['f1']:.4f} | "
            f"{result['precision']:.4f} | {result['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
