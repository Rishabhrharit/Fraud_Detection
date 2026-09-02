"""Evaluate the model, tune the decision threshold, and save plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.metrics import auc as sklearn_auc

from src.train import load_split_data

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = ROOT / "models"


def load_evaluation_data() -> tuple[pd.DataFrame, pd.Series, object]:
    """Load evaluation data and the trained XGBoost model.

    Args:
        None.

    Returns:
        The X_test matrix, y_test labels, and the saved XGBoost model.
    """
    import joblib

    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["Class"]
    model = joblib.load(MODEL_DIR / "xgb_model.pkl")
    return X_test, y_test, model


def save_roc_and_pr_curves(y_true: pd.Series, y_prob: np.ndarray) -> None:
    """Generate and save ROC and precision-recall curves.

    Args:
        y_true: Ground-truth labels.
        y_prob: Positive-class probabilities.

    Returns:
        None.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "roc_curve.png", dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = sklearn_auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pr_curve.png", dpi=200)
    plt.close()


def tune_threshold(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    """Tune threshold to minimize false positives under recall constraints.

    Args:
        y_true: Ground-truth labels.
        y_prob: Positive-class probabilities.

    Returns:
        A dictionary with threshold and selected metrics.
    """
    best = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "false_positives": float("inf"),
    }

    for threshold in np.arange(0.10, 0.91, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        false_positives = ((y_pred == 1) & (y_true == 0)).sum()

        if recall >= 0.7 and (
            false_positives < best["false_positives"]
            or (false_positives == best["false_positives"] and f1 > best["f1"])
        ):
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_positives": int(false_positives),
            }

    return best


def save_confusion_matrix(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> None:
    """Generate the confusion matrix at the tuned threshold.

    Args:
        y_true: Ground-truth labels.
        y_prob: Positive-class probabilities.
        threshold: Decision threshold used for classification.

    Returns:
        None.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix (threshold = {threshold:.2f})")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def main() -> None:
    """Run evaluation, threshold tuning, and artifact saving."""
    X_test, y_test, model = load_evaluation_data()
    y_prob = model.predict_proba(X_test)[:, 1]
    save_roc_and_pr_curves(y_test, y_prob)

    default_threshold = 0.5
    default_pred = (y_prob >= default_threshold).astype(int)
    default_precision = precision_score(y_test, default_pred, zero_division=0)
    default_recall = recall_score(y_test, default_pred, zero_division=0)
    default_f1 = f1_score(y_test, default_pred, zero_division=0)
    default_false_positives = ((default_pred == 1) & (y_test == 0)).sum()

    tuned = tune_threshold(y_test, y_prob)
    tuned_pred = (y_prob >= tuned["threshold"]).astype(int)
    tuned_precision = precision_score(y_test, tuned_pred, zero_division=0)
    tuned_recall = recall_score(y_test, tuned_pred, zero_division=0)
    tuned_f1 = f1_score(y_test, tuned_pred, zero_division=0)
    tuned_false_positives = ((tuned_pred == 1) & (y_test == 0)).sum()
    false_positive_reduction = (
        ((default_false_positives - tuned_false_positives) / default_false_positives) * 100.0
        if default_false_positives > 0
        else 0.0
    )

    print("Default threshold (0.5):")
    print({
        "precision": default_precision,
        "recall": default_recall,
        "f1": default_f1,
        "false_positives": int(default_false_positives),
    })
    print("Tuned threshold:")
    print({
        "threshold": tuned["threshold"],
        "precision": tuned_precision,
        "recall": tuned_recall,
        "f1": tuned_f1,
        "false_positives": int(tuned_false_positives),
    })
    print(f"False positive reduction: {false_positive_reduction:.2f}%")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "threshold.json").write_text(json.dumps({"threshold": float(tuned["threshold"])}), encoding="utf-8")
    save_confusion_matrix(y_test, y_prob, tuned["threshold"])


if __name__ == "__main__":
    main()
