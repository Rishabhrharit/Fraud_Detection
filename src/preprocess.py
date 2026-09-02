"""Engineer features, split data, and fit the scaler."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_PARQUET_PATH = PROCESSED_DIR / "creditcard.parquet"
MODEL_DIR = ROOT / "models"
FEATURE_NAMES_PATH = PROCESSED_DIR / "feature_names.json"
AMOUNT_STATS_PATH = MODEL_DIR / "amount_stats.json"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time and amount features used by the fraud model.

    Args:
        df: Raw credit card dataset.

    Returns:
        A DataFrame with engineered fraud features.
    """
    engineered = df.copy()
    engineered["hour_of_day"] = (engineered["Time"] % 86400) // 3600
    engineered["amt_log"] = np.log1p(engineered["Amount"])
    engineered["amt_zscore"] = (
        engineered["Amount"] - engineered["Amount"].mean()
    ) / engineered["Amount"].std()
    engineered = engineered.drop(columns=["Time", "Amount"])
    return engineered


def main() -> None:
    """Train/test split and scaling for the model pipeline."""
    if not RAW_PARQUET_PATH.exists():
        print("Please place creditcard.csv in data/raw/ and re-run")
        raise SystemExit(1)

    df = pd.read_parquet(RAW_PARQUET_PATH)
    print(f"Preparing data from {RAW_PARQUET_PATH}")

    engineered = engineer_features(df)
    feature_names = [col for col in engineered.columns if col.startswith("V")] + [
        "hour_of_day",
        "amt_log",
        "amt_zscore",
    ]

    X = engineered[feature_names]
    y = engineered["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train_scaled.to_parquet(PROCESSED_DIR / "X_train.parquet", index=False)
    X_test_scaled.to_parquet(PROCESSED_DIR / "X_test.parquet", index=False)
    pd.DataFrame({"Class": y_train}).to_parquet(PROCESSED_DIR / "y_train.parquet", index=False)
    pd.DataFrame({"Class": y_test}).to_parquet(PROCESSED_DIR / "y_test.parquet", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    FEATURE_NAMES_PATH.write_text(json.dumps(feature_names), encoding="utf-8")
    amount_stats = {
        "amount_mean": float(df["Amount"].mean()),
        "amount_std": float(df["Amount"].std()),
    }
    AMOUNT_STATS_PATH.write_text(json.dumps(amount_stats), encoding="utf-8")

    print("Train class distribution:")
    print(y_train.value_counts())
    print("Test class distribution:")
    print(y_test.value_counts())
    print(f"Saved scaled train/test features to {PROCESSED_DIR}")
    print(f"Saved scaler to {MODEL_DIR / 'scaler.pkl'}")
    print(f"Saved feature names to {FEATURE_NAMES_PATH}")


if __name__ == "__main__":
    main()
