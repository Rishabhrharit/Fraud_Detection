"""Load, validate, and persist the raw fraud detection dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DATA_PATH = PROCESSED_DIR / "creditcard.parquet"


def load_creditcard_data() -> pd.DataFrame:
    """Load the dataset, validate it, and save a processed parquet file.

    Args:
        None.

    Returns:
        The loaded pandas DataFrame.
    """
    if not RAW_DATA_PATH.exists():
        print("Please place creditcard.csv in data/raw/ and re-run")
        raise SystemExit(1)

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['Class'].mean():.6f}")
    print("Class distribution:")
    print(df["Class"].value_counts())

    if "Class" not in df.columns:
        raise ValueError("The dataset must contain a Class column.")
    if df.isnull().any().any():
        raise ValueError("The dataset contains missing values.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")
    return df


def main() -> None:
    """Entry point for the ingestion pipeline."""
    load_creditcard_data()


if __name__ == "__main__":
    main()
