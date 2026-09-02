"""Pydantic schemas for the fraud-detection API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Schema for a single credit-card transaction prediction request."""

    Time: float = Field(..., description="Transaction timestamp in seconds from Unix epoch")
    Amount: float = Field(..., description="Transaction amount")
    V1: float = Field(...)
    V2: float = Field(...)
    V3: float = Field(...)
    V4: float = Field(...)
    V5: float = Field(...)
    V6: float = Field(...)
    V7: float = Field(...)
    V8: float = Field(...)
    V9: float = Field(...)
    V10: float = Field(...)
    V11: float = Field(...)
    V12: float = Field(...)
    V13: float = Field(...)
    V14: float = Field(...)
    V15: float = Field(...)
    V16: float = Field(...)
    V17: float = Field(...)
    V18: float = Field(...)
    V19: float = Field(...)
    V20: float = Field(...)
    V21: float = Field(...)
    V22: float = Field(...)
    V23: float = Field(...)
    V24: float = Field(...)
    V25: float = Field(...)
    V26: float = Field(...)
    V27: float = Field(...)
    V28: float = Field(...)


class TopFeature(BaseModel):
    """SHAP top feature entry returned by the API."""

    feature: str
    shap_value: float
    direction: str


class PredictionOutput(BaseModel):
    """API response payload describing the model prediction."""

    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    top_features: List[TopFeature]
