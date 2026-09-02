"""Unit tests for the fraud-detection FastAPI application."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def _base_payload(**overrides):
    """Return a valid transaction payload for testing."""
    payload = {
        "Time": 1000.0,
        "Amount": 50.0,
        "V1": -0.3,
        "V2": 0.2,
        "V3": 0.0,
        "V4": 0.1,
        "V5": -0.2,
        "V6": 0.4,
        "V7": -0.1,
        "V8": 0.3,
        "V9": 0.2,
        "V10": -0.5,
        "V11": 0.1,
        "V12": 0.0,
        "V13": -0.2,
        "V14": 0.2,
        "V15": -0.1,
        "V16": 0.3,
        "V17": -0.2,
        "V18": 0.0,
        "V19": 0.4,
        "V20": -0.1,
        "V21": 0.2,
        "V22": 0.0,
        "V23": -0.2,
        "V24": 0.1,
        "V25": -0.3,
        "V26": 0.2,
        "V27": -0.1,
        "V28": 0.0,
    }
    payload.update(overrides)
    return payload


def test_health(client: TestClient) -> None:
    """The health endpoint should report model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True


def test_predict_valid(client: TestClient) -> None:
    """A valid payload should be accepted and scored."""
    payload = _base_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_predict_missing_field(client: TestClient) -> None:
    """Missing required fields should return a 422 validation error."""
    payload = _base_payload()
    payload.pop("V28")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_high_amount(client: TestClient) -> None:
    """A very large amount should still return a valid prediction."""
    payload = _base_payload(Amount=50000.0, V1=0.0, V2=0.0, V3=0.0, V4=0.0, V5=0.0, V6=0.0, V7=0.0, V8=0.0, V9=0.0,
                            V10=0.0, V11=0.0, V12=0.0, V13=0.0, V14=0.0, V15=0.0, V16=0.0, V17=0.0,
                            V18=0.0, V19=0.0, V20=0.0, V21=0.0, V22=0.0, V23=0.0, V24=0.0, V25=0.0,
                            V26=0.0, V27=0.0, V28=0.0)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
