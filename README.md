# Credit Card Fraud Detection — End-to-End ML Pipeline

Production-ready fraud detection system built on the ULB Credit Card Fraud Detection dataset, using XGBoost, model explainability, MLflow tracking, and a FastAPI prediction service.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-FFB000?logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?logo=mlflow&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.45-8A2BE2)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)

## Overview

This project demonstrates an end-to-end fraud detection workflow for a highly imbalanced classification problem. The dataset contains real credit card transactions, where genuine purchases dominate and fraudulent activity is a rare anomaly. Because false positives are operationally expensive in financial products, the project emphasizes threshold tuning, explainable predictions, and a deployable API service.

The system covers the full lifecycle: ingesting data, engineering features, training multiple models, comparing performance in MLflow, tuning the decision threshold for business constraints, generating SHAP explanations, and serving predictions through a REST API.

## Business context

Fraud detection is one of the most important applications of machine learning in financial services. In practice, the class imbalance is severe: most transactions are legitimate, while only a small fraction are malicious. A model that maximizes raw accuracy is not useful here because it often becomes biased toward the majority class. A production-grade system must be tuned to minimize false positives without sacrificing recall, while also explaining why a transaction was flagged.

This project balances classification quality and operational usefulness by combining:
- strong tree-based modeling with XGBoost
- class weighting instead of synthetic oversampling
- tuned decision thresholds for fraud-prioritized precision/recall trade-offs
- SHAP-based explanations for human review and trust

## Architecture

```text
creditcard.csv
    ↓
ingest.py
    ↓
preprocess.py
    ↓
train.py
    ↓
MLflow tracking + model registry
    ↓
model evaluation + threshold tuning
    ↓
FastAPI /predict service
    ↓
SHAP explanation outputs
```

## Project structure

```text
credit-card-fraud-detection/
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── api/
│   ├── main.py
│   └── schemas.py
├── models/
├── outputs/
│   ├── plots/
│   └── shap/
├── tests/
│   └── test_api.py
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md
└── .env
```

## Key results

This project is designed for the official ULB dataset with the following expected characteristics:
- 284,807 transactions
- 30 feature columns: Time, Amount, V1-V28, Class
- Fraud rate: approximately 0.17%
- Class target: 0 = genuine, 1 = fraud

Local validation run in this workspace showed:
- Dataset size: 284,807 transactions
- XGBoost ROC-AUC: 0.9831
- XGBoost precision at threshold 0.5: 0.3415
- XGBoost recall at threshold 0.5: 0.8571
- XGBoost F1 at threshold 0.5: 0.4884
- Tuned threshold: 0.90
- Precision at tuned threshold: 0.8488
- Recall at tuned threshold: 0.7449
- F1 at tuned threshold: 0.7935
- Pytest result: 4 passed, 0 failed

> The real ULB CSV is not downloaded programmatically; the user places it in data/raw/creditcard.csv manually before running the pipeline.

## Setup and execution

### 1) Place the dataset

Put the downloaded CSV at:

```bash
data/raw/creditcard.csv
```

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the pipeline in order

```bash
python src/ingest.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/explain.py
```

### 5) Start the API

```bash
uvicorn api.main:app --reload
```

### 6) Run tests

```bash
pytest tests/ -v
```

## API usage

### Health check

```bash
curl http://localhost:8000/health
```

### Fraud prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 1000,
    "Amount": 50.0,
    "V1": -0.3,
    "V2": 0.2,
    "V3": 0.1,
    "V4": -0.2,
    "V5": 0.1,
    "V6": -0.3,
    "V7": 0.2,
    "V8": 0.1,
    "V9": -0.2,
    "V10": 0.0,
    "V11": -0.1,
    "V12": 0.2,
    "V13": 0.1,
    "V14": -0.3,
    "V15": 0.2,
    "V16": 0.0,
    "V17": -0.2,
    "V18": 0.1,
    "V19": 0.2,
    "V20": -0.1,
    "V21": 0.0,
    "V22": 0.2,
    "V23": -0.1,
    "V24": 0.3,
    "V25": -0.2,
    "V26": 0.0,
    "V27": 0.1,
    "V28": -0.1
  }'
```

## Model comparison

| Model | ROC-AUC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9717 | 0.1059 | 0.0562 | 0.9082 |
| Random Forest | 0.9624 | 0.8539 | 0.9500 | 0.7755 |
| XGBoost | 0.9831 | 0.4884 | 0.3415 | 0.8571 |

## Design decisions

- XGBoost over neural networks: strong performance, fast inference, and better interpretability for tabular fraud data
- Threshold tuning: fraud detection has an asymmetric cost profile, so false positives are more damaging than missed detections in many operational settings
- SHAP TreeExplainer: exact feature attribution for tree-based models, making predictions easier to explain and audit
- Class weighting over SMOTE: handles imbalance without generating synthetic minority samples that may distort the real pattern
- PCA features (V1-V28): already anonymized and reduced, so no additional dimensionality reduction is needed

## Portfolio value

This project highlights practical ML engineering skills expected in production data science teams:
- data validation and preprocessing
- feature engineering for imbalanced data
- model comparison and training pipelines
- MLflow experiment tracking and model registry usage
- evaluation and threshold optimization
- SHAP explainability for stakeholder trust
- deployment-ready REST API
- test coverage for endpoint behavior

## Docker

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

## Notes

- The real dataset must be placed manually in data/raw/creditcard.csv before running the pipeline.
- The project is intentionally structured for GitHub portfolio use and can be extended with CI/CD, monitoring, and model retraining workflows.
