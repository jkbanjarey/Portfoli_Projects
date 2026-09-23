# PaySim Fraud Detection — End-to-End MLOps

An end-to-end fraud-alerting project based on the PaySim mobile-money simulation dataset. It trains a time-aware classifier, tracks experiments in MLflow, serves calibrated fraud scores with FastAPI, records privacy-conscious prediction events, and produces local drift reports.

## What it includes

- Chronological train/validation/test splitting using transaction `step`.
- ID-safe feature engineering: account IDs are never model features or prediction logs.
- Logistic-regression baseline and histogram-gradient-boosting candidate comparison.
- Local MLflow experiment tracking and `paysim-fraud-detector` model registration.
- High-recall alert threshold selected from validation data.
- Single-transaction and batch CSV FastAPI scoring endpoints.
- Prediction audit logging and PSI-based feature drift report.
- Docker Compose, automated tests, and GitHub Actions CI.

## Quick start

Create and activate a Python 3.11 virtual environment, then install the package:

```powershell
pip install -e .
python -m src.pipeline.train
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Training reads `PaySim_Dataset/PS_20174392719_1491204439457_log.csv` by default. It writes the champion bundle to `artifacts/champion/` and the MLflow database to `mlflow.db`.

Open `http://localhost:8000/docs` to use the interactive API. For a batch request, upload [sample_transactions.csv](sample_transactions.csv) to `POST /predict/batch`.

## Docker workflow

The data is mounted read-only and is never embedded in the image:

```powershell
docker compose run --rm train
docker compose up api mlflow
```

The scoring API is available at `http://localhost:8000`; MLflow is at `http://localhost:5000`.

## Operational commands

```powershell
pytest -q
python -m src.monitoring
```

The drift command compares non-identifying logged feature values against the champion's training reference profile and writes `reports/drift_report.json` plus `reports/drift_report.html`. A numeric PSI of 0.2 or higher is marked as drift.

## API contract

`POST /predict` accepts the PaySim raw transaction columns excluding `isFraud` and returns:

```json
{
  "fraud_probability": 0.91,
  "fraud_alert": true,
  "threshold": 0.42,
  "model_version": "<mlflow-run-id>"
}
```

`GET /health` exposes model readiness, and `GET /model-info` returns the champion's feature contract and evaluation metrics.

## Notes

The alert is designed to route transactions for review, not automatically block them. Its threshold intentionally favors recall, so false-positive alerts are an expected trade-off. The included DVC pointer remains available for dataset provenance; this project uses local MLflow rather than a DVC pipeline.
