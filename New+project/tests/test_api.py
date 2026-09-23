import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src import api


class FakeModel:
    def predict_proba(self, features):
        return np.column_stack([np.full(len(features), .2), np.full(len(features), .8)])


def transaction():
    return {"step": 1, "type": "TRANSFER", "amount": 100.0, "nameOrig": "C1", "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0, "nameDest": "M1", "oldbalanceDest": 0.0, "newbalanceDest": 100.0, "isFlaggedFraud": 0}


def configured_client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "append_prediction_events", lambda *args, **kwargs: None)
    api._model = FakeModel()
    api._metadata = {"model_name": "paysim-fraud-detector", "candidate": "test", "threshold": .5, "features": [], "validation_metrics": {}, "test_metrics": {}, "mlflow_run_id": "test-run"}
    return TestClient(api.app)


def test_single_prediction(tmp_path, monkeypatch):
    client = configured_client(tmp_path, monkeypatch)
    response = client.post("/predict", json=transaction())
    assert response.status_code == 200
    assert response.json()["fraud_alert"] is True


def test_batch_prediction(tmp_path, monkeypatch):
    client = configured_client(tmp_path, monkeypatch)
    payload = pd.DataFrame([transaction()]).to_csv(index=False).encode()
    response = client.post("/predict/batch", files={"file": ("input.csv", payload, "text/csv")})
    assert response.status_code == 200
    assert "fraud_probability" in response.text


def test_invalid_payload_is_rejected(tmp_path, monkeypatch):
    client = configured_client(tmp_path, monkeypatch)
    bad = transaction()
    bad["amount"] = -1
    assert client.post("/predict", json=bad).status_code == 422
