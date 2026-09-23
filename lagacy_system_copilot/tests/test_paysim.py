import csv

import pytest

from cold_chain.paysim import PaySimRepository

HEADERS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
    "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


@pytest.fixture
def paysim_csv(tmp_path):
    path = tmp_path / "paysim.csv"
    rows = [
        [1, "PAYMENT", 100.0, "C1", 500.0, 400.0, "M1", 0.0, 0.0, 0, 0],
        [2, "TRANSFER", 200.0, "C2", 300.0, 100.0, "C3", 50.0, 250.0, 1, 0],
        [2, "CASH_OUT", 300.0, "C2", 100.0, 0.0, "C4", 0.0, 300.0, 1, 1],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADERS)
        writer.writerows(rows)
    return path


def test_metrics_and_risk_analysis(paysim_csv):
    repo = PaySimRepository(str(paysim_csv), max_rows=10)
    assert '"transaction_count": 2' in repo.transaction_metrics(2, 2, None)
    risk = repo.fraud_risk(1, 2)
    assert '"TRANSFER"' in risk
    assert '"fraud_rate_percent": 100.0' in risk


def test_account_activity_and_anomalies(paysim_csv):
    repo = PaySimRepository(str(paysim_csv), max_rows=10)
    assert '"nameOrig": "C2"' in repo.account_activity("C2", 1, 2)
    assert '"isFraud": 1' in repo.balance_anomalies(1, 2, None)


def test_preview_returns_top_rows(paysim_csv):
    preview = PaySimRepository(str(paysim_csv), max_rows=10).preview_transactions(1, 2, None, 2)
    assert preview.count('"step"') == 2
    assert '"nameOrig": "C1"' in preview


def test_rejects_invalid_step_range(paysim_csv):
    with pytest.raises(ValueError, match="start_step"):
        PaySimRepository(str(paysim_csv), max_rows=10).transaction_metrics(3, 2, None)
