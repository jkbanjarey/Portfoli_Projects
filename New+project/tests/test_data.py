import pandas as pd
import pytest

from src.data import temporal_split, validate_frame
from src.features import MODEL_FEATURES, build_features


def frame():
    rows = []
    for step in range(1, 11):
        rows.append({"step": step, "type": "TRANSFER", "amount": 100.0, "nameOrig": f"C{step}", "oldbalanceOrg": 100.0, "newbalanceOrig": 0.0, "nameDest": f"M{step}", "oldbalanceDest": 0.0, "newbalanceDest": 100.0, "isFraud": step % 2, "isFlaggedFraud": 0})
    return pd.DataFrame(rows)


def test_features_exclude_raw_identifiers():
    features = build_features(frame())
    assert list(features.columns) == MODEL_FEATURES
    assert "nameOrig" not in features and "nameDest" not in features


def test_temporal_split_has_no_time_overlap():
    train, validation, test = temporal_split(frame())
    assert train.step.max() < validation.step.min() < test.step.min()


def test_invalid_transaction_type_is_rejected():
    bad = frame()
    bad.loc[0, "type"] = "INVALID"
    with pytest.raises(ValueError, match="Invalid transaction type"):
        validate_frame(bad)
