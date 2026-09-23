"""Read-only, local access to the PaySim CSV archive using Polars lazy queries."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import polars as pl

REQUIRED_COLUMNS = {
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
}


class PaySimRepository:
    def __init__(self, dataset_path: str, max_rows: int) -> None:
        self.dataset_path = Path(dataset_path)
        self.max_rows = max_rows

    def _csv_path(self) -> Path:
        if self.dataset_path.suffix.lower() == ".csv":
            csv_path = self.dataset_path
        elif self.dataset_path.suffix.lower() == ".zip":
            target_dir = Path("data") / self.dataset_path.stem
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self.dataset_path) as archive:
                csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(csv_names) != 1:
                    raise ValueError("PaySim archive must contain exactly one CSV file.")
                csv_path = target_dir / Path(csv_names[0]).name
                if not csv_path.exists():
                    with archive.open(csv_names[0]) as source, csv_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
        else:
            raise ValueError("PAYSIM_DATASET_PATH must point to a .zip or .csv file.")
        if not csv_path.is_file():
            raise FileNotFoundError(f"PaySim dataset not found: {csv_path}")
        return csv_path

    def _data(self) -> pl.LazyFrame:
        frame = pl.scan_csv(
            self._csv_path(),
            schema_overrides={
                "step": pl.Int64, "amount": pl.Float64, "oldbalanceOrg": pl.Float64,
                "newbalanceOrig": pl.Float64, "oldbalanceDest": pl.Float64,
                "newbalanceDest": pl.Float64, "isFraud": pl.Int64, "isFlaggedFraud": pl.Int64,
            },
        )
        missing = REQUIRED_COLUMNS - set(frame.collect_schema().names())
        if missing:
            raise ValueError(f"PaySim CSV is missing required columns: {sorted(missing)}")
        return frame

    @staticmethod
    def _filters(start_step: int, end_step: int, transaction_type: str | None) -> list[pl.Expr]:
        if start_step > end_step:
            raise ValueError("start_step must not be after end_step")
        filters = [pl.col("step").is_between(start_step, end_step)]
        if transaction_type:
            filters.append(pl.col("type") == transaction_type.upper())
        return filters

    @staticmethod
    def _json(frame: pl.DataFrame) -> str:
        return json.dumps(frame.to_dicts(), default=str)

    def transaction_metrics(self, start_step: int, end_step: int, transaction_type: str | None) -> str:
        result = self._data().filter(*self._filters(start_step, end_step, transaction_type)).select(
            pl.len().alias("transaction_count"),
            pl.col("amount").sum().round(2).alias("total_amount"),
            pl.col("amount").mean().round(2).alias("average_amount"),
            pl.col("isFraud").sum().alias("fraud_count"),
            pl.col("isFlaggedFraud").sum().alias("flagged_fraud_count"),
        ).collect()
        return self._json(result)

    def preview_transactions(
        self, start_step: int, end_step: int, transaction_type: str | None, limit: int
    ) -> str:
        """Return a small, ordered, read-only sample of raw PaySim records."""
        if limit < 1 or limit > self.max_rows:
            raise ValueError(f"limit must be between 1 and {self.max_rows}")
        result = self._data().filter(*self._filters(start_step, end_step, transaction_type)).select(
            "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
            "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
        # The source CSV is ordered by simulation step; avoid a global 6M-row sort.
        ).limit(limit).collect()
        return self._json(result)

    def fraud_risk(self, start_step: int, end_step: int) -> str:
        result = self._data().filter(*self._filters(start_step, end_step, None)).group_by("type").agg(
            pl.len().alias("transaction_count"),
            pl.col("amount").mean().round(2).alias("average_amount"),
            pl.col("isFraud").sum().alias("fraud_count"),
            pl.col("isFlaggedFraud").sum().alias("flagged_fraud_count"),
        ).with_columns(
            (pl.col("fraud_count") / pl.col("transaction_count") * 100).round(4).alias("fraud_rate_percent")
        ).sort("fraud_rate_percent", descending=True).collect()
        return self._json(result)

    def balance_anomalies(self, start_step: int, end_step: int, transaction_type: str | None) -> str:
        result = self._data().filter(*self._filters(start_step, end_step, transaction_type)).with_columns(
            (pl.col("oldbalanceOrg") - pl.col("newbalanceOrig") - pl.col("amount")).abs().round(2).alias("origin_balance_gap")
        ).filter((pl.col("isFraud") == 1) | (pl.col("origin_balance_gap") > 0.01)).select(
            "step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "origin_balance_gap", "isFraud", "isFlaggedFraud",
        ).limit(self.max_rows).collect()
        return self._json(result)

    def account_activity(self, account_id: str, start_step: int, end_step: int) -> str:
        if not account_id.strip():
            raise ValueError("account_id is required")
        result = self._data().filter(*self._filters(start_step, end_step, None)).filter(
            (pl.col("nameOrig") == account_id) | (pl.col("nameDest") == account_id)
        ).select(
            "step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
        ).limit(self.max_rows).collect()
        return self._json(result)
