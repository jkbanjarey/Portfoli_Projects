from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from cold_chain.paysim import PaySimRepository


def build_tools(repository: PaySimRepository):
    """Expose bounded, read-only PaySim retrieval and analysis tools to the agent."""

    @tool
    def preview_transactions(
        start_step: Annotated[int, "First inclusive PaySim hour/step; use 1 for the dataset start"] = 1,
        end_step: Annotated[int, "Last inclusive PaySim hour/step; use 743 for the full dataset"] = 743,
        transaction_type: Annotated[str | None, "Optional type: PAYMENT, TRANSFER, CASH_OUT, DEBIT, or CASH_IN"] = None,
        limit: Annotated[int, "Number of rows to return, from 1 to 100"] = 10,
    ) -> str:
        """Show top/sample raw transaction rows. Use for requests to view, list, preview, or retrieve data."""
        return repository.preview_transactions(start_step, end_step, transaction_type, limit)

    @tool
    def query_transaction_metrics(
        start_step: Annotated[int, "First inclusive PaySim hour/step; PaySim spans approximately 1 to 743"],
        end_step: Annotated[int, "Last inclusive PaySim hour/step"],
        transaction_type: Annotated[str | None, "Optional type: PAYMENT, TRANSFER, CASH_OUT, DEBIT, or CASH_IN"] = None,
    ) -> str:
        """Get total count, amount, and fraud labels for a step range and optional transaction type."""
        return repository.transaction_metrics(start_step, end_step, transaction_type)

    @tool
    def analyze_balance_anomalies(
        start_step: Annotated[int, "First inclusive PaySim hour/step"],
        end_step: Annotated[int, "Last inclusive PaySim hour/step"],
        transaction_type: Annotated[str | None, "Optional transaction type"] = None,
    ) -> str:
        """Find fraud-labelled or origin-balance-inconsistent transactions in a step range."""
        return repository.balance_anomalies(start_step, end_step, transaction_type)

    @tool
    def assess_fraud_risk(
        start_step: Annotated[int, "First inclusive PaySim hour/step"],
        end_step: Annotated[int, "Last inclusive PaySim hour/step"],
    ) -> str:
        """Compare fraud count, fraud rate, flagged fraud, and average amount by transaction type."""
        return repository.fraud_risk(start_step, end_step)

    @tool
    def check_account_activity(
        account_id: Annotated[str, "Exact PaySim anonymized account ID, e.g. C1231006815"],
        start_step: Annotated[int, "First inclusive PaySim hour/step"],
        end_step: Annotated[int, "Last inclusive PaySim hour/step"],
    ) -> str:
        """Retrieve capped activity records for one origin or destination account."""
        return repository.account_activity(account_id, start_step, end_step)

    return [
        preview_transactions,
        query_transaction_metrics,
        analyze_balance_anomalies,
        assess_fraud_risk,
        check_account_activity,
    ]
