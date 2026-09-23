"""LangGraph orchestrator for the cold-chain logistics assistant."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from cold_chain.config import Settings, get_settings
from cold_chain.paysim import PaySimRepository
from cold_chain.tools import build_tools

SYSTEM_PROMPT = """You are a PaySim financial-transaction and fraud analyst. Use a tool for every data claim.
PaySim's `step` is an hourly simulation time-step, not a calendar date; clarify this when needed. It has
five types: CASH_IN, CASH_OUT, DEBIT, PAYMENT, and TRANSFER. For broad questions use steps 1 through 743.
If the user asks to view, show, list, preview, retrieve, or display dataset rows, you MUST call
`preview_transactions`; do not say that retrieval is unsupported. For requests for top rows, use steps 1
through 743 and a limit of 10 unless the user gives other filters. Never claim a capability is unavailable
without first checking the tool descriptions.
Never claim data that tools did not return. Keep summaries concise, include step ranges, units, and fraud
counts/rates where relevant. Explain empty results plainly."""


@dataclass
class ToolTrace:
    tool: str
    parameters: dict[str, Any]
    elapsed_ms: int | None = None
    result_count: int | None = None


class SafeTraceHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self._start_times: dict[str, float] = {}
        self.traces: list[ToolTrace] = []

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, run_id: Any, **_: Any) -> None:
        import json

        try:
            parameters = json.loads(input_str)
        except json.JSONDecodeError:
            parameters = {}
        self._start_times[str(run_id)] = monotonic()
        self.traces.append(ToolTrace(serialized.get("name", "database_tool"), parameters))

    def on_tool_end(self, output: Any, run_id: Any, **_: Any) -> None:
        import json

        trace = next((entry for entry in reversed(self.traces) if entry.elapsed_ms is None), None)
        if trace:
            trace.elapsed_ms = round((monotonic() - self._start_times.pop(str(run_id), monotonic())) * 1000)
            try:
                parsed = json.loads(str(output))
                trace.result_count = len(parsed) if isinstance(parsed, list) else None
            except json.JSONDecodeError:
                trace.result_count = None


@dataclass
class AgentResponse:
    answer: str
    traces: list[ToolTrace] = field(default_factory=list)


def build_agent(settings: Settings | None = None):
    settings = settings or get_settings()
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not configured. Add it to .env and restart Streamlit.")
    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0,
    )
    repository = PaySimRepository(settings.paysim_dataset_path, settings.max_query_rows)
    tools = build_tools(repository)
    return create_react_agent(model, tools, prompt=SYSTEM_PROMPT)


def ask(question: str, settings: Settings | None = None) -> AgentResponse:
    tracer = SafeTraceHandler()
    result = build_agent(settings).invoke({"messages": [HumanMessage(content=question)]}, config={"callbacks": [tracer]})
    answer = result["messages"][-1].content
    if isinstance(answer, list):
        answer = "".join(str(part) for part in answer)
    return AgentResponse(answer=str(answer), traces=tracer.traces)
