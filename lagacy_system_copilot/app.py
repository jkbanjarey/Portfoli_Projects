from __future__ import annotations

import streamlit as st

from agent import AgentResponse, ask
from cold_chain.config import get_settings

st.set_page_config(page_title="PaySim Fraud Copilot", page_icon="🛡️", layout="centered")
st.title("🛡️ PaySim Fraud Copilot")
st.caption("Ask about transactions, fraud risk, balance anomalies, or anonymized account activity.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("traces"):
            with st.expander("Safe execution trace"):
                st.json(message["traces"])

if prompt := st.chat_input("e.g. Which transaction types have the highest fraud rate across all steps?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Checking the logistics data…"):
            try:
                response: AgentResponse = ask(prompt, get_settings())
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 - avoid leaking credential or transport details to users.
                st.error("The agent could not complete the request. Check your OpenAI API key, network access, and PaySim dataset path.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")
            else:
                st.markdown(response.answer)
                traces = [trace.__dict__ for trace in response.traces]
                if traces:
                    with st.expander("Safe execution trace"):
                        st.json(traces)
                st.session_state.messages.append({"role": "assistant", "content": response.answer, "traces": traces})
