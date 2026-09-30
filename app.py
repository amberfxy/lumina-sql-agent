"""Streamlit frontend for LuminaSQL-Agent."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from config import DatabaseBackend, get_settings

settings = get_settings()

st.set_page_config(
    page_title="LuminaSQL-Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top left, #101826 0%, #05070d 45%, #020308 100%);
        color: #e8eefc;
    }
    .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }
    .lumina-title {
        font-family: "SF Mono", "Fira Code", monospace;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #7dd3fc, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .lumina-subtitle {
        color: #94a3b8;
        font-family: "SF Mono", "Fira Code", monospace;
        margin-bottom: 1.5rem;
    }
    .debug-card {
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        background: rgba(15, 23, 42, 0.75);
        margin-bottom: 0.75rem;
    }
    div[data-testid="stCodeBlock"] {
        border: 1px solid #1e293b;
        border-radius: 10px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown('<div class="lumina-title">LuminaSQL-Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lumina-subtitle">Natural Language → SQL/PartiQL with self-correcting execution</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Runtime")
    backend_label = st.radio(
        "Database backend",
        options=["PostgreSQL", "DynamoDB"],
        horizontal=True,
    )
    backend = DatabaseBackend.POSTGRES if backend_label == "PostgreSQL" else DatabaseBackend.DYNAMODB
    allow_mutations = st.toggle("Allow mutating statements", value=False)
    api_base_url = st.text_input("API base URL", value=settings.api_base_url)
    st.caption(f"LLM provider: `{settings.llm_provider.value}`")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a data question in plain English...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    stream_placeholder = st.empty()
    query_placeholder = st.empty()
    debug_placeholder = st.empty()
    result_placeholder = st.empty()

    streamed_text = ""
    final_query = ""
    debug_events: list[dict[str, Any]] = []
    agent_result: dict[str, Any] | None = None

    payload = {
        "query": prompt,
        "backend": backend.value,
        "allow_mutations": allow_mutations,
    }

    try:
        with (
            httpx.Client(timeout=None) as client,
            client.stream(
                "POST",
                f"{api_base_url.rstrip('/')}/api/v1/query/stream",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response,
        ):
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                event = json.loads(line.removeprefix("data: "))
                event_type = event.get("type")
                content = event.get("content")

                if event_type == "status":
                    stream_placeholder.info(content)
                elif event_type == "schema":
                    with st.expander("Pruned schema context", expanded=False):
                        st.markdown(content)
                elif event_type == "llm_token":
                    streamed_text += content
                    stream_placeholder.markdown(f"**LLM stream**\n\n```\n{streamed_text}\n```")
                elif event_type == "query":
                    final_query = content
                    query_placeholder.code(final_query, language="sql")
                elif event_type == "error":
                    debug_events.append(content)
                    with debug_placeholder.container():
                        st.warning("Self-debug loop intercepted an execution error")
                        for item in debug_events:
                            st.markdown(
                                f'<div class="debug-card"><b>Attempt {item["attempt"]}</b><br/>'
                                f"<code>{item['query']}</code><br/><br/>"
                                f'<span style="color:#fca5a5;">{item["raw_error"]}</span></div>',
                                unsafe_allow_html=True,
                            )
                elif event_type == "result":
                    agent_result = content
                    st.session_state.last_result = content

    except httpx.HTTPError as exc:
        st.error(f"API request failed: {exc}")
        st.stop()

    with st.chat_message("assistant"):
        if agent_result and agent_result.get("success"):
            execution = agent_result.get("execution") or {}
            rows = execution.get("rows", [])
            columns = execution.get("columns", [])
            st.success(agent_result.get("message", "Success"))
            if final_query:
                st.code(final_query, language="sql")
            if rows:
                dataframe = pd.DataFrame(rows, columns=columns or None)
                result_placeholder.dataframe(dataframe, use_container_width=True)
            else:
                result_placeholder.info("Query executed successfully but returned no rows.")
            assistant_message = f"Executed successfully.\n\n```sql\n{final_query}\n```"
        else:
            message = (agent_result or {}).get("message", "Agent failed to produce a valid query.")
            st.error(message)
            if final_query:
                st.code(final_query, language="sql")
            assistant_message = message

        st.session_state.messages.append({"role": "assistant", "content": assistant_message})

if st.session_state.last_result:
    with st.expander("Raw agent payload", expanded=False):
        st.json(st.session_state.last_result)
