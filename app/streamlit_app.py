"""Streamlit UI: latest weekly report + an ad hoc chat box.

Run with:
    uv run streamlit run app/streamlit_app.py

Reads from the same Chroma store and LangGraph app used by the CLIs in
src/app_review_agent — this is a thin UI layer, not a separate
implementation of the agent.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app_review_agent.config import REPORTS_DIR, REVIEWS_CSV_PATH, TARGET_APP_ID
from app_review_agent.graph.graph import build_app

st.set_page_config(page_title="App Review Insights Agent", page_icon="📱", layout="wide")


@st.cache_resource
def get_graph_app():
    return build_app()


def latest_report_paths() -> tuple[Path, Path] | None:
    reports = sorted(REPORTS_DIR.glob("report_*.md"))
    if not reports:
        return None
    report_path = reports[-1]
    stats_path = report_path.with_suffix("").with_suffix(".stats.json")
    return report_path, stats_path


with st.sidebar:
    st.header("App Review Insights Agent")
    st.caption("RAG + LangGraph agent for mobile app review analysis")
    st.markdown(f"**Target app:** `{TARGET_APP_ID}`")

    if REVIEWS_CSV_PATH.exists():
        df = pd.read_csv(REVIEWS_CSV_PATH)
        st.metric("Reviews ingested", len(df))
        st.caption(f"Rating avg: {df['rating'].mean():.2f} / 5")
    else:
        st.warning("No reviews.csv found — run the scraper first.")

    st.divider()
    st.caption("Architecture: scrape → embed (Chroma) → LangGraph "
               "(retrieve → classify → analyze → report, or a tool-calling "
               "chat agent with memory).")

tab_report, tab_chat = st.tabs(["📊 Weekly Report", "💬 Ask a question"])

with tab_report:
    st.subheader("Latest weekly report")

    existing = latest_report_paths()
    col1, col2 = st.columns([3, 1])
    with col2:
        generate_clicked = st.button("Generate new report", type="primary", use_container_width=True)

    if generate_clicked:
        with st.spinner("Retrieving, classifying, and analyzing reviews... this calls the LLM and can take a minute."):
            graph_app = get_graph_app()
            thread_id = str(uuid.uuid4())
            result = graph_app.invoke({"mode": "batch"}, config={"configurable": {"thread_id": thread_id}})

            import json
            from datetime import datetime, timezone

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
            report_path = REPORTS_DIR / f"report_{timestamp}.md"
            stats_path = REPORTS_DIR / f"report_{timestamp}.stats.json"
            report_path.write_text(result["report"], encoding="utf-8")
            stats_path.write_text(json.dumps(result["stats"], indent=2, default=str), encoding="utf-8")
        st.rerun()

    existing = latest_report_paths()
    if existing is None:
        st.info("No report generated yet — click **Generate new report** to run the batch pipeline.")
    else:
        report_path, stats_path = existing
        st.caption(f"Generated: {report_path.stem.removeprefix('report_')}")
        st.markdown(report_path.read_text(encoding="utf-8"))

with tab_chat:
    st.subheader("Ask about the reviews")
    st.caption("Answers are grounded in retrieved reviews, not guesses — the agent decides when to search.")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    question = st.chat_input("e.g. What are people saying about crashes on the latest version?")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                graph_app = get_graph_app()
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = graph_app.invoke(
                    {"mode": "chat", "messages": [{"role": "user", "content": question}]},
                    config=config,
                )
                answer = result["messages"][-1].content
            st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))
