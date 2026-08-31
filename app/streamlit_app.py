"""Streamlit UI: an app picker, a weekly report tab, and a chat tab.

Run with:
    uv run streamlit run app/streamlit_app.py

Reads from the same Chroma store and LangGraph app used by the CLIs in
src/app_review_agent — this is a thin UI layer, not a separate
implementation of the agent. Supports multiple apps: reviews are
tagged with app_id end-to-end, and every report/chat call is scoped to
whichever app is selected in the sidebar.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app_review_agent.config import TARGET_APP_ID
from app_review_agent.graph.graph import build_app
from app_review_agent.graph.reports import latest_report_paths, new_report_paths, report_timestamp
from app_review_agent.ingestion.run_scrape import scrape_app
from app_review_agent.vectorstore.embedder import get_known_app_ids, get_vectorstore, ingest_dataframe

st.set_page_config(page_title="App Review Insights Agent", page_icon="📱", layout="wide")


@st.cache_resource
def get_graph_app():
    return build_app()


with st.sidebar:
    st.header("App Review Insights Agent")
    st.caption("RAG + LangGraph agent for mobile app review analysis")

    known_apps = get_known_app_ids() or [TARGET_APP_ID]
    default_index = known_apps.index(TARGET_APP_ID) if TARGET_APP_ID in known_apps else 0
    selected_app = st.selectbox("Analyzing app", known_apps, index=default_index)

    if selected_app != st.session_state.get("selected_app"):
        # Switching apps starts a fresh chat thread rather than mixing
        # one app's context into another's conversation memory.
        st.session_state.selected_app = selected_app
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []

    reviews_df = None
    if Path("data/reviews.csv").exists():
        reviews_df = pd.read_csv("data/reviews.csv")
        app_df = reviews_df[reviews_df["app_id"] == selected_app]
        st.metric("Reviews ingested", len(app_df))
        if len(app_df):
            st.caption(f"Rating avg: {app_df['rating'].mean():.2f} / 5")

    with st.expander("➕ Add another app"):
        st.caption("Scrape and embed a new Google Play app id, e.g. `com.spotify.music`.")
        new_app_id = st.text_input("Play Store package name", key="new_app_id")
        new_app_count = st.number_input("Reviews to fetch", min_value=50, max_value=1000, value=200, step=50)
        if st.button("Scrape & embed", use_container_width=True):
            if not new_app_id.strip():
                st.error("Enter a package name first.")
            else:
                try:
                    with st.spinner(f"Scraping {new_app_id}..."):
                        combined_df = scrape_app(new_app_id.strip(), count=int(new_app_count))
                        new_rows = combined_df[combined_df["app_id"] == new_app_id.strip()]
                    with st.spinner("Embedding into the vector store..."):
                        ingest_dataframe(new_rows, get_vectorstore())
                    st.success(f"Added {len(new_rows)} reviews for {new_app_id}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Couldn't scrape {new_app_id}: {e}")

    st.divider()
    st.caption("Architecture: scrape → embed (Chroma) → LangGraph "
               "(retrieve → classify → analyze → report, or a tool-calling "
               "chat agent with memory), scoped per app.")

tab_report, tab_chat = st.tabs(["📊 Weekly Report", "💬 Ask a question"])

with tab_report:
    st.subheader(f"Latest report — {selected_app}")

    col1, col2 = st.columns([3, 1])
    with col2:
        generate_clicked = st.button("Generate new report", type="primary", use_container_width=True)

    if generate_clicked:
        with st.spinner("Retrieving, classifying, and analyzing reviews... this calls the LLM and can take a minute."):
            graph_app = get_graph_app()
            thread_id = str(uuid.uuid4())
            result = graph_app.invoke(
                {"mode": "batch", "app_id": selected_app},
                config={"configurable": {"thread_id": thread_id}},
            )

            report_path, stats_path = new_report_paths(selected_app)
            report_path.write_text(result["report"], encoding="utf-8")
            stats_path.write_text(json.dumps(result["stats"], indent=2, default=str), encoding="utf-8")
        st.rerun()

    existing = latest_report_paths(selected_app)
    if existing is None:
        st.info("No report generated yet for this app — click **Generate new report** to run the batch pipeline.")
    else:
        report_path, _ = existing
        st.caption(f"Generated: {report_timestamp(report_path, selected_app)}")
        st.markdown(report_path.read_text(encoding="utf-8"))

with tab_chat:
    st.subheader(f"Ask about {selected_app}'s reviews")
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
                    {"mode": "chat", "app_id": selected_app, "messages": [{"role": "user", "content": question}]},
                    config=config,
                )
                answer = result["messages"][-1].content
            st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))
