"""Retrieval exposed as a tool the chat agent can decide to call.

This is what the brief means by "retrieve_node (as a tool)" in the
interactive flow: unlike the batch branch, where retrieval always runs
first, here the LLM decides whether/when a question needs a lookup
against the review store at all.
"""
from __future__ import annotations

from langchain_core.tools import tool

from app_review_agent.vectorstore.embedder import get_vectorstore

SEARCH_K = 8


@tool
def search_reviews(query: str) -> str:
    """Search the app's review store for reviews relevant to `query`.

    Use this whenever the user asks about what reviewers said, are
    saying, or feel about something specific — e.g. a feature, a
    complaint, a version, or a rating trend. Do not guess; look it up.
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=SEARCH_K)
    if not docs:
        return "No matching reviews found."

    lines = [
        f"- rating={d.metadata.get('rating')} version={d.metadata.get('app_version')} "
        f"at={d.metadata.get('at')}: {d.page_content}"
        for d in docs
    ]
    return "\n".join(lines)
