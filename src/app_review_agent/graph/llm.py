"""Single place that constructs the Groq chat model.

Every node that needs an LLM imports get_llm() from here instead of
constructing ChatGroq directly, so model/temperature changes (or a
future provider swap) happen in one place.
"""
from __future__ import annotations

from langchain_groq import ChatGroq

from app_review_agent.config import GROQ_API_KEY, GROQ_MODEL


def get_llm(temperature: float = 0.0) -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set — add it to .env before running LLM-backed nodes.")
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=temperature)
