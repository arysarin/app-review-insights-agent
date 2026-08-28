"""Shared state schema for the LangGraph agent.

One schema backs both branches of the graph (see graph/graph.py):
batch fields (retrieved/classified/stats/report) are populated by the
weekly-report flow, `messages` is populated by the interactive chat
flow. `total=False` lets each branch touch only the keys it needs.
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class RetrievedReview(TypedDict):
    review_id: str
    content: str
    rating: int
    app_version: str
    at: str


class ClassifiedReview(RetrievedReview):
    category: str


class GraphState(TypedDict, total=False):
    mode: Literal["batch", "chat"]

    # batch branch: retrieve -> classify -> analyze -> report
    retrieved: list[RetrievedReview]
    classified: list[ClassifiedReview]
    stats: dict
    report: str

    # chat branch: agent <-> tools loop, memory via checkpointer
    messages: Annotated[list, add_messages]
