"""The LangGraph agent: one graph, two branches.

    mode="batch" -> retrieve -> classify -> analyze -> report -> END
    mode="chat"  -> chat_agent <-> tools (loop until no tool call) -> END

Routing happens once, at the entry point, based on state["mode"]. This
is the "real orchestration, not two scripts" the brief calls for: both
flows share one state schema and one compiled graph, they just take
different paths through it.
"""
from __future__ import annotations

import sqlite3
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app_review_agent.config import CHECKPOINT_DB_PATH
from app_review_agent.graph.chat_nodes import chat_agent_node
from app_review_agent.graph.nodes import analyze_node, classify_node, report_node, retrieve_node
from app_review_agent.graph.state import GraphState
from app_review_agent.graph.tools import search_reviews
from app_review_agent.vectorstore.embedder import ensure_vectorstore_seeded


def _route_by_mode(state: GraphState) -> Literal["retrieve", "chat_agent"]:
    return "retrieve" if state.get("mode") == "batch" else "chat_agent"


def _build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("classify", classify_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("report", report_node)

    graph.add_node("chat_agent", chat_agent_node)
    graph.add_node("tools", ToolNode([search_reviews]))

    graph.add_conditional_edges(START, _route_by_mode, {"retrieve": "retrieve", "chat_agent": "chat_agent"})

    graph.add_edge("retrieve", "classify")
    graph.add_edge("classify", "analyze")
    graph.add_edge("analyze", "report")
    graph.add_edge("report", END)

    graph.add_conditional_edges("chat_agent", tools_condition, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "chat_agent")

    return graph


def build_app(checkpoint_path: str | None = None):
    """Compile the graph with a sqlite-backed checkpointer.

    The checkpointer gives the chat branch memory across turns (via
    thread_id in the invoke config); the batch branch ignores it since
    each report run is self-contained.
    """
    ensure_vectorstore_seeded()
    conn = sqlite3.connect(checkpoint_path or str(CHECKPOINT_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return _build_graph().compile(checkpointer=checkpointer)
