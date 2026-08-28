"""Interactive flow: a tool-calling chat agent with conversation memory.

Memory itself is not implemented here — it comes from compiling the
graph with a checkpointer (see graph.py) and invoking with a stable
thread_id, so follow-up questions in the same thread see prior turns.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage

from app_review_agent.graph.llm import get_llm
from app_review_agent.graph.state import GraphState
from app_review_agent.graph.tools import search_reviews

_SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an assistant that answers questions about a mobile app's "
        "store reviews. Use the search_reviews tool to look up relevant "
        "reviews before answering any question about what users think, "
        "report, or feel — never guess or invent review content. Cite "
        "specifics (ratings, versions) from the tool results when you have "
        "them. If the reviews don't cover something, say so plainly."
    )
)

_TOOLS = [search_reviews]


def chat_agent_node(state: GraphState) -> dict:
    llm = get_llm().bind_tools(_TOOLS)
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [_SYSTEM_PROMPT, *messages]
    response = llm.invoke(messages)
    return {"messages": [response]}
