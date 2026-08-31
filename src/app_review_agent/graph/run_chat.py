"""CLI entry point for Step 4: interactive chat with memory, in the terminal.

Run with:
    uv run python -m app_review_agent.graph.run_chat

Each run uses a fresh thread_id, so history doesn't carry over between
CLI invocations — that's expected; the Streamlit app assigns one
thread_id per browser session instead. Type 'exit' to quit.
"""
from __future__ import annotations

import argparse
import logging
import uuid

from app_review_agent.config import TARGET_APP_ID
from app_review_agent.graph.graph import build_app

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chat over one app's reviews.")
    parser.add_argument("--app-id", default=TARGET_APP_ID, help="Which scraped app to chat about")
    args = parser.parse_args()

    app = build_app()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"App Review Insights chat ({args.app_id}). Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = app.invoke(
            {"mode": "chat", "app_id": args.app_id, "messages": [{"role": "user", "content": question}]},
            config=config,
        )
        print(f"Agent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
