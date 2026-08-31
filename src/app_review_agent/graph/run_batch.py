"""CLI entry point for Step 3: run the batch flow and save a weekly report.

Run with:
    uv run python -m app_review_agent.graph.run_batch
    uv run python -m app_review_agent.graph.run_batch --app-id com.spotify.music

Requires reviews already embedded into Chroma (Step 2) and a working
GROQ_API_KEY in .env (Step 3 is the first step that makes LLM calls).
"""
from __future__ import annotations

import argparse
import json
import logging
import uuid

from app_review_agent.config import TARGET_APP_ID
from app_review_agent.graph.graph import build_app
from app_review_agent.graph.reports import new_report_paths

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the batch retrieve->classify->analyze->report flow.")
    parser.add_argument("--app-id", default=TARGET_APP_ID, help="Which scraped app to report on")
    args = parser.parse_args()

    app = build_app()
    thread_id = str(uuid.uuid4())
    result = app.invoke(
        {"mode": "batch", "app_id": args.app_id},
        config={"configurable": {"thread_id": thread_id}},
    )

    report_path, stats_path = new_report_paths(args.app_id)
    report_path.write_text(result["report"], encoding="utf-8")
    stats_path.write_text(json.dumps(result["stats"], indent=2, default=str), encoding="utf-8")

    print(f"Report saved to {report_path}")
    print(f"Stats saved to {stats_path}")
    print()
    print(result["report"])


if __name__ == "__main__":
    main()
