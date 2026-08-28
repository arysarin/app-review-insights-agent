"""CLI entry point for Step 3: run the batch flow and save a weekly report.

Run with:
    uv run python -m app_review_agent.graph.run_batch

Requires reviews already embedded into Chroma (Step 2) and a working
GROQ_API_KEY in .env (Step 3 is the first step that makes LLM calls).
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from app_review_agent.config import REPORTS_DIR
from app_review_agent.graph.graph import build_app

logging.basicConfig(level=logging.INFO)


def main() -> None:
    app = build_app()
    thread_id = str(uuid.uuid4())
    result = app.invoke(
        {"mode": "batch"},
        config={"configurable": {"thread_id": thread_id}},
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.md"
    stats_path = REPORTS_DIR / f"report_{timestamp}.stats.json"

    report_path.write_text(result["report"], encoding="utf-8")
    stats_path.write_text(json.dumps(result["stats"], indent=2, default=str), encoding="utf-8")

    print(f"Report saved to {report_path}")
    print(f"Stats saved to {stats_path}")
    print()
    print(result["report"])


if __name__ == "__main__":
    main()
