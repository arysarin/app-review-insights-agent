"""Filename conventions for saved weekly reports, shared by the batch
CLI and the Streamlit app so both agree on where a given app's latest
report lives.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app_review_agent.config import REPORTS_DIR


def _sanitize(app_id: str) -> str:
    return app_id.replace(".", "_")


def new_report_paths(app_id: str) -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    stem = f"report_{_sanitize(app_id)}_{timestamp}"
    return REPORTS_DIR / f"{stem}.md", REPORTS_DIR / f"{stem}.stats.json"


def latest_report_paths(app_id: str) -> tuple[Path, Path] | None:
    reports = sorted(REPORTS_DIR.glob(f"report_{_sanitize(app_id)}_*.md"))
    if not reports:
        return None
    report_path = reports[-1]
    stats_path = report_path.with_suffix("").with_suffix(".stats.json")
    return report_path, stats_path


def report_timestamp(report_path: Path, app_id: str) -> str:
    """The `2026-08-28_154149` portion of a report filename."""
    prefix = f"report_{_sanitize(app_id)}_"
    return report_path.stem.removeprefix(prefix)
