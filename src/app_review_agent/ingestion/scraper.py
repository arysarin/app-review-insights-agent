"""Fetches reviews for a given app from the Google Play Store.

This is Step 1 of the build: get real review data onto disk so every
later stage (embedding, classification, reporting) has something real
to work with. No LLM calls happen here on purpose — keep ingestion
and reasoning cleanly separated.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd
from google_play_scraper import Sort, reviews

logger = logging.getLogger(__name__)


@dataclass
class Review:
    review_id: str
    user_name: str
    content: str
    rating: int
    thumbs_up: int
    app_version: str | None
    at: datetime


def fetch_reviews(
    app_id: str,
    count: int = 500,
    lang: str = "en",
    country: str = "in",
) -> list[Review]:
    """Fetch up to `count` recent reviews for `app_id` from Google Play.

    Uses Sort.NEWEST so repeated runs can be de-duplicated by
    review_id and appended to the existing dataset — see
    run_scrape.py, which handles that merge.
    """
    logger.info("Fetching up to %s reviews for %s", count, app_id)
    raw, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=count,
    )

    parsed = [
        Review(
            review_id=r["reviewId"],
            user_name=r["userName"],
            content=r["content"],
            rating=r["score"],
            thumbs_up=r["thumbsUpCount"],
            app_version=r.get("reviewCreatedVersion"),
            at=r["at"],
        )
        for r in raw
    ]
    logger.info("Fetched %s reviews", len(parsed))
    return parsed


def reviews_to_dataframe(reviews_list: list[Review]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in reviews_list])
