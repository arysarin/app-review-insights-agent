"""Sanity checks for the scraper's data shaping — not live network tests.

These don't call the Play Store; they check that reviews_to_dataframe
produces the columns every later pipeline stage depends on.
"""
from datetime import datetime

from app_review_agent.ingestion.scraper import Review, reviews_to_dataframe


def test_reviews_to_dataframe_columns():
    sample = [
        Review(
            review_id="abc123",
            app_id="com.example.app",
            user_name="Test User",
            content="Great app but crashes on checkout",
            rating=3,
            thumbs_up=5,
            app_version="1.2.0",
            at=datetime(2026, 1, 1),
        )
    ]
    df = reviews_to_dataframe(sample)
    assert list(df.columns) == [
        "review_id",
        "app_id",
        "user_name",
        "content",
        "rating",
        "thumbs_up",
        "app_version",
        "at",
    ]
    assert len(df) == 1


def test_reviews_to_dataframe_empty():
    df = reviews_to_dataframe([])
    assert df.empty
