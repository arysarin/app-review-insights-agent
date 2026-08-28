"""CLI entry point for Step 1: scrape reviews and save them to disk.

Run with:
    uv run python -m app_review_agent.ingestion.run_scrape

Safe to run repeatedly — new reviews get appended and de-duplicated
against whatever is already in data/reviews.csv.
"""
import logging

import pandas as pd

from app_review_agent.config import REVIEWS_CSV_PATH, TARGET_APP_ID
from app_review_agent.ingestion.scraper import fetch_reviews, reviews_to_dataframe

logging.basicConfig(level=logging.INFO)


def main() -> None:
    reviews_list = fetch_reviews(TARGET_APP_ID, count=500)
    new_df = reviews_to_dataframe(reviews_list)

    if REVIEWS_CSV_PATH.exists():
        existing_df = pd.read_csv(REVIEWS_CSV_PATH)
        combined = pd.concat([existing_df, new_df]).drop_duplicates(subset="review_id")
    else:
        combined = new_df

    combined.to_csv(REVIEWS_CSV_PATH, index=False)
    print(
        f"Saved {len(combined)} total reviews "
        f"({len(new_df)} fetched this run) to {REVIEWS_CSV_PATH}"
    )


if __name__ == "__main__":
    main()
