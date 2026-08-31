"""CLI entry point for Step 1: scrape reviews and save them to disk.

Run with:
    uv run python -m app_review_agent.ingestion.run_scrape
    uv run python -m app_review_agent.ingestion.run_scrape --app-id com.spotify.music --count 300

Safe to run repeatedly, including for multiple apps — new reviews get
appended and de-duplicated (by app_id + review_id) against whatever is
already in data/reviews.csv.
"""
import argparse
import logging

import pandas as pd

from app_review_agent.config import REVIEWS_CSV_PATH, TARGET_APP_ID
from app_review_agent.ingestion.scraper import fetch_reviews, reviews_to_dataframe

logging.basicConfig(level=logging.INFO)


def scrape_app(app_id: str, count: int = 500) -> pd.DataFrame:
    """Scrape `app_id`, merge into reviews.csv, and return the combined dataframe."""
    reviews_list = fetch_reviews(app_id, count=count)
    new_df = reviews_to_dataframe(reviews_list)

    if REVIEWS_CSV_PATH.exists():
        existing_df = pd.read_csv(REVIEWS_CSV_PATH)
        if "app_id" not in existing_df.columns:
            # Back-fill rows scraped before multi-app support existed.
            existing_df["app_id"] = TARGET_APP_ID
        combined = pd.concat([existing_df, new_df]).drop_duplicates(subset=["app_id", "review_id"])
    else:
        combined = new_df

    combined.to_csv(REVIEWS_CSV_PATH, index=False)
    print(
        f"Saved {len(combined)} total reviews "
        f"({len(new_df)} fetched this run for {app_id}) to {REVIEWS_CSV_PATH}"
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Google Play reviews for one app.")
    parser.add_argument("--app-id", default=TARGET_APP_ID, help="Google Play package name")
    parser.add_argument("--count", type=int, default=500, help="Number of recent reviews to fetch")
    args = parser.parse_args()

    scrape_app(args.app_id, count=args.count)


if __name__ == "__main__":
    main()
