"""CLI entry point for Step 2: embed data/reviews.csv into Chroma.

Run with:
    uv run python -m app_review_agent.vectorstore.run_embed

Safe to run repeatedly after new scrapes — upserts by review_id.
"""
import logging

import pandas as pd

from app_review_agent.config import REVIEWS_CSV_PATH
from app_review_agent.vectorstore.embedder import get_vectorstore, ingest_dataframe

logging.basicConfig(level=logging.INFO)


def main() -> None:
    if not REVIEWS_CSV_PATH.exists():
        raise SystemExit(f"{REVIEWS_CSV_PATH} not found — run Step 1 (scraping) first.")

    df = pd.read_csv(REVIEWS_CSV_PATH)
    vectorstore = get_vectorstore()
    count = ingest_dataframe(df, vectorstore)
    total_in_store = vectorstore._collection.count()
    print(f"Embedded {count} reviews from {REVIEWS_CSV_PATH} ({total_in_store} total in Chroma collection)")


if __name__ == "__main__":
    main()
