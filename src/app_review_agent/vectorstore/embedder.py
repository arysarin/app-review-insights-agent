"""Step 2: embed reviews and upsert them into a local Chroma collection.

Kept separate from the LangGraph reasoning layer for the same reason
ingestion is separate from scraping: embedding is a mechanical,
repeatable step, not a reasoning step. `retrieve_node` (graph/nodes.py)
reads from the store this module builds.
"""
from __future__ import annotations

import logging

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app_review_agent.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    REVIEWS_CSV_PATH,
)

logger = logging.getLogger(__name__)

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def get_vectorstore() -> Chroma:
    """Return the singleton Chroma store, creating it on first access."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_PERSIST_DIR,
        )
    return _vectorstore


def review_row_to_document(row: pd.Series) -> Document:
    return Document(
        page_content=str(row["content"]),
        metadata={
            "review_id": str(row["review_id"]),
            "rating": int(row["rating"]),
            "thumbs_up": int(row["thumbs_up"]),
            "app_version": str(row["app_version"]) if pd.notna(row["app_version"]) else "unknown",
            "at": str(row["at"]),
        },
    )


def ingest_dataframe(df: pd.DataFrame, vectorstore: Chroma | None = None, batch_size: int = 200) -> int:
    """Embed and upsert every row of `df` into Chroma, keyed by review_id.

    Upserting on review_id means re-running this after a fresh scrape is
    safe and idempotent — existing reviews get overwritten in place
    rather than duplicated.
    """
    vectorstore = vectorstore or get_vectorstore()
    documents = [review_row_to_document(row) for _, row in df.iterrows()]
    ids = [doc.metadata["review_id"] for doc in documents]

    total = 0
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        vectorstore.add_documents(batch_docs, ids=batch_ids)
        total += len(batch_docs)
        logger.info("Embedded %s/%s reviews", total, len(documents))
    return total


def ensure_vectorstore_seeded() -> None:
    """Embed data/reviews.csv into Chroma if the collection is empty.

    Chroma's persist directory is gitignored (it's a regenerable local
    cache), but reviews.csv is committed — so a fresh checkout or a
    fresh Streamlit Cloud deploy has no vector store yet on first boot.
    Called from every entry point (CLIs and the Streamlit app) so none
    of them require a manual `run_embed` step first.
    """
    vectorstore = get_vectorstore()
    if vectorstore._collection.count() > 0:
        return
    if not REVIEWS_CSV_PATH.exists():
        logger.warning("%s not found — nothing to seed the vector store with.", REVIEWS_CSV_PATH)
        return

    logger.info("Chroma collection is empty — seeding from %s", REVIEWS_CSV_PATH)
    df = pd.read_csv(REVIEWS_CSV_PATH)
    ingest_dataframe(df, vectorstore)
