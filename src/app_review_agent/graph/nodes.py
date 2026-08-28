"""Batch flow nodes: retrieve -> classify -> analyze -> report.

Each node reads/writes only the GraphState keys it owns, so the graph
in graph.py can wire them into a sequence without any node needing to
know about the others.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from app_review_agent.config import CATEGORIES
from app_review_agent.graph.classify import CATEGORY_DESCRIPTIONS, classify_reviews
from app_review_agent.graph.llm import get_llm
from app_review_agent.graph.state import GraphState
from app_review_agent.vectorstore.embedder import get_vectorstore

logger = logging.getLogger(__name__)

RETRIEVE_K_PER_CATEGORY = 15

_CATEGORY_QUERIES = {name: desc for name, desc in CATEGORY_DESCRIPTIONS.items()}


def retrieve_node(state: GraphState) -> dict:
    """Pull the most relevant reviews per theme from Chroma.

    Rather than dumping the entire dataset into the LLM, we run one
    similarity query per category and merge the results — this is the
    RAG step the rest of the batch flow reasons over.
    """
    vectorstore = get_vectorstore()
    by_id: dict[str, dict] = {}

    for category, query in _CATEGORY_QUERIES.items():
        docs = vectorstore.similarity_search(query, k=RETRIEVE_K_PER_CATEGORY)
        for doc in docs:
            review_id = doc.metadata["review_id"]
            if review_id not in by_id:
                by_id[review_id] = {
                    "review_id": review_id,
                    "content": doc.page_content,
                    "rating": doc.metadata.get("rating"),
                    "app_version": doc.metadata.get("app_version"),
                    "at": doc.metadata.get("at"),
                }

    logger.info("retrieve_node: %s unique reviews across %s category queries", len(by_id), len(_CATEGORY_QUERIES))
    return {"retrieved": list(by_id.values())}


def classify_node(state: GraphState) -> dict:
    """Tag each retrieved review into one of CATEGORIES via an LLM call."""
    retrieved = state.get("retrieved", [])
    llm = get_llm()
    id_to_category = classify_reviews(retrieved, llm)

    classified = [
        {**review, "category": id_to_category.get(review["review_id"], "unclassified")}
        for review in retrieved
    ]
    logger.info("classify_node: classified %s/%s reviews", len(id_to_category), len(retrieved))
    return {"classified": classified}


def analyze_node(state: GraphState) -> dict:
    """Pure stats over the classified set — no LLM call in this node."""
    classified = state.get("classified", [])
    if not classified:
        return {"stats": {"category_counts": {}, "avg_rating_by_category": {}, "avg_rating_by_version": {}}}

    df = pd.DataFrame(classified)

    category_counts = df["category"].value_counts().to_dict()
    avg_rating_by_category = df.groupby("category")["rating"].mean().round(2).to_dict()

    version_stats = (
        df.groupby("app_version")
        .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))
        .round(2)
    )
    version_stats = version_stats[version_stats["review_count"] >= 2]
    avg_rating_by_version = version_stats.sort_index().to_dict(orient="index")

    examples_by_category = defaultdict(list)
    for row in classified:
        if len(examples_by_category[row["category"]]) < 3:
            examples_by_category[row["category"]].append(row["content"])

    stats = {
        "total_classified": len(classified),
        "category_counts": category_counts,
        "avg_rating_by_category": avg_rating_by_category,
        "avg_rating_by_version": avg_rating_by_version,
        "examples_by_category": dict(examples_by_category),
    }
    logger.info("analyze_node: computed stats over %s classified reviews", len(classified))
    return {"stats": stats}


_REPORT_PROMPT = """You are writing a concise weekly insight report for a mobile app's \
product team, based on a sample of recent app store reviews that have been \
retrieved and classified by an automated pipeline.

Categories and their meaning:
{category_list}

Computed statistics (JSON):
{stats_json}

Write a markdown report with these sections:
1. **Overview** — one paragraph, total reviews analyzed and overall tone.
2. **By category** — for each category with reviews, one short paragraph:
   volume, average rating, and 1-2 direct example quotes.
3. **Version notes** — any notable rating differences across app versions
   present in the data (if the differences are small or the sample per
   version is thin, say so honestly rather than overstating a trend).
4. **Recommendations** — 2-4 concrete, prioritized suggestions for the
   product team based on the above.

Be factual and grounded only in the statistics and quotes given. Do not
invent numbers or claim trends the data doesn't support.
"""


def report_node(state: GraphState) -> dict:
    """Write the natural-language weekly report from computed stats."""
    import json

    stats = state.get("stats", {})
    category_list = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORY_DESCRIPTIONS.items())
    prompt = _REPORT_PROMPT.format(category_list=category_list, stats_json=json.dumps(stats, indent=2, default=str))

    llm = get_llm(temperature=0.2)
    response = llm.invoke(prompt)
    logger.info("report_node: generated report (%s chars)", len(response.content))
    return {"report": response.content}
