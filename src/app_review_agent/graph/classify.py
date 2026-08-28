"""Review classification — shared by classify_node and the eval harness.

Kept as a standalone function (not just inline in classify_node) so
eval/run_eval.py can call the exact same code path the graph uses,
rather than a re-implementation that could silently drift from it.
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app_review_agent.config import CATEGORIES

logger = logging.getLogger(__name__)

CATEGORY_DESCRIPTIONS = {
    "crash": "app crashes, freezes, force closes, won't open, or otherwise fails to run",
    "ux": "confusing interface, navigation problems, design/usability complaints",
    "billing": "payment, subscription, pricing, refund, or being charged money",
    "praise": "positive, complimentary reviews with no specific complaint",
    "feature_request": "explicitly asks for a new feature or a change to existing behavior",
}

_Category = Literal[tuple(CATEGORIES)]  # type: ignore[valid-type]


class ReviewClassification(BaseModel):
    review_id: str
    category: _Category = Field(description="Best-fitting single category for this review")


class ClassificationBatch(BaseModel):
    classifications: list[ReviewClassification]


_PROMPT_TEMPLATE = """You are classifying mobile app store reviews into exactly one category each.

Categories:
{category_list}

Classify every review below into the single best-fitting category, even if
it is an imperfect fit — always choose the closest one. Return one
classification per review_id given.

Reviews:
{reviews_block}
"""


def _format_categories() -> str:
    return "\n".join(f"- {name}: {desc}" for name, desc in CATEGORY_DESCRIPTIONS.items())


def _format_reviews(reviews: list[dict]) -> str:
    return "\n".join(f"[{r['review_id']}] (rating={r.get('rating', '?')}) {r['content']}" for r in reviews)


def classify_reviews(
    reviews: list[dict],
    llm: BaseChatModel,
    batch_size: int = 10,
) -> dict[str, str]:
    """Classify a list of {review_id, content, ...} dicts.

    Returns a review_id -> category mapping. Any review the model fails
    to return a classification for is omitted (callers should treat a
    missing id as unclassified rather than guessing).
    """
    if not reviews:
        return {}

    structured_llm = llm.with_structured_output(ClassificationBatch)
    results: dict[str, str] = {}

    for start in range(0, len(reviews), batch_size):
        batch = reviews[start : start + batch_size]
        prompt = _PROMPT_TEMPLATE.format(
            category_list=_format_categories(),
            reviews_block=_format_reviews(batch),
        )
        try:
            output: ClassificationBatch = structured_llm.invoke(prompt)
        except Exception:
            logger.exception("Classification batch starting at %s failed", start)
            continue

        valid_ids = {r["review_id"] for r in batch}
        for c in output.classifications:
            if c.review_id in valid_ids:
                results[c.review_id] = c.category
            else:
                logger.warning("Model returned unknown review_id %s, dropping", c.review_id)

    return results
