"""analyze_node is pure (no LLM, no network) so it's fully unit-testable."""
from app_review_agent.graph.nodes import analyze_node


def test_analyze_node_computes_stats_from_classified_reviews():
    classified = [
        {"review_id": "1", "content": "crashes constantly", "rating": 1, "app_version": "1.0", "at": "t", "category": "crash"},
        {"review_id": "2", "content": "great app", "rating": 5, "app_version": "1.0", "at": "t", "category": "praise"},
        {"review_id": "3", "content": "great app too", "rating": 5, "app_version": "1.1", "at": "t", "category": "praise"},
    ]

    result = analyze_node({"classified": classified})
    stats = result["stats"]

    assert stats["total_classified"] == 3
    assert stats["category_counts"]["crash"] == 1
    assert stats["category_counts"]["praise"] == 2
    assert stats["avg_rating_by_category"]["crash"] == 1.0
    assert stats["avg_rating_by_category"]["praise"] == 5.0


def test_analyze_node_handles_empty_input():
    result = analyze_node({"classified": []})
    assert result["stats"]["category_counts"] == {}
