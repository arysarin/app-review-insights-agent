"""Phase 5: score classify_node's classification logic against ground truth.

Run with:
    uv run python eval/run_eval.py

Loads eval/labeled_reviews.csv (hand-labeled, see build_labeled_set.py),
runs the exact same classify_reviews() function the batch graph uses,
and reports precision/recall/F1 per category plus a confusion matrix —
so classification accuracy is a measured number, not a claim.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from app_review_agent.graph.classify import classify_reviews
from app_review_agent.graph.llm import get_llm

LABELED_PATH = Path(__file__).resolve().parent / "labeled_reviews.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "results.md"


def main() -> None:
    df = pd.read_csv(LABELED_PATH)
    reviews = df[["review_id", "content", "rating"]].to_dict(orient="records")

    llm = get_llm()
    predictions = classify_reviews(reviews, llm)

    df["predicted"] = df["review_id"].map(predictions)
    unclassified = df["predicted"].isna().sum()
    df["predicted"] = df["predicted"].fillna("unclassified")

    y_true = df["label"].tolist()
    y_pred = df["predicted"].tolist()
    labels = sorted(set(y_true) | set(y_pred))

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = (df["label"] == df["predicted"]).mean()

    matrix_lines = ["Confusion matrix (rows=true, cols=predicted):", "labels: " + ", ".join(labels)]
    for label, row in zip(labels, matrix):
        matrix_lines.append(f"{label:>16}: {[int(v) for v in row]}")

    output_lines = [
        f"# Classification eval — {len(df)} hand-labeled reviews",
        "",
        f"Model: {llm.model_name if hasattr(llm, 'model_name') else llm}",
        f"Overall accuracy: {accuracy:.2%}",
        f"Unclassified (model returned no answer): {unclassified}",
        "",
        "```",
        report,
        "```",
        "",
        "```",
        *matrix_lines,
        "```",
        "",
        "## Misclassifications",
    ]
    for _, row in df[df["label"] != df["predicted"]].iterrows():
        output_lines.append(
            f"- true=`{row['label']}` predicted=`{row['predicted']}`: \"{row['content'][:100]}\""
        )

    output = "\n".join(output_lines)
    print(output)
    RESULTS_PATH.write_text(output, encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
