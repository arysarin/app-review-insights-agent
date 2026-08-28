# Evaluation

## Method

1. `build_labeled_set.py` — 19 real reviews from `data/reviews.csv` were
   hand-labeled by reading their content (not model-generated), producing
   `labeled_reviews.csv`. Selection was keyword-assisted (e.g. searching for
   "crash"/"not opening" for the crash category) but every label was verified
   by reading the full review text.
2. `run_eval.py` — runs `classify_reviews()` from
   `src/app_review_agent/graph/classify.py`, the *exact* function
   `classify_node` uses in the real graph, against the labeled set, then
   scores predictions with `sklearn.metrics.classification_report` and a
   confusion matrix. Results are written to `results.md`.

## Known limitation: thin `billing` category

The target app (WhatsApp) is free with no in-app purchases, so genuine
billing/payment complaints are essentially absent from its review corpus —
only one loose match ("ads") turned up after searching for payment,
subscription, refund, pricing, and cost-related keywords. Rather than
inventing synthetic billing examples to pad the eval set, this is
documented as a real finding: for a free app, expect `billing` to be a
low-volume category in both the eval set and the batch report, and don't
over-index on its precision/recall with n=1.

## Result snapshot

See `results.md` for the latest run's numbers (overall accuracy, per-category
precision/recall/F1, confusion matrix, and misclassified examples). Because
classification is a single temperature-0 LLM call per batch, re-running
`run_eval.py` may shift by a misclassification or two — that variance is
itself worth noting in write-ups rather than treating one run as ground
truth.

## LangSmith tracing

Set `LANGCHAIN_TRACING_V2=true` and a real `LANGCHAIN_API_KEY` in `.env` to
get a trace of every node's input/output (retrieve → classify → analyze →
report, or the chat agent's tool-calling loop) in the LangSmith UI under the
`LANGCHAIN_PROJECT` name. This is the "not just a demoed output" artifact
the project brief calls for — a reviewer can open a trace and see exactly
which reviews were retrieved and how the LLM classified/reported on them.
