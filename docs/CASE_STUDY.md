# Case Study: App Review Insights Agent

*A RAG + LangGraph agent for mobile app review analysis, built as a
portfolio project supporting a transition from mobile development into
GenAI / agentic AI engineering.*

## 1. The problem, and why it's real

Every app on the Play Store collects hundreds or thousands of reviews.
Buried in that noise is real signal — recurring crashes, UX friction,
pricing complaints, feature requests — but reading them manually doesn't
scale, and the star-rating average hides almost all of it. As a mobile
developer with production experience shipping enterprise apps, this isn't
a hypothetical: "what are users actually complaining about, and has it
gotten better or worse since the last release" is a question every mobile
team has and few answer well without either manual triage or an
enterprise analytics tool that's overkill for one app.

This build targeted `com.whatsapp` — 500 real, recent Play Store reviews —
specifically to force the pipeline to deal with messy, informal,
inconsistent real-world text rather than a clean synthetic dataset.

## 2. Architecture, and why RAG + LangGraph over a single prompt

```
Ingestion (no LLM):  Google Play  →  google-play-scraper  →  dedupe/append  →  reviews.csv

Embedding (no LLM):  reviews.csv  →  sentence-transformers  →  Chroma (upsert by review_id)

LangGraph agent — one graph, one state schema, two branches:

  mode="batch"  retrieve → classify → analyze → report
                (per-category similarity search) (LLM, structured output) (pandas, no LLM) (LLM)

  mode="chat"   chat_agent ⇄ tools ("search_reviews")   [SqliteSaver gives this branch memory]
```

**Why RAG instead of dumping all 500 reviews into the prompt context:**
cost and precision. The batch flow runs five targeted similarity searches
(one per category — crash, UX, billing, praise, feature request) instead
of reasoning over the entire dataset every run, so classification is
grounded in the reviews actually relevant to each theme. The chat flow
takes this further: `search_reviews` is exposed as a *tool*, and the LLM
decides whether a question needs a lookup at all, rather than always
re-retrieving.

**Why LangGraph instead of a linear chain:** the batch and interactive
flows are genuinely different control flows — one is a fixed pipeline,
the other is an open-ended tool-calling loop that terminates when the
model stops requesting tools — and they need to share memory
infrastructure and a vector store. Routing both through one compiled
graph, branching once at the entry point on `state["mode"]`, is real
orchestration: adding a third flow (e.g. auto-replying to reviews) would
mean adding a branch, not a new script.

**Why pandas for `analyze_node` instead of asking the LLM for stats:**
LLMs are unreliable arithmetic engines. Category counts and per-version
rating averages are exact, cheap, and auditable when computed directly —
the LLM's job in `report_node` is turning already-correct numbers into
prose, not computing them.

## 3. What broke, and how it was found

- **The default Groq model didn't exist.** The brief's original LLM
  choice (Claude API) was swapped for Groq mid-build to match API keys
  already in `.env`. The first obvious model name,
  `llama-3.3-70b-versatile`, returned a 404 — it had been deprecated from
  Groq's catalog. Fix: called `Groq(...).models.list()` directly to see
  what was actually being served, and picked `openai/gpt-oss-120b` from
  the live list instead of trusting a remembered model name.
- **`langchain` had jumped a major version underneath the pinned
  `>=0.3.0` range.** The environment had `langchain` 1.x / `langgraph`
  1.x already installed, which meant several integrations
  (`langchain_community`, the old `HuggingFaceEmbeddings` import path)
  had moved to standalone packages (`langchain-huggingface`). Fix: added
  the split-out packages explicitly instead of assuming the 0.x import
  layout from the brief still applied.
- **Chroma's `add_documents` needed to be an upsert, not an insert**, so
  that re-running ingestion after a fresh scrape wouldn't duplicate
  reviews already in the store. Verified by reading `langchain_chroma`'s
  source directly rather than assuming — it does call `collection.upsert`
  keyed on the `ids` you pass, which is what makes ingestion idempotent
  by `review_id`.
- **`uv add`/`uv remove` silently dropped dev dependencies** (`pytest`,
  `ruff`) from the virtualenv because they live in an optional `dev`
  extra that isn't installed by a plain `uv sync`. Fix: `uv sync --extra
  dev` after every dependency change.
- **Deploying to Streamlit Community Cloud would have shipped an empty
  app.** `data/reviews.csv` and the Chroma store were both gitignored,
  which is fine locally but means a fresh clone (or a fresh cloud deploy)
  has nothing to retrieve against. Fix: committed the review CSV (public
  review text, not sensitive) and added `ensure_vectorstore_seeded()`,
  which re-embeds it automatically on first run if the Chroma collection
  is empty — so `git clone && uv sync && streamlit run` works without a
  manual embedding step.

## 4. Evaluation results — actual numbers

19 real reviews were hand-labeled by reading their content (see
`eval/build_labeled_set.py` for exactly which ones and why), then scored
against `classify_reviews()` — the same function `classify_node` calls in
the live graph, not a re-implementation.

```
Overall accuracy: 84.21%

                 precision    recall  f1-score   support
          crash       1.00      0.80      0.89         5
feature_request       1.00      0.75      0.86         4
         praise       1.00      1.00      1.00         5
             ux       0.67      1.00      0.80         4
        billing       0.00      0.00      0.00         1
```

Praise and crash are classified reliably; UX has lower precision because
the model pulls in some feature-request-shaped complaints as UX friction.
`billing` is the honest weak point — WhatsApp is free with no in-app
purchases, so the review corpus has essentially zero genuine billing
complaints (one loose match, "ads", after searching every payment-related
keyword). Rather than inventing synthetic billing examples to make the
eval set look balanced, that scarcity is documented as a real finding in
`eval/README.md`: for a free app, expect the billing category to be
low-volume in both the eval set and the weekly report, and don't
over-index on n=1 precision/recall.

Because classification is a single temperature-0 LLM call per batch, a
second `run_eval.py` run produced 89.47% instead — a reminder that one run
isn't ground truth, which is itself worth stating plainly rather than
picking the better-looking number.

## 5. Demo

- Local: `uv run streamlit run app/streamlit_app.py`
- Live: *[add the Streamlit Community Cloud URL here after deploying]*

## Resume bullet

> Built a RAG-powered multi-agent system (LangGraph, Chroma, Groq) that
> ingests and analyzes mobile app store reviews, producing evaluated,
> trend-aware insight reports — applying production experience shipping
> enterprise iOS/Flutter apps to a problem mobile teams face directly.
