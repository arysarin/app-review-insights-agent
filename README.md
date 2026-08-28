# App Review Insights Agent

A RAG + LangGraph agent that ingests mobile app store reviews, retrieves and
classifies them, and produces weekly insight reports — plus an interactive
chat mode for ad hoc questions like *"what are people saying about the new
checkout flow?"*

See `docs/PROJECT_BRIEF.md` for the full rationale, architecture, and roadmap.

## Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph (one graph, two branches — batch report + interactive chat) |
| Vector DB | Chroma (local, persisted to `data/chroma/`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, local, free |
| LLM | Groq (`openai/gpt-oss-120b` by default) — fast, free-tier, used for classification, report writing, and the chat agent |
| Memory | LangGraph `SqliteSaver` checkpointer, keyed by thread_id |
| Scraping | `google-play-scraper` |
| Eval | hand-labeled set + `scikit-learn` precision/recall, optional LangSmith tracing |
| UI | Streamlit |

## Setup

```bash
uv sync
cp .env.example .env
# fill in GROQ_API_KEY (https://console.groq.com/keys — free tier) and TARGET_APP_ID
```

## Step 1 — Scrape reviews

```bash
uv run python -m app_review_agent.ingestion.run_scrape
```

Fetches recent reviews for `TARGET_APP_ID` (a Google Play package name, e.g.
`com.whatsapp`) and saves/appends them to `data/reviews.csv`, de-duplicated
by review_id. Safe to re-run any time to pick up new reviews.

## Step 2 — Embed into Chroma

```bash
uv run python -m app_review_agent.vectorstore.run_embed
```

Embeds every review in `data/reviews.csv` and upserts it into a local
Chroma collection at `CHROMA_PERSIST_DIR`, keyed by review_id (so re-running
after a fresh scrape updates in place rather than duplicating). Every
LLM-backed entry point below also auto-seeds this if the collection is
empty, so this step is optional but useful to run explicitly after scraping.

## Step 3 — Batch report (retrieve → classify → analyze → report)

```bash
uv run python -m app_review_agent.graph.run_batch
```

Runs the batch branch of the graph: retrieves the most relevant reviews per
theme from Chroma, classifies each into `crash` / `ux` / `billing` /
`praise` / `feature_request` via a Groq call, computes volume and
rating-delta statistics with pandas (no LLM involved in this step), then
has the LLM write a markdown report from those stats. Output goes to
`data/reports/report_<timestamp>.md` (+ a `.stats.json` alongside it).

## Step 4 — Interactive chat (terminal)

```bash
uv run python -m app_review_agent.graph.run_chat
```

A REPL against the chat branch of the graph: the agent decides when to call
`search_reviews` (retrieval exposed as a tool) before answering, and a
`SqliteSaver` checkpointer gives it memory across turns in the same thread.

## Step 5 — Evaluation

```bash
uv run python eval/run_eval.py
```

Scores the exact classification function `classify_node` uses against
`eval/labeled_reviews.csv` (19 hand-labeled real reviews). Writes
precision/recall/F1 per category and a confusion matrix to
`eval/results.md`. See `eval/README.md` for methodology and a documented
limitation (the target app has almost no genuine billing complaints, so
that category is thin by design, not by omission).

Optional: set `LANGCHAIN_TRACING_V2=true` and a real `LANGCHAIN_API_KEY` in
`.env` for full LangSmith traces of every node.

## Step 6 — Streamlit app

```bash
uv run streamlit run app/streamlit_app.py
```

Two tabs: the latest weekly report (with a button to regenerate it live),
and a chat box wired to the same graph and memory as Step 4. On a machine
with no `data/chroma/` yet (e.g. right after cloning), the app auto-seeds
the vector store from the committed `data/reviews.csv` on first load.

### Deploying to Streamlit Community Cloud

1. Push this repo (it commits `data/reviews.csv` and a seed
   `data/reports/*.md` so the demo isn't empty on first load; `data/chroma/`
   and the sqlite checkpoint DB are gitignored and rebuild automatically).
2. On share.streamlit.io, point at `app/streamlit_app.py`.
3. Add `GROQ_API_KEY` (and `TARGET_APP_ID` if different) under the app's
   Secrets — `config.py` reads them the same way whether they come from
   `.env` locally or Streamlit secrets in the cloud, since Streamlit
   injects secrets as environment variables.

## Run tests

```bash
uv run pytest
```

## Roadmap

1. ✅ Scrape reviews → CSV
2. ✅ Embed reviews → Chroma vector store
3. ✅ LangGraph batch flow: retrieve → classify → analyze → report
4. ✅ Interactive chat node with memory (LangGraph checkpointer)
5. ✅ Hand-labeled eval set + precision/recall, LangSmith tracing support
6. ✅ Streamlit app, ready to deploy to Streamlit Community Cloud

## Project layout

```
app-review-insights-agent/
├── src/app_review_agent/
│   ├── config.py            # central env-based settings
│   ├── ingestion/           # Step 1 — scraping
│   ├── vectorstore/         # Step 2 — Chroma embedding + auto-seed
│   └── graph/
│       ├── state.py         # shared GraphState (batch + chat fields)
│       ├── llm.py           # single place ChatGroq is constructed
│       ├── classify.py      # classification logic shared by classify_node and eval
│       ├── nodes.py         # retrieve / classify / analyze / report nodes
│       ├── tools.py         # search_reviews — retrieval exposed as a tool
│       ├── chat_nodes.py    # tool-calling chat agent node
│       ├── graph.py         # the graph: batch + chat branches, one state schema
│       ├── run_batch.py     # CLI — Step 3
│       └── run_chat.py      # CLI — Step 4
├── eval/
│   ├── build_labeled_set.py # how labeled_reviews.csv was produced
│   ├── labeled_reviews.csv  # 19 hand-labeled real reviews
│   ├── run_eval.py          # Step 5
│   └── results.md           # latest eval run's numbers
├── data/                    # reviews.csv (committed), reports/ (committed),
│                             # chroma/ + checkpoints.sqlite (gitignored, regenerated)
├── tests/
├── app/streamlit_app.py     # Step 6
└── docs/PROJECT_BRIEF.md    # why / how / significance
```
