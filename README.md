# App Review Insights Agent

**Live demo: https://app-review-insights-agent.streamlit.app/**

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
uv run python -m app_review_agent.ingestion.run_scrape --app-id com.spotify.music --count 300
```

Fetches recent reviews for an app (`TARGET_APP_ID` by default, or any
Google Play package name via `--app-id`) and saves/appends them to
`data/reviews.csv`, tagged with `app_id` and de-duplicated by
`(app_id, review_id)`. Safe to re-run any time, for any number of apps —
everything downstream (retrieval, reports, chat) is scoped per app_id.

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
uv run python -m app_review_agent.graph.run_batch --app-id com.spotify.music
```

Runs the batch branch of the graph, scoped to one app: retrieves the most
relevant reviews per theme from that app's slice of Chroma, classifies each
into `crash` / `ux` / `billing` / `praise` / `feature_request` via a Groq
call, computes volume and rating-delta statistics with pandas (no LLM
involved in this step), then has the LLM write a markdown report from those
stats. Output goes to `data/reports/report_<app_id>_<timestamp>.md` (+ a
`.stats.json` alongside it).

## Step 4 — Interactive chat (terminal)

```bash
uv run python -m app_review_agent.graph.run_chat
uv run python -m app_review_agent.graph.run_chat --app-id com.spotify.music
```

A REPL against the chat branch of the graph: the agent decides when to call
`search_reviews` (retrieval exposed as a tool, scoped to `--app-id` via
LangGraph's `InjectedState` so the model can't search another app's
reviews) before answering, and a `SqliteSaver` checkpointer gives it memory
across turns in the same thread.

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

A sidebar app picker scopes everything below it — the report tab, stats,
and chat — to whichever scraped app is selected. An **"Add another app"**
expander lets you scrape and embed a brand-new Google Play package name
live from the UI, no CLI needed; it shows up in the picker immediately
after. On a machine with no `data/chroma/` yet (e.g. right after cloning),
the app auto-seeds the vector store from the committed `data/reviews.csv`
on first load.

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
6. ✅ Streamlit app, deployed to Streamlit Community Cloud
7. ✅ Multi-app support: every review tagged with `app_id`, an in-UI app
   picker, and a live "scrape a new app" flow — reports/retrieval/chat are
   all scoped per app

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
│       ├── reports.py       # per-app report filename conventions
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
