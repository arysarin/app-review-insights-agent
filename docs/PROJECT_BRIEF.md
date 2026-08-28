# Project Brief: App Review Insights Agent

**A RAG + LangGraph agent for mobile app review analysis**
*Prepared as a portfolio project to support a transition from mobile
development into GenAI / agentic AI engineering.*

---

## 1. Why This Project

### 1.1 The problem it solves
Every app on the Play Store or App Store collects hundreds or thousands of
user reviews over time. Buried in that noise is real signal — recurring
crashes, UX friction, pricing complaints, feature requests — but reading
them manually doesn't scale, and star-rating averages hide almost all of
the useful detail. Teams either ignore this signal or pay for enterprise
analytics tools that are overkill for a single app or a solo developer.

### 1.2 Why this problem, specifically
Three reasons this was chosen over the other candidate ideas (quick-commerce
price comparison, App Store review *triage* with GitHub/reply automation):

1. **It's authentic, not borrowed.** As a mobile developer with production
   experience shipping enterprise apps, the pain of "what are users actually
   complaining about" is a problem I've encountered directly — not a
   problem invented for the sake of having a project.
2. **It's genuinely agentic, not a wrapper.** It requires retrieval, tool
   use, multi-step reasoning, and branching — not a single prompt-in,
   text-out call. That distinction matters to anyone evaluating agentic
   AI skill.
3. **It's scoped correctly.** The more ambitious version of this idea
   (auto-filing GitHub tickets, auto-posting replies) was deliberately cut
   for v1 — better to ship a smaller, fully-working, well-evaluated agent
   than a larger one that's half-finished.

### 1.3 Why it matters for a career pivot
A hiring manager screening for a GenAI/agentic AI role is not primarily
checking "does this person know the LangGraph API." They're checking
whether the candidate can:
- Design a system with real state and branching logic, not just chain
  prompts
- Justify architectural decisions (why RAG, why this vector store)
- Evaluate whether the agent actually works, and prove it with data
- Ship something a user could actually run

This project is built specifically to produce evidence for all four —
alongside a resume narrative that ties directly to two years of shipped
mobile experience: *"Built a multi-agent system for mobile app review
analysis, applying production experience shipping enterprise iOS/Flutter
apps to a problem mobile teams actually have."*

---

## 2. What It Does — Significance

| Without this agent | With this agent |
|---|---|
| Manually scroll through hundreds of reviews | Reviews are automatically fetched, embedded, and indexed |
| Skim for a general impression | Reviews are retrieved and classified by theme (crash, UX, billing, praise, feature request) |
| No sense of trend over time | Weekly report shows volume and sentiment shift by category and app version |
| Can't ask follow-up questions | Interactive chat answers ad hoc questions ("what's changed since v3.2?") using retrieval, not guesswork |
| No way to know if the analysis is even reliable | A hand-labeled eval set and LangSmith traces measure and expose classification accuracy |

The significance isn't the scraping (that part is commodity engineering).
It's that the agent **reasons over retrieved evidence rather than an entire
dumped context**, **remembers conversation state**, and **is measured, not
just demoed** — the three things that separate an agentic AI system from a
prompt template.

---

## 3. How It Will Be Built — Architecture

Two connected flows:

### 3.1 Ingestion pipeline (scheduled, no LLM calls)
```
Google Play reviews  →  fetch (google-play-scraper)  →  dedupe/append  →  data/reviews.csv
```
Kept LLM-free and separate from the agent on purpose — ingestion should be
cheap, repeatable, and independently testable.

### 3.2 LangGraph agent (the reasoning core)
```
                 ┌─────────────────┐
   new reviews → │  embed + upsert │ → Chroma vector store
                 └─────────────────┘

Batch flow (scheduled weekly report):
  retrieve_node → classify_node → analyze_node → report_node

Interactive flow (ad hoc questions, has memory):
  user question → retrieve_node (as a tool) → chat_node → answer
```

- **`retrieve_node`** — pulls the most relevant reviews from Chroma for the
  current theme or question, instead of re-reading the entire dataset every
  time.
- **`classify_node`** — an LLM call that tags retrieved reviews into
  categories (crash, UX, billing, praise, feature request).
- **`analyze_node`** — computes trend statistics: volume per category over
  time, rating deltas across app versions.
- **`report_node`** — writes the weekly summary in natural language from
  the computed stats and classified themes.
- **`chat_node`** — LangGraph checkpointer gives this node conversation
  memory, so follow-up questions ("what about before v3.0?") work without
  re-explaining context.

The batch and interactive flows are separate branches in the same graph,
which is itself a demonstration of real orchestration — not just two
scripts.

### 3.3 Tech stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | LangGraph | State machine with branching, not a linear chain |
| Vector DB | Chroma (local) | Free, runs locally, zero infra to manage |
| Embeddings | `sentence-transformers` | Free, local, no API cost for iterating |
| LLM | Claude API | Reasoning, classification, report writing |
| Scraping | `google-play-scraper` | No API key required, works against public review data |
| Tracing / eval | LangSmith (free tier) | Makes agent reasoning inspectable — a concrete interview artifact |
| UI | Streamlit | Fast to build, easy to deploy, good enough for a demo |
| Deployment | Streamlit Community Cloud | Free, public URL, no server management |

---

## 4. Skills Map

What each part of the build teaches, mapped to core agentic AI competencies:

| Core skill | Where it's learned in this build |
|---|---|
| Tool use / function calling | Scraper tool, retriever-as-tool in chat flow |
| Retrieval-augmented generation | Chroma ingestion + `retrieve_node` |
| State machines / orchestration | The full LangGraph: branching between batch and interactive flows |
| Memory | Checkpointer powering the `chat_node` |
| Planning | Multi-step reasoning: retrieve → classify → analyze → report |
| Evaluation | Hand-labeled review set scored against `classify_node` output |
| Observability | LangSmith traces on every node's decision |
| Deployment | Streamlit Cloud, scheduled ingestion |

---

## 5. Build Roadmap

| Phase | Deliverable | Definition of done |
|---|---|---|
| 1. Ingestion | Scraper + CSV storage | `data/reviews.csv` has real reviews, de-duplicated across runs |
| 2. Embedding | Reviews in Chroma | Manual retrieval queries return sensible, relevant reviews |
| 3. Batch graph | `retrieve → classify → report` runs end-to-end | Produces a readable weekly report from real data |
| 4. Interactive graph | `chat_node` with memory | Follow-up questions correctly use prior conversation context |
| 5. Evaluation | LangSmith tracing + 10–20 hand-labeled reviews | Classification accuracy is measured, not assumed |
| 6. Deployment | Streamlit app, live URL | A stranger can open the link and get real insights without setup |

*(Phase 1 is scaffolded and ready to run — see the accompanying project
files and `README.md`.)*

---

## 6. Evaluation & Observability Plan

- Hand-label 10–20 reviews across categories (crash, UX, billing, praise,
  feature request) as ground truth.
- Run `classify_node` against them and compute basic precision/recall per
  category.
- Log every node's input/output as a LangSmith trace, so a reviewer (or an
  interviewer) can see exactly how the agent reasoned from raw review to
  final report — not just the final output.
- Track this over iterations: if classification accuracy improves after a
  prompt change, that's a documented before/after, not a claim.

---

## 7. Deployment Plan

- Streamlit app reads from the existing Chroma store and exposes:
  - the latest weekly report
  - a chat box for ad hoc questions
- Deployed on Streamlit Community Cloud (free tier) for a public,
  no-login demo URL.
- Ingestion runs on a schedule (manually triggered for v1; a scheduled
  GitHub Action is a natural v2 addition).

---

## 8. Presenting This for Job Applications

**Resume bullet (draft):**
> Built a RAG-powered multi-agent system (LangGraph, Chroma, Claude) that
> ingests and analyzes mobile app store reviews, producing evaluated,
> trend-aware insight reports — applying production experience shipping
> enterprise iOS/Flutter apps to a problem mobile teams face directly.

**Case study to write alongside the code** (this matters more than the
repo itself for getting noticed):
1. The problem and why it's real
2. Architecture diagram + why RAG/LangGraph over a single prompt
3. What broke during the build and how it was debugged
4. Evaluation results — actual numbers, not "it works well"
5. Link to the live demo

**Demo strategy:** a working public URL beats a GitHub link every time in
a first screen — lead with it.

---

## 9. Scope Discipline

Explicitly out of scope for v1, to keep this shippable:
- Auto-filing GitHub issues from classified bugs (natural v2 extension,
  not required to demonstrate the core skills)
- Auto-posting replies to the store listing
- Multi-app / multi-store support (Play Store only for v1; App Store
  scraping can be added later)

The goal is one small, fully-working, properly evaluated agent — not a
large, half-finished platform.
