# Shifa42 — Agentic Clinical Intelligence Copilot

A clinical decision-support copilot that answers questions grounded in published
guideline literature and de-identified emergency-department data — and **refuses to
answer when the retrieved evidence doesn't support a claim**, escalating to a
clinician instead of guessing.

> **Research and educational demonstration. Not a certified medical device.**
> It does not provide medical advice. All data is synthetic or open, de-identified
> research data (MIMIC-IV-ED Demo, PhysioNet; `epfl-llm/guidelines`). No protected
> health information is used anywhere in this system. Any real clinical use
> requires licensed physician oversight.

---

## The engineering thesis

Off-the-shelf LLMs hallucinate confidently in this domain. That is not a UX flaw,
it is a safety failure mode. So the central design claim here is:

**a clinical AI system's most important feature is its ability to decline.**

Everything below serves one of three goals — grounding, verification, or
auditability. The system has exactly two terminal states: a cited answer that
passed an automated groundedness check, or an escalation. There is no third path
where a low-confidence answer ships with a hedge attached, because hedged clinical
answers get read as answers.

---

## Architecture

```
                              ┌─────────────────────────┐
                              │   Next.js 14 Frontend    │
                              │  (chat UI + dashboard)   │
                              └────────────┬─────────────┘
                                           │ REST / SSE
                              ┌────────────▼─────────────┐
                              │   FastAPI Gateway         │
                              │  /query /risk /health     │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │   LangGraph Agent Graph   │
                              └────────────┬─────────────┘
        ┌──────────────┬───────────────┬──┴───────────┬──────────────┐
        ▼              ▼               ▼              ▼              ▼
   Intake Agent   Retrieval      Vector Retriever  Graph Retriever  Risk Model
   (normalize +   Planner        (pgvector +       (NetworkX        (calibrated LR
    PHI screen)   (adaptive:     bge-large +       traversal over   on MIMIC-IV-ED
                  vector/graph/  BM25 hybrid)      condition-drug-  triage features)
                  both)                            symptom graph)
        └──────────────┴───────────────┴──────────────┴──────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  Context Fusion Agent     │
                              │  (RRF → dedup → MMR)      │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  Draft Composer Agent     │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  Critic / Groundedness    │──low confidence──┐
                              │  Agent (RAGAS-style)      │                  │
                              └────────────┬─────────────┘                  │
                                    high confidence                          ▼
                                           │                    ┌────────────────────┐
                              ┌────────────▼─────────────┐      │ Reformulate query   │
                              │  Final Composer           │      │ & retry (max 2x)    │
                              │  (binds inline citations) │      └──────────┬──────────┘
                              └────────────┬─────────────┘                 │
                                           │                    still low? │
                              ┌────────────▼─────────────┐                 ▼
                              │  Answer + citations +      │    ┌────────────────────┐
                              │  groundedness score        │    │ Escalation Agent:   │
                              └────────────────────────────┘    │ "Insufficient       │
                                                                 │ evidence — escalate │
                                                                 │ to physician."      │
                                                                 └────────────────────┘
```

Every box is a real LangGraph node with its own system prompt, retry policy, and
audit row. The retriever fan-out is genuinely parallel — on the relationship route,
vector search and graph traversal have no dependency on each other.

---

## Measured facts

Every number below was produced by a script in this repo and read from a manifest
in `backend/data/`. Nothing here is estimated or rounded up.

### Guideline corpus (`scripts/ingest_guidelines.py`)

| | |
|---|---|
| Rows scanned | 37,970 |
| Documents kept | 2,000 |
| Parent passages | 12,183 |
| Child chunks (retrieval unit) | 53,026 |
| Child tokens — mean / p95 / max | 383 / 453 / 499 |
| Rejected | 6,050 too short, 15 low text ratio |

Sources are **stratified**, not head-N: the corpus is ordered by publisher, so
taking the first 2,000 rows yields an almost entirely oncology corpus. Quotas are
allocated across all nine sources by available supply.

`cma` 287 · `cdc` 286 · `nice` 286 · `pubmed` 286 · `wikidoc` 286 · `who` 221 ·
`spor` 216 · `cco` 83 · `icrc` 49

### ED risk stratification (`scripts/train_risk.py`)

Task: predict admission from **triage-time** data only.
Cohort: 222 ED stays from **64 patients**, 67.6% admitted.

| Model | AUC-ROC (95% CI) | PR-AUC | Brier | Calibration slope |
|---|---|---|---|---|
| Logistic regression | 0.710 (0.635–0.783) | 0.821 | 0.218 | 0.53 |
| XGBoost | 0.638 (0.557–0.719) | 0.745 | 0.213 | 0.52 |
| **Calibrated LR** ← selected | 0.674 (0.603–0.747) | 0.802 | **0.207** | **0.78** |
| *Baseline:* triage acuity alone | 0.677 | 0.775 | — | — |
| *Baseline:* always-admit | 0.500 | — | — | — |

**Read this table honestly:**

- **XGBoost lost to logistic regression.** At n=222 with 22 features, gradient
  boosting overfits and a regularized linear model wins. That is the truthful
  outcome at this data scale.
- **The model barely beats the triage nurse.** 0.674–0.710 against an acuity-only
  baseline of 0.677, with heavily overlapping confidence intervals. There is no
  evidence here that this model adds clinical value over the score already being
  assigned at triage.
- **Selection is by Brier score, not AUC.** AUC measures ranking only and is
  invariant to monotone transforms, so a model can top the AUC table while its
  "0.9" corresponds to a 70% event rate. Brier is a strictly proper scoring rule
  covering calibration *and* discrimination. The two criteria disagree here, which
  is exactly why the rule is stated before the results.
- **Calibration was a real defect, and was fixed.** The uncalibrated models sit at
  a slope of ~0.53 — badly overconfident. Platt scaling moves it to 0.78 at a cost
  of 0.036 AUC. For a probability displayed at a bedside, that is the right trade.
- **Grouped validation is not optional here.** 222 stays come from 64 patients, so
  a random split puts the same patient on both sides. Ignoring patient grouping
  inflates XGBoost's AUC by **+0.030** — measured, not assumed.
- **Race is excluded as a predictor.** It exists in the source data. Using it would
  likely improve AUC by encoding access and referral disparities as if they were
  clinical facts.
- **Post-triage features are excluded.** `los_hours`, `n_vital_checks`, `hr_std`
  and friends are measured *after* the admission decision. `--include-leaky`
  reproduces the inflated numbers to size the effect.

### Benchmarks

See **[RESULTS.md](RESULTS.md)**. The runner (`scripts/run_benchmarks.py`) evaluates
MedQA / PubMedQA / MedMCQA across three retrieval paths — `vector_only`,
`graph_assisted`, `agentic` — so the agentic loop's accuracy gain can be weighed
against its latency and token cost rather than assumed.

---

## Quick start

### Prerequisites

- Python 3.11 (provisioned automatically by `uv`)
- Node 20+
- **macOS only:** `brew install libomp` — xgboost's wheel links against OpenMP and
  `import xgboost` fails without it.

### Backend

```bash
cd backend
uv venv --python 3.11
uv pip install -e ".[dev]"
cp .env.example .env        # optional — see "Degraded mode" below
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                 # http://localhost:3000
```

### Populate the index

```bash
python scripts/ingest_guidelines.py     # ~1 min  (downloads + chunks the corpus)
python scripts/ingest_mimic.py          # ~30 s   (MIMIC-IV-ED Demo)
python scripts/ingest_rxnorm.py         # ~10 s   (14,663 RxNorm ingredients)
python scripts/embed_index.py           # 45-90 min on Apple Silicon — resumable
python scripts/train_risk.py            # ~30 s
```

These need a real LLM key (see below):

```bash
python scripts/build_graph.py           # LLM triple extraction → NetworkX graph
python scripts/run_benchmarks.py --n 50 # writes RESULTS.md inputs
```

### Or use the Makefile

```bash
make install     # both packages
make pipeline    # data → index → risk  (everything needing no API key)
make api         # backend on :8000
make web         # frontend on :3000
make check       # lint → types → tests → build
```

---

## Degraded mode — the repo runs with zero credentials

Cloning and running this project requires no API key and no database. Two seams
make that true, and **both are loud, never silent**:

| Missing | Fallback | How you know |
|---|---|---|
| No LLM key | `StubProvider` — deterministic placeholder text | `degraded: true` in every API response, banner in the UI, `/api/health` reports `degraded` |
| No `DATABASE_URL` | `LocalVectorStore` — NumPy brute-force cosine | reported in `/api/health` under `vector_store.backend` |

The local vector store is not a token gesture. At 53k chunks × 1024 dims (~217 MB
in float32) an exhaustive search is a single matmul returning in milliseconds —
faster than a round trip to a cloud Postgres. What pgvector buys is a scaling
ceiling and SQL filtering, not demo-scale latency, and saying so is more useful
than pretending otherwise.

`guard_real_llm()` hard-fails every eval and benchmark script when the stub is
active. A metric computed against placeholder text is not a measurement, and
publishing one would make every other number here untrustworthy.

### To enable real inference

```bash
# https://console.groq.com/keys  — free, 60 seconds
echo 'GROQ_API_KEY=gsk_...' >> backend/.env

# https://console.neon.tech — free tier; use the POOLED connection string
echo 'DATABASE_URL=postgresql://...-pooler.../neondb' >> backend/.env
```

---

## Design decisions worth defending

**Parent–child chunking.** We retrieve on ~450-token children (one idea per
embedding, so dense retrieval is precise) and compose on ~2,000-token parents (wide
enough to actually answer from). Retrieving on large chunks blurs the embedding;
composing on small ones produces confident answers assembled from fragments.

**RRF, not weighted score blending.** BM25 scores are unbounded and corpus-
dependent; cosine sits in [-1, 1]. Blending needs a normalization scheme and a
weight, both of which require a labelled set we don't have at this scale.
Reciprocal Rank Fusion discards magnitudes and uses rank only — one parameter,
robust to the scale mismatch by construction.

**MMR after fusion.** Guideline corpora restate the same recommendation across many
documents. A rank-ordered top-k returns five phrasings of one idea, which inflates
apparent support while giving the composer nothing new.

**The critic fails closed.** If the groundedness check itself throws, the score is
0.0 and the request escalates. A verification step that fails open is not a
verification step.

**Retries re-enter at retrieval, not drafting.** A groundedness failure means we
found the wrong evidence. Re-drafting against the same context just produces a
differently-worded ungrounded answer.

**The escalation message is a template, not generated.** A model asked to write a
refusal tends to smuggle a partial answer into it.

**RAGAS metrics implemented natively.** RAGAS pulls a LangChain provider stack and
expects an OpenAI-shaped client. The metric *definitions* are what matter and they
are short, so the safety-critical critic path doesn't depend on a heavy third-party
eval library. `pip install -e ".[ragas]"` installs upstream RAGAS for cross-checking.

**SSE is parsed by hand over `fetch`, with line endings normalized.** `EventSource`
cannot issue a POST, and the query body carries the question plus an optional
patient id — so the stream is read off `fetch` and framed manually. Two traps
there, both hit during this build: a network chunk has no relationship to an SSE
frame (a `data:` line routinely splits across two reads, so the parser must hold a
buffer across them), and `sse-starlette` emits **CRLF** — a parser that splits on
`"\n\n"` alone matches nothing against `"\r\n\r\n"` and silently emits zero events
while the server logs a perfectly successful run. `curl` hides this because it
prints raw bytes.

**CSS sticky instead of GSAP ScrollTrigger pinning.** The "how it thinks" section
was built on ScrollTrigger's `pin` first. It doesn't compose with Lenis: Lenis
animates `scrollTop` on its own RAF while ScrollTrigger measures native scroll, and
the pin-spacer sizes against a position that no longer matches the screen — the
section collapsed to a blank viewport. `position: sticky` pins natively with no
spacer to keep in sync, and Motion's `useScroll` advances the steps. Fewer moving
parts, and it cannot desynchronize by construction.

---

## Safety layer

- **Non-dismissible disclaimer** on every page. No close button and no state behind
  it — a dismissible version would be reliably absent exactly when a user is
  deciding how much to trust an answer.
- **Identifier screening at intake**, before the query reaches any retriever, log
  line, or third-party API. Regex screen plus an LLM opinion; a hit from either
  raises the flag, which is recorded in the audit trail and surfaced in the API.
- **No PHI**, enforced by tests rather than asserted. `test_safety.py` scans the
  ingested cohort for identifier-shaped columns, verifies race is not a predictor,
  verifies post-triage leakage is excluded, and verifies grouped validation.
- **Escalation is first-class in the UI**, with its reason and score — not hidden
  in a log.
- **Audit trail** of every node transition, retrieval source, and confidence score,
  written to Postgres when available and JSONL otherwise, and never raising into
  the request path.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/query` | Run the agent. SSE-streams the trace, then the result. |
| `GET /api/risk-score/{id}` | ED risk score with the model's own validation numbers attached |
| `GET /api/risk-model` | Full model card: metrics, baselines, leakage check |
| `GET /api/health` | Per-component readiness — degraded if any dependency is stubbed |
| `GET /api/info` | Corpus / cohort / model manifests (what the UI displays) |
| `GET /api/audit` | Recent audit entries + aggregate stats |
| `GET /api/patients` | Sample MIMIC-IV-ED encounter ids for the demo |
| `GET /metrics` | Prometheus exposition format |
| `GET /metrics/summary` | Same counters as JSON, for the dashboard |

The metric worth alerting on is `shifa42_escalations_total / shifa42_runs_total`.
A sudden climb means retrieval has degraded — a stale index, a provider returning
junk — and it surfaces before any user reports a bad answer.

---

## Project layout

```
backend/
  app/
    agent/          LangGraph state, graph assembly, one module per node group
    retrieval/      chunking, embeddings, pgvector + local stores, BM25, RRF/MMR, graph
    llm/            provider chain + stub fallback, all prompts in one file
    eval/           RAGAS-definition metrics (shared by the critic and the harness)
    risk/           shared feature engineering + serving
    api/            FastAPI routers
    audit/          governance trail
  scripts/          ingestion, indexing, training, graph build, benchmarks
  tests/            59 tests — chunking, retrieval, safety gates, API contract
frontend/
  app/              Next.js 14 App Router
  components/       hero + R3F scene, console, trace strip, bento dossier, evidence
  lib/              SSE client, tokens, effects/theme state
docs/specs/         design spec
```

---

## Honest limitations

1. **The ED cohort is tiny.** 222 stays, 64 patients. Risk numbers are directional
   at demo scale, reported with confidence intervals, and the model does not
   convincingly beat triage acuity alone.
2. **Knowledge-graph edges are LLM-extracted and noisy.** `build_graph.py` holds
   out a sample with source text for manual checking; extraction precision is
   reported as measured or not at all.
3. **Free-tier rate limits bottleneck the agentic path.** Per-path latency and call
   counts are reported alongside accuracy so the complexity cost is visible.
4. **Benchmarks run at n=50 by default.** Confidence intervals are Wilson score
   intervals and they are wide. `--n 200` is supported for a longer run.
5. **MPS is not fully stable for hour-long embedding jobs.** An internal Metal
   command-buffer error was hit at ~30k chunks. The indexer now checkpoints to
   shards, retries, and falls back to CPU — but the underlying flakiness is a
   platform issue, not something this code fixes.
6. **Render's free tier will not run this.** 512 MB cannot hold bge-large's ~1.3 GB
   working set. See the caveats in `render.yaml`.

---

## Deployment

Artifacts are written and correct; **nothing has been deployed**.

- `.github/workflows/ci.yml` — lint → type-check → tests → build
- `render.yaml` (backend, native Python runtime), `frontend/vercel.json` (frontend)
- `Makefile` — one-command setup, pipeline, run, and quality targets
- `.env.example` in both packages; real secrets are never committed

There is intentionally **no Docker** in this project. The backend's dependency tree
is torch-dominated (~2.5 GB image even on CPU wheels), which buys little over
`uv` + a pinned Python for a single-service demo, and Render's native Python runtime
deploys it directly.

CI runs with no LLM key and no database on purpose, exercising the degraded path
end-to-end — that path is a supported mode, so a regression in it fails the build.

---

## Interview talking points

- *"The most important design decision was the Critic/Groundedness Agent. The
  system is built to say 'I don't know, escalate to a physician' rather than
  hallucinate, because a confident wrong answer is worse than no answer here. It
  fails closed: if the check itself errors, the score is zero and the request
  escalates."*
- *"XGBoost lost to logistic regression on the risk model, and the model barely
  beats the triage nurse's own acuity score. I'm reporting that, with confidence
  intervals, rather than picking the framing that flatters it."*
- *"I selected the risk model on Brier score, not AUC, because AUC is invariant to
  monotone transforms — a model can top the AUC table while its probabilities are
  badly calibrated, and this one displays a probability to a clinician."*
- *"222 stays come from 64 patients, so I used GroupKFold. Ignoring that inflates
  AUC by 0.030 — I measured it rather than hand-waving."*
- *"The Adaptive Router means the agentic loop only runs when the query actually
  needs it, and the three-way benchmark measures the accuracy gain against the
  added latency, so the complexity has to earn its keep."*
