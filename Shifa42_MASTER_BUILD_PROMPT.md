# SHIFA42 — Agentic Clinical Intelligence Copilot
## Master Build Prompt for Claude Code — v1.0 (Production-Ready Spec)

> **Paste this entire document into Claude Code as your project brief.** It contains the full architecture, real datasets, backend pipeline, frontend spec, and deployment plan. Build phase-by-phase in the order given in Section 9. Zero paid API keys required anywhere in this stack.

---

## 0. Executive Summary (for non-technical / C-level review)

Shifa42 is a portfolio-grade clinical decision-support copilot that answers clinical questions grounded in real, published guideline literature and real (de-identified, open-access) emergency-department data — and, critically, **knows when it doesn't know enough to answer**, escalating to a human clinician instead of guessing. It demonstrates the exact engineering pattern hospital systems (including G42/M42's own Med42 clinical LLM) need to deploy AI safely: retrieval grounded in verifiable sources, an automated groundedness check before any answer ships, and full audit logging of every autonomous decision.

**This is a research/educational portfolio project, not a certified medical device.** It uses only synthetic and open, de-identified research datasets — never real patient data — and every response in the UI carries a visible disclaimer that outputs require licensed clinician review. That constraint is treated as a feature: the whole architecture is built around proving the system respects it.

---

## 1. Problem Statement

Clinicians and care coordinators need fast answers grounded in clinical guidelines and a patient's own history. Off-the-shelf LLMs hallucinate confidently in this domain, and that's not a UX flaw — it's a safety failure mode. Shifa42's core engineering thesis: **a clinical AI system's most important feature is its ability to refuse to answer.**

---

## 2. Tech Stack (locked — zero paid API keys anywhere)

| Layer | Choice | Why |
|---|---|---|
| Frontend framework | Next.js 14 (App Router) + TypeScript | Your established stack |
| Styling | Tailwind CSS + shadcn/ui | Fast, accessible component primitives |
| Animation | Motion (`motion/react` v12) | Cheap-property animation, scroll reveals |
| 3D | React Three Fiber + `@react-three/drei` | Progressive-enhancement hero scene |
| Complex scroll timelines | GSAP ScrollTrigger | Only for the pinned multi-step "how it works" sequence |
| Smooth scroll | Lenis | Buttery scroll, reduced-motion aware |
| Theme | `next-themes` | Dark-first light/dark toggle |
| Client state | Zustand | Effects toggle, theme, session state |
| Backend framework | FastAPI (Python 3.11+) | Async, typed, fast |
| Agent orchestration | LangGraph | Multi-agent graph with retries/reflection |
| LLM routing | LiteLLM | Single interface across free-tier providers |
| LLM providers (free tier, rotate as needed) | Groq (Llama 3.3 70B), Google Gemini 2.0 Flash free tier, Cerebras free tier, Ollama (local fallback) | Zero cost |
| Embeddings | `BAAI/bge-large-en-v1.5` (local, sentence-transformers) | Free, strong retrieval baseline; no embedding API cost |
| Vector store | PostgreSQL + pgvector | Free tier on Supabase/Neon, or local Docker |
| Graph store | NetworkX (default, zero-infra) → Neo4j Community (optional upgrade) | Start simple, upgrade only if the graph queries need it |
| Evaluation | RAGAS | Faithfulness, context precision/recall, answer relevance |
| Containerization | Docker + Docker Compose | One-command local stack |
| CI/CD | GitHub Actions | Lint → type-check → test → build → deploy |
| Frontend deploy | Vercel (free tier) | |
| Backend deploy | Render or Fly.io (free tier) | |
| Managed Postgres | Neon or Supabase (free tier) | |

---

## 3. Datasets — real, concrete, zero-cost, no credentialing barrier

This is the part that makes the accuracy numbers defensible instead of invented. Every dataset below is real, publicly downloadable today, and requires **no paid access and no institutional credentialing** (this matters — full MIMIC-IV requires a completed CITI training + data use agreement; the pieces below don't).

| Purpose | Dataset | What it actually is | Access |
|---|---|---|---|
| RAG knowledge base (guideline corpus) | **`epfl-llm/guidelines`** (Hugging Face) | Real curated clinical practice guideline corpus, used in real clinical-LLM research (the Meditron project) | Open, `datasets.load_dataset("epfl-llm/guidelines")` |
| Structured patient/encounter context | **MIMIC-IV-ED Demo** (PhysioNet) | Real, de-identified subset of ~100 patients from a database of 400,000+ real emergency-department admissions at Beth Israel Deaconess Medical Center — triage vitals, chief complaints, diagnosis codes, disposition | Open access, ODbL license, no credentialing: `physionet.org/content/mimic-iv-ed-demo/2.2/` |
| Medication normalization / interaction reference | **RxNorm** (U.S. National Library of Medicine) | Real, authoritative drug nomenclature and interaction reference | Free, `www.nlm.nih.gov/research/umls/rxnorm` |
| **Accuracy benchmark #1** | **MedQA (USMLE)** | 1,273 real USMLE board-exam questions | HF: `GBaker/MedQA-USMLE-4-options` |
| **Accuracy benchmark #2** | **PubMedQA** | Real biomedical yes/no/maybe QA grounded in PubMed abstracts | HF: `bigbio/pubmed_qa` |
| **Accuracy benchmark #3** | **MedMCQA** | 194k+ real AIIMS/NEET-PG medical entrance exam questions | HF: `openlifescienceai/medmcqa` |

**Why this matters for your "need real dataset for accuracy" requirement:** Med42 (M42's actual clinical LLM) reported a 72% zero-shot USMLE-style score as its headline accuracy claim. You're going to do the same thing at smaller scale: run Shifa42's full retrieval+generation pipeline against held-out slices of MedQA/PubMedQA/MedMCQA and **report the real number you get — including if it's mediocre.** That honesty is the differentiator, not the score itself.

---

## 4. Full System Architecture

```
                              ┌─────────────────────────┐
                              │   Next.js 14 Frontend    │
                              │  (chat UI + dashboard)   │
                              └────────────┬─────────────┘
                                           │ REST/SSE
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
   (structures    Planner        (pgvector +       (NetworkX        (XGBoost on
   the query)     (adaptive:     bge-large-en)     traversal over   MIMIC-IV-ED
                  vector/graph/                    condition-drug-  features)
                  both)                             symptom graph)
        └──────────────┴───────────────┴──────────────┴──────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  Context Fusion Agent     │
                              │  (dedup, MMR diversity)   │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  Draft Composer Agent     │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │  Critic / Groundedness    │──low confidence──┐
                              │  Agent (RAGAS-style score)│                  │
                              └────────────┬─────────────┘                  │
                                    high confidence                          ▼
                                           │                    ┌────────────────────┐
                              ┌────────────▼─────────────┐      │ Reformulate query   │
                              │  Final Composer           │      │ & retry (max 2x)    │
                              │  (adds inline citations)  │      └──────────┬──────────┘
                              └────────────┬─────────────┘                 │
                                           │                    still low? │
                              ┌────────────▼─────────────┐                 ▼
                              │  Response to user with     │    ┌────────────────────┐
                              │  source citations + score  │    │ Escalation Agent:   │
                              └────────────────────────────┘    │ "Insufficient       │
                                                                  │ evidence — escalate │
                                                                  │ to physician."      │
                                                                  └────────────────────┘
```

Every box above is a real LangGraph node with its own prompt, retries, and logged decision — this graph diagram *is* your architecture documentation, use it directly in the README and in interviews.

---

## 5. Backend Pipeline — Full Workflow, End to End

### Phase A — Data ingestion & knowledge graph construction
1. Download `epfl-llm/guidelines` and the MIMIC-IV-ED Demo tables.
2. Chunk guideline text using **parent-child chunking** (small chunks for retrieval precision, linked to larger parent chunks for context — a documented 2026 best practice, not just simple fixed-size chunking).
3. Run LLM-assisted entity/relation extraction (via LiteLLM → Groq free tier) over guideline chunks to build a knowledge graph: nodes = {conditions, medications, symptoms, guideline sections}, edges = {treats, contraindicated_with, symptom_of, references}.
4. Cross-reference medication nodes against RxNorm for canonical naming.
5. Persist the graph as a NetworkX pickle (simple, zero-infra) — document a Neo4j migration path for anyone who wants to scale it.

### Phase B — Embedding & indexing
6. Embed all chunks locally with `bge-large-en-v1.5` (no API cost).
7. Store in pgvector with metadata (source doc, section, parent chunk id).
8. Build a BM25 sparse index alongside it (`rank_bm25` or Postgres full-text search) for hybrid retrieval.

### Phase C — Risk-stratification ML model
9. Engineer features from MIMIC-IV-ED Demo: age, triage vitals, chief complaint category, prior condition flags.
10. Train and compare a baseline logistic regression against XGBoost for acuity/risk prediction. Report both AUC-ROC and calibration — **and report honestly if XGBoost's edge is marginal on a 100-patient demo set**, since that's the truthful outcome at this data scale and exactly the kind of honest limitation your existing DL labs already model well.

### Phase D — Agent graph (LangGraph)
11. Build each node from the architecture diagram in Section 4 as a distinct LangGraph node with its own system prompt.
12. **Retrieval Planner** = your Adaptive RAG router: a cheap few-shot LLM call classifies the query as simple-factual / needs-patient-context / needs-relationship-reasoning, and routes accordingly.
13. **Critic/Groundedness Agent** is the safety-critical node: score the draft against retrieved sources using a RAGAS-style faithfulness check; below threshold → reformulate and retry (cap at 2 retries); still low → hand off to the Escalation Agent, which returns a clear "insufficient evidence, escalate to a physician" response rather than a hedged guess.
14. Log every node transition, retrieval source, and confidence score to a structured audit table — this is your governance layer and your strongest interview talking point.

### Phase E — API layer
15. FastAPI endpoints: `POST /query` (SSE streaming response), `GET /risk-score/{patient_id}`, `GET /health`, `GET /metrics` (Prometheus-format, even if you're not running full Prometheus — the format alone signals production awareness).

### Phase F — Evaluation harness
16. Run RAGAS metrics (faithfulness, context precision, context recall, answer relevance) across a held-out query set.
17. Run the full pipeline against held-out slices of MedQA (200 questions), PubMedQA (200), and MedMCQA (200) and report accuracy per benchmark, per retrieval path (vector-only vs. graph-assisted vs. full agentic loop) — this three-way comparison is what proves the agentic loop is earning its complexity cost, not just adding latency.

---

## 6. Frontend Spec — Modern UI, 3D, Effects, Theming, Mobile

### Visual direction
Dark-first design (per 2026 best practice — designed as the primary theme, not an inverted afterthought), clinical-but-not-clinical-boring: deep navy/graphite base, a single confident accent color (teal or a muted medical-blue — avoid the AI-generic acid-green-on-black look), generous whitespace, restrained glassmorphism only on the nav bar and citation-source modals.

### Layout
- **Hero section:** kinetic typography headline ("Ask. We'll show our work.") with a React Three Fiber 3D scene behind it — a slowly rotating abstract molecular/neural-node structure in the accent color, mouse-reactive tilt, **lazy-loaded and progressively enhanced** (page is fully functional and still looks intentional with WebGL absent, per the quality bar — critical on mobile/low-power devices).
- **Query interface:** chat-style input, streaming response with the retrieval path visualized live (a small animated indicator showing "vector search → graph traversal → critic check → done," so the user watches the agent actually work — this is your best demo moment).
- **Dashboard (bento grid):** cards for citation sources (expandable), groundedness score (animated counter/gauge), risk-stratification output, and an audit-log panel — bento grids scan and dwell well and are still winning in production per current design consensus.
- **Effects toggle:** a header control labeled "Motion & 3D" (on/off) — separate from the theme toggle, wired to `useReducedMotion()` plus a manual override in Zustand, disables the 3D scene and large transforms while keeping the app fully usable. This is both an accessibility requirement and a real performance lever on low-end devices.
- **Theme toggle:** animated sun/moon morph switch (`next-themes`), persisted across sessions, dark mode designed with deliberate contrast rather than a simple CSS invert.

### Motion
- Scroll-triggered reveals via `whileInView` for section entrances.
- A single GSAP ScrollTrigger pinned sequence for the "how the agent thinks" explainer section (the one place a complex pinned timeline earns its dependency weight).
- Lenis smooth scroll site-wide, disabled automatically under `prefers-reduced-motion`.
- Magnetic hover on primary CTAs, animated counters for stats ("guideline chunks indexed," "avg groundedness score," "benchmark accuracy") pulled live from the `/metrics` endpoint — real numbers, not placeholder text.

### Mobile
- Fully responsive to 360px width, fluid type/spacing via `clamp()`.
- 3D hero scene either simplified (fewer particles/geometry) or replaced with a static gradient-mesh fallback below a defined breakpoint — never a broken or clipped layout.
- Bottom sheet pattern for the citation/audit panels on mobile instead of side panels.
- Real `<button>`/`<a>` elements throughout, visible focus rings, logical tab order — no motion at the expense of the underlying document.

---

## 7. Safety, Compliance & Disclaimer Layer (build this, don't skip it)

- A persistent, non-dismissible footer disclaimer: *"Shifa42 is a research and educational demonstration. It is not a certified medical device and does not provide medical advice. All data shown is synthetic or open, de-identified research data (MIMIC-IV-ED Demo, PhysioNet). Any real clinical use requires licensed physician oversight."*
- No real PHI anywhere in the codebase or demo data — confirm this explicitly in the README.
- The Escalation Agent path (Section 5, step 13) is your proof that the system is designed to defer to humans rather than overreach — make this visible in the UI, not just in logs.

---

## 8. "C-Level" Production Deployment

- **Docker Compose** — one command (`docker compose up`) brings up frontend, backend, Postgres+pgvector, and the agent service locally.
- **GitHub Actions CI/CD** — on push to `main`: lint (ruff + eslint) → type-check (mypy + tsc) → test (pytest + vitest) → build → deploy.
- **Production targets:** Vercel (frontend), Render or Fly.io (FastAPI + LangGraph backend), Neon or Supabase (managed Postgres/pgvector) — all free-tier.
- **Environment management:** `.env.example` committed, real secrets never committed, documented in README.
- **Observability:** structured JSON logging, a `/health` endpoint, and a `/metrics` endpoint in Prometheus exposition format (wire it to a free Grafana Cloud dashboard if you want the visual, or just leave the endpoint — the format itself signals you understand production monitoring).
- **Security baseline:** locked CORS origins, request rate limiting (`slowapi`), input validation on every endpoint (Pydantic v2 models), no PHI persisted.

---

## 9. Build Order (give this to Claude Code as your execution sequence)

1. Scaffold repo: `/backend` (FastAPI + LangGraph), `/frontend` (Next.js 14), `docker-compose.yml`, `.env.example`, `README.md`.
2. Data ingestion scripts (Phase A) — pull and process the three real datasets from Section 3.
3. Embedding + indexing pipeline (Phase B).
4. Risk model training script + saved artifact (Phase C).
5. LangGraph agent graph, node by node, matching Section 4's diagram exactly (Phase D).
6. FastAPI endpoints wired to the agent graph (Phase E).
7. Evaluation harness + benchmark runner, produce a `RESULTS.md` with real numbers (Phase F).
8. Frontend: static structure and Tailwind styling first, get it looking right before it moves (per the frontend quality workflow) — then layer in Motion, the 3D hero, theme toggle, and effects toggle.
9. Wire frontend to backend via SSE streaming for the live "agent thinking" visualization.
10. Docker Compose + CI/CD + deploy to Vercel/Render/Neon.
11. Write the README with the architecture diagram, real benchmark numbers, honest limitations section, and viva/interview Q&A prep (matching your standard project-documentation format).

---

## 10. Honest Limitations to Report Up Front (write these into RESULTS.md, don't wait to discover them)

- MIMIC-IV-ED Demo is only ~100 patients — the risk-stratification model's numbers should be reported as directional/demo-scale, not production-grade, exactly like your FraudShield PR-AUC caveat.
- Knowledge-graph entity extraction quality depends on the free-tier LLM used for extraction — expect noisier edges than a paid frontier model would produce; report extraction precision on a manually-checked sample rather than assuming it's clean.
- Free-tier LLM rate limits will bottleneck the agentic (multi-retry) path under load — document the latency/cost tradeoff of each routing path from Section 5, step 17, honestly.

---

## 11. Viva / Interview Talking Points

- "The most important design decision was the Critic/Groundedness Agent — the system is built to say 'I don't know, escalate to a physician' rather than hallucinate, because in this domain a wrong confident answer is worse than no answer."
- "I benchmarked against real USMLE/PubMed/AIIMS question sets, not made-up test cases, and I'm reporting the real number including where it's weak."
- "I used MIMIC-IV-ED Demo specifically because it's real de-identified hospital data that's openly accessible without a credentialing process, which let me build something grounded in real clinical data structure without any PHI risk."
- "The agentic RAG loop only kicks in when the Adaptive Router decides the query actually needs it — I measured the accuracy gain against the added latency/cost to prove the complexity is earning its keep, not just showing off."
