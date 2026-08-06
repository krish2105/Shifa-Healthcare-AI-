# Shifa42 — Design Spec

**Date:** 2026-08-06
**Status:** Approved, in implementation
**Source brief:** `Shifa42_MASTER_BUILD_PROMPT.md` (v1.0)

This document records the decisions made on top of the master brief. Where this
document and the brief disagree, this document wins.

---

## 1. What we are building

An agentic clinical decision-support copilot that answers clinical questions grounded
in published guideline literature and de-identified emergency-department data, and
that refuses to answer when the retrieved evidence does not support a claim.

The engineering thesis: **the ability to decline is the primary feature.** Every
architectural choice below serves either grounding, verification, or auditability.

**This is a research and educational demonstration. It is not a certified medical
device.** No real PHI enters the codebase or the demo data.

---

## 2. Decisions made against the brief

| Area | Decision | Rationale |
|---|---|---|
| Python runtime | 3.11.15, provisioned by `uv` | Host runs 3.13; several ML dependencies still lag it. `uv` isolates this without touching system Python. |
| Vector store | Neon free-tier Postgres + `pgvector` | Same engine locally and in production, so there is no dev/prod drift. |
| Vector store fallback | Local NumPy brute-force store | See §4.2. The system must run end-to-end with zero credentials. |
| LLM provider | Groq (`llama-3.3-70b-versatile`) via LiteLLM | Free tier, fastest inference. Gemini and Cerebras remain configured as fallbacks in the router chain. |
| LLM fallback | Deterministic stub provider | See §4.1. Enables frontend work, CI, and tests without a key. |
| Embeddings | `BAAI/bge-large-en-v1.5`, local via sentence-transformers | 1024-dim, no API cost. Runs on Apple Silicon MPS. |
| Guideline corpus | ~2,000 documents, quality-filtered from `epfl-llm/guidelines` | Verified public and ungated (checked 2026-08-06). Fits the free Neon tier and embeds in well under an hour on an M4. |
| Chunking | Parent–child | Child chunks (~450 tok) for retrieval precision; parent chunks (~2,000 tok) supplied to the composer for context. |
| Sparse retrieval | BM25 via `rank_bm25`, fused with dense using Reciprocal Rank Fusion | RRF needs no score normalization tuning, which matters because BM25 and cosine scores are not comparable. |
| Graph store | NetworkX pickle | Zero infrastructure. Neo4j migration path documented, not built. |
| Critic threshold | faithfulness ≥ 0.75; max 2 reformulations; then escalate | Config value, not a constant. RESULTS.md reports the threshold-vs-escalation-rate curve. |
| Benchmarks | Runner supports `--n`, built for 200; executed at n=50 | Groq free-tier rate limits. Results checkpoint to disk so a stall loses nothing. |
| Deployment | Artifacts written, nothing deployed | Dockerfiles, compose, CI, `render.yaml`, `vercel.json` exist and are correct. Only the local stack is run. |
| Art direction | A — graphite base, cool teal accent | Selected from three built mockups. |
| Typography | Geist + Geist Mono, self-hosted via `next/font` | Variable, zero layout shift, mono handles dense numeric readouts. |

---

## 3. Architecture

The LangGraph node topology is exactly the diagram in §4 of the master brief. Each
box is a real node with its own system prompt, its own retry policy, and its own
audit row.

```
intake → planner ─┬→ vector_retriever ─┐
                  ├→ graph_retriever  ─┼→ fusion → draft → critic ─┬→ compose → done
                  └→ risk_node        ─┘                           │
                                                                    ├→ reformulate → (loop, max 2)
                                                                    └→ escalate → done
```

**Routing.** The planner classifies each query as `simple_factual`,
`needs_patient_context`, or `needs_relationship_reasoning` and activates only the
retrievers that classification requires. This is the mechanism that lets us measure
whether the agentic path earns its latency, rather than assuming it does.

**Termination.** The graph has exactly two terminal states: a composed answer with
inline citations, or an escalation. There is no third path where a low-confidence
answer ships with a hedge.

---

## 4. Degraded-mode design

The system must be runnable and demonstrable by someone who has cloned the repo and
supplied no credentials. Two seams make that true. Both are explicit and labelled in
the UI and logs — never silent.

### 4.1 LLM router

`app/llm/router.py` resolves a provider chain at startup:

1. `GROQ_API_KEY` present → Groq
2. else `GEMINI_API_KEY` present → Gemini
3. else `CEREBRAS_API_KEY` present → Cerebras
4. else → `StubProvider`

`StubProvider` returns deterministic, clearly-marked placeholder completions. It
exists so the frontend, the API contract, and the test suite can be exercised
without a key. Any response produced through it carries `degraded: true` in the API
payload and renders a visible banner in the UI. It is never used to produce a number
that appears in RESULTS.md.

### 4.2 Vector store

`app/retrieval/store.py` defines a `VectorStore` protocol with two implementations:

- `PgVectorStore` — used when `DATABASE_URL` is set. HNSW index, cosine distance.
- `LocalVectorStore` — a NumPy matrix on disk, brute-force cosine.

At our corpus scale (~40–60k chunks × 1024 dims ≈ 240 MB) brute-force search is a
single matmul and returns in a few milliseconds, so the fallback is genuinely usable
rather than a token gesture. The retrieval *results* are equivalent; only the index
structure and the scaling ceiling differ.

---

## 5. Data

| Purpose | Source | Access |
|---|---|---|
| Guideline corpus | `epfl-llm/guidelines` (HF) | Public, ungated — verified 2026-08-06 |
| ED encounters | MIMIC-IV-ED Demo v2.2 (PhysioNet) | Open, ODbL, no credentialing |
| Drug nomenclature | RxNorm (NLM) | Free |
| Benchmark | MedQA-USMLE-4-options | HF |
| Benchmark | PubMedQA | HF |
| Benchmark | MedMCQA | HF |

The ingester logs the exact document and chunk counts it actually produced. No count
is reported anywhere in the project that was not measured.

---

## 6. Safety layer

- Non-dismissible footer disclaimer on every page, wording per master brief §7.
- No PHI in the repository, the demo data, or the logs. Asserted in the README and
  covered by a test that scans ingested records for identifier-shaped fields.
- The escalation path is rendered in the UI as a first-class outcome with its reason
  and score, not hidden in a log file.
- Degraded mode (§4) is always visible to the user when active.

---

## 7. Frontend

Dark-first, graphite base, single teal accent. Built static-first, then motion.

- **Hero** — kinetic headline, React Three Fiber node-lattice scene, lazy-loaded and
  progressively enhanced. Page is complete and intentional with WebGL absent.
- **Query interface** — chat input, SSE-streamed response, live agent-trace strip
  showing each node as it fires.
- **Dashboard** — bento grid: groundedness gauge, expandable citations, risk output,
  audit log, escalation state.
- **Effects toggle** — "Motion & 3D" header control, Zustand-backed, defaulting from
  `useReducedMotion()`. Disables the 3D scene and large transforms; app stays fully
  usable.
- **Theme toggle** — `next-themes`, dark designed first.
- **Mobile** — responsive to 360px, `clamp()` fluid type, bottom sheets replacing
  side panels, static gradient-mesh fallback for the hero below the breakpoint.

Quality gate: keyboard-accessible, visible focus rings, `prefers-reduced-motion`
respected, only `transform`/`opacity` animated on scroll, body contrast ≥ 4.5:1.

---

## 8. Honest limitations (carried into RESULTS.md)

1. MIMIC-IV-ED Demo is ~100 patients. Risk-model numbers are directional at demo
   scale, not production-grade, and are reported with confidence intervals.
2. Knowledge-graph edges are extracted by a free-tier LLM. Extraction precision is
   measured on a manually-checked sample and reported as measured, not assumed.
3. Free-tier rate limits bottleneck the multi-retry agentic path. Per-path latency
   and call counts are reported alongside accuracy so the complexity cost is visible.
4. Benchmarks are executed at n=50 per set. Confidence intervals are reported and are
   wide. The runner supports n=200 for a longer run.

---

## 9. Out of scope

- Live deployment to any host.
- Neo4j (migration path documented only).
- Authentication and multi-tenancy.
- Any use of real patient data.
