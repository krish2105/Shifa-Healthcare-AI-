# Shifa42 — Results

**Run date:** 2026-08-06
**Hardware:** Apple M4 Pro, 24 GB, macOS 15.5
**Embeddings:** `BAAI/bge-large-en-v1.5`, local, fp16 on MPS
**Vector store:** local NumPy (no `DATABASE_URL` supplied for this run)

Every figure in this document was produced by a script in this repository and read
from a manifest in `backend/data/`. Sections that have not been run say so
explicitly rather than carrying a placeholder that could be mistaken for a result.

---

## 1. Status of each measurement

| Measurement | Status | Blocker |
|---|---|---|
| Guideline corpus ingestion | ✅ measured | — |
| ED cohort assembly | ✅ measured | — |
| Risk stratification model | ✅ measured | — |
| RxNorm normalization table | ✅ measured | — |
| Embedding + hybrid index | ✅ measured | — |
| Knowledge-graph extraction | ⏳ **not run** | needs `GROQ_API_KEY` |
| RAGAS retrieval metrics | ⏳ **not run** | needs `GROQ_API_KEY` |
| MedQA / PubMedQA / MedMCQA | ⏳ **not run** | needs `GROQ_API_KEY` |

`guard_real_llm()` blocks all four of the pending items while the stub provider is
active. A metric computed against deterministic placeholder text is not a
measurement, and reporting one would make every other number here untrustworthy.

To produce them:

```bash
echo 'GROQ_API_KEY=gsk_...' >> backend/.env
python scripts/build_graph.py --chunks 900 --audit-sample 40
python scripts/run_benchmarks.py --n 50
```

---

## 2. Guideline corpus

Source: `epfl-llm/guidelines` (Hugging Face) — public, ungated, verified 2026-08-06.

| | |
|---|---|
| Rows scanned | 37,970 |
| Documents kept | 2,000 |
| Parent passages | 12,183 |
| Child chunks (retrieval unit) | 53,026 |
| Child tokens — mean / median / p95 / max | 383 / 425 / 453 / **499** |
| Rejected — too short | 6,050 |
| Rejected — low text ratio | 15 |

The max child chunk is 499 tokens, under bge-large's 512-token window. This matters:
a chunk over the limit is silently truncated by the encoder rather than raising, so
its tail would become permanently unsearchable. `test_chunking.py` asserts the bound.

### Source distribution (stratified)

| Source | Docs | | Source | Docs |
|---|---|---|---|---|
| `cma` | 287 | | `who` | 221 |
| `cdc` | 286 | | `spor` | 216 |
| `nice` | 286 | | `cco` | 83 |
| `pubmed` | 286 | | `icrc` | 49 |
| `wikidoc` | 286 | | | |

**Why stratification was necessary.** The corpus is ordered by publisher. An initial
smoke run taking the first 32 eligible rows returned 30 documents from a single
source (`cco`, oncology). Head-N sampling at n=2,000 would have produced a corpus
that looks excellent on cancer questions and collapses on everything else — with the
benchmark average hiding it. Quotas are allocated across all nine sources by
available supply, and within a source documents are taken on an even stride rather
than from the head, since documents within a publisher are themselves topic-ordered.

---

## 3. ED cohort

Source: MIMIC-IV-ED Demo v2.2 (PhysioNet), ODbL, open access, no credentialing.

| | |
|---|---|
| ED stays | 222 |
| Unique patients | **64** |
| Admitted | 150 (67.6%) |
| Triage acuity 1 / 2 / 3 / 4 | 18 / 97 / 90 / 2 (15 missing) |

The 222-stays-from-64-patients ratio is the single most consequential fact about
this dataset, and it dictates the validation scheme in §4.

---

## 4. Risk stratification

**Task:** predict admission from the emergency department, using triage-time data only.
**Validation:** `GroupKFold(n_splits=5)` on `subject_id`.

| Model | AUC-ROC (95% CI) | PR-AUC | Brier | Calibration slope |
|---|---|---|---|---|
| Logistic regression | 0.710 (0.635–0.783) | 0.821 | 0.218 | 0.53 |
| XGBoost | 0.638 (0.557–0.719) | 0.745 | 0.213 | 0.52 |
| **Calibrated LR** ← selected | 0.674 (0.603–0.747) | 0.802 | **0.207** | **0.78** |
| *Baseline:* triage acuity alone | 0.677 | 0.775 | — | — |
| *Baseline:* always-admit | 0.500 | (acc 0.676) | — | — |

Top features (calibrated LR): `o2sat`, `acuity`, `cc_dyspnea`,
`arrival_transport_WALK IN`, `cc_fever`. 22 features total.

### 4.1 XGBoost lost

At n=222 with 22 features, gradient boosting overfits and a regularized linear model
wins by a clear margin (0.710 vs 0.638). This is the honest outcome at this data
scale, and it is the result regardless of which model would have made a better story.

### 4.2 The model barely beats the triage nurse

Triage acuity used directly as a risk score scores **0.677**. The best model scores
0.674–0.710 with confidence intervals that overlap it almost entirely. **There is no
evidence in this data that the model adds clinical value over the score already
being assigned at triage.** A version of this project that omitted the acuity
baseline would look considerably more impressive and would be considerably less
true.

### 4.3 Selection criterion: Brier, not AUC

AUC measures ranking only and is invariant to any monotone transform of the
predictions — a model can top the AUC table while its "0.9" corresponds to a 70%
event rate. Shifa42 shows this number to a clinician as a probability, so what
matters is whether the number means what it says.

Brier score is a strictly proper scoring rule, decomposing into calibration and
refinement, and minimized only by correct probabilities. The two criteria disagree
on this dataset, which is precisely why the rule is stated before the results:

- AUC ranks plain LR first (0.710 vs 0.674)
- Brier ranks calibrated LR first (0.207 vs 0.218), with slope 0.78 vs 0.53

### 4.4 Calibration was a defect, and was fixed

Both uncalibrated models sit near a calibration slope of **0.53** — substantially
overconfident. Platt scaling (sigmoid, not isotonic: 222 rows is far too few for a
non-parametric mapping) moves it to **0.78** and improves Brier from 0.218 to 0.207,
at a cost of 0.036 AUC that sits well inside the confidence interval.

### 4.5 Measured leakage from ignoring patient grouping

| Split | XGBoost AUC |
|---|---|
| `StratifiedKFold` (ignores patients) | 0.667 |
| `GroupKFold` on `subject_id` | 0.638 |
| **Inflation** | **+0.030** |

The ungrouped figure is what a naive random split would have reported. It is
recorded only to size the bias; every number elsewhere in this document is grouped.

### 4.6 Exclusions

- **Race** is present in the source data and is **not** used as a predictor. It
  would likely improve AUC by encoding access and referral disparities as clinical
  facts.
- **Post-triage features** (`los_hours`, `n_vital_checks`, `hr_std`, `sbp_min`,
  `o2sat_min`, `n_diagnoses`) are measured after the admission decision and are
  excluded. Length of stay in particular is close to a restatement of the target.
  `--include-leaky` reproduces the inflated numbers.

---

## 5. Embedding and index

| | |
|---|---|
| Model | `BAAI/bge-large-en-v1.5` (1024-dim) |
| Precision | fp16 on MPS |
| Chunks indexed | 53,026 |
| Vector store | local NumPy (`DATABASE_URL` unset for this run) |
| Sparse index | BM25 (`rank_bm25`) over the same chunks |

### 5.1 Throughput, measured

Benchmarked on this machine over the real corpus (mean ~460 tokens/chunk):

| Configuration | Throughput | Projected for 53k chunks |
|---|---|---|
| bge-large, fp32, batch 32 | 7.6 chunks/s | ~116 min |
| bge-large, fp32, batch 128 | 7.1 chunks/s | ~125 min |
| **bge-large, fp16, batch 128** | **9.3 chunks/s** | **~95 min** |
| bge-base, fp16, batch 128 | 30.3 chunks/s | ~29 min |
| bge-large, fp32, CPU | 4.4 chunks/s | ~200 min |

fp16 buys ~22% for no meaningful retrieval cost — embeddings are L2-normalized
immediately afterwards, so half-precision rounding lands far below the granularity
that affects ranking. Batches past 128 stop helping; the MPS backend is
memory-bandwidth bound here, not compute bound.

bge-base is 3.3× faster and was rejected: it is a 768-dim model with a weaker
retrieval baseline, and the corpus is indexed once.

### 5.2 MPS instability — a real failure, and the fix

The first full run **crashed at 29,696 / 53,026 chunks** (56%) after ~37 minutes of
sustained GPU work:

```
Error: command buffer exited with error status.
  The Metal Performance Shaders operations encoded on it may not have completed.
  Internal Error (00000001:Internal Error)
```

The original script saved only at the end, so the entire 37 minutes was lost. That
was a design flaw in a job of that length, not merely bad luck. `embed_index.py` was
rewritten to:

- checkpoint to 2,048-chunk shards on disk, written to a temp file and renamed so an
  interrupted write cannot leave a truncated shard a later resume would trust;
- resume by default, skipping shards already present and validating their shape;
- flush the MPS allocator periodically — the failure correlates with accumulated
  allocator pressure rather than any single batch;
- retry a failed batch once after rebuilding the model, then **permanently fall back
  to CPU** for the remainder rather than aborting.

This is a platform flakiness issue, not something this code fixes; it makes the job
survivable rather than making MPS reliable.

---

## 6. RxNorm

| | |
|---|---|
| Source | RxNav `allconcepts` (`tty=IN`), NLM — free, no key |
| Ingredient concepts | **14,663** |

Used to collapse medication nodes onto a single RXCUI identity. Without it
"paracetamol", "acetaminophen" and a brand name become three unconnected graph
nodes and a query about one finds none of the others' edges.

---

## 7. Knowledge graph — NOT YET RUN

Requires `GROQ_API_KEY`. When run, `scripts/build_graph.py` writes
`graph_manifest.json` (node/edge counts by type) and
`graph_extraction_audit.json` — a held-out sample of extracted triples with their
source text and a `manual_verdict` field.

**Extraction precision will be reported only from that manually-labelled file.**
Edges come from a free-tier LLM reading guideline prose, not from a curated ontology
such as UMLS or SNOMED, and they will be noisier. A precision figure derived from
anything other than hand-checking would be an assumption dressed as a measurement.

---

## 8. Benchmarks — NOT YET RUN

Requires `GROQ_API_KEY`. `scripts/run_benchmarks.py` evaluates three benchmarks ×
three retrieval paths:

| Benchmark | Source | Split |
|---|---|---|
| MedQA (USMLE) | `GBaker/MedQA-USMLE-4-options` | test, 1,273 available |
| PubMedQA | `qiaojin/PubMedQA` (`pqa_labeled`) | 1,000 available |
| MedMCQA | `openlifescienceai/medmcqa` | validation, 4,183 available |

All three loaders were verified against the live schemas on 2026-08-06.

**Paths:** `vector_only` → `graph_assisted` → `agentic`. The three-way split is the
point: it measures whether the agentic loop earns its latency and token cost against
the simpler paths, rather than assuming it does.

**Scoring.** The agentic path can decline to answer, so three figures are reported
together:

- `accuracy_strict` — escalations counted as incorrect. **The headline number.**
- `accuracy_answered` — accuracy on the answered subset only.
- `coverage` — the fraction answered at all.

Counting a refusal as correct would let the system score well by refusing
everything. Counting it as merely wrong erases the distinction between "confidently
wrong" and "knew it didn't know" — the behaviour this project exists to
demonstrate. `accuracy_answered` is never reported without `coverage` beside it.

Confidence intervals are **Wilson score intervals**, not the normal approximation:
at n=50 with proportions near 0 or 1 the normal interval runs outside [0,1] and
understates uncertainty exactly where these benchmarks are likely to land.

Expect the intervals to be wide at n=50. `--n 200` is supported for a longer run and
checkpoints after every question, so a free-tier rate-limit stall loses nothing.

---

## 9. Honest limitations

1. **The ED cohort is 64 patients.** Risk numbers are directional at demo scale, and
   the model does not convincingly beat triage acuity alone.
2. **Knowledge-graph edges will be noisy.** Precision reported as measured, or not
   at all.
3. **Free-tier rate limits bottleneck the agentic path.** Per-path latency and call
   counts are reported alongside accuracy.
4. **Benchmarks default to n=50.** Wide intervals, honestly stated.
5. **MPS is not stable for hour-long jobs** on this hardware. Mitigated, not solved.
6. **The local NumPy vector store was used for this run.** Retrieval *results* are
   equivalent to pgvector; only the index structure and scaling ceiling differ.
7. **No live deployment.** Deploy artifacts are written and correct; nothing was
   deployed. Render's 512 MB free tier cannot hold bge-large's working set.
