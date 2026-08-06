#!/usr/bin/env python
"""Phase F — benchmark the pipeline against real medical exam question sets.

Three benchmarks (MedQA/USMLE, PubMedQA, MedMCQA) x three retrieval paths:

  1. `vector_only`      dense + BM25 retrieval, answer composed directly
  2. `graph_assisted`   adds knowledge-graph traversal, still no verification
  3. `agentic`          the full graph: adaptive routing, critic, retry, escalation

The three-way split is the point. Anyone can add an agentic loop and report that
the system is sophisticated; this measures whether the loop earns its latency and
token cost against the simpler paths, and reports the answer either way.

**How escalations are scored.** The agentic path can decline to answer. Counting a
refusal as a correct answer would let the system score well by refusing everything;
counting it as simply wrong hides the distinction between "confidently wrong" and
"knew it didn't know" — which is the behaviour this project exists to demonstrate.
So both are reported:

  * `accuracy_strict`   escalations counted as incorrect (the headline number)
  * `accuracy_answered` accuracy on the answered subset only
  * `coverage`          fraction of questions answered at all

`accuracy_answered` without `coverage` beside it is meaningless, so they are always
emitted together.

Results checkpoint to disk after every question, so a free-tier rate-limit stall
loses nothing and `--resume` continues where it stopped.

Usage:
    python scripts/run_benchmarks.py --n 50
    python scripts/run_benchmarks.py --n 200 --paths agentic
    python scripts/run_benchmarks.py --n 50 --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.llm import router as llm
from app.logging_conf import get_logger

log = get_logger("bench")

SEED = 42
PATHS = ("vector_only", "graph_assisted", "agentic")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_runs"


# --------------------------------------------------------------------- datasets


def load_medqa(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    rng = random.Random(SEED)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idx:
        row = ds[i]
        opts = row["options"]
        # options is a dict {"A": "...", ...}
        choices = {k: str(v) for k, v in opts.items()} if isinstance(opts, dict) else {}
        if not choices:
            continue
        out.append(
            {
                "id": f"medqa_{i}",
                "question": row["question"],
                "choices": choices,
                "answer": str(row.get("answer_idx") or "").strip().upper(),
                "kind": "mcq",
            }
        )
    return out


def load_pubmedqa(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rng = random.Random(SEED)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idx:
        row = ds[i]
        out.append(
            {
                "id": f"pubmedqa_{i}",
                "question": row["question"],
                "choices": {"YES": "yes", "NO": "no", "MAYBE": "maybe"},
                "answer": str(row["final_decision"]).strip().upper(),
                "kind": "yesno",
            }
        )
    return out


def load_medmcqa(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    rng = random.Random(SEED)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    letters = ["A", "B", "C", "D"]
    out = []
    for i in idx:
        row = ds[i]
        choices = {
            "A": str(row["opa"]), "B": str(row["opb"]),
            "C": str(row["opc"]), "D": str(row["opd"]),
        }
        cop = row.get("cop")
        if cop is None or not (0 <= int(cop) < 4):
            continue
        out.append(
            {
                "id": f"medmcqa_{i}",
                "question": row["question"],
                "choices": choices,
                "answer": letters[int(cop)],
                "kind": "mcq",
            }
        )
    return out


LOADERS = {"medqa": load_medqa, "pubmedqa": load_pubmedqa, "medmcqa": load_medmcqa}


# ----------------------------------------------------------------- answer paths

ANSWER_SYSTEM = """You are answering a medical exam question using retrieved clinical guideline excerpts.

Respond with JSON only:
{"answer": "<the letter or YES/NO/MAYBE>", "reasoning": "<one sentence>"}

Use the SOURCES where they help. If they do not cover the question, answer from
clinical knowledge rather than refusing — this is an exam, and an unanswered
question is scored as wrong.
"""


def format_question(item: dict) -> str:
    lines = [item["question"], ""]
    for k, v in item["choices"].items():
        lines.append(f"{k}. {v}")
    return "\n".join(lines)


async def retrieve(query: str, use_graph: bool) -> list[str]:
    """Retrieval shared by the two non-agentic paths."""
    from app.retrieval.bm25 import get_bm25
    from app.retrieval.embedder import embed_query
    from app.retrieval.fusion import deduplicate, mmr_select, reciprocal_rank_fusion
    from app.retrieval.store import get_store

    store = get_store()
    lists = []
    weights = []

    try:
        dense = store.search(embed_query(query), settings.retrieval_candidate_k)
        if dense:
            lists.append(dense)
            weights.append(1.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("bench.dense_failed", error=str(exc)[:120])

    bm = get_bm25()
    if bm:
        sparse = bm.search(query, settings.retrieval_candidate_k)
        if sparse:
            lists.append(sparse)
            weights.append(0.8)

    if use_graph:
        from app.agent.nodes.retrievers import _resolve_chunks
        from app.retrieval.graph_store import get_graph

        kg = get_graph()
        if kg:
            seeds = kg.match_entities(query, limit=6)
            paths = kg.traverse(seeds) if seeds else []
            chunk_ids: list[str] = []
            for p in paths[: settings.retrieval_top_k]:
                chunk_ids.extend(p.chunk_ids)
            g_hits = _resolve_chunks(list(dict.fromkeys(chunk_ids))[: settings.retrieval_candidate_k])
            if g_hits:
                lists.append(g_hits)
                weights.append(0.6)

    if not lists:
        return []

    fused = deduplicate(reciprocal_rank_fusion(lists, weights=weights))
    selected = mmr_select(fused, None, k=settings.retrieval_top_k)
    return [h.chunk.text for h in selected]


async def run_simple_path(item: dict, use_graph: bool) -> dict:
    t0 = time.perf_counter()
    q = format_question(item)
    contexts = await retrieve(item["question"], use_graph)

    sources = "\n\n".join(f"[{i}] {c[:1200]}" for i, c in enumerate(contexts[:8], start=1))
    result = await llm.complete_json(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"SOURCES:\n{sources}\n\nQUESTION:\n{q}"},
        ],
        tier=llm.Tier.STRONG,
        default={"answer": ""},
        max_tokens=350,
    )
    return {
        "predicted": normalize_answer(str(result.get("answer") or ""), item),
        "escalated": False,
        "latency_s": round(time.perf_counter() - t0, 2),
        "sources": len(contexts),
    }


async def run_agentic_path(item: dict) -> dict:
    from app.agent.graph import run_agent

    t0 = time.perf_counter()
    q = format_question(item)
    res = await run_agent(
        f"{q}\n\nWhich option is correct? Answer with the letter or YES/NO/MAYBE.",
    )
    predicted = "" if res["escalated"] else normalize_answer(res["answer"], item)
    return {
        "predicted": predicted,
        "escalated": bool(res["escalated"]),
        "groundedness": res.get("groundedness", 0.0),
        "route": res.get("route", ""),
        "attempts": res.get("attempts", 0),
        "latency_s": round(time.perf_counter() - t0, 2),
        "sources": res.get("sources_reviewed", 0),
    }


def normalize_answer(raw: str, item: dict) -> str:
    """Pull a choice label out of free-form model output."""
    import re

    text = raw.strip().upper()
    if not text:
        return ""

    if item["kind"] == "yesno":
        for token in ("YES", "NO", "MAYBE"):
            if re.search(rf"\b{token}\b", text):
                return token
        return ""

    valid = set(item["choices"].keys())
    m = re.search(r"\b([A-D])\b", text)
    if m and m.group(1) in valid:
        return m.group(1)

    # Fall back to matching the option text itself, for answers that state the
    # content rather than the label.
    for letter, body in item["choices"].items():
        if body and body.upper()[:40] in text:
            return letter
    return ""


# --------------------------------------------------------------------- scoring


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Not the normal approximation: at n=50 with proportions near 0 or 1 the normal
    interval runs past [0,1] and understates uncertainty exactly where these
    benchmarks are most likely to land.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def summarize(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    correct = sum(1 for r in rows if r["correct"])
    escalated = sum(1 for r in rows if r.get("escalated"))
    answered = [r for r in rows if not r.get("escalated") and r["predicted"]]
    answered_correct = sum(1 for r in answered if r["correct"])

    lo, hi = wilson_ci(correct, n)
    a_lo, a_hi = wilson_ci(answered_correct, len(answered)) if answered else (0.0, 0.0)

    return {
        "n": n,
        "accuracy_strict": round(correct / n, 4),
        "accuracy_strict_ci95": [round(lo, 4), round(hi, 4)],
        "accuracy_answered": round(answered_correct / len(answered), 4) if answered else None,
        "accuracy_answered_ci95": [round(a_lo, 4), round(a_hi, 4)] if answered else None,
        "coverage": round(len(answered) / n, 4),
        "escalated": escalated,
        "escalation_rate": round(escalated / n, 4),
        "unparseable": sum(1 for r in rows if not r["predicted"] and not r.get("escalated")),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "mean_sources": round(sum(r.get("sources", 0) for r in rows) / n, 1),
    }


# ------------------------------------------------------------------------ main


async def main_async(args: argparse.Namespace) -> int:
    settings.ensure_dirs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    llm.guard_real_llm("Benchmarking")

    from app.retrieval.store import get_store

    if get_store().count() == 0:
        log.error("bench.empty_index", hint="Run scripts/embed_index.py first.")
        return 1

    benchmarks = args.benchmarks.split(",")
    paths = args.paths.split(",")

    checkpoint = RESULTS_DIR / f"checkpoint_n{args.n}.jsonl"
    done_keys: set[str] = set()
    rows: list[dict] = []

    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
            done_keys.add(f"{r['benchmark']}|{r['path']}|{r['id']}")
        log.info("bench.resumed", completed=len(rows))

    fh = checkpoint.open("a")

    try:
        for bench in benchmarks:
            loader = LOADERS.get(bench)
            if not loader:
                log.warning("bench.unknown", benchmark=bench)
                continue

            log.info("bench.loading", benchmark=bench, n=args.n)
            try:
                items = loader(args.n)
            except Exception as exc:  # noqa: BLE001
                log.error("bench.load_failed", benchmark=bench, error=str(exc)[:220])
                continue
            log.info("bench.loaded", benchmark=bench, items=len(items))

            for path in paths:
                for i, item in enumerate(items):
                    key = f"{bench}|{path}|{item['id']}"
                    if key in done_keys:
                        continue

                    try:
                        if path == "agentic":
                            out = await run_agentic_path(item)
                        else:
                            out = await run_simple_path(item, use_graph=(path == "graph_assisted"))
                    except Exception as exc:  # noqa: BLE001
                        log.warning("bench.item_failed", key=key, error=str(exc)[:160])
                        out = {"predicted": "", "escalated": False, "latency_s": 0.0, "sources": 0}

                    row = {
                        "benchmark": bench,
                        "path": path,
                        "id": item["id"],
                        "expected": item["answer"],
                        "correct": out["predicted"] == item["answer"],
                        **out,
                    }
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()

                    if (i + 1) % 10 == 0:
                        subset = [r for r in rows if r["benchmark"] == bench and r["path"] == path]
                        acc = sum(1 for r in subset if r["correct"]) / max(len(subset), 1)
                        log.info("bench.progress", benchmark=bench, path=path,
                                 done=i + 1, of=len(items), running_acc=round(acc, 3))
    finally:
        fh.close()

    # ------------------------------------------------------------- summarize
    report: dict[str, Any] = {
        "config": {
            "n_per_benchmark": args.n,
            "benchmarks": benchmarks,
            "paths": paths,
            "groundedness_threshold": settings.groundedness_threshold,
            "max_reformulations": settings.max_reformulations,
            "retrieval_top_k": settings.retrieval_top_k,
            "embedding_model": settings.embedding_model,
            "llm": llm.provider_info(),
            "seed": SEED,
        },
        "results": {},
        "totals": {},
        "notes": [
            "accuracy_strict counts escalations as incorrect — this is the headline number.",
            "accuracy_answered is accuracy on answered questions only and is meaningless "
            "without coverage beside it.",
            "Confidence intervals are Wilson score intervals, which stay inside [0,1] at "
            "small n where the normal approximation does not.",
        ],
    }

    for bench in benchmarks:
        report["results"][bench] = {}
        for path in paths:
            subset = [r for r in rows if r["benchmark"] == bench and r["path"] == path]
            if subset:
                report["results"][bench][path] = summarize(subset)

    for path in paths:
        subset = [r for r in rows if r["path"] == path]
        if subset:
            report["totals"][path] = summarize(subset)

    report["llm_usage"] = llm.USAGE.snapshot()

    out_path = settings.artifacts_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(report, indent=2))

    log.info("bench.complete", rows=len(rows), path=str(out_path))
    print("\n" + json.dumps(report["totals"], indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Questions per benchmark.")
    ap.add_argument("--benchmarks", default="medqa,pubmedqa,medmcqa")
    ap.add_argument("--paths", default=",".join(PATHS))
    ap.add_argument("--resume", action="store_true", help="Continue from the checkpoint file.")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
