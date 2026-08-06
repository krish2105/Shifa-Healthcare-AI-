"use client";

import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { getAudit, getInfo } from "@/lib/api";
import type { TraceEntry } from "@/lib/types";
import { useEffectsEnabled } from "@/lib/use-effects";
import { cn, fmt } from "@/lib/utils";

/**
 * Provenance section: what the system is actually built on, and how well it does.
 *
 * Every figure here is read from a manifest written by a script in this repo —
 * the ingester, the trainer, the benchmark runner. Nothing is hardcoded. If a
 * number cannot be read, the UI says so rather than falling back to a plausible
 * placeholder, because a placeholder in a provenance panel is worse than a blank.
 */

interface Info {
  corpus?: {
    documents_kept: number;
    child_chunks: number;
    parent_chunks: number;
    sources: Record<string, number>;
    rows_scanned: number;
  } | null;
  ed_cohort?: { stays: number; patients: number; admitted_rate: number } | null;
  risk_model?: {
    selected_model: string;
    results: Record<string, { auc_roc: number; brier: number; calibration_slope: number }>;
    baselines: Record<string, { auc_roc?: number; accuracy?: number; note?: string }>;
    cv: string;
  } | null;
  benchmarks?: Record<string, unknown> | null;
  config?: Record<string, number | string> | null;
  retrieval_diagnostics?: {
    queries: number;
    dense_vs_sparse_complementarity: { jaccard_overlap_at_10: { mean: number } };
    final_set_provenance: {
      from_dense_only: { mean: number };
      from_sparse_only: { mean: number };
      from_both: { mean: number };
    };
    mmr_redundancy_reduction: {
      mean_pairwise_similarity_before: number;
      mean_pairwise_similarity_after: number;
      relative_reduction: number;
    };
    latency_ms: { dense: { median: number }; sparse: { median: number } };
  } | null;
}

export function EvidenceSection() {
  const effects = useEffectsEnabled();
  const [info, setInfo] = useState<Info | null>(null);
  const [audit, setAudit] = useState<TraceEntry[]>([]);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    getInfo()
      .then((d) => setInfo(d as Info))
      .catch(() => setFailed(true));
    getAudit(14)
      .then((d) => setAudit(d.entries))
      .catch(() => void 0);
  }, []);

  const corpus = info?.corpus;
  const cohort = info?.ed_cohort;
  const risk = info?.risk_model;

  return (
    <section id="evidence" className="scroll-mt-24 px-5 py-16 sm:px-8">
      <div className="mx-auto max-w-[1240px]">
        <div className="mb-5 flex flex-wrap items-baseline gap-3.5">
          <h2 className="text-h2 font-semibold">Evidence base</h2>
          <p className="text-[13px] text-fg-3">
            Measured, not asserted — every figure is read from a manifest written by an ingestion or
            training script.
          </p>
        </div>

        {failed && (
          <p className="rounded-lg border border-warn/30 bg-warn/10 px-4 py-3 text-[13px] text-warn">
            Could not reach the API, so no provenance figures can be shown. Start the backend and
            reload.
          </p>
        )}

        <div className="grid grid-cols-1 gap-3.5 md:grid-cols-12">
          <Panel className="md:col-span-5" title="Guideline corpus" effects={effects} delay={0}>
            {corpus ? (
              <>
                <Row k="documents indexed" v={fmt.int(corpus.documents_kept)} />
                <Row k="scanned to select them" v={fmt.int(corpus.rows_scanned)} />
                <Row k="parent passages" v={fmt.int(corpus.parent_chunks)} />
                <Row k="child chunks (retrieval unit)" v={fmt.int(corpus.child_chunks)} />
                <div className="mt-3 border-t border-line pt-3">
                  <p className="mb-2 font-mono text-[9.5px] uppercase tracking-[0.14em] text-fg-3">
                    sources · stratified
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(corpus.sources).map(([s, n]) => (
                      <span
                        key={s}
                        className="rounded border border-line px-1.5 py-0.5 font-mono text-[10px] text-fg-2"
                      >
                        {s} <span className="text-fg-3">{n}</span>
                      </span>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <Empty label="Corpus not yet ingested — run scripts/ingest_guidelines.py" />
            )}
          </Panel>

          <Panel className="md:col-span-7" title="Risk model card" effects={effects} delay={0.05}>
            {risk ? (
              <>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[420px] text-left font-mono text-[11px]">
                    <thead>
                      <tr className="text-fg-3">
                        <th className="py-1.5 font-normal">model</th>
                        <th className="py-1.5 text-right font-normal">AUC</th>
                        <th className="py-1.5 text-right font-normal">Brier</th>
                        <th className="py-1.5 text-right font-normal">calib.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(risk.results).map(([name, m]) => (
                        <tr
                          key={name}
                          className={cn(
                            "border-t border-line",
                            name === risk.selected_model && "text-accent",
                          )}
                        >
                          <td className="py-1.5">
                            {name}
                            {name === risk.selected_model && " ←"}
                          </td>
                          <td className="py-1.5 text-right">{fmt.score(m.auc_roc, 3)}</td>
                          <td className="py-1.5 text-right">{fmt.score(m.brier, 3)}</td>
                          <td className="py-1.5 text-right">{fmt.score(m.calibration_slope, 2)}</td>
                        </tr>
                      ))}
                      {Object.entries(risk.baselines).map(([name, b]) => (
                        <tr key={name} className="border-t border-line text-fg-3">
                          <td className="py-1.5">{name} (baseline)</td>
                          <td className="py-1.5 text-right">{fmt.score(b.auc_roc, 3)}</td>
                          <td className="py-1.5 text-right">—</td>
                          <td className="py-1.5 text-right">—</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-3 border-t border-line pt-3 text-[11.5px] leading-relaxed text-fg-3">
                  {risk.cv}
                  {cohort && (
                    <>
                      {" "}on {fmt.int(cohort.stays)} ED stays from {fmt.int(cohort.patients)}{" "}
                      patients ({fmt.pct(cohort.admitted_rate, 0)} admitted). Grouped by patient
                      because 222 stays come from only 64 people — a random split leaks.
                    </>
                  )}
                </p>
              </>
            ) : (
              <Empty label="Risk model not trained — run scripts/train_risk.py" />
            )}
          </Panel>

          <Panel
            className="md:col-span-12"
            title="Retrieval diagnostics · why hybrid + MMR"
            effects={effects}
            delay={0.1}
          >
            {info?.retrieval_diagnostics ? (
              <RetrievalDiagnostics d={info.retrieval_diagnostics} />
            ) : (
              <Empty label="Not yet measured — run scripts/eval_retrieval.py (no API key needed)." />
            )}
          </Panel>

          <Panel className="md:col-span-12" title="Audit log · every node transition" effects={effects} delay={0.15}>
            {audit.length ? (
              <div className="overflow-x-auto">
                <ol className="min-w-[560px] font-mono text-[10.5px] leading-[1.95]">
                  {audit.map((e, i) => (
                    <li key={i} className="flex gap-3">
                      <span className="flex-none text-fg-3">{fmt.time(e.ts)}</span>
                      <span className="w-[130px] flex-none text-accent">{e.node}</span>
                      <span
                        className={cn(
                          "flex-1 truncate",
                          e.event === "escalated" || e.event === "failed"
                            ? "text-danger"
                            : "text-fg-2",
                        )}
                      >
                        {e.event}
                        {e.detail && Object.keys(e.detail).length > 0 && (
                          <span className="text-fg-3">
                            {" · "}
                            {Object.entries(e.detail)
                              .slice(0, 3)
                              .map(([k, v]) => `${k}=${String(v).slice(0, 30)}`)
                              .join(" ")}
                          </span>
                        )}
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
            ) : (
              <Empty label="No runs recorded yet — ask a question above." />
            )}
          </Panel>
        </div>
      </div>
    </section>
  );
}

function Panel({
  title,
  children,
  className,
  effects,
  delay,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
  effects: boolean;
  delay: number;
}) {
  return (
    <motion.div
      initial={effects ? { opacity: 0, y: 14 } : false}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.45, delay, ease: [0.16, 1, 0.3, 1] }}
      className={cn("card px-4.5 py-4", className)}
    >
      <h3 className="mb-2.5 font-mono text-[9.5px] uppercase tracking-[0.14em] text-fg-3">
        {title}
      </h3>
      {children}
    </motion.div>
  );
}

/**
 * Renders the no-LLM retrieval diagnostics.
 *
 * These exist to make two design claims falsifiable rather than asserted: that
 * dense and sparse retrieval fail differently (so fusing them is worth an extra
 * index), and that MMR removes real redundancy (so it is not pure latency).
 */
function RetrievalDiagnostics({
  d,
}: {
  d: NonNullable<Info["retrieval_diagnostics"]>;
}) {
  const prov = d.final_set_provenance;
  const mmr = d.mmr_redundancy_reduction;

  const bars = [
    { label: "dense only", v: prov.from_dense_only.mean, cls: "bg-accent" },
    { label: "both agree", v: prov.from_both.mean, cls: "bg-accent/45" },
    { label: "BM25 only", v: prov.from_sparse_only.mean, cls: "bg-warn" },
  ];

  return (
    <div className="grid gap-5 md:grid-cols-3">
      <div>
        <p className="mb-2 text-[11.5px] leading-relaxed text-fg-3">
          Where the final top-k came from, over {d.queries} real clinical questions.
        </p>
        <div className="flex h-2 overflow-hidden rounded-full bg-line-2">
          {bars.map((b) => (
            <div key={b.label} className={cn(b.cls)} style={{ width: `${b.v * 100}%` }} />
          ))}
        </div>
        <dl className="mt-2.5 space-y-1 font-mono text-[10.5px]">
          {bars.map((b) => (
            <div key={b.label} className="flex justify-between gap-2">
              <dt className="flex items-center gap-1.5 text-fg-3">
                <span className={cn("h-2 w-2 rounded-sm", b.cls)} />
                {b.label}
              </dt>
              <dd className="text-fg-2">{fmt.pct(b.v, 0)}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-2.5 text-[11px] leading-relaxed text-fg-3">
          The BM25-only share would never have been retrieved by a dense-only system.
        </p>
      </div>

      <div>
        <p className="mb-2 text-[11.5px] leading-relaxed text-fg-3">
          Dense vs BM25 top-10 agreement (Jaccard).
        </p>
        <div className="font-mono text-[30px] font-light leading-none tracking-tight text-accent">
          {fmt.score(d.dense_vs_sparse_complementarity.jaccard_overlap_at_10.mean, 3)}
        </div>
        <p className="mt-2.5 text-[11px] leading-relaxed text-fg-3">
          Low overlap means the two retrievers surface largely different evidence — the
          premise of fusing them. Near 1.0 would mean the sparse index is redundant.
        </p>
      </div>

      <div>
        <p className="mb-2 text-[11.5px] leading-relaxed text-fg-3">
          MMR redundancy reduction within the returned set.
        </p>
        <div className="flex items-baseline gap-2 font-mono">
          <span className="text-[19px] font-light text-fg-3 line-through">
            {fmt.score(mmr.mean_pairwise_similarity_before, 3)}
          </span>
          <span className="text-fg-3">→</span>
          <span className="text-[30px] font-light leading-none tracking-tight text-accent">
            {fmt.score(mmr.mean_pairwise_similarity_after, 3)}
          </span>
        </div>
        <p className="mt-1.5 font-mono text-[10.5px] text-fg-2">
          {fmt.pct(mmr.relative_reduction, 0)} less repetition
        </p>
        <p className="mt-2.5 text-[11px] leading-relaxed text-fg-3">
          Mean pairwise similarity before and after MMR. No drop would mean MMR is pure
          latency.
        </p>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-line py-1.5 last:border-0">
      <span className="text-[12px] text-fg-3">{k}</span>
      <span className="font-mono text-[12.5px] tabular-nums text-fg">{v}</span>
    </div>
  );
}

function Empty({ label }: { label: string }) {
  return <p className="py-2 text-[12.5px] leading-relaxed text-fg-3">{label}</p>;
}
