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

          <Panel className="md:col-span-12" title="Audit log · every node transition" effects={effects} delay={0.1}>
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
