"use client";

import { motion } from "motion/react";
import { ChevronDown, ExternalLink } from "lucide-react";
import { useState } from "react";
import type { QueryResult } from "@/lib/types";
import { useEffectsEnabled } from "@/lib/use-effects";
import { bandColor, cn, fmt } from "@/lib/utils";
import { GroundednessGauge } from "./groundedness-gauge";

/**
 * The answer dossier — a bento grid of everything the agent used and decided.
 *
 * The organizing idea: a clinician should be able to audit an answer without
 * leaving the page or reading a log file. Sources, the groundedness score and its
 * unsupported claims, the risk model with its own validation numbers, and the
 * graph paths are all first-class, not disclosure-triangle afterthoughts.
 */
export function Dossier({ result }: { result: QueryResult }) {
  const effects = useEffectsEnabled();

  const cards = [
    <GroundednessCard key="g" result={result} />,
    <CitationsCard key="c" result={result} />,
    result.risk ? <RiskCard key="r" result={result} /> : null,
    result.graph_paths.length ? <GraphCard key="gr" result={result} /> : null,
    <DecisionCard key="d" result={result} />,
  ].filter(Boolean);

  return (
    <section id="dossier" className="scroll-mt-24 pt-10">
      <div className="mb-4 flex flex-wrap items-baseline gap-3.5">
        <h2 className="text-h2 font-semibold">Answer dossier</h2>
        <p className="text-[13px] text-fg-3">
          Everything the agent retrieved, scored, and decided — visible, not buried in logs.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3.5 md:grid-cols-12">
        {cards.map((card, i) => (
          <motion.div
            key={i}
            initial={effects ? { opacity: 0, y: 14 } : false}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.45, delay: i * 0.05, ease: [0.16, 1, 0.3, 1] }}
            className={cn(
              "md:col-span-6",
              i === 0 && "md:col-span-4",
              i === 1 && "md:col-span-8",
            )}
          >
            {card}
          </motion.div>
        ))}
      </div>
    </section>
  );
}

function CardShell({
  title,
  children,
  className,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("card h-full px-4.5 py-4", className)}>
      <h3 className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-fg-3">{title}</h3>
      {children}
    </div>
  );
}

function GroundednessCard({ result }: { result: QueryResult }) {
  const r = result.critic_report ?? {};
  const passed = !result.escalated;
  return (
    <CardShell title="Groundedness">
      <div className="mt-3.5 flex items-center gap-4">
        <GroundednessGauge value={result.groundedness} passed={passed} />
        <div>
          <div className="font-mono text-[34px] font-light leading-none tracking-tight">
            {fmt.score(result.groundedness)}
          </div>
          <p className="mt-2 text-[11.5px] leading-snug text-fg-3">
            {r.method === "llm"
              ? "LLM-judged faithfulness of the draft against its retrieved sources"
              : r.method === "lexical"
                ? "Lexical fallback — the LLM judge was unavailable"
                : r.method === "drafter_abstained"
                  ? "The drafter declined to answer from these sources"
                  : "Faithfulness of the draft against retrieved sources"}
          </p>
        </div>
      </div>

      {typeof r.total_claims === "number" && r.total_claims > 0 && (
        <p className="mt-3 font-mono text-[10.5px] text-fg-3">
          {r.supported_claims}/{r.total_claims} claims supported
        </p>
      )}

      <div className="mt-3.5 flex items-center justify-between border-t border-line pt-3 font-mono text-[10.5px] text-fg-3">
        <span>Ship threshold</span>
        <span className={passed ? "text-accent" : "text-danger"}>
          ≥ 0.75 · {passed ? "PASS" : "FAIL"}
        </span>
      </div>

      {!!r.unsupported?.length && (
        <ul className="mt-3 space-y-1.5 border-t border-line pt-3">
          {r.unsupported.slice(0, 3).map((u, i) => (
            <li key={i} className="text-[11.5px] leading-snug text-fg-3">
              <span className="text-warn">unsupported:</span> {u.claim}
            </li>
          ))}
        </ul>
      )}
    </CardShell>
  );
}

function CitationsCard({ result }: { result: QueryResult }) {
  const [open, setOpen] = useState<number | null>(null);

  if (!result.citations.length) {
    return (
      <CardShell title="Sources cited · 0">
        <p className="mt-3 text-[12.5px] leading-relaxed text-fg-3">
          {result.escalated
            ? "No sources are cited on an escalation. Attaching references to a refusal invites it to be read as a partial answer."
            : "No sources were cited by this answer."}
        </p>
      </CardShell>
    );
  }

  return (
    <CardShell title={`Sources cited · ${result.citations.length}`}>
      <ul className="mt-1">
        {result.citations.map((c) => (
          <li key={c.index} id={`source-${c.index}`} className="scroll-mt-28 border-b border-line py-2.5 last:border-0">
            <button
              type="button"
              onClick={() => setOpen(open === c.index ? null : c.index)}
              aria-expanded={open === c.index}
              className="flex w-full items-start gap-2.5 text-left"
            >
              <span className="mt-px grid h-5 w-5 flex-none place-items-center rounded-md border border-line-2 bg-accent/10 font-mono text-[10px] font-semibold text-accent">
                {c.index}
              </span>
              <span className="flex-1">
                <span className="block text-[12.5px] leading-snug text-fg">{c.title}</span>
                <span className="mt-1 block font-mono text-[10px] text-fg-3">
                  {c.source}
                  {c.section ? ` · ${c.section}` : ""} · via {c.retriever}
                </span>
              </span>
              <span className="flex flex-none items-center gap-1.5 self-center font-mono text-[10.5px] text-accent">
                {fmt.score(c.score, 3)}
                <ChevronDown
                  className={cn("h-3 w-3 transition-transform", open === c.index && "rotate-180")}
                  aria-hidden
                />
              </span>
            </button>

            {open === c.index && (
              <div className="mt-2.5 rounded-md border border-line bg-bg-elev px-3 py-2.5">
                <p className="text-[12px] leading-relaxed text-fg-2">{c.snippet}</p>
                {c.url && (
                  <a
                    href={c.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-2 inline-flex items-center gap-1 font-mono text-[10.5px] text-accent hover:underline"
                  >
                    source <ExternalLink className="h-2.5 w-2.5" aria-hidden />
                  </a>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
    </CardShell>
  );
}

function RiskCard({ result }: { result: QueryResult }) {
  const risk = result.risk!;
  const perf = risk.model_performance ?? {};
  return (
    <CardShell title="Risk stratification">
      <div className={cn("mt-3 font-mono text-[40px] font-light leading-none tracking-tight", bandColor(risk.band))}>
        {fmt.score(risk.risk_score)}
        <span className="ml-1.5 text-[13px] text-fg-3">/ 1.0</span>
      </div>
      <p className="mt-2 text-[11.5px] text-fg-3">
        {risk.band} · {risk.outcome_predicted}
      </p>

      <dl className="mt-3.5 grid grid-cols-2 gap-x-4 gap-y-1.5 border-t border-line pt-3 font-mono text-[10.5px]">
        {Object.entries(risk.observed_triage)
          .filter(([k, v]) => v !== null && k !== "chiefcomplaint")
          .slice(0, 6)
          .map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2">
              <dt className="text-fg-3">{k}</dt>
              <dd className="text-fg-2">{String(v)}</dd>
            </div>
          ))}
      </dl>

      <p className="mt-3 border-t border-line pt-3 text-[10.5px] leading-relaxed text-fg-3">
        {risk.model} · AUC {fmt.score(Number(perf.auc_roc))} · calibration slope{" "}
        {fmt.score(Number(perf.calibration_slope))} · {String(perf.n_stays ?? "?")} stays from{" "}
        {String(perf.n_patients ?? "?")} patients. Directional only.
      </p>
    </CardShell>
  );
}

function GraphCard({ result }: { result: QueryResult }) {
  return (
    <CardShell title={`Knowledge-graph paths · ${result.graph_paths.length}`}>
      <ul className="mt-2.5 space-y-2">
        {result.graph_paths.slice(0, 5).map((p, i) => (
          <li key={i} className="font-mono text-[11px] leading-relaxed text-fg-2">
            <span className="text-accent">→</span> {p.describe}
          </li>
        ))}
      </ul>
      <p className="mt-3 border-t border-line pt-3 text-[10.5px] leading-relaxed text-fg-3">
        Edges are extracted by an LLM from guideline text and are noisier than a curated ontology.
        Verify against the cited source before relying on any relationship.
      </p>
    </CardShell>
  );
}

function DecisionCard({ result }: { result: QueryResult }) {
  return (
    <CardShell title="Routing decision">
      <p className="mt-3 font-mono text-[13px] text-accent">{result.route || "—"}</p>
      <p className="mt-1.5 text-[11.5px] leading-relaxed text-fg-3">{result.route_reasoning || "—"}</p>

      <dl className="mt-3.5 grid grid-cols-2 gap-x-4 gap-y-1.5 border-t border-line pt-3 font-mono text-[10.5px]">
        <div className="flex justify-between gap-2">
          <dt className="text-fg-3">confidence</dt>
          <dd className="text-fg-2">{fmt.score(result.route_confidence)}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-fg-3">attempts</dt>
          <dd className="text-fg-2">{result.attempts}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-fg-3">sources</dt>
          <dd className="text-fg-2">{result.sources_reviewed}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-fg-3">outcome</dt>
          <dd className={result.escalated ? "text-danger" : "text-accent"}>{result.outcome}</dd>
        </div>
      </dl>

      {result.entities.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5 border-t border-line pt-3">
          {result.entities.slice(0, 8).map((e) => (
            <span
              key={e}
              className="rounded border border-line px-1.5 py-0.5 font-mono text-[10px] text-fg-3"
            >
              {e}
            </span>
          ))}
        </div>
      )}
    </CardShell>
  );
}
