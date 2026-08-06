"use client";

import { AnimatePresence, motion } from "motion/react";
import type { TraceEntry } from "@/lib/types";
import { useEffectsEnabled } from "@/lib/use-effects";
import { cn, fmt } from "@/lib/utils";

/**
 * The live agent trace.
 *
 * This is the component the whole project is arguing for: a user watching the
 * agent work, one real node at a time. Every row here corresponds to an SSE frame
 * emitted when a LangGraph node completed — nothing is simulated, and the same
 * events are what the audit log persists.
 *
 * Failure states are coloured distinctly rather than being folded into a generic
 * "step done" style, because "critic failed" and "critic passed" are the two
 * outcomes a reader most needs to tell apart at a glance.
 */

function toneFor(entry: TraceEntry): string {
  const e = entry.event;
  if (e === "escalated" || e === "error_failed_closed" || e === "failed")
    return "border-danger/40 bg-danger/10 text-danger";
  if (e === "drafter_abstained" || e === "rewritten" || e === "empty" || e === "no_context")
    return "border-warn/35 bg-warn/10 text-warn";
  if (e === "skipped")
    return "border-line bg-transparent text-fg-3";
  if (e === "passed")
    return "border-accent/40 bg-accent/10 text-accent";
  return "border-line-2 bg-surface-2 text-fg-2";
}

/** One-line summary of a node's payload — the number that matters, not the blob. */
function summarize(entry: TraceEntry): string | null {
  const d = entry.detail ?? {};
  switch (entry.node) {
    case "planner":
      return d.route ? `${d.route} · conf ${fmt.score(Number(d.confidence), 2)}` : null;
    case "vector_retriever":
      return `${d.dense ?? 0} dense · ${d.sparse ?? 0} sparse`;
    case "graph_retriever":
      return entry.event === "skipped"
        ? String(d.reason ?? "").slice(0, 60)
        : `${d.paths ?? 0} paths · ${d.chunks ?? 0} chunks`;
    case "fusion":
      return entry.event === "empty" ? "no results" : `${d.fused ?? 0} → ${d.selected ?? 0} chunks`;
    case "draft":
      return Array.isArray(d.citations_used)
        ? `${(d.citations_used as unknown[]).length} citations`
        : null;
    case "critic":
      return d.faithfulness !== undefined
        ? `faithfulness ${fmt.score(Number(d.faithfulness), 2)} / ${fmt.score(Number(d.threshold), 2)}`
        : String(d.reason ?? "").slice(0, 60);
    case "reformulate":
      return String(d.strategy ?? "").slice(0, 70);
    case "risk_node":
      return d.score !== undefined && d.score !== null ? `risk ${fmt.score(Number(d.score), 2)}` : "no patient";
    case "final_composer":
      return `${d.citations_bound ?? 0} citations bound`;
    case "escalate":
      return `best ${fmt.score(Number(d.best_groundedness), 2)} < ${fmt.score(Number(d.threshold), 2)}`;
    default:
      return null;
  }
}

export function TraceStrip({
  entries,
  running,
  totalMs,
}: {
  entries: TraceEntry[];
  running: boolean;
  totalMs: number;
}) {
  const effects = useEffectsEnabled();
  if (!entries.length && !running) return null;

  return (
    <div className="card mt-4 px-4 py-3.5">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-fg-3">
          Live trace
        </span>
        {running && (
          <span className="inline-flex items-center gap-1.5 font-mono text-[10px] text-accent">
            <span className="h-[5px] w-[5px] animate-blink rounded-full bg-accent" />
            running
          </span>
        )}
        <span className="ml-auto font-mono text-[10.5px] text-fg-3">
          {fmt.ms(totalMs)}
          {entries.length > 0 && ` · ${entries.length} nodes`}
        </span>
      </div>

      <ol className="mt-3 flex flex-wrap items-center gap-1.5">
        <AnimatePresence initial={false}>
          {entries.map((e, i) => {
            const detail = summarize(e);
            return (
              <motion.li
                key={`${e.node}-${e.seq ?? i}`}
                initial={effects ? { opacity: 0, scale: 0.94, y: 4 } : false}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                className={cn(
                  "inline-flex items-center gap-2 rounded-lg border px-2.5 py-1.5 text-[11.5px] font-medium",
                  toneFor(e),
                )}
                title={detail ? `${e.label ?? e.node} — ${detail}` : e.label ?? e.node}
              >
                <span className="h-[5px] w-[5px] flex-none rounded-full bg-current" />
                <span>{e.label ?? e.node}</span>
                {detail && (
                  <span className="hidden font-mono text-[10px] opacity-70 sm:inline">{detail}</span>
                )}
                <span className="font-mono text-[9.5px] opacity-50">{fmt.ms(e.duration_ms)}</span>
              </motion.li>
            );
          })}
        </AnimatePresence>

        {running && (
          <motion.li
            initial={effects ? { opacity: 0 } : false}
            animate={{ opacity: 1 }}
            className="inline-flex items-center gap-2 rounded-lg border border-line px-2.5 py-1.5 text-[11.5px] text-fg-3"
          >
            <span className="h-[5px] w-[5px] animate-blink rounded-full bg-fg-3" />
            working…
          </motion.li>
        )}
      </ol>
    </div>
  );
}
