"use client";

import { motion } from "motion/react";
import { AlertTriangle, ShieldCheck, UserRound } from "lucide-react";
import type { QueryResult } from "@/lib/types";
import { useEffectsEnabled } from "@/lib/use-effects";
import { cn, fmt } from "@/lib/utils";

/**
 * The answer, or the refusal.
 *
 * Escalations get a visually distinct treatment rather than being rendered as
 * "an answer that happens to say no". Making a refusal look like a normal
 * response is how a refusal gets skimmed and misread as guidance — the whole
 * safety argument collapses at the last rendering step.
 *
 * Citation markers in the prose are turned into anchor links down to the source
 * list, so a claim can be checked in one click instead of by eye.
 */
export function AnswerPanel({ result }: { result: QueryResult }) {
  const effects = useEffectsEnabled();
  const escalated = result.escalated;

  return (
    <motion.article
      initial={effects ? { opacity: 0, y: 12 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      aria-live="polite"
      className={cn(
        "mt-4 rounded-lg border px-5 py-4.5",
        escalated ? "border-danger/35 bg-danger/[0.06]" : "border-line bg-surface",
      )}
    >
      <header className="mb-3 flex flex-wrap items-center gap-2.5">
        {escalated ? (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-danger/40 bg-danger/10 px-2.5 py-1 text-[11px] font-semibold text-danger">
            <AlertTriangle className="h-3 w-3" aria-hidden />
            Escalated to physician
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/40 bg-accent/10 px-2.5 py-1 text-[11px] font-semibold text-accent">
            <ShieldCheck className="h-3 w-3" aria-hidden />
            Groundedness {fmt.score(result.groundedness)} · passed
          </span>
        )}

        <span className="font-mono text-[10.5px] text-fg-3">
          route: {result.route || "—"} · {result.attempts} attempt
          {result.attempts === 1 ? "" : "s"} · {result.sources_reviewed} sources
        </span>

        {result.risk && (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-line-2 px-2.5 py-1 font-mono text-[10.5px] text-fg-2">
            <UserRound className="h-3 w-3" aria-hidden />
            risk {fmt.score(result.risk.risk_score)} ({result.risk.band})
          </span>
        )}

        {result.degraded && (
          <span className="rounded-full border border-warn/40 bg-warn/10 px-2.5 py-1 text-[10.5px] font-medium text-warn">
            degraded — placeholder output
          </span>
        )}
        {result.contains_identifiers && (
          <span className="rounded-full border border-warn/40 bg-warn/10 px-2.5 py-1 text-[10.5px] font-medium text-warn">
            identifier-shaped text detected in query
          </span>
        )}
      </header>

      <div className="prose-clinical text-[14.5px] leading-relaxed text-fg">
        {renderWithCitations(result.answer)}
      </div>
    </motion.article>
  );
}

/**
 * Render markdown-lite prose with [n] citation markers as jump links.
 * Deliberately not a full markdown pipeline — the composer emits bold, bullets
 * and citations, and a whole parser for that surface is unnecessary weight.
 */
function renderWithCitations(text: string) {
  const lines = text.split("\n");

  return lines.map((line, li) => {
    if (!line.trim()) return <div key={li} className="h-2.5" />;

    const bullet = /^\s*[-*•]\s+/.test(line);
    const content = bullet ? line.replace(/^\s*[-*•]\s+/, "") : line;

    // Split on bold and citation tokens in one pass.
    const parts = content.split(/(\*\*[^*]+\*\*|\[\d{1,2}\])/g).filter(Boolean);

    const rendered = parts.map((part, pi) => {
      if (/^\*\*[^*]+\*\*$/.test(part)) {
        return (
          <strong key={pi} className="font-semibold">
            {part.slice(2, -2)}
          </strong>
        );
      }
      const m = part.match(/^\[(\d{1,2})\]$/);
      if (m) {
        return (
          <a
            key={pi}
            href={`#source-${m[1]}`}
            className="mx-0.5 inline-flex h-[17px] min-w-[17px] items-center justify-center rounded border border-accent/40 bg-accent/10 px-1 align-baseline font-mono text-[10px] font-semibold text-accent transition-colors hover:bg-accent/20"
            aria-label={`Jump to source ${m[1]}`}
          >
            {m[1]}
          </a>
        );
      }
      return <span key={pi}>{part}</span>;
    });

    return bullet ? (
      <div key={li} className="flex gap-2.5 py-0.5">
        <span aria-hidden className="mt-[7px] h-1 w-1 flex-none rounded-full bg-accent" />
        <p className="flex-1">{rendered}</p>
      </div>
    ) : (
      <p key={li} className="py-0.5">
        {rendered}
      </p>
    );
  });
}
