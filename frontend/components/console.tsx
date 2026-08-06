"use client";

import { AnimatePresence, motion } from "motion/react";
import { ArrowRight, Loader2, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { getPatients, streamQuery } from "@/lib/api";
import { useSessionStore } from "@/lib/store";
import type { QueryResult, TraceEntry } from "@/lib/types";
import { useEffectsEnabled } from "@/lib/use-effects";
import { cn } from "@/lib/utils";
import { AnswerPanel } from "./answer-panel";
import { Dossier } from "./dossier";
import { TraceStrip } from "./trace-strip";

const EXAMPLES = [
  "First-line antibiotic for community-acquired pneumonia in adults",
  "Initial resuscitation targets in suspected sepsis",
  "When should anticoagulation be started in atrial fibrillation?",
  "Which analgesic is safer in a patient with stage 4 chronic kidney disease?",
];

export function Console() {
  const effects = useEffectsEnabled();
  const [value, setValue] = useState("");
  const [running, setRunning] = useState(false);
  const [trace, setTrace] = useState<TraceEntry[]>([]);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [patients, setPatients] = useState<{ stay_id: string; chiefcomplaint: string }[]>([]);

  const patientId = useSessionStore((s) => s.patientId);
  const setPatientId = useSessionStore((s) => s.setPatientId);
  const setLastRunId = useSessionStore((s) => s.setLastRunId);

  const abortRef = useRef<(() => void) | null>(null);
  const startedAt = useRef<number>(0);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    getPatients()
      .then((d) => setPatients(d.patients.slice(0, 8)))
      .catch(() => setPatients([]));
  }, []);

  // Abort any in-flight stream when the component unmounts, so a navigation
  // away does not leave a reader pinned on a half-consumed response body.
  useEffect(() => () => abortRef.current?.(), []);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setElapsed(Date.now() - startedAt.current), 100);
    return () => window.clearInterval(id);
  }, [running]);

  const submit = useCallback(
    (q: string) => {
      const question = q.trim();
      if (!question || running) return;

      abortRef.current?.();
      setRunning(true);
      setTrace([]);
      setResult(null);
      setError(null);
      startedAt.current = Date.now();
      setElapsed(0);

      abortRef.current = streamQuery(
        question,
        {
          onStart: (d) => setLastRunId(d.run_id),
          onTrace: (t) => setTrace((prev) => [...prev, t]),
          onDone: (r) => {
            setResult(r);
            setRunning(false);
          },
          onError: (msg) => {
            setError(msg);
            setRunning(false);
          },
        },
        { patientId },
      );
    },
    [running, patientId, setLastRunId],
  );

  return (
    <section id="query" className="relative z-10 mx-auto max-w-[1240px] px-5 sm:px-8">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(value);
        }}
      >
        <label htmlFor="clinical-question" className="sr-only">
          Clinical question
        </label>
        <div className="flex items-center gap-2.5 rounded-[14px] border border-line-2 bg-surface-2 py-1.5 pl-4 pr-1.5 shadow-[0_0_0_4px_rgb(var(--accent-dim)),0_18px_46px_rgba(0,0,0,0.35)] transition-shadow focus-within:shadow-[0_0_0_5px_rgb(var(--accent)/0.18),0_18px_46px_rgba(0,0,0,0.45)]">
          <span aria-hidden className="font-mono text-[13px] font-semibold text-accent">
            ›
          </span>
          <input
            id="clinical-question"
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="Ask a clinical question…"
            autoComplete="off"
            disabled={running}
            className="flex-1 bg-transparent py-2.5 text-[14.5px] text-fg outline-none placeholder:text-fg-3 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={running || !value.trim()}
            className="inline-flex items-center gap-1.5 rounded-[10px] bg-accent px-4 py-2.5 text-[12.5px] font-semibold text-accent-fg transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {running ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                Tracing
              </>
            ) : (
              <>
                Trace answer
                <ArrowRight className="h-3.5 w-3.5" aria-hidden />
              </>
            )}
          </button>
        </div>
      </form>

      {/* Examples + optional patient context */}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-fg-3">Try</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            disabled={running}
            onClick={() => {
              setValue(ex);
              submit(ex);
            }}
            className="rounded-full border border-line px-2.5 py-1 text-[11.5px] text-fg-3 transition-colors hover:border-strong hover:text-fg-2 disabled:opacity-50"
          >
            {ex.length > 52 ? `${ex.slice(0, 52)}…` : ex}
          </button>
        ))}
      </div>

      {patients.length > 0 && (
        <div className="mt-2.5 flex flex-wrap items-center gap-2">
          <span className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-fg-3">
            Patient context
          </span>
          <button
            type="button"
            onClick={() => setPatientId(null)}
            className={cn(
              "rounded-full border px-2.5 py-1 text-[11.5px] transition-colors",
              !patientId
                ? "border-accent/50 bg-accent/10 text-accent"
                : "border-line text-fg-3 hover:text-fg-2",
            )}
          >
            none
          </button>
          {patients.slice(0, 5).map((p) => (
            <button
              key={p.stay_id}
              type="button"
              onClick={() => setPatientId(patientId === p.stay_id ? null : p.stay_id)}
              title={p.chiefcomplaint || "ED encounter"}
              className={cn(
                "rounded-full border px-2.5 py-1 font-mono text-[11px] transition-colors",
                patientId === p.stay_id
                  ? "border-accent/50 bg-accent/10 text-accent"
                  : "border-line text-fg-3 hover:text-fg-2",
              )}
            >
              {p.stay_id}
            </button>
          ))}
          <span className="text-[10.5px] text-fg-3">
            MIMIC-IV-ED Demo · de-identified
          </span>
        </div>
      )}

      <TraceStrip
        entries={trace}
        running={running}
        totalMs={result ? trace.reduce((a, t) => a + (t.duration_ms || 0), 0) : elapsed}
      />

      <AnimatePresence mode="wait">
        {error && (
          <motion.div
            key="err"
            initial={effects ? { opacity: 0, y: 8 } : false}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            role="alert"
            className="mt-4 flex items-start gap-3 rounded-lg border border-danger/35 bg-danger/10 px-4 py-3 text-[13px] text-danger"
          >
            <span className="flex-1">{error}</span>
            <button
              type="button"
              onClick={() => submit(value)}
              className="inline-flex items-center gap-1.5 rounded-md border border-danger/40 px-2 py-1 text-[11.5px] hover:bg-danger/10"
            >
              <RotateCcw className="h-3 w-3" aria-hidden />
              Retry
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {result && (
        <>
          <AnswerPanel result={result} />
          <Dossier result={result} />
        </>
      )}
    </section>
  );
}
