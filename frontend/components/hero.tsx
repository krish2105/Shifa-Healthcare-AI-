"use client";

import { motion } from "motion/react";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { getMetrics } from "@/lib/api";
import type { MetricsSummary } from "@/lib/types";
import { useEffectsEnabled } from "@/lib/use-effects";
import { AnimatedCounter } from "./animated-counter";

/**
 * The WebGL scene loads only in the browser, only after mount, and only when
 * effects are enabled — so the ~600 kB three.js bundle is never fetched for a
 * user who has motion off or is on a small screen.
 */
const HeroScene = dynamic(() => import("./hero-scene"), {
  ssr: false,
  loading: () => null,
});

/** Static fallback: a CSS-only lattice. Intentional-looking, not an empty box. */
function SceneFallback() {
  return (
    <div aria-hidden className="absolute inset-0 grid place-items-center">
      <div className="relative aspect-square w-full max-w-[420px]">
        {[12, 24, 36].map((inset, i) => (
          <div
            key={inset}
            className="absolute rounded-full border border-line-2"
            style={{ inset: `${inset}%`, opacity: 1 - i * 0.25 }}
          />
        ))}
        <div className="absolute left-1/2 top-1/2 h-24 w-24 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle_at_34%_30%,rgb(var(--accent)),rgb(var(--accent-2))_58%,transparent_74%)] shadow-[0_0_68px_rgb(var(--glow))]" />
        <svg viewBox="0 0 400 400" className="absolute inset-0 h-full w-full overflow-visible">
          <g stroke="rgb(var(--line-2))" fill="none" strokeWidth="1">
            <path d="M200 200 L74 116M200 200 L330 148M200 200 L118 312M200 200 L306 306M200 200 L200 56" />
          </g>
          {[
            [74, 116, 5.5],
            [330, 148, 5],
            [118, 312, 4.5],
            [306, 306, 4],
            [200, 56, 4],
          ].map(([cx, cy, r]) => (
            <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r={r} fill="rgb(var(--accent))" opacity={0.85} />
          ))}
        </svg>
      </div>
    </div>
  );
}

const STAT_FALLBACK: MetricsSummary["runs"] = {
  runs_total: 0,
  answered: 0,
  escalated: 0,
  escalation_rate: 0,
  avg_groundedness: 0,
};

export function Hero() {
  const effects = useEffectsEnabled();
  const [mounted, setMounted] = useState(false);
  const [wide, setWide] = useState(false);
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [apiDown, setApiDown] = useState(false);

  useEffect(() => {
    setMounted(true);
    const mq = window.matchMedia("(min-width: 768px)");
    const onChange = () => setWide(mq.matches);
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch(() => setApiDown(true));
  }, []);

  // 3D is gated on all three: mounted (no SSR), effects on, and a viewport wide
  // enough that the scene is worth its cost.
  const show3D = mounted && effects && wide;

  const runs = metrics?.runs ?? STAT_FALLBACK;

  const stats: { value: number; label: string; format: "int" | "score" | "pct"; accentSuffix?: boolean }[] = [
    { value: metrics?.chunks_indexed ?? 0, label: "Guideline chunks indexed", format: "int" },
    { value: runs.avg_groundedness, label: "Avg groundedness", format: "score" },
    { value: runs.escalation_rate, label: "Escalation rate", format: "pct" },
    { value: metrics?.graph?.edges ?? 0, label: "Knowledge-graph edges", format: "int" },
  ];

  return (
    <section id="top" className="relative overflow-hidden">
      <div aria-hidden className="pointer-events-none absolute inset-x-0 -top-40 h-[780px] mesh-bg" />
      <div aria-hidden className="pointer-events-none absolute inset-0 grid-bg opacity-50" />

      <div className="relative z-10 mx-auto grid max-w-[1240px] items-center gap-12 px-5 py-14 sm:px-8 md:grid-cols-[1.08fr_0.92fr] md:py-20">
        <div>
          <motion.span
            initial={effects ? { opacity: 0, y: 8 } : false}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="inline-flex items-center gap-2 rounded-full border border-line-2 bg-accent/10 px-3 py-1.5 font-mono text-[10.5px] uppercase tracking-[0.12em] text-accent"
          >
            <span className="h-[5px] w-[5px] rounded-full bg-accent shadow-[0_0_8px_rgb(var(--accent))]" />
            Agentic clinical RAG
          </motion.span>

          <motion.h1
            initial={effects ? { opacity: 0, y: 14 } : false}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.06, ease: [0.16, 1, 0.3, 1] }}
            className="mt-5 text-display font-semibold"
          >
            Ask.
            <br />
            <span className="text-accent">We&apos;ll show</span>
            <br />
            <span className="font-light text-fg-2">our work.</span>
          </motion.h1>

          <motion.p
            initial={effects ? { opacity: 0, y: 14 } : false}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.12, ease: [0.16, 1, 0.3, 1] }}
            className="mt-5 max-w-[52ch] text-pretty text-[15.5px] leading-relaxed text-fg-2"
          >
            Every answer is traced to a published guideline and scored for groundedness before it
            ships.{" "}
            <strong className="font-medium text-fg">
              When the evidence isn&apos;t there, it escalates to a physician instead of guessing.
            </strong>{" "}
            Refusing to answer is the feature.
          </motion.p>

          <motion.dl
            initial={effects ? { opacity: 0 } : false}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.24 }}
            className="mt-9 flex flex-wrap gap-x-9 gap-y-5"
          >
            {stats.map((s) => (
              <div key={s.label}>
                <dd className="font-mono text-[27px] font-light leading-none tracking-tight">
                  {apiDown ? (
                    <span className="text-fg-3">—</span>
                  ) : (
                    <AnimatedCounter value={s.value} format={s.format} animate={effects} />
                  )}
                </dd>
                <dt className="mt-2 font-mono text-[9.5px] uppercase tracking-[0.13em] text-fg-3">
                  {s.label}
                </dt>
              </div>
            ))}
          </motion.dl>

          {apiDown && (
            <p className="mt-5 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-[12px] text-warn">
              Backend unreachable. Start it with{" "}
              <code className="font-mono">uvicorn app.main:app</code> in{" "}
              <code className="font-mono">/backend</code>.
            </p>
          )}
          {metrics?.degraded && (
            <p className="mt-5 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-[12px] text-warn">
              <strong className="font-medium">Degraded mode.</strong> No LLM credential is
              configured, so answers are deterministic placeholders. Set{" "}
              <code className="font-mono">GROQ_API_KEY</code> in{" "}
              <code className="font-mono">backend/.env</code> for real inference.
            </p>
          )}
        </div>

        <div className="relative order-first aspect-square w-full md:order-none">
          {show3D ? <HeroScene /> : <SceneFallback />}
          <p className="absolute bottom-1 left-1/2 -translate-x-1/2 whitespace-nowrap font-mono text-[9.5px] uppercase tracking-[0.14em] text-fg-3">
            condition ↔ drug ↔ symptom graph
          </p>
        </div>
      </div>
    </section>
  );
}
