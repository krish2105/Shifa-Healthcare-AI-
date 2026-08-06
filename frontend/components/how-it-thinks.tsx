"use client";

import { motion, useMotionValueEvent, useScroll } from "motion/react";
import { useRef, useState } from "react";
import { useEffectsEnabled } from "@/lib/use-effects";
import { cn } from "@/lib/utils";

/**
 * The "how the agent thinks" scroll sequence: a section that holds still while
 * scroll advances through six steps.
 *
 * **Implementation note — why not GSAP ScrollTrigger `pin`.**
 * The original plan used ScrollTrigger's `pin`, which is the usual tool for this.
 * It does not compose with Lenis here: Lenis animates `scrollTop` on its own RAF,
 * ScrollTrigger measures the native scroll position, and the pin-spacer ends up
 * sized against a position that no longer matches the screen — the section
 * collapsed to a blank viewport. Wiring `ScrollTrigger.update` to Lenis's scroll
 * event fixes the tracking but not the spacer geometry against this section's
 * flex/min-h-screen container.
 *
 * CSS `position: sticky` does the pinning natively, with no spacer to compute and
 * nothing to keep in sync — it is immune to whoever owns the scroll loop. Motion's
 * `useScroll` then reads real document scroll (which Lenis genuinely drives) to
 * advance the steps. Fewer moving parts, one less dependency in the critical
 * path, and it cannot desynchronize by construction.
 *
 * With effects off it degrades to a plain two-column list: no sticky, no
 * scroll-linked behaviour, identical content.
 */

const STEPS = [
  {
    n: "01",
    node: "intake",
    title: "Normalize and screen",
    body: "The question is rewritten to be self-contained, clinical entities are extracted, and the text is screened for identifier-shaped content before it reaches any retriever, log line, or third-party API.",
  },
  {
    n: "02",
    node: "planner",
    title: "Route adaptively",
    body: "A cheap classifier decides whether this is a simple lookup, a patient-context question, or genuine relationship reasoning — and activates only the retrievers that classification requires. Simple questions never pay for graph traversal.",
  },
  {
    n: "03",
    node: "retrieval",
    title: "Search two ways at once",
    body: "Dense search over bge-large embeddings finds paraphrase; BM25 finds the rare drug names and codes the embedding smoothed away. On relationship queries, a multi-hop graph traversal runs alongside them.",
  },
  {
    n: "04",
    node: "fusion",
    title: "Fuse, dedupe, diversify",
    body: "Results merge by rank via Reciprocal Rank Fusion — BM25 and cosine scores are not comparable, so magnitudes are discarded. MMR then trades a little relevance for coverage, because five paraphrases of one recommendation is not five pieces of evidence.",
  },
  {
    n: "05",
    node: "critic",
    title: "Verify before shipping",
    body: "The draft is decomposed into atomic claims and each is checked against the retrieved sources. Correct-but-uncited counts as unsupported: the check is about traceability, not truth in general. Below 0.75, the answer does not ship.",
  },
  {
    n: "06",
    node: "escalate",
    title: "Refuse, or answer",
    body: "A failed check reformulates the query and retries, at most twice. Still short, and the system returns no answer at all — it escalates to a clinician. There is no path that ships a low-confidence answer with a hedge attached.",
  },
];

export function HowItThinks() {
  const effects = useEffectsEnabled();
  const trackRef = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);

  const { scrollYProgress } = useScroll({
    target: trackRef,
    offset: ["start start", "end end"],
  });

  useMotionValueEvent(scrollYProgress, "change", (p) => {
    // Clamp so the last step stays selected through the tail of the track rather
    // than flicking past it at p === 1.
    const i = Math.min(STEPS.length - 1, Math.floor(p * STEPS.length));
    setActive(Math.max(0, i));
  });

  // NB: one DOM structure for both modes, switched by className.
  //
  // An early `return` for the non-effects branch looks cleaner but silently
  // breaks the scroll tracking: `useEffectsEnabled()` is false on the first
  // render (motion preference is unknowable during SSR), so the early-return
  // branch would render, `useScroll` would bind to a null `target` ref, and it
  // never re-measures when the sticky branch appears a tick later. The steps
  // then freeze on the first card forever.
  return (
    <section id="how-it-thinks" className="scroll-mt-24">
      <div
        ref={trackRef}
        style={effects ? { height: `${STEPS.length * 62}vh` } : undefined}
        className={cn("relative", !effects && "px-5 py-16 sm:px-8")}
      >
        <div
          className={cn(
            effects ? "sticky top-0 flex h-screen items-center px-5 sm:px-8" : "",
          )}
        >
          <div className="mx-auto w-full max-w-[1240px]">
            <Heading />

            {effects && (
              <ol className="mb-6 flex items-center gap-1.5" aria-hidden>
                {STEPS.map((s, i) => (
                  <li
                    key={s.n}
                    className={cn(
                      "h-[3px] flex-1 rounded-full transition-colors duration-300",
                      i <= active ? "bg-accent" : "bg-line-2",
                    )}
                  />
                ))}
              </ol>
            )}

            <div
              className={cn(
                effects
                  ? "relative min-h-[240px] sm:min-h-[210px]"
                  : "grid gap-3.5 md:grid-cols-2",
              )}
            >
              {STEPS.map((s, i) => (
                <motion.div
                  key={s.n}
                  className={cn(effects && "absolute inset-x-0 top-0 mx-auto max-w-3xl")}
                  initial={false}
                  animate={
                    effects
                      ? {
                          opacity: i === active ? 1 : 0,
                          y: i === active ? 0 : i < active ? -20 : 22,
                        }
                      : { opacity: 1, y: 0 }
                  }
                  transition={{ duration: 0.42, ease: [0.16, 1, 0.3, 1] }}
                  style={effects ? { pointerEvents: i === active ? "auto" : "none" } : undefined}
                  aria-hidden={effects && i !== active}
                >
                  <StepCard step={s} />
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Heading() {
  return (
    <div className="mb-7 flex flex-wrap items-baseline gap-3.5">
      <h2 className="text-h2 font-semibold">How the agent thinks</h2>
      <p className="text-[13px] text-fg-3">
        Six nodes, each with its own prompt, retry policy, and audit row.
      </p>
    </div>
  );
}

function StepCard({ step }: { step: (typeof STEPS)[number] }) {
  return (
    <article className="card px-5 py-5">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-[11px] font-semibold text-accent">{step.n}</span>
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-fg-3">
          {step.node}
        </span>
      </div>
      <h3 className="mt-2 text-[17px] font-semibold tracking-[-0.015em]">{step.title}</h3>
      <p className="mt-2 max-w-[62ch] text-pretty text-[13.5px] leading-relaxed text-fg-2">
        {step.body}
      </p>
    </article>
  );
}
