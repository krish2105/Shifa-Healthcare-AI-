"use client";

import { animate, useInView } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { fmt } from "@/lib/utils";

/**
 * Count-up for a real metric.
 *
 * Two things it does that a naive version does not:
 *
 * 1. It writes to state at ~60fps but renders formatted text, so the digit count
 *    never changes mid-animation — otherwise the surrounding layout shifts on
 *    every frame as "9" becomes "10" becomes "100".
 * 2. It respects `animate={false}` completely — no RAF loop is started at all,
 *    rather than being started and visually suppressed.
 */
export function AnimatedCounter({
  value,
  format = "int",
  animate: shouldAnimate = true,
  duration = 1.1,
}: {
  value: number;
  format?: "int" | "score" | "pct";
  animate?: boolean;
  duration?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  const [display, setDisplay] = useState(shouldAnimate ? 0 : value);

  useEffect(() => {
    if (!shouldAnimate) {
      setDisplay(value);
      return;
    }
    if (!inView) return;

    const controls = animate(0, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => setDisplay(v),
    });
    return () => controls.stop();
  }, [value, inView, shouldAnimate, duration]);

  const text =
    format === "int" ? fmt.int(Math.round(display))
    : format === "pct" ? fmt.pct(display, 1)
    : fmt.score(display, 2);

  return (
    <span ref={ref} className="tabular-nums">
      {text}
    </span>
  );
}
