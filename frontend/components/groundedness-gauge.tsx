"use client";

import { motion } from "motion/react";
import { useEffectsEnabled } from "@/lib/use-effects";

/**
 * Radial gauge for the groundedness score.
 *
 * The threshold is drawn as a tick on the arc, not merely stated in text beside
 * it. A score of 0.71 means nothing on its own; a score sitting just short of a
 * visible line means "this failed, and barely" at a glance.
 */
export function GroundednessGauge({
  value,
  passed,
  threshold = 0.75,
  size = 86,
}: {
  value: number;
  passed: boolean;
  threshold?: number;
  size?: number;
}) {
  const effects = useEffectsEnabled();
  const stroke = 6;
  const r = (size - stroke) / 2 - 3;
  const c = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(1, value || 0));

  // Threshold tick position on the same rotated coordinate system as the arc.
  const tickAngle = -90 + threshold * 360;
  const tickRad = (tickAngle * Math.PI) / 180;
  const cx = size / 2;
  const cy = size / 2;

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      className="flex-none"
      role="img"
      aria-label={`Groundedness ${value.toFixed(2)} of 1.0, threshold ${threshold}, ${passed ? "passed" : "failed"}`}
    >
      <circle cx={cx} cy={cy} r={r} fill="none" strokeWidth={stroke} stroke="rgb(var(--line-2))" />

      <motion.circle
        cx={cx}
        cy={cy}
        r={r}
        fill="none"
        strokeWidth={stroke}
        strokeLinecap="round"
        stroke={passed ? "rgb(var(--accent))" : "rgb(var(--danger))"}
        strokeDasharray={c}
        initial={effects ? { strokeDashoffset: c } : false}
        animate={{ strokeDashoffset: c * (1 - clamped) }}
        transition={{ duration: effects ? 1.1 : 0, ease: [0.16, 1, 0.3, 1] }}
        transform={`rotate(-90 ${cx} ${cy})`}
        style={{ filter: `drop-shadow(0 0 7px rgb(var(--${passed ? "glow" : "danger"})))` }}
      />

      <line
        x1={cx + Math.cos(tickRad) * (r - stroke / 2 - 2)}
        y1={cy + Math.sin(tickRad) * (r - stroke / 2 - 2)}
        x2={cx + Math.cos(tickRad) * (r + stroke / 2 + 2)}
        y2={cy + Math.sin(tickRad) * (r + stroke / 2 + 2)}
        stroke="rgb(var(--fg-2))"
        strokeWidth={1.5}
      />
    </svg>
  );
}
