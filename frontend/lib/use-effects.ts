"use client";

import { useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";
import { useEffectsStore } from "./store";

/**
 * The single source of truth for "should this animate?".
 *
 * Resolution order: an explicit user choice wins; otherwise the OS
 * `prefers-reduced-motion` setting decides. Components never read the store
 * directly, so there is exactly one place where that precedence is defined.
 *
 * Returns `false` until mounted. Motion state cannot be known during SSR, and
 * guessing produces a hydration mismatch — starting from "no motion" also means
 * the first paint is the static layout, which is the correct fallback anyway.
 */
export function useEffectsEnabled(): boolean {
  const systemReduced = useReducedMotion();
  const override = useEffectsStore((s) => s.effects);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);
  if (!mounted) return false;
  return override ?? !systemReduced;
}

export function useSystemReduced(): boolean {
  return !!useReducedMotion();
}
