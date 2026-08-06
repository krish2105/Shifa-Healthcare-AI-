"use client";

import type Lenis from "lenis";

/**
 * Module-level handle on the live Lenis instance.
 *
 * GSAP ScrollTrigger and Lenis both want to own the scroll loop. Left
 * un-integrated, Lenis intercepts the wheel and animates `scrollTop` on its own
 * RAF, while ScrollTrigger keeps reading the native scroll position on a
 * different tick — so pinned sections compute their spacer against a position
 * that no longer matches what the user sees, and the layout collapses.
 *
 * The fix is to make one drive the other: Lenis emits scroll, ScrollTrigger
 * updates from it, and GSAP's ticker drives Lenis's RAF so both run on the same
 * clock. That wiring needs a reference to the instance from a component that did
 * not create it, which is what this tiny registry provides — a full React context
 * would be more ceremony for a single mutable value that is not render-relevant.
 */

let instance: Lenis | null = null;
const listeners = new Set<(l: Lenis | null) => void>();

export function setLenis(l: Lenis | null) {
  instance = l;
  listeners.forEach((fn) => fn(l));
}

export function getLenis(): Lenis | null {
  return instance;
}

/** Subscribe to instance changes. Fires immediately with the current value. */
export function onLenisChange(fn: (l: Lenis | null) => void): () => void {
  listeners.add(fn);
  fn(instance);
  return () => {
    listeners.delete(fn);
  };
}
