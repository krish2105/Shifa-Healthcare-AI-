"use client";

import Lenis from "lenis";
import { ThemeProvider } from "next-themes";
import { useEffect, type ReactNode } from "react";
import { setLenis } from "@/lib/lenis";
import { useEffectsEnabled } from "@/lib/use-effects";

/**
 * Lenis smooth scroll, gated on the effects toggle.
 *
 * Smooth scroll is the one "premium feel" effect that actively harms users who
 * asked for reduced motion — it takes over the scroll they explicitly control. So
 * it is created only when effects are on, and fully destroyed (not paused) when
 * they are switched off, restoring native scrolling with no residual RAF loop.
 *
 * The instance is published to `lib/lenis` so ScrollTrigger can synchronize with
 * it; see that module for why the two must share a clock.
 */
function SmoothScroll({ children }: { children: ReactNode }) {
  const enabled = useEffectsEnabled();

  useEffect(() => {
    if (!enabled) {
      setLenis(null);
      return;
    }

    const lenis = new Lenis({
      duration: 1.05,
      easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
      touchMultiplier: 1.6,
    });
    setLenis(lenis);

    let frame = 0;
    const raf = (time: number) => {
      lenis.raf(time);
      frame = requestAnimationFrame(raf);
    };
    frame = requestAnimationFrame(raf);

    return () => {
      cancelAnimationFrame(frame);
      lenis.destroy();
      setLenis(null);
    };
  }, [enabled]);

  return <>{children}</>;
}

export function Providers({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider
      attribute="data-theme"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange
    >
      <SmoothScroll>{children}</SmoothScroll>
    </ThemeProvider>
  );
}
