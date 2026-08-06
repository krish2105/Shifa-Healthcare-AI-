"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { useEffectsStore } from "@/lib/store";
import { useEffectsEnabled, useSystemReduced } from "@/lib/use-effects";
import { cn } from "@/lib/utils";

/**
 * "Motion & 3D" control — separate from the theme toggle on purpose.
 *
 * These are different needs. Theme is preference; motion is often accessibility
 * or a low-powered device, and bundling them would force a user who needs
 * stillness to also accept a light interface. Turning this off disables the WebGL
 * scene and every large transform while leaving the app fully usable.
 */
export function EffectsToggle() {
  const enabled = useEffectsEnabled();
  const systemReduced = useSystemReduced();
  const toggle = useEffectsStore((s) => s.toggleEffects);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <button
      type="button"
      onClick={() => toggle(systemReduced)}
      aria-pressed={enabled}
      title={
        systemReduced && mounted
          ? "Your system requests reduced motion. Shifa42 follows that by default."
          : "Toggle animations and the 3D hero scene"
      }
      className="inline-flex items-center gap-2 rounded-full border border-line-2 px-2.5 py-1.5 text-[11px] font-medium text-fg-2 transition-colors hover:bg-surface-2 hover:text-fg"
    >
      <span className="hidden sm:inline">Motion &amp; 3D</span>
      <span className="sm:hidden">
        <Monitor className="h-3.5 w-3.5" aria-hidden />
      </span>
      <span
        aria-hidden
        className={cn(
          "relative h-[15px] w-[26px] rounded-full transition-colors",
          enabled ? "bg-accent" : "bg-fg-3/50",
        )}
      >
        <motion.span
          className={cn(
            "absolute top-[2px] h-[11px] w-[11px] rounded-full",
            enabled ? "bg-accent-fg" : "bg-bg",
          )}
          animate={{ left: enabled ? 13 : 2 }}
          transition={
            enabled ? { type: "spring", stiffness: 500, damping: 32 } : { duration: 0 }
          }
        />
      </span>
      <span className="sr-only">{enabled ? "Disable" : "Enable"} motion and 3D effects</span>
    </button>
  );
}

/** Sun/moon morph. Renders a stable placeholder pre-mount to avoid a theme flash. */
export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const enabled = useEffectsEnabled();
  useEffect(() => setMounted(true), []);

  const isDark = resolvedTheme !== "light";

  if (!mounted) {
    return <div className="h-[31px] w-[31px] rounded-md border border-line-2" aria-hidden />;
  }

  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      aria-label={`Switch to ${isDark ? "light" : "dark"} theme`}
      className="grid h-[31px] w-[31px] place-items-center overflow-hidden rounded-md border border-line-2 text-fg-2 transition-colors hover:bg-surface-2 hover:text-fg"
    >
      <motion.span
        key={isDark ? "moon" : "sun"}
        initial={enabled ? { y: 12, opacity: 0, rotate: -35 } : false}
        animate={{ y: 0, opacity: 1, rotate: 0 }}
        transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
        className="grid place-items-center"
      >
        {isDark ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
      </motion.span>
    </button>
  );
}
