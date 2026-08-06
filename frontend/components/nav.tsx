"use client";

import { useMotionValueEvent, useScroll } from "motion/react";
import { useState } from "react";
import { EffectsToggle, ThemeToggle } from "./toggles";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "#how-it-thinks", label: "How it thinks" },
  { href: "#dossier", label: "Answer dossier" },
  { href: "#evidence", label: "Evidence" },
];

export function Nav() {
  const { scrollY } = useScroll();
  const [lifted, setLifted] = useState(false);

  // Only the border/shadow change on scroll. Animating blur or background-color
  // here would repaint a full-width backdrop-filter element on every frame.
  useMotionValueEvent(scrollY, "change", (v) => setLifted(v > 24));

  return (
    <header className="sticky top-0 z-50 px-4 pt-4 sm:px-8">
      <nav
        className={cn(
          "mx-auto flex max-w-[1240px] items-center gap-5 rounded-[15px] px-4 py-2.5 transition-shadow duration-300 glass",
          lifted && "shadow-[0_10px_40px_-12px_rgba(0,0,0,0.55)]",
        )}
        aria-label="Primary"
      >
        <a href="#top" className="mr-auto flex items-center gap-2.5">
          <span
            aria-hidden
            className="grid h-[27px] w-[27px] flex-none place-items-center rounded-lg bg-gradient-to-br from-accent to-accent-2 text-[12px] font-extrabold text-accent-fg shadow-[0_0_22px_rgb(var(--accent-dim))]"
          >
            S
          </span>
          <span className="leading-none">
            <span className="block text-[15px] font-semibold tracking-[-0.015em]">Shifa42</span>
            <span className="mt-1 block font-mono text-[9.5px] tracking-[0.15em] text-fg-3">
              CLINICAL COPILOT
            </span>
          </span>
        </a>

        <div className="hidden gap-6 md:flex">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-[13px] font-medium text-fg-2 transition-colors hover:text-fg"
            >
              {l.label}
            </a>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <EffectsToggle />
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
