import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const fmt = {
  score: (n: number | undefined | null, digits = 2) =>
    n === undefined || n === null || Number.isNaN(n) ? "—" : n.toFixed(digits),

  pct: (n: number | undefined | null, digits = 1) =>
    n === undefined || n === null || Number.isNaN(n) ? "—" : `${(n * 100).toFixed(digits)}%`,

  int: (n: number | undefined | null) =>
    n === undefined || n === null || Number.isNaN(n) ? "—" : n.toLocaleString("en-US"),

  ms: (n: number | undefined | null) => {
    if (n === undefined || n === null || Number.isNaN(n)) return "—";
    return n < 1000 ? `${Math.round(n)}ms` : `${(n / 1000).toFixed(2)}s`;
  },

  time: (ts: number | undefined) => {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString("en-GB", { hour12: false }) + `.${String(d.getMilliseconds()).padStart(3, "0")}`;
  },
};

/** Band → token color, shared by the gauge and the risk card. */
export function bandColor(band: string): string {
  switch (band) {
    case "high":
      return "text-danger";
    case "elevated":
      return "text-warn";
    case "moderate":
      return "text-fg-2";
    default:
      return "text-accent";
  }
}
