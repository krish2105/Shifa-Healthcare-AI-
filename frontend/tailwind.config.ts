import type { Config } from "tailwindcss";

/**
 * Tokens live as CSS variables holding raw RGB triplets, surfaced here through
 * Tailwind's `<alpha-value>` mechanism. One source of truth in globals.css, while
 * still allowing opacity modifiers like `bg-accent/10` everywhere.
 */
const rgb = (v: string) => `rgb(var(${v}) / <alpha-value>)`;

const config: Config = {
  darkMode: ["class", '[data-theme="dark"]'],
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: rgb("--bg"),
        "bg-elev": rgb("--bg-elev"),
        fg: rgb("--fg"),
        "fg-2": rgb("--fg-2"),
        "fg-3": rgb("--fg-3"),
        accent: rgb("--accent"),
        "accent-2": rgb("--accent-2"),
        "accent-fg": rgb("--accent-fg"),
        warn: rgb("--warn"),
        danger: rgb("--danger"),
        ok: rgb("--ok"),
        line: "rgb(var(--line))",
        "line-2": "rgb(var(--line-2))",
      },
      backgroundColor: {
        surface: "rgb(var(--surface))",
        "surface-2": "rgb(var(--surface-2))",
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "ui-monospace", "monospace"],
      },
      fontSize: {
        // Fluid type: interpolates smoothly rather than jumping at breakpoints.
        display: [
          "clamp(2.4rem, 5.6vw, 4.25rem)",
          { lineHeight: "0.98", letterSpacing: "-0.035em" },
        ],
        h2: [
          "clamp(1.35rem, 2.4vw, 1.9rem)",
          { lineHeight: "1.15", letterSpacing: "-0.02em" },
        ],
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 4px)",
        sm: "calc(var(--radius) - 8px)",
      },
      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(12px)" },
          to: { opacity: "1", transform: "none" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "0.9", transform: "scale(1)" },
          "50%": { opacity: "1", transform: "scale(1.06)" },
        },
        blink: { "50%": { opacity: "0.32" } },
        flow: { to: { strokeDashoffset: "-40" } },
      },
      animation: {
        "fade-up": "fade-up 0.5s cubic-bezier(0.16,1,0.3,1) both",
        "pulse-soft": "pulse-soft 4.5s ease-in-out infinite",
        blink: "blink 1.15s ease-in-out infinite",
        flow: "flow 2.4s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
