import type { Metadata, Viewport } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Providers } from "@/components/providers";
import { Disclaimer } from "@/components/disclaimer";
import "./globals.css";

export const metadata: Metadata = {
  title: "Shifa42 — Agentic Clinical Intelligence Copilot",
  description:
    "Clinical decision support grounded in published guideline literature, with an automated groundedness check before any answer ships and a physician-escalation path when the evidence is insufficient. Research demonstration — not a medical device.",
  applicationName: "Shifa42",
  // A demo that answers clinical questions has no business in a search index.
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0a0b0d" },
    { media: "(prefers-color-scheme: light)", color: "#fafbfc" },
  ],
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      data-theme="dark"
      suppressHydrationWarning
      className={`${GeistSans.variable} ${GeistMono.variable}`}
    >
      <body className="min-h-screen font-sans antialiased">
        <Providers>
          {/* Skip link: the first tab stop, so keyboard users are not dragged
              through the whole nav to reach the query input. */}
          <a
            href="#query"
            className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-md focus:bg-accent focus:px-4 focus:py-2 focus:text-sm focus:font-semibold focus:text-accent-fg"
          >
            Skip to query
          </a>
          {children}
          <Disclaimer />
        </Providers>
      </body>
    </html>
  );
}
