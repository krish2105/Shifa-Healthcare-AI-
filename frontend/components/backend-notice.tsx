"use client";

// NB: no brand icons — this version of lucide-react dropped them.
import { AlertTriangle, ExternalLink } from "lucide-react";
import { API_BASE } from "@/lib/api";

/**
 * Explains an absent backend.
 *
 * This matters more than a normal error state, because on a frontend-only
 * deployment (Vercel, no API host) it *is* the experience every visitor gets.
 * The dev-oriented "run uvicorn" message is wrong there — a recruiter opening the
 * link has no local checkout and no intention of getting one.
 *
 * So the copy is chosen by context: if the page is served from a real domain but
 * configured to talk to localhost, the backend was never deployed, and we say
 * that plainly and explain why (a 1.3 GB embedding model does not fit in a
 * serverless function). If we are on localhost, the developer just hasn't started
 * the API, and the fix is the command.
 */
export function BackendNotice({ variant = "banner" }: { variant?: "banner" | "inline" }) {
  const pointsAtLocalhost = /localhost|127\.0\.0\.1/.test(API_BASE);
  const servedFromRemote =
    typeof window !== "undefined" && !/localhost|127\.0\.0\.1/.test(window.location.hostname);

  const frontendOnlyDeploy = pointsAtLocalhost && servedFromRemote;

  return (
    <div
      role="status"
      className={
        variant === "banner"
          ? "mt-5 rounded-lg border border-warn/30 bg-warn/10 px-4 py-3"
          : "rounded-lg border border-warn/30 bg-warn/10 px-4 py-3"
      }
    >
      <p className="flex items-start gap-2.5 text-[12.5px] leading-relaxed text-warn">
        <AlertTriangle className="mt-px h-3.5 w-3.5 flex-none" aria-hidden />
        {frontendOnlyDeploy ? (
          <span>
            <strong className="font-semibold">This is a frontend-only deployment.</strong> The
            agent backend isn&apos;t hosted here — it loads a 1.3 GB embedding model and a 53k-chunk
            index, which doesn&apos;t fit in a serverless function. The UI below is fully built and
            wired; run the stack locally to see it answer.
          </span>
        ) : (
          <span>
            <strong className="font-semibold">Backend unreachable</strong> at{" "}
            <code className="font-mono">{API_BASE}</code>. Start it with{" "}
            <code className="font-mono">make api</code> (or{" "}
            <code className="font-mono">uvicorn app.main:app</code> in{" "}
            <code className="font-mono">/backend</code>).
          </span>
        )}
      </p>

      {frontendOnlyDeploy && (
        <a
          href="https://github.com/krish2105/Shifa-Healthcare-AI-"
          target="_blank"
          rel="noopener noreferrer"
          className="mt-2.5 inline-flex items-center gap-1.5 rounded-md border border-warn/40 px-2.5 py-1 font-mono text-[11px] text-warn transition-colors hover:bg-warn/10"
        >
          Source, architecture and measured results
          <ExternalLink className="h-3 w-3" aria-hidden />
        </a>
      )}
    </div>
  );
}
