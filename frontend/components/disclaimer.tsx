"use client";

/**
 * Persistent, non-dismissible disclaimer.
 *
 * There is no close button and no `useState` behind it, by design. This is the
 * one piece of UI whose entire purpose is to be present when a user is deciding
 * how much to trust an answer — a dismissible version would be reliably absent
 * exactly then. Wording is fixed by the project's safety spec.
 */
export function Disclaimer() {
  return (
    <footer
      role="contentinfo"
      className="relative z-20 border-t border-line bg-bg-elev/80 backdrop-blur-sm"
    >
      <div className="mx-auto flex max-w-[1240px] items-start gap-3 px-5 py-4 sm:px-8">
        <span aria-hidden className="mt-px select-none text-base leading-none text-accent">
          ⚕
        </span>
        <p className="text-pretty text-[11.5px] leading-relaxed text-fg-3">
          <strong className="font-medium text-fg-2">
            Shifa42 is a research and educational demonstration. It is not a certified medical
            device and does not provide medical advice.
          </strong>{" "}
          All data shown is synthetic or open, de-identified research data (MIMIC-IV-ED Demo,
          PhysioNet; clinical guideline corpus, <code className="font-mono">epfl-llm/guidelines</code>
          ). No protected health information is used anywhere in this system. Any real clinical use
          requires licensed physician oversight.
        </p>
      </div>
    </footer>
  );
}
