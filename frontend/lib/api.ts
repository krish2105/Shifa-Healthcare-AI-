import type { Health, MetricsSummary, QueryResult, TraceEntry } from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!res.ok) throw new Error(`${path} → ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const getHealth = () => getJSON<Health>("/api/health");
export const getMetrics = () => getJSON<MetricsSummary>("/metrics/summary");
export const getInfo = () => getJSON<Record<string, unknown>>("/api/info");
export const getAudit = (limit = 40) =>
  getJSON<{ entries: TraceEntry[]; stats: Record<string, number | string> }>(
    `/api/audit?limit=${limit}`,
  );
export const getPatients = () =>
  getJSON<{ patients: { stay_id: string; chiefcomplaint: string; acuity: number | null }[] }>(
    "/api/patients",
  );

export interface StreamHandlers {
  onStart?: (d: { run_id: string; query: string }) => void;
  onTrace?: (t: TraceEntry) => void;
  onDone?: (r: QueryResult) => void;
  onError?: (message: string) => void;
}

/**
 * Stream a query over SSE.
 *
 * `EventSource` cannot issue a POST, and the query body carries the question plus
 * an optional patient id — so we read the SSE stream off `fetch` and parse frames
 * by hand. The parser holds a buffer across chunk boundaries because a network
 * chunk has no relationship to an SSE frame: a single `data:` line routinely
 * arrives split across two reads, and splitting per-chunk silently drops events.
 *
 * Returns an abort function so a component can cancel in flight — on unmount, or
 * when the user submits a new question before the previous one finished.
 */
export function streamQuery(
  query: string,
  handlers: StreamHandlers,
  opts: { patientId?: string | null } = {},
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({
          query,
          patient_id: opts.patientId || null,
          stream: true,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        handlers.onError?.(
          res.status === 429
            ? "Rate limit reached. Wait a moment and try again."
            : `Request failed (${res.status}). ${text.slice(0, 200)}`,
        );
        return;
      }
      if (!res.body) {
        handlers.onError?.("No response stream from the server.");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;

        // Normalize line endings before framing. The SSE spec allows CRLF, LF or
        // a bare CR as a line terminator, and sse-starlette emits CRLF — so a
        // parser that splits on "\n\n" alone matches nothing against
        // "\r\n\r\n" and silently never emits a single event.
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n|\r/g, "\n");

        // Frames are separated by a blank line. Keep the trailing partial frame
        // in the buffer for the next read.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          if (!frame.trim()) continue;

          let event = "message";
          const dataLines: string[] = [];
          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
            // ':' comment lines are SSE keep-alive pings — ignore.
          }
          if (!dataLines.length) continue;

          let payload: unknown;
          try {
            payload = JSON.parse(dataLines.join("\n"));
          } catch {
            continue;
          }

          if (event === "start") handlers.onStart?.(payload as never);
          else if (event === "trace") handlers.onTrace?.(payload as TraceEntry);
          else if (event === "done") handlers.onDone?.(payload as QueryResult);
          else if (event === "error")
            handlers.onError?.((payload as { error?: string }).error ?? "Unknown error");
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      handlers.onError?.(
        (err as Error).message.includes("fetch")
          ? `Cannot reach the API at ${API_BASE}. Is the backend running?`
          : (err as Error).message,
      );
    }
  })();

  return () => controller.abort();
}
