"""Audit trail for autonomous decisions.

The governance claim this project makes is that every autonomous decision is
recorded and reconstructable. That is only true if the record is written somewhere
durable and is written even when the request fails — so this module writes to
Postgres when it is available and to a local JSONL file when it is not, and never
raises into the request path.

What gets recorded, per run: the query, the route chosen and why, every retrieval
and its source, every groundedness score, every retry, and the terminal outcome. In
other words, enough to answer "why did the system say this?" months later without
the original session.

**No PHI.** Only the query text the user submitted is stored, and intake flags
identifier-shaped content before it reaches here. This is a demonstration over open
data; a production deployment would need a retention policy and access controls that
are deliberately out of scope.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.config import settings
from app.logging_conf import get_logger

log = get_logger("audit")

JSONL_PATH = settings.artifacts_dir / "audit_log.jsonl"


class AuditLog:
    def __init__(self) -> None:
        self._pg_ok: bool | None = None

    # -- backend selection --------------------------------------------------

    def _try_pg(self):
        if not settings.database_url:
            self._pg_ok = False
            return None
        try:
            from app.retrieval.store import PgVectorStore, get_store

            store = get_store()
            if isinstance(store, PgVectorStore):
                self._pg_ok = True
                return store
        except Exception as exc:  # noqa: BLE001
            log.warning("audit.pg_unavailable", error=str(exc)[:160])
        self._pg_ok = False
        return None

    # -- writes -------------------------------------------------------------

    def record_events(self, run_id: str, events: list[dict[str, Any]]) -> None:
        """Persist a run's trace events. Never raises."""
        if not events:
            return
        try:
            store = self._try_pg() if self._pg_ok is not False else None
            if store is not None:
                self._write_pg(store, run_id, events)
            else:
                self._write_jsonl(run_id, events)
        except Exception as exc:  # noqa: BLE001 — auditing must never break a request
            log.error("audit.write_failed", error=str(exc)[:200], run_id=run_id)

    def _write_pg(self, store, run_id: str, events: list[dict[str, Any]]) -> None:
        rows = [
            (
                run_id,
                e.get("node", ""),
                e.get("event", ""),
                float(e.get("duration_ms", 0.0)),
                json.dumps(e.get("detail", {}), default=str),
            )
            for e in events
        ]
        with store.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO audit_log (run_id, node, event, duration_ms, payload) "
                "VALUES (%s,%s,%s,%s,%s::jsonb)",
                rows,
            )

    def _write_jsonl(self, run_id: str, events: list[dict[str, Any]]) -> None:
        JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with JSONL_PATH.open("a") as fh:
            for e in events:
                fh.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "ts": e.get("ts", time.time()),
                            "node": e.get("node", ""),
                            "event": e.get("event", ""),
                            "duration_ms": e.get("duration_ms", 0.0),
                            "detail": e.get("detail", {}),
                        },
                        default=str,
                    )
                    + "\n"
                )

    def record_run(self, result: dict[str, Any]) -> None:
        """Record the run summary alongside its per-node events."""
        run_id = result.get("run_id", "")
        self.record_events(run_id, result.get("trace", []))
        self.record_events(
            run_id,
            [
                {
                    "node": "_run",
                    "event": result.get("outcome", "unknown"),
                    "ts": time.time(),
                    "duration_ms": 0.0,
                    "detail": {
                        "query": result.get("query", "")[:500],
                        "route": result.get("route"),
                        "groundedness": result.get("groundedness"),
                        "attempts": result.get("attempts"),
                        "citations": len(result.get("citations", [])),
                        "degraded": result.get("degraded"),
                        "identifiers_flagged": result.get("contains_identifiers"),
                    },
                }
            ],
        )

    # -- reads --------------------------------------------------------------

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent audit entries, newest first. Powers the UI audit panel."""
        try:
            store = self._try_pg() if self._pg_ok is not False else None
            if store is not None:
                with store.pool.connection() as conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT run_id, ts, node, event, duration_ms, payload "
                        "FROM audit_log ORDER BY id DESC LIMIT %s",
                        (limit,),
                    )
                    return [
                        {
                            "run_id": r[0],
                            "ts": r[1].timestamp() if hasattr(r[1], "timestamp") else r[1],
                            "node": r[2],
                            "event": r[3],
                            "duration_ms": r[4],
                            "detail": r[5] or {},
                        }
                        for r in cur.fetchall()
                    ]
        except Exception as exc:  # noqa: BLE001
            log.warning("audit.read_pg_failed", error=str(exc)[:160])

        if not JSONL_PATH.exists():
            return []
        lines = JSONL_PATH.read_text().strip().splitlines()
        out: list[dict[str, Any]] = []
        for line in reversed(lines[-(limit * 3) :]):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out

    def stats(self) -> dict[str, Any]:
        """Aggregate counters for /metrics and the dashboard."""
        entries = self.recent(limit=2000)
        runs = [e for e in entries if e.get("node") == "_run"]
        answered = sum(1 for r in runs if r.get("event") == "answered")
        escalated = sum(1 for r in runs if r.get("event") == "escalated")
        total = answered + escalated
        scores = [
            float(r["detail"]["groundedness"])
            for r in runs
            if isinstance(r.get("detail"), dict) and r["detail"].get("groundedness") is not None
        ]
        return {
            "runs_total": total,
            "answered": answered,
            "escalated": escalated,
            "escalation_rate": round(escalated / total, 4) if total else 0.0,
            "avg_groundedness": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "events_recorded": len(entries),
            "backend": "postgres" if self._pg_ok else "jsonl",
        }


audit = AuditLog()
