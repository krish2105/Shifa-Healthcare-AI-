"""POST /query — run the agent, streaming its trace over SSE.

The stream is the demo. A user watching "route → retrieve → fuse → critic → compose"
land one event at a time is seeing the actual node transitions, not a scripted
animation: each SSE frame is emitted as LangGraph finishes the corresponding node,
and the same events are what the audit table persists.

We stream `stream_mode="updates"`, which yields one payload per completed node. That
is coarser than token-level streaming, and it is the right granularity here — the
interesting unit is "the critic returned 0.86", not the individual tokens of a draft
that may yet be rejected and never shown.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import get_agent, serialize_result
from app.agent.state import initial_state
from app.api.metrics import record_run as record_metrics
from app.audit.store import audit
from app.logging_conf import get_logger
from app.schemas import QueryRequest, QueryResponse

log = get_logger("api.query")
router = APIRouter()

# Human-readable labels for the trace strip. Keeping them server-side means the UI
# renders whatever the graph actually did, including nodes added later.
NODE_LABELS = {
    "intake": "Normalizing query",
    "planner": "Choosing retrieval strategy",
    "vector_retriever": "Hybrid vector + keyword search",
    "graph_retriever": "Traversing knowledge graph",
    "risk_node": "Scoring patient risk",
    "fusion": "Fusing and deduplicating evidence",
    "draft": "Drafting grounded answer",
    "critic": "Checking groundedness",
    "reformulate": "Reformulating query and retrying",
    "final_composer": "Binding citations",
    "escalate": "Escalating to physician",
}


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, default=str)}


@router.post("/query", response_model=QueryResponse, response_model_exclude_none=False)
async def query(req: QueryRequest, request: Request):
    run_id = uuid.uuid4().hex[:16]

    if not req.stream:
        from app.agent.graph import run_agent

        try:
            result = await run_agent(req.query, patient_id=req.patient_id, run_id=run_id)
        except Exception as exc:
            log.error("query.failed", run_id=run_id, error=str(exc)[:300])
            raise HTTPException(status_code=502, detail=f"Agent run failed: {exc}") from exc
        audit.record_run(result)
        record_metrics(result)
        return result

    agent = get_agent()

    async def event_stream():
        seen = 0
        final_state: dict[str, Any] = {}
        try:
            yield _sse("start", {"run_id": run_id, "query": req.query})

            async for chunk in agent.astream(
                initial_state(req.query, run_id, req.patient_id),
                config={"recursion_limit": 60},
                stream_mode="updates",
            ):
                if await request.is_disconnected():
                    log.info("query.client_disconnected", run_id=run_id)
                    break

                for node_name, delta in (chunk or {}).items():
                    if not isinstance(delta, dict):
                        continue
                    # Accumulate so the terminal payload is a complete state even
                    # though updates arrive as per-node deltas.
                    for k, v in delta.items():
                        if k == "trace":
                            final_state.setdefault("trace", [])
                            final_state["trace"].extend(v or [])
                        else:
                            final_state[k] = v

                    events = delta.get("trace") or []
                    for ev in events:
                        seen += 1
                        yield _sse(
                            "trace",
                            {
                                "seq": seen,
                                "node": ev.get("node", node_name),
                                "label": NODE_LABELS.get(ev.get("node", node_name), node_name),
                                "event": ev.get("event", ""),
                                "duration_ms": ev.get("duration_ms", 0.0),
                                "detail": ev.get("detail", {}),
                            },
                        )
                    # Yield to the event loop so frames flush as they are produced
                    # rather than batching at the end of a fast node sequence.
                    await asyncio.sleep(0)

            final_state.setdefault("run_id", run_id)
            final_state.setdefault("query", req.query)
            result = serialize_result(final_state)  # type: ignore[arg-type]
            audit.record_run(result)
            record_metrics(result)
            yield _sse("done", result)

        except asyncio.CancelledError:
            log.info("query.cancelled", run_id=run_id)
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("query.stream_failed", run_id=run_id, error=str(exc)[:300])
            yield _sse("error", {"run_id": run_id, "error": str(exc)[:300]})

    return EventSourceResponse(event_stream(), ping=15000)
