"""Structured JSON logging.

Two consumers: humans reading the console during development, and the audit trail.
The audit trail is not this module's job (see app/audit/), but every audit event is
*also* emitted here so a single log stream reconstructs a full agent run.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import settings

_configured = False


def configure_logging(force_json: bool | None = None) -> None:
    global _configured
    if _configured:
        return

    use_json = force_json if force_json is not None else settings.app_env != "local"

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        # NB: no stdlib.add_logger_name — it reads `logger.name`, which PrintLogger
        # does not have. get_logger() binds the name into the event dict instead.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )
    # These are chatty and say nothing we need.
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> Any:
    configure_logging()
    return structlog.get_logger().bind(logger=name)
