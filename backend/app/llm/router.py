"""LLM routing across free-tier providers, with a credential-free degraded mode.

Design notes
------------
1. **Provider chain.** Resolved once at import: groq -> gemini -> cerebras -> stub.
   The first provider holding a key wins. Others stay in the chain as failover for
   rate limits, which is the failure mode that actually bites on free tiers.

2. **Degraded mode is loud.** With no key at all we fall through to `StubProvider`,
   which returns deterministic placeholder text. That makes the frontend, the API
   contract and the test suite exercisable by anyone who clones the repo. Every
   response produced this way is flagged `degraded=True`, surfaced in the API
   payload, and rendered as a banner in the UI. `StubProvider` output must never
   reach RESULTS.md — `guard_real_llm()` enforces that for eval scripts.

3. **Two model tiers.** Routing and classification calls use the fast/cheap model;
   drafting and critique use the strong one. On a free tier the cheap calls are the
   difference between a demo that runs and one that spends its whole quota on
   query classification.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.config import settings
from app.logging_conf import get_logger

log = get_logger(__name__)


class Provider(StrEnum):
    GROQ = "groq"
    GEMINI = "gemini"
    CEREBRAS = "cerebras"
    STUB = "stub"


class Tier(StrEnum):
    FAST = "fast"  # routing, classification, entity extraction
    STRONG = "strong"  # drafting, critique, final composition


@dataclass
class LLMUsage:
    """Process-wide counters. Exposed on /metrics; also reported per-run in evals."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    retries: int = 0
    latency_ms_total: float = 0.0
    by_provider: dict[str, int] = field(default_factory=dict)

    def record(self, provider: str, pt: int, ct: int, ms: float) -> None:
        self.calls += 1
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.latency_ms_total += ms
        self.by_provider[provider] = self.by_provider.get(provider, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "errors": self.errors,
            "retries": self.retries,
            "avg_latency_ms": round(self.latency_ms_total / self.calls, 1) if self.calls else 0.0,
            "by_provider": dict(self.by_provider),
        }


USAGE = LLMUsage()


@dataclass(frozen=True)
class ProviderConfig:
    provider: Provider
    api_key: str | None
    strong_model: str
    fast_model: str

    def model_for(self, tier: Tier) -> str:
        return self.fast_model if tier is Tier.FAST else self.strong_model


def _build_chain() -> list[ProviderConfig]:
    chain: list[ProviderConfig] = []
    if settings.groq_api_key:
        chain.append(
            ProviderConfig(Provider.GROQ, settings.groq_api_key, settings.groq_model, settings.groq_model_fast)
        )
    if settings.gemini_api_key:
        chain.append(
            ProviderConfig(Provider.GEMINI, settings.gemini_api_key, settings.gemini_model, settings.gemini_model)
        )
    if settings.cerebras_api_key:
        chain.append(
            ProviderConfig(
                Provider.CEREBRAS, settings.cerebras_api_key, settings.cerebras_model, settings.cerebras_model
            )
        )
    if not chain:
        chain.append(ProviderConfig(Provider.STUB, None, "stub", "stub"))
    return chain


PROVIDER_CHAIN: list[ProviderConfig] = _build_chain()
IS_DEGRADED: bool = PROVIDER_CHAIN[0].provider is Provider.STUB

if IS_DEGRADED:
    log.warning(
        "llm.degraded_mode",
        msg="No LLM API key found. Running with the deterministic stub provider. "
        "Responses are placeholders and are flagged degraded=True. "
        "Set GROQ_API_KEY in backend/.env for real inference.",
    )
else:
    log.info("llm.chain_resolved", chain=[c.provider.value for c in PROVIDER_CHAIN])


# --------------------------------------------------------------------------- stub


_STUB_NOTICE = (
    "[DEGRADED MODE — no LLM credential configured. This is deterministic placeholder "
    "text, not a model output, and carries no clinical meaning.]"
)


class StubProvider:
    """Deterministic stand-in so the stack runs end-to-end without any credential.

    Returns shape-correct output for each prompt kind the graph asks for, keyed on a
    hash of the prompt so runs are reproducible. It is intentionally obvious: nobody
    should ever mistake this for a model response.
    """

    @staticmethod
    def complete(messages: list[dict[str, str]], tier: Tier, json_mode: bool) -> str:
        prompt = "\n".join(m.get("content", "") for m in messages)
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        low = prompt.lower()

        if json_mode:
            # Match the JSON contract each node expects.
            if "route" in low or "classify" in low:
                routes = ["simple_factual", "needs_patient_context", "needs_relationship_reasoning"]
                return json.dumps(
                    {
                        "route": routes[seed % 3],
                        "confidence": 0.5,
                        "reasoning": "stub: deterministic route, not a model decision",
                        "entities": ["stub_entity"],
                    }
                )
            if "faithful" in low or "grounded" in low or "critic" in low:
                return json.dumps(
                    {
                        "faithfulness": 0.5,
                        "supported_claims": 0,
                        "total_claims": 0,
                        "unsupported": [],
                        "reasoning": "stub: no real groundedness assessment performed",
                    }
                )
            if "entit" in low or "relation" in low or "triple" in low:
                return json.dumps({"entities": [], "relations": []})
            return json.dumps({"result": "stub", "confidence": 0.5})

        return (
            f"{_STUB_NOTICE}\n\n"
            "A grounded answer would appear here, with inline citations [1][2] bound to "
            "the retrieved guideline chunks shown in the sources panel."
        )


# --------------------------------------------------------------------------- core


class RateLimitedError(RuntimeError):
    pass


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("rate" in s and "limit" in s) or "429" in s or "quota" in s


async def _call_litellm(
    cfg: ProviderConfig,
    messages: list[dict[str, str]],
    tier: Tier,
    json_mode: bool,
    temperature: float,
    max_tokens: int | None,
) -> tuple[str, int, int]:
    import litellm

    litellm.drop_params = True  # providers differ on supported params; don't hard-fail
    litellm.suppress_debug_info = True

    kwargs: dict[str, Any] = {
        "model": cfg.model_for(tier),
        "messages": messages,
        "api_key": cfg.api_key,
        "temperature": temperature,
        "timeout": settings.llm_timeout_s,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = await litellm.acompletion(**kwargs)
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    return text, pt, ct


async def complete(
    messages: list[dict[str, str]],
    *,
    tier: Tier = Tier.STRONG,
    json_mode: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Run a completion across the provider chain with backoff.

    Walks the chain on rate limits rather than sleeping on a single provider — the
    whole point of configuring more than one free tier.
    """
    temp = settings.llm_temperature if temperature is None else temperature
    last_exc: Exception | None = None

    for cfg in PROVIDER_CHAIN:
        if cfg.provider is Provider.STUB:
            return StubProvider.complete(messages, tier, json_mode)

        for attempt in range(settings.llm_max_retries):
            t0 = time.perf_counter()
            try:
                text, pt, ct = await _call_litellm(cfg, messages, tier, json_mode, temp, max_tokens)
                USAGE.record(cfg.provider.value, pt, ct, (time.perf_counter() - t0) * 1000)
                return text
            except Exception as exc:  # noqa: BLE001 — provider SDKs raise broadly
                last_exc = exc
                USAGE.errors += 1
                if _is_rate_limit(exc):
                    log.warning("llm.rate_limited", provider=cfg.provider.value, attempt=attempt)
                    break  # try the next provider immediately
                if attempt < settings.llm_max_retries - 1:
                    USAGE.retries += 1
                    await asyncio.sleep(1.5 * (2**attempt))
                else:
                    log.warning("llm.provider_failed", provider=cfg.provider.value, error=str(exc)[:200])

    log.error("llm.chain_exhausted", error=str(last_exc)[:300])
    raise RateLimitedError(f"All LLM providers failed. Last error: {last_exc}")


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


async def complete_json(
    messages: list[dict[str, str]],
    *,
    tier: Tier = Tier.FAST,
    default: dict[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Completion that must yield an object.

    Models disregard `response_format` often enough that a bare `json.loads` is not
    a safe contract. We strip fences, then salvage the outermost brace-balanced
    block. If everything fails we return `default` rather than raising, because a
    malformed router response should degrade the route, not kill the request.
    """
    raw = await complete(
        messages, tier=tier, json_mode=True, temperature=temperature, max_tokens=max_tokens
    )
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    m = _JSON_BLOCK.search(text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    log.warning("llm.json_parse_failed", preview=text[:180])
    return default if default is not None else {}


def guard_real_llm(context: str) -> None:
    """Refuse to produce reportable numbers from the stub provider.

    Called by every eval and benchmark script. A metric computed against
    deterministic placeholder text is not a measurement, and publishing one would
    make every other number in RESULTS.md untrustworthy.
    """
    if IS_DEGRADED:
        raise RuntimeError(
            f"{context} requires a real LLM provider, but none is configured.\n"
            "The stub provider returns placeholder text; any metric derived from it "
            "would be meaningless.\n"
            "Set GROQ_API_KEY in backend/.env and re-run."
        )


def provider_info() -> dict[str, Any]:
    return {
        "degraded": IS_DEGRADED,
        "active_provider": PROVIDER_CHAIN[0].provider.value,
        "chain": [c.provider.value for c in PROVIDER_CHAIN],
        "strong_model": PROVIDER_CHAIN[0].strong_model,
        "fast_model": PROVIDER_CHAIN[0].fast_model,
    }
