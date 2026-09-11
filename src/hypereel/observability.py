"""Opt-in graph tracing; never changes graph state or retries application work.

Attach once to an invocation config and reuse that config across interrupts.
LangGraph propagates the callback to its nodes for both invoke and stream.
Imports and client creation happen only when tracing is explicitly enabled.
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from uuid import uuid4
from collections.abc import Callable
from typing import TypeVar
from contextlib import contextmanager
from contextvars import ContextVar

from .config import Settings

_log = logging.getLogger(__name__)
_T = TypeVar("_T")
_provider_failures: ContextVar[list[str] | None] = ContextVar("provider_failures", default=None)
_provider_budget: ContextVar[dict | None] = ContextVar("provider_budget", default=None)


class ProviderBudgetExceeded(RuntimeError):
    """Raised before a provider request would exceed a configured run cap."""


@contextmanager
def provider_budget_scope(settings: Settings):
    """Track provider usage and enforce request/spend caps for one evaluation case."""
    ledger_path = settings.provider_spend_ledger_path
    spent_before = 0.0
    if ledger_path:
        try:
            spent_before = float(json.loads(Path(ledger_path).read_text()).get("estimated_spend_usd", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            spent_before = 0.0
    state = {
        "calls": [],
        "attempted_calls": 0,
        "estimated_spend_usd": 0.0,
        "estimated_spend_before_run_usd": spent_before,
        "ledger_path": ledger_path,
        "max_calls": settings.max_provider_calls,
        "max_spend_usd": settings.max_provider_spend_usd,
        "reserve_usd": settings.provider_call_reserve_usd,
        "input_rate": settings.nebius_input_cost_per_million_usd,
        "output_rate": settings.nebius_output_cost_per_million_usd,
    }
    token = _provider_budget.set(state)
    try:
        yield state
    finally:
        _provider_budget.reset(token)


def authorize_provider_call(*, provider: str, model: str, operation: str) -> None:
    state = _provider_budget.get()
    if state is None:
        return
    if state["max_calls"] > 0 and state["attempted_calls"] >= state["max_calls"]:
        raise ProviderBudgetExceeded("provider request cap reached")
    if (state["max_spend_usd"] > 0
            and state["estimated_spend_before_run_usd"]
            + state["estimated_spend_usd"] + state["reserve_usd"]
            > state["max_spend_usd"]):
        raise ProviderBudgetExceeded("provider spend cap reached")
    state["attempted_calls"] += 1
    state["calls"].append({
        "provider": provider,
        "model": model,
        "operation": operation,
        "status": "started",
        "reserved_cost_usd": state["reserve_usd"],
    })


def record_provider_usage(completion, *, status: str = "success") -> None:
    state = _provider_budget.get()
    if state is None or not state["calls"]:
        return
    usage = getattr(completion, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    output = int(getattr(usage, "completion_tokens", 0) or 0)
    estimated = (
        prompt * state["input_rate"] / 1_000_000
        + output * state["output_rate"] / 1_000_000
    )
    call = state["calls"][-1]
    call.update({
        "status": status,
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": prompt + output,
        "estimated_cost_usd": estimated,
    })
    state["estimated_spend_usd"] += estimated
    if state["ledger_path"]:
        path = Path(state["ledger_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        total = state["estimated_spend_before_run_usd"] + state["estimated_spend_usd"]
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps({"estimated_spend_usd": total}, indent=2) + "\n")
        os.replace(temporary, path)


def record_provider_failure(error: Exception) -> None:
    """Record a caught failure without changing the provider's fallback return.

    Only exception class names are recorded, never API error bodies or secrets.
    The context is local to this call/thread and reset even when a call raises.
    """
    failures = _provider_failures.get()
    if failures is not None:
        failures.append(type(error).__name__)


def _start_provider_span(name: str, metadata: dict):
    """Use the enclosing node's callbacks; never create a standalone client."""
    try:
        from langgraph.config import get_config
        from langchain_core.runnables.config import get_callback_manager_for_config

        config = get_config()
        # Only trace within a run explicitly instrumented by with_tracing().
        if not config.get("metadata", {}).get("hypereel_execution_id"):
            return None
        manager = get_callback_manager_for_config({
            **config, "metadata": {**config.get("metadata", {}), **metadata},
        })
        return manager.on_chain_start(
            None, {}, name=name,
            run_type="chain" if metadata.get("provider_mock") else "llm",
        )
    except Exception:
        return None


def trace_provider_call(call: Callable[[], _T], *, name: str, metadata: dict) -> _T:
    """Record a provider operation, delegating exactly once even if tracing fails.

    Live calls are LLM spans; mocks are chain spans. No token usage is invented.
    Neither call arguments nor returned content are sent to the callbacks.
    """
    span = _start_provider_span(name, metadata)
    failures: list[str] = []
    token = _provider_failures.set(failures)
    try:
        result = call()
    except BaseException as exc:
        if span is not None:
            try:
                span.on_chain_error(RuntimeError(type(exc).__name__))
            except Exception:
                pass
        raise
    finally:
        _provider_failures.reset(token)
    if span is not None:
        try:
            if failures:
                span.on_chain_error(RuntimeError("caught provider failure: " + ", ".join(failures)))
            else:
                span.on_chain_end({})
        except Exception:
            pass
    return result


def _make_tracer(settings: Settings):
    from langchain_core.tracers.langchain import LangChainTracer
    from langsmith import Client

    client = Client(
        api_key=settings.langsmith_api_key,
        api_url=settings.langsmith_endpoint,
        hide_inputs=True,
        hide_outputs=True,
        timeout_ms=1000,
    )
    tracer = LangChainTracer(project_name=settings.langsmith_project, client=client)
    # Callback failures must never propagate into the graph (including streams).
    tracer.raise_error = False
    return tracer


def with_tracing(config: dict, settings: Settings, *, recipe_id: str, entrypoint: str) -> dict:
    """Return a copied config with native LangSmith callbacks, or the original.

    Call at run creation, not on every resume. The telemetry execution ID is
    independent of the existing checkpoint thread ID, which is left untouched.
    Full graph inputs/outputs are hidden; metadata contains no Settings object,
    source URL, subject description, or frame contents. Native error traces may
    still contain exception messages. SDK uploads happen in the background.
    """
    if not settings.tracing_enabled:
        return config
    if not settings.langsmith_api_key:
        _log.warning("HypeReel tracing disabled: LANGSMITH_API_KEY is missing.")
        return config
    try:
        tracer = _make_tracer(settings)
        callbacks = config.get("callbacks")
        if callbacks is None or isinstance(callbacks, (list, tuple)):
            callbacks = [*(callbacks or []), tracer]
        else:
            callbacks = callbacks.copy()
            callbacks.add_handler(tracer, inherit=True)
        return {
            **config,
            "callbacks": callbacks,
            "tags": [*config.get("tags", []), "hypereel"],
            "metadata": {
                **config.get("metadata", {}),
                "hypereel_execution_id": str(uuid4()),
                "hypereel_entrypoint": entrypoint,
                "recipe_id": recipe_id,
                "requested_vision_provider": settings.vision_provider,
                "requested_llm_provider": settings.llm_provider,
            },
        }
    except Exception:
        # Don't log exception text: client errors can include endpoint credentials.
        _log.warning("HypeReel tracing unavailable; continuing without its callback.")
        return config
