"""Assemble the HypeReel LangGraph state machine and a convenience runner.

Graph shape::

    plan -> ingest -> propose -> scoreboard -> classify -> select -> judge
          -> (judge: revise -> select  |  accept -> approve_clips)
          -> approve_clips [HUMAN GATE] -> render -> summarize
          -> approve_share [HUMAN GATE] -> deliver -> END

``scoreboard`` grounds candidate windows in game state (clock live/frozen,
score changes) read OCR-free from the on-screen scoreboard, so the selector can
drop timeout/dead-ball/sideline windows and confirm real made baskets. It is a
transparent pass-through for recipes that don't declare a ``scoreboard`` signal.

``judge`` is an LLM-as-judge critic: it scores the assembled reel against the
recipe's acceptance criteria and can bounce it back to ``select`` once (bounded
by ``_MAX_REVISIONS``) with stricter selection overrides before the human ever
sees it.

``approve_clips`` and ``approve_share`` are pass-through nodes; the graph is
compiled with ``interrupt_before=["approve_clips", "approve_share"]`` so
execution pauses right before each one, giving a human (or the CLI/UI layer)
a chance to inspect ``state`` and either approve or redirect. Resuming uses
the standard LangGraph checkpointer pattern verified against the installed
langgraph==1.2.11 API: ``update_state(config, partial)`` followed by
``invoke(None, config)``.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional

# LangGraph's default msgpack serde logs a per-type WARNING when it checkpoints
# our own Pydantic models (Recipe, CandidateWindow, ...). They round-trip fine;
# the message is just forward-compat noise that would clutter the CLI/demo
# output, so we raise that specific logger's threshold above WARNING.
logging.getLogger("langgraph.checkpoint.serde.jsonplus").setLevel(logging.ERROR)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from ..config import Settings, get_settings
from ..models import Recipe, ReelResult
from ..observability import with_tracing
from .nodes import (
    approve_clips_node,
    approve_share_node,
    classify_node,
    deliver_node,
    ingest_node,
    judge_node,
    plan_node,
    propose_node,
    record_selection_feedback,
    render_node,
    route_after_judge,
    scoreboard_node,
    select_node,
    summarize_node,
)
from .state import ReelState, new_state


def build_graph(checkpointer=None, settings: Optional[Settings] = None):
    """Construct and compile the HypeReel pipeline graph.

    Uses an in-memory checkpointer by default so ``run_pipeline`` (and tests)
    work fully offline without any external persistence.

    ``settings`` is resolved once here and injected into the nodes that need it
    (via ``functools.partial``), so a caller's :class:`Settings` genuinely flows
    through the pipeline instead of every node re-reading the environment. Nodes
    still default to ``get_settings()`` when called standalone (e.g. in tests).
    """
    checkpointer = checkpointer or InMemorySaver()
    settings = settings or get_settings()

    graph = StateGraph(ReelState)
    graph.add_node("plan", plan_node)
    graph.add_node("ingest", partial(ingest_node, settings=settings))
    graph.add_node("propose", partial(propose_node, settings=settings))
    graph.add_node("scoreboard", scoreboard_node)
    graph.add_node("classify", partial(classify_node, settings=settings))
    graph.add_node("select", select_node)
    graph.add_node("judge", partial(judge_node, settings=settings))
    graph.add_node("approve_clips", approve_clips_node)
    graph.add_node("render", partial(render_node, settings=settings))
    graph.add_node("summarize", partial(summarize_node, settings=settings))
    graph.add_node("approve_share", approve_share_node)
    graph.add_node("deliver", deliver_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "ingest")
    graph.add_edge("ingest", "propose")
    graph.add_edge("propose", "scoreboard")
    graph.add_edge("scoreboard", "classify")
    graph.add_edge("classify", "select")
    # select -> judge (the LLM critic). select still sets human_gate to
    # 'underfilled' vs 'approve_clips' for the UI; the judge decides whether the
    # reel is good enough to show a human or needs one stricter re-selection.
    graph.add_edge("select", "judge")
    graph.add_conditional_edges(
        "judge", route_after_judge, {"revise": "select", "accept": "approve_clips"}
    )
    graph.add_edge("approve_clips", "render")
    graph.add_edge("render", "summarize")
    graph.add_edge("summarize", "approve_share")
    graph.add_edge("approve_share", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["approve_clips", "approve_share"])


def _values(snapshot_or_dict) -> dict:
    """Both ``invoke()`` and ``get_state()`` results normalize to a plain dict."""
    if isinstance(snapshot_or_dict, dict):
        return snapshot_or_dict
    return dict(getattr(snapshot_or_dict, "values", {}) or {})


def run_pipeline(
    source: str,
    recipe: Recipe,
    settings: Optional[Settings] = None,
    *,
    auto_approve: bool = True,
    max_duration: Optional[float] = None,
    audience: Optional[str] = None,
    subject_description: Optional[str] = None,
    max_candidates: Optional[int] = None,
    thread_id: str = "demo",
) -> ReelResult:
    """Build the graph, run it end-to-end, and return a :class:`ReelResult`.

    With ``auto_approve=True`` (the default, used by the CLI ``--demo`` mode
    and tests) both human gates are auto-approved so the pipeline completes
    without any interaction, entirely offline in mock mode.

    With ``auto_approve=False`` the run stops at the *first* gate it hits and
    returns a partial ``ReelResult`` describing the state at that pause point
    — the caller (a UI/CLI) is expected to inspect it, then resume the graph
    itself via the same ``thread_id``.

    Never raises: any unexpected failure is captured into ``ReelResult.notes``.
    """
    settings = settings or get_settings()
    runtime_notes: list[str] = []

    try:
        app = build_graph(settings=settings)
        config = with_tracing(
            {"configurable": {"thread_id": thread_id}}, settings,
            recipe_id=recipe.id, entrypoint="runner",
        )
        initial = new_state(
            source,
            recipe,
            max_duration=max_duration,
            audience=audience,
            subject_description=subject_description,
            max_candidates=max_candidates,
        )

        current = _values(app.invoke(initial, config))

        while True:
            snapshot = app.get_state(config)
            if not snapshot.next:
                current = _values(snapshot)
                break

            gate_node = snapshot.next[0]
            current = _values(snapshot)

            if not auto_approve:
                runtime_notes.append(
                    f"paused for human approval at gate '{gate_node}' "
                    f"(human_gate={current.get('human_gate')!r})"
                )
                break

            if gate_node == "approve_clips":
                try:
                    record_selection_feedback(
                        recipe,
                        current.get("scored_clips", []),
                        current.get("selected_clips", []),
                        settings,
                    )
                except Exception:
                    pass
                app.update_state(config, {"approved": True, "needs_human": False})
            elif gate_node == "approve_share":
                app.update_state(config, {"shared": True, "needs_human": False})
            else:
                app.update_state(config, {"needs_human": False})

            current = _values(app.invoke(None, config))

        return ReelResult(
            output_path=current.get("output_path"),
            clips=current.get("selected_clips", []),
            total_duration=current.get("proposed_duration", 0.0),
            recipe_id=recipe.id,
            audience=current.get("audience") or recipe.subject_selector.audience,
            summary=current.get("summary", ""),
            notes=list(current.get("notes", [])) + list(current.get("errors", [])) + runtime_notes,
        )
    except Exception as exc:  # run_pipeline must never raise
        return ReelResult(
            output_path=None,
            clips=[],
            total_duration=0.0,
            recipe_id=recipe.id,
            audience=audience or recipe.subject_selector.audience,
            notes=[f"run_pipeline failed unexpectedly: {exc}"],
        )
