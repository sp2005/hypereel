"""LangGraph node functions.

Each node is a plain ``def <name>_node(state: ReelState) -> dict`` that reads
what it needs from ``state`` and the leaf modules, then returns a *partial*
state update (LangGraph merges it into the running state). Nodes never raise —
every leaf module they call is already non-raising by design, so a node's job
is mostly bookkeeping: pick a provider, call the leaf function, translate the
result into state fields, and append a human-readable note.

``approve_clips_node`` / ``approve_share_node`` are deliberate no-ops: they
exist only as named targets for ``compile(interrupt_before=...)`` in
:mod:`hypereel.graph.build`, so the graph pauses *before* running them.
"""

from __future__ import annotations

import json
import os
import re

from ..analyze.classifier import classify_candidates
from ..config import get_settings
from ..ingest.source_resolver import IngestError, parse_video_quality, resolve_source
from ..memory.store import MemoryStore
from ..models import Clip, Recipe
from ..providers.factory import get_llm_provider, get_vision_provider
from ..render.renderer import render_reel
from ..select.selector import score_candidates, select_clips
from ..signals.audio import detect_commentary
from ..signals.merge import propose_candidates
from ..signals.scoreboard import annotate_candidates
from .state import GATE_APPROVE_CLIPS, GATE_APPROVE_SHARE, GATE_UNDERFILLED, ReelState

_FALLBACK_DURATION = 120.0


def plan_node(state: ReelState) -> dict:
    """Pick a strategy family + the active proposer signals for this recipe."""
    recipe: Recipe = state["recipe"]

    if recipe.highlight_model == "quality_based":
        strategy = "quality_based"
    elif recipe.highlight_model == "hybrid":
        strategy = "quality_based" if recipe.scoring_rubric else "event_based"
    else:
        strategy = "event_based"

    active_signals = [s.type for s in recipe.proposer_signals()]

    return {
        "strategy": strategy,
        "active_signals": active_signals,
        "notes": state.get("notes", [])
        + [f"plan: strategy='{strategy}', active_signals={active_signals}"],
    }


def ingest_node(state: ReelState, settings=None) -> dict:
    """Resolve the source to a local video path (or degrade gracefully)."""
    settings = settings or get_settings()
    source = state.get("source", "")

    # The user's free-text brief can also carry output-quality intent
    # ("use HD if available", "quick 360p preview"); honor it on downloads.
    recipe: Recipe = state.get("recipe")
    brief = recipe.subject_selector.description if recipe else None
    max_height = parse_video_quality(brief)

    try:
        path, duration = resolve_source(source, settings, max_height=max_height)
        has_commentary = detect_commentary(path)
        notes = [f"ingest: resolved '{source}' -> {path} ({duration:.1f}s)"]
        if max_height is not None:
            notes.append(f"ingest: prompt requested video quality <= {max_height}p")
        return {
            "video_path": path,
            "video_duration": duration if duration and duration > 0 else _FALLBACK_DURATION,
            "has_commentary": has_commentary,
            "notes": state.get("notes", []) + notes,
        }
    except IngestError as exc:
        return {
            "video_path": None,
            "video_duration": _FALLBACK_DURATION,
            "has_commentary": False,
            "errors": state.get("errors", []) + [f"ingest error: {exc}"],
            "needs_human": True,
            "human_gate": "ingest_failed",
            "notes": state.get("notes", [])
            + ["ingest failed; continuing with a synthetic timeline so the demo can proceed"],
        }
    except Exception as exc:  # belt-and-suspenders: this node must never raise
        return {
            "video_path": None,
            "video_duration": _FALLBACK_DURATION,
            "has_commentary": False,
            "errors": state.get("errors", []) + [f"ingest unexpected error: {exc}"],
            "needs_human": True,
            "human_gate": "ingest_failed",
            "notes": state.get("notes", []) + ["ingest failed unexpectedly; falling back."],
        }


def _apply_signal_conditions(recipe: Recipe, context: dict) -> Recipe:
    """Resolve each signal's ``enabled_if`` against detected runtime conditions.

    A signal declared ``enabled_if: commentary_present`` only stays active if
    the pipeline actually detected commentary; otherwise it's switched off so it
    neither proposes windows nor contributes to scoring. This is the concrete
    wiring behind the recipe's conditional-signal feature: cheap detections
    (e.g. ``has_commentary`` from ingest) gate which signals the planner runs.

    Recipes with no conditional signals are returned unchanged (no copy), so the
    common case — and both shipped sample recipes — is completely unaffected.
    """
    if not any(s.enabled_if for s in recipe.signals):
        return recipe
    resolved = recipe.model_copy(deep=True)
    for sig in resolved.signals:
        if sig.enabled_if and not context.get(sig.enabled_if, False):
            sig.enabled = False
    return resolved


def propose_node(state: ReelState, settings=None) -> dict:
    """Run the proposer signals and merge them into one candidate timeline."""
    settings = settings or get_settings()
    # Gate conditional signals on what ingest detected, then run the survivors.
    context = {"commentary_present": bool(state.get("has_commentary"))}
    recipe: Recipe = _apply_signal_conditions(state["recipe"], context)
    candidates = propose_candidates(
        state.get("video_path"), state.get("video_duration", 0.0), recipe, settings
    )
    uncapped_candidates = list(candidates)
    notes = state.get("notes", []) + [f"propose: {len(candidates)} candidate window(s) generated"]

    slice_start = state.get("evaluation_start_seconds")
    slice_end = state.get("evaluation_end_seconds")
    if slice_start is not None or slice_end is not None:
        candidates = [w for w in candidates
                      if (slice_start is None or w.end > slice_start)
                      and (slice_end is None or w.start < slice_end)]
        notes.append(
            f"propose: evaluation slice [{slice_start or 0:.1f}, "
            f"{slice_end if slice_end is not None else float('inf'):.1f}) kept "
            f"{len(candidates)} window(s)"
        )

    # Quick-test cap: classify only the first N windows so a prompt/recipe change
    # can be validated in ~1 min of API spend instead of a full pass. Windows are
    # kept in chronological order (earliest first); None/<=0 means no cap.
    cap = state.get("max_candidates")
    if cap and cap > 0 and len(candidates) > cap:
        ordered = sorted(candidates, key=lambda w: w.start)[:cap]
        if state.get("candidate_sampling") == "spread":
            source = sorted(candidates, key=lambda w: w.start)
            indexes = [round(i * (len(source) - 1) / (cap - 1)) for i in range(cap)] \
                if cap > 1 else [len(source) // 2]
            ordered = [source[i] for i in indexes]
        notes.append(
            f"propose: quick-test cap ON — kept {cap} of {len(candidates)} windows "
            f"using {state.get('candidate_sampling', 'chronological')} sampling"
        )
        candidates = ordered

    update = {
        "candidates": candidates,
        "uncapped_candidates": uncapped_candidates,
        "active_signals": [s.type for s in recipe.proposer_signals()],
        "notes": notes,
    }
    # Only thread the resolved recipe back into state when conditions actually
    # changed it, so downstream scoring/render use the same active signal set.
    if recipe is not state["recipe"]:
        update["recipe"] = recipe
        update["notes"].append(
            f"propose: conditional signals resolved against {context}"
        )
    return update


def _scoreboard_signal(recipe: Recipe):
    """The recipe's ``scoreboard`` signal config, or None if it doesn't use one."""
    for sig in recipe.signals:
        if sig.type == "scoreboard" and sig.enabled:
            return sig
    return None


def scoreboard_node(state: ReelState) -> dict:
    """Ground candidates in game state via the scoreboard overlay (OCR-free).

    Only does work when the recipe declares a ``scoreboard`` signal and a real
    video is present; otherwise it's a transparent pass-through. Annotates each
    candidate window with clock-live / score-change flags the selector uses to
    crush dead-ball (timeout/sideline) windows and confirm made baskets.
    """
    recipe: Recipe = state["recipe"]
    sig = _scoreboard_signal(recipe)
    candidates = state.get("candidates", [])
    if sig is None or not candidates or not state.get("video_path"):
        return {}

    candidates, stats = annotate_candidates(state.get("video_path"), candidates, sig.params or None)
    if not stats.get("ok"):
        note = "scoreboard: could not read overlay (no cv2/video/regions); skipping game-state grounding"
    else:
        note = (
            f"scoreboard: annotated {stats['annotated']} window(s) — "
            f"{stats['live']} live, {stats['dead']} dead-ball/timeout, "
            f"{stats['score_changes']} with a score change"
        )
    return {"candidates": candidates, "notes": state.get("notes", []) + [note]}


def classify_node(state: ReelState, settings=None) -> dict:
    """Judge every candidate window with the active vision provider."""
    settings = settings or get_settings()
    recipe: Recipe = state["recipe"]
    provider = get_vision_provider(settings)
    classifications = classify_candidates(
        state.get("candidates", []), recipe, provider, settings, state.get("video_path")
    )
    return {
        "classifications": classifications,
        "mode": provider.name,
        "notes": state.get("notes", [])
        + [f"classify: {len(classifications)} window(s) judged by '{provider.name}' provider"],
    }


def _effective_recipe(recipe: Recipe, overrides: dict, audience: str | None = None) -> Recipe:
    """Apply the judge's numeric selection overrides (and a runtime audience) to a copy.

    Only ``min_score`` is a recipe-level knob; ``require_moment`` is applied as a
    post-score filter in :func:`select_node` (it isn't part of the recipe schema).
    A runtime ``audience`` override is baked onto ``subject_selector.audience`` so
    it actually reaches :func:`select.selector.score_candidates`'s ``require_subject``
    rule — i.e. choosing ``"team"`` genuinely turns the one-team filter off (and
    ``"individual"`` turns it on). Returns the recipe unchanged when there's
    nothing to override, so the common first pass never pays for a deep copy.
    """
    needs_min_score = overrides.get("min_score") is not None
    needs_audience = bool(audience) and audience != recipe.subject_selector.audience
    if not needs_min_score and not needs_audience:
        return recipe
    eff = recipe.model_copy(deep=True)
    if needs_min_score:
        eff.selection.min_score = float(overrides["min_score"])
    if needs_audience:
        eff.subject_selector.audience = audience  # type: ignore[assignment]
    return eff


def select_node(state: ReelState) -> dict:
    """Score + budget-fit the final clip set, then decide which human gate to raise.

    Honors any ``selection_overrides`` the quality judge wrote on a revision pass:
    a higher ``min_score`` (drop weak clips) and/or ``require_moment`` (keep only
    clips the vision scorer gave a real moment_type, i.e. drop padding). On the
    first pass ``selection_overrides`` is empty and behavior is unchanged.
    """
    recipe: Recipe = state["recipe"]
    overrides = state.get("selection_overrides") or {}
    eff_recipe = _effective_recipe(recipe, overrides, audience=state.get("audience"))

    scored = score_candidates(
        state.get("candidates", []),
        state.get("classifications", []),
        eff_recipe,
        video_duration=state.get("video_duration"),
    )
    if overrides.get("require_moment"):
        scored = [c for c in scored if c.moment_type]

    selected = select_clips(
        scored, eff_recipe, max_duration=state.get("max_duration"), audience=state.get("audience")
    )
    proposed_duration = sum(c.duration for c in selected)
    budget = state.get("max_duration") or recipe.selection.max_duration

    underfilled = (not selected) or (proposed_duration < 0.8 * budget)
    human_gate = GATE_UNDERFILLED if underfilled else GATE_APPROVE_CLIPS

    note = (
        f"select: {len(selected)} clip(s) chosen, "
        f"{proposed_duration:.1f}s of {budget:.1f}s budget "
        f"({'underfilled' if underfilled else 'ready for review'})"
    )
    if overrides:
        note += f" [judge overrides applied: {overrides}]"

    return {
        "scored_clips": scored,
        "selected_clips": selected,
        "proposed_duration": proposed_duration,
        "needs_human": True,
        "human_gate": human_gate,
        "notes": state.get("notes", []) + [note],
    }


# How many times the judge is allowed to bounce the reel back to selection.
# One revision is enough to demonstrate self-correction without risking a long
# (or, on a flaky provider, non-converging) loop before the human gate. The
# single revision is made *smart* (see judge_node): it broadens rather than
# tightening whenever tightening would leave too few real plays.
_MAX_REVISIONS = 1

# Below either floor the reel reads as "too thin" and the judge broadens instead
# of shrinking further — the fix for the "revise -> 1 clip -> stuck" regression.
_MIN_REEL_CLIPS = 3
_MIN_REEL_SECONDS = 20.0

_JUDGE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_judge_prompt(recipe: Recipe, clips: list[Clip], proposed: float, budget: float) -> str:
    """Prompt the LLM to critique the assembled reel against the recipe's intent."""
    accept = "\n".join(f"  - {a}" for a in recipe.acceptance) or "  - (none specified)"
    n_conf = sum(1 for c in clips if c.moment_type)
    fill_pct = (proposed / budget * 100.0) if budget else 0.0
    clip_lines = "\n".join(
        f"  {i}. {c.moment_type or 'UNCLASSIFIED'} | {c.start:.0f}-{c.end:.0f}s "
        f"| score {c.score:.2f} | {(c.reason or '')[:80]}"
        for i, c in enumerate(clips, 1)
    ) or "  (no clips selected)"

    return (
        "You are the QUALITY JUDGE for a highlight-reel agent — the last check "
        "before a human reviews the reel. Judge whether the selected clips form a "
        f"compelling, on-theme reel for the recipe '{recipe.name}' ({recipe.domain}).\n\n"
        f"Acceptance criteria the reel should satisfy:\n{accept}\n\n"
        f"Reel under review: {len(clips)} clip(s), {proposed:.0f}s of a {budget:.0f}s "
        f"budget ({fill_pct:.0f}% filled). {n_conf} clip(s) have a confirmed moment "
        f"type; {len(clips) - n_conf} are UNCLASSIFIED (the vision scorer saw the "
        "subject but could not confirm a specific play).\n"
        f"Clips:\n{clip_lines}\n\n"
        "Judge on TWO axes:\n"
        "1) Quality: a strong sports reel is built from real, varied plays (for "
        "basketball: made baskets, threes, dunks/layups, blocks, steals, rebounds), "
        "not padded with UNCLASSIFIED windows where nothing clearly happens.\n"
        "2) Length: a good reel shows SEVERAL plays and uses a meaningful share of "
        "the budget. A reel of only 1-3 clips, or well under ~40% of the budget, is "
        "too thin if the game surely had more real plays to show.\n\n"
        "Choose ONE action:\n"
        "- 'require_confirmed_moments' — if more than about a third of the clips are "
        "UNCLASSIFIED (drops padding).\n"
        "- 'raise_threshold' — if the clips are mostly confirmed but low-scoring/weak.\n"
        "- 'broaden' — if the reel is too short/thin (too few clips, well under "
        "budget) and should include more real plays.\n"
        "- 'none' — if the reel is already a good length and quality; then verdict "
        "'accept'.\n\n"
        "Respond with STRICT JSON only, no prose or markdown fences, exactly:\n"
        '{"quality_score": <number 0..1>, "verdict": "accept"|"revise", '
        '"action": "none"|"require_confirmed_moments"|"raise_threshold"|"broaden", '
        '"issues": [<short strings>], "feedback": <one short sentence>}'
    )


def _parse_judge_json(text: str) -> dict:
    """Tolerantly parse the judge's JSON; return {} on any failure (never raises)."""
    try:
        match = _JUDGE_JSON_RE.search(text or "")
        if not match:
            return {}
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def judge_node(state: ReelState, settings=None) -> dict:
    """LLM-as-judge: critique the selected reel and, if weak, request one revision.

    Runs a real LLM call to score the reel against the recipe's acceptance
    criteria. When the critic rejects it (e.g. the reel is padded with
    unclassified windows) and we haven't already revised, it writes
    ``selection_overrides`` and routes back to ``select`` for one cheaper,
    stricter pass. Degrades to 'accept' if the LLM is unavailable or returns
    unparseable output, so the pipeline always progresses to the human gate.
    """
    settings = settings or get_settings()
    recipe: Recipe = state["recipe"]
    clips = state.get("selected_clips", [])
    budget = state.get("max_duration") or recipe.selection.max_duration
    proposed = state.get("proposed_duration", 0.0)
    revision_count = state.get("revision_count", 0)

    llm = get_llm_provider(settings)
    try:
        verdict = _parse_judge_json(llm.generate(_build_judge_prompt(recipe, clips, proposed, budget)))
    except Exception as exc:
        verdict = {"feedback": f"judge unavailable: {exc}"}

    verdict.setdefault("verdict", "accept")
    verdict.setdefault("action", "none")

    # Decide the *corrective* action from the reel's actual state, treating the
    # LLM's suggested action as advisory. The critical rule: never tighten into a
    # corner — if dropping unclassified clips would leave too few real plays, or
    # the reel is already too thin, BROADEN instead of shrinking further. This is
    # what stops the "revise -> 1 clip -> stuck" failure.
    overrides_now = dict(state.get("selection_overrides") or {})
    llm_action = verdict.get("action")
    n_clips = len(clips)
    n_confirmed = sum(1 for c in clips if c.moment_type)
    too_short = n_clips < _MIN_REEL_CLIPS or proposed < _MIN_REEL_SECONDS

    if verdict.get("verdict") != "revise":
        action = "none"
    elif too_short:
        action = "broaden"
    elif llm_action == "require_confirmed_moments":
        action = "require_confirmed_moments" if n_confirmed >= _MIN_REEL_CLIPS else "broaden"
    elif llm_action == "raise_threshold":
        action = "raise_threshold" if n_clips > _MIN_REEL_CLIPS else "broaden"
    elif llm_action == "broaden":
        action = "broaden"
    else:
        action = "none"

    # Avoid oscillation / no-op repeats across the (bounded) loop.
    if action == "require_confirmed_moments" and (overrides_now.get("require_moment") or overrides_now.get("broadened")):
        action = "none"
    if action == "broaden" and overrides_now.get("broadened"):
        action = "none"

    wants_revision = (
        action != "none"
        and revision_count < _MAX_REVISIONS
        # 'broaden' may fire on an empty reel (exactly when we want to relax);
        # the tightening actions need clips to act on.
        and (n_clips > 0 or action == "broaden")
    )

    note = (
        f"judge[{llm.name}]: verdict={verdict.get('verdict')} "
        f"score={verdict.get('quality_score')} llm_action={llm_action} "
        f"-> action={action} (clips={n_clips}, confirmed={n_confirmed}) "
        f"— {str(verdict.get('feedback', ''))[:140]}"
    )
    update: dict = {
        "judge_verdict": verdict,
        "judge_decision": "revise" if wants_revision else "accept",
        "notes": state.get("notes", []) + [note],
    }

    if wants_revision:
        overrides = overrides_now
        if action == "require_confirmed_moments":
            overrides["require_moment"] = True
        elif action == "raise_threshold":
            overrides["min_score"] = round(min(0.95, recipe.selection.min_score + 0.1), 2)
        elif action == "broaden":
            # Reel too thin: relax the filters so more *real* plays get in. The
            # scoreboard gate + subject check still keep dead-ball/off-subject
            # windows out, so lowering the bar admits live plays, not padding.
            overrides["require_moment"] = False
            overrides["broadened"] = True
            overrides["min_score"] = round(max(0.3, recipe.selection.min_score - 0.1), 2)
        update["selection_overrides"] = overrides
        update["revision_count"] = revision_count + 1
        update["notes"].append(
            f"judge: requesting revision #{revision_count + 1} ({action}) with overrides={overrides}"
        )

    return update


def route_after_judge(state: ReelState) -> str:
    """Conditional-edge router: loop back to ``select`` to revise, else proceed."""
    return "revise" if state.get("judge_decision") == "revise" else "accept"


def approve_clips_node(state: ReelState) -> dict:
    """Pass-through interrupt target — the graph pauses *before* this node runs."""
    return {}


def render_node(state: ReelState, settings=None) -> dict:
    """Render (or manifest-fallback) the approved clips, then gate before sharing."""
    settings = settings or get_settings()
    recipe: Recipe = state["recipe"]
    out_path = os.path.join(settings.output_dir, f"reel_{recipe.id}.mp4")
    result = render_reel(state.get("selected_clips", []), state.get("video_path"), recipe, out_path, settings)
    return {
        "output_path": result.output_path,
        "needs_human": True,
        "human_gate": GATE_APPROVE_SHARE,
        "notes": state.get("notes", [])
        + list(result.notes)
        + [f"render: output at {result.output_path}"],
    }


def summarize_node(state: ReelState, settings=None) -> dict:
    """Generate a short natural-language summary of the reel via the LLM provider."""
    settings = settings or get_settings()
    recipe: Recipe = state["recipe"]
    clips = state.get("selected_clips", [])
    lines = [
        f"- {c.moment_type or 'moment'} ({c.start:.1f}-{c.end:.1f}s, score {c.score:.2f})"
        for c in clips
    ]
    prompt = (
        f"Write a short, upbeat highlight-reel summary for the recipe '{recipe.name}' "
        f"({recipe.domain}). {len(clips)} clip(s) totaling "
        f"{state.get('proposed_duration', 0.0):.1f}s were selected:\n" + "\n".join(lines)
    )

    llm = get_llm_provider(settings)
    try:
        summary = llm.generate(prompt)
    except Exception as exc:
        summary = f"(summary unavailable: {exc})"

    return {
        "summary": summary,
        "notes": state.get("notes", []) + [f"summarize: generated with '{llm.name}' provider"],
    }


def approve_share_node(state: ReelState) -> dict:
    """Pass-through interrupt target — the graph pauses *before* this node runs."""
    return {}


def deliver_node(state: ReelState) -> dict:
    """Terminal node: mark the reel as delivered."""
    status = "shared" if state.get("shared") else "ready (not shared)"
    return {
        "notes": state.get("notes", []) + [f"deliver: reel {status}"],
    }


def route_after_select(state: ReelState) -> str:
    """Conditional-edge router: flag underfilled reels distinctly from ready ones."""
    return "underfilled" if state.get("human_gate") == GATE_UNDERFILLED else "approve"


def record_selection_feedback(recipe: Recipe, scored_clips: list[Clip], selected_clips: list[Clip], settings) -> None:
    """Best-effort: log kept vs. dropped clips to the memory store on human approval.

    Never raises — persistence failures must not affect the pipeline.
    """
    try:
        store = MemoryStore(settings.memory_path)
        selected_keys = {(c.start, c.end) for c in selected_clips}
        dropped = [c for c in scored_clips if (c.start, c.end) not in selected_keys]
        store.record_feedback(recipe.id, selected_clips, dropped)
    except Exception:
        pass
