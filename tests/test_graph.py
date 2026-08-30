"""Tests for the LangGraph integration layer: nodes, router, and run_pipeline.

Everything here is offline (mock providers, no network, no media libs). Nodes
are unit-tested directly against hand-built ``ReelState`` dicts for
determinism; the compiled graph is only exercised end-to-end through
``run_pipeline``.
"""

from __future__ import annotations

import pytest

from hypereel.config import get_settings
from hypereel.graph.build import build_graph, run_pipeline
from hypereel.graph import nodes as nodes_mod
from hypereel.graph.nodes import (
    approve_clips_node,
    approve_share_node,
    classify_node,
    deliver_node,
    ingest_node,
    judge_node,
    plan_node,
    propose_node,
    render_node,
    route_after_judge,
    route_after_select,
    select_node,
    summarize_node,
)
from hypereel.graph.state import (
    GATE_APPROVE_CLIPS,
    GATE_APPROVE_SHARE,
    GATE_UNDERFILLED,
    new_state,
)
from hypereel.models import Clip, SignalConfig
from hypereel.providers.factory import get_vision_provider
from hypereel.analyze.classifier import classify_candidates


def _base_state(basketball_recipe, source: str = "irrelevant_source.mp4", **overrides) -> dict:
    return new_state(source, basketball_recipe, **overrides)


# --------------------------------------------------------------------------- #
#  Individual nodes
# --------------------------------------------------------------------------- #


def test_plan_node_sets_strategy_and_signals(basketball_recipe):
    state = _base_state(basketball_recipe)
    update = plan_node(state)

    assert update["strategy"] == "event_based"
    assert set(update["active_signals"]) == {"audio_peak", "motion_intensity"}
    assert update["notes"]


def test_new_state_bakes_subject_description_into_recipe_copy(basketball_recipe):
    brief = "the team in black jerseys with red trim; #23 if legible"
    state = _base_state(basketball_recipe, subject_description=brief)

    # The run's recipe carries the brief...
    assert state["recipe"].subject_selector.description == brief
    # ...without mutating the caller's original recipe (deep copy).
    assert basketball_recipe.subject_selector.description is None
    assert state["recipe"] is not basketball_recipe


def test_new_state_blank_subject_description_is_a_noop(basketball_recipe):
    # Empty / whitespace-only brief must not trigger a copy or set a value.
    state = _base_state(basketball_recipe, subject_description="   ")
    assert state["recipe"] is basketball_recipe
    assert state["recipe"].subject_selector.description is None


def test_ingest_node_handles_missing_source(basketball_recipe):
    state = _base_state(basketball_recipe, source="nonexistent_local.mp4")
    update = ingest_node(state)

    assert update["video_path"] is None
    assert update["video_duration"] > 0  # fallback duration so the demo can proceed
    assert update["needs_human"] is True
    assert update["human_gate"]
    assert update["errors"]


def test_ingest_node_resolves_existing_local_file(tmp_path, basketball_recipe):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a real video, just needs to exist")

    state = _base_state(basketball_recipe, source=str(video))
    update = ingest_node(state)

    assert update["video_path"] == str(video)
    assert update["video_duration"] > 0  # probe fails -> fallback, but always > 0
    assert isinstance(update["has_commentary"], bool)
    assert not update.get("errors")


def test_propose_node_never_empty(basketball_recipe):
    state = _base_state(basketball_recipe, video_path=None)
    state["video_duration"] = 120.0
    update = propose_node(state)

    assert update["candidates"]
    assert update["notes"]


def _recipe_with_commentary_signal(basketball_recipe):
    recipe = basketball_recipe.model_copy(deep=True)
    recipe.signals.append(
        SignalConfig(type="transcript_keyword", role="proposer", weight=0.5, enabled_if="commentary_present")
    )
    return recipe


def test_propose_node_disables_conditional_signal_when_condition_false(basketball_recipe):
    recipe = _recipe_with_commentary_signal(basketball_recipe)
    state = _base_state(recipe, video_path=None)
    state["video_duration"] = 120.0
    state["has_commentary"] = False  # no commentary detected

    update = propose_node(state)

    assert "recipe" in update, "resolved recipe must be threaded back into state"
    assert "transcript_keyword" not in update["active_signals"], "commentary-gated signal must be off"
    assert "audio_peak" in update["active_signals"], "unconditional signals stay on"


def test_propose_node_keeps_conditional_signal_when_condition_true(basketball_recipe):
    recipe = _recipe_with_commentary_signal(basketball_recipe)
    state = _base_state(recipe, video_path=None)
    state["video_duration"] = 120.0
    state["has_commentary"] = True  # commentary detected -> signal activates

    update = propose_node(state)

    assert "transcript_keyword" in update["active_signals"]


def test_propose_node_leaves_recipe_unchanged_without_conditional_signals(basketball_recipe):
    # Neither shipped recipe uses enabled_if; the common path must not copy/alter it.
    state = _base_state(basketball_recipe, video_path=None)
    state["video_duration"] = 120.0
    update = propose_node(state)
    assert "recipe" not in update


def test_new_state_carries_max_candidates(basketball_recipe):
    assert _base_state(basketball_recipe)["max_candidates"] is None
    assert _base_state(basketball_recipe, max_candidates=5)["max_candidates"] == 5


def test_propose_node_caps_to_first_n_windows_chronologically(basketball_recipe):
    # A long fake video yields many windows; the cap should keep only the
    # earliest N, in start-time order — the quick-test dry-run path.
    state = _base_state(basketball_recipe, video_path=None, max_candidates=3)
    state["video_duration"] = 1200.0

    full = propose_node(_base_state(basketball_recipe, video_path=None))["candidates"]
    assert len(full) > 3, "need more than the cap to prove truncation"

    capped = propose_node(state)["candidates"]
    assert len(capped) == 3
    starts = [w.start for w in capped]
    assert starts == sorted(starts), "kept windows must stay chronological"
    assert max(starts) <= sorted(w.start for w in full)[2], "kept the EARLIEST windows"
    assert any("quick-test cap ON" in n for n in propose_node(state)["notes"])


def test_propose_node_cap_unset_or_zero_is_unbounded(basketball_recipe):
    base = _base_state(basketball_recipe, video_path=None)
    base["video_duration"] = 1200.0
    n_all = len(propose_node(base)["candidates"])

    zero = _base_state(basketball_recipe, video_path=None, max_candidates=0)
    zero["video_duration"] = 1200.0
    assert len(propose_node(zero)["candidates"]) == n_all
    assert not any("quick-test cap" in n for n in propose_node(zero)["notes"])


def test_classify_node_sets_mode_and_classifications(basketball_recipe, sample_candidates):
    state = _base_state(basketball_recipe)
    state["candidates"] = sample_candidates
    update = classify_node(state)

    assert update["mode"] == "mock"
    assert len(update["classifications"]) == len(sample_candidates)


def test_select_node_underfilled_when_no_candidates(basketball_recipe):
    state = _base_state(basketball_recipe)
    state["candidates"] = []
    state["classifications"] = []
    update = select_node(state)

    assert update["selected_clips"] == []
    assert update["human_gate"] == GATE_UNDERFILLED
    assert update["needs_human"] is True


def test_select_node_approve_when_candidates_fill_budget(basketball_recipe, sample_candidates):
    settings = get_settings()
    provider = get_vision_provider(settings)
    classifications = classify_candidates(sample_candidates, basketball_recipe, provider, settings)

    state = _base_state(basketball_recipe)
    state["candidates"] = sample_candidates
    state["classifications"] = classifications
    update = select_node(state)

    assert update["human_gate"] in (GATE_APPROVE_CLIPS, GATE_UNDERFILLED)
    assert update["proposed_duration"] >= 0.0
    assert isinstance(update["selected_clips"], list)
    assert isinstance(update["scored_clips"], list)


# --------------------------------------------------------------------------- #
#  LLM-as-judge critic + revision loop
# --------------------------------------------------------------------------- #


class _FakeLLM:
    """Stub LLM provider returning a canned string from generate()."""

    def __init__(self, text: str, name: str = "fake"):
        self._text = text
        self.name = name

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        return self._text


def _selected_state(basketball_recipe, clips, **overrides):
    state = _base_state(basketball_recipe, **overrides)
    state["selected_clips"] = clips
    state["proposed_duration"] = sum(c.duration for c in clips)
    return state


def test_judge_accepts_by_default_with_mock_llm(basketball_recipe):
    # The mock LLM returns non-JSON prose -> parse fails -> safe 'accept'.
    clips = [Clip(start=0.0, end=8.0, moment_type="made_basket", score=0.8, subject_present=True)]
    update = judge_node(_selected_state(basketball_recipe, clips))

    assert update["judge_decision"] == "accept"
    assert "selection_overrides" not in update  # nothing to revise
    assert route_after_judge({"judge_decision": update["judge_decision"]}) == "accept"


def test_judge_requests_revision_and_sets_require_moment(monkeypatch, basketball_recipe):
    verdict = (
        '{"quality_score": 0.3, "verdict": "revise", '
        '"action": "require_confirmed_moments", "issues": ["too much padding"], '
        '"feedback": "half the clips are unclassified"}'
    )
    monkeypatch.setattr(nodes_mod, "get_llm_provider", lambda settings: _FakeLLM(verdict))

    # A reel with enough confirmed plays to stand on (3) PLUS an unclassified
    # padding clip: tightening is the right call here (it won't gut the reel).
    clips = [
        Clip(start=0.0, end=8.0, moment_type="made_basket", score=0.8, subject_present=True),
        Clip(start=20.0, end=28.0, moment_type="three_pointer", score=0.8, subject_present=True),
        Clip(start=40.0, end=48.0, moment_type="layup_or_dunk", score=0.8, subject_present=True),
        Clip(start=60.0, end=68.0, moment_type=None, score=0.6, subject_present=True),  # padding
    ]
    update = judge_node(_selected_state(basketball_recipe, clips))

    assert update["judge_decision"] == "revise"
    assert update["selection_overrides"]["require_moment"] is True
    assert update["revision_count"] == 1
    assert route_after_judge(update) == "revise"


def test_judge_raise_threshold_action_bumps_min_score(monkeypatch, basketball_recipe):
    verdict = '{"verdict": "revise", "action": "raise_threshold", "feedback": "weak clips"}'
    monkeypatch.setattr(nodes_mod, "get_llm_provider", lambda settings: _FakeLLM(verdict))

    # A full-enough reel (>3 clips, well over the length floor) so raising the
    # threshold is safe — otherwise the judge would broaden instead.
    clips = [
        Clip(start=0.0, end=8.0, moment_type="made_basket", score=0.51, subject_present=True),
        Clip(start=20.0, end=28.0, moment_type="made_basket", score=0.52, subject_present=True),
        Clip(start=40.0, end=48.0, moment_type="made_basket", score=0.53, subject_present=True),
        Clip(start=60.0, end=68.0, moment_type="made_basket", score=0.54, subject_present=True),
    ]
    update = judge_node(_selected_state(basketball_recipe, clips))

    assert update["judge_decision"] == "revise"
    bumped = update["selection_overrides"]["min_score"]
    assert bumped == pytest.approx(basketball_recipe.selection.min_score + 0.1)


def test_judge_does_not_revise_past_max_revisions(monkeypatch, basketball_recipe):
    verdict = '{"verdict": "revise", "action": "require_confirmed_moments", "feedback": "still weak"}'
    monkeypatch.setattr(nodes_mod, "get_llm_provider", lambda settings: _FakeLLM(verdict))

    clips = [Clip(start=0.0, end=8.0, moment_type=None, score=0.6, subject_present=True)]
    state = _selected_state(basketball_recipe, clips)
    state["revision_count"] = 1  # already revised once -> must not loop again
    update = judge_node(state)

    assert update["judge_decision"] == "accept"
    assert "selection_overrides" not in update


def test_select_node_require_moment_override_drops_unclassified(basketball_recipe, sample_candidates):
    from hypereel.analyze.classifier import classify_candidates as _cc
    from hypereel.providers.factory import get_vision_provider as _gvp

    settings = get_settings()
    classifications = _cc(sample_candidates, basketball_recipe, _gvp(settings), settings)

    base = _base_state(basketball_recipe)
    base["candidates"] = sample_candidates
    base["classifications"] = classifications

    # Without the override, unclassified (moment_type=None) clips may be kept.
    base["selection_overrides"] = {"require_moment": True}
    update = select_node(base)

    assert all(c.moment_type for c in update["selected_clips"]), (
        "require_moment must drop every clip with no confirmed moment_type"
    )


def test_select_node_team_audience_override_disables_subject_filter(basketball_recipe):
    """A runtime audience='team' override must reach require_subject via the
    effective recipe, so an off-subject window is kept instead of dropped."""
    from hypereel.graph.nodes import _effective_recipe
    from hypereel.models import CandidateWindow, Classification

    # A high-scoring window whose subject is NOT present (would be dropped by the
    # individual-audience one-team filter).
    window = CandidateWindow(start=0.0, end=8.0, signal_scores={"audio_peak": 1.0, "motion_intensity": 1.0})
    classification = Classification(moment_type="made_basket", subject_present=False, confidence=1.0)

    # _effective_recipe bakes the override onto subject_selector.audience.
    assert _effective_recipe(basketball_recipe, {}, audience="team").subject_selector.audience == "team"
    # No override (or a matching one) returns the recipe unchanged (no copy).
    assert _effective_recipe(basketball_recipe, {}, audience=None) is basketball_recipe

    base = _base_state(basketball_recipe)
    base["candidates"] = [window]
    base["classifications"] = [classification]

    # Default (individual) audience: the off-subject window is filtered out.
    assert select_node(base)["selected_clips"] == []

    # team override: the subject gate is off, so the window survives.
    base_team = _base_state(basketball_recipe, audience="team")
    base_team["candidates"] = [window]
    base_team["classifications"] = [classification]
    assert select_node(base_team)["selected_clips"], "team override must keep off-subject windows"


def test_judge_broaden_action_relaxes_filters_when_too_short(monkeypatch, basketball_recipe):
    verdict = '{"verdict": "revise", "action": "broaden", "feedback": "only 1 clip, way under budget"}'
    monkeypatch.setattr(nodes_mod, "get_llm_provider", lambda settings: _FakeLLM(verdict))

    clips = [Clip(start=0.0, end=8.0, moment_type="made_basket", score=0.9, subject_present=True)]
    update = judge_node(_selected_state(basketball_recipe, clips))

    assert update["judge_decision"] == "revise"
    ov = update["selection_overrides"]
    assert ov["require_moment"] is False, "broaden must lift the confirmed-moment filter"
    assert ov["min_score"] == pytest.approx(basketball_recipe.selection.min_score - 0.1)
    assert route_after_judge(update) == "revise"


def test_judge_broaden_fires_even_on_empty_reel(monkeypatch, basketball_recipe):
    # 'broaden' is exactly the action for an empty/too-thin reel, so unlike the
    # tightening actions it must fire even when no clips were selected.
    verdict = '{"verdict": "revise", "action": "broaden", "feedback": "nothing selected"}'
    monkeypatch.setattr(nodes_mod, "get_llm_provider", lambda settings: _FakeLLM(verdict))

    update = judge_node(_selected_state(basketball_recipe, []))

    assert update["judge_decision"] == "revise"
    assert "selection_overrides" in update


def test_scoreboard_node_passthrough_without_signal(basketball_recipe):
    # basketball_recipe declares no 'scoreboard' signal -> node is a no-op.
    from hypereel.models import CandidateWindow

    state = _base_state(basketball_recipe)
    state["candidates"] = [CandidateWindow(start=0.0, end=8.0)]
    state["video_path"] = "irrelevant.mp4"
    assert nodes_mod.scoreboard_node(state) == {}


def test_scoreboard_node_noop_without_video(basketball_recipe):
    # Even with a scoreboard signal, no video_path means nothing to read.
    recipe = basketball_recipe.model_copy(deep=True)
    recipe.signals.append(SignalConfig(type="scoreboard", role="scorer", weight=0.0))
    state = _base_state(recipe)
    from hypereel.models import CandidateWindow
    state["candidates"] = [CandidateWindow(start=0.0, end=8.0)]
    state["video_path"] = None
    assert nodes_mod.scoreboard_node(state) == {}


def test_approve_gate_nodes_are_pure_passthroughs(basketball_recipe):
    state = _base_state(basketball_recipe)
    assert approve_clips_node(state) == {}
    assert approve_share_node(state) == {}


def test_render_node_falls_back_to_manifest(tmp_path, monkeypatch, basketball_recipe):
    monkeypatch.setenv("HYPEREEL_OUTPUT_DIR", str(tmp_path))
    clips = [
        Clip(start=0.0, end=5.0, moment_type="made_basket", score=0.9, subject_present=True),
    ]
    state = _base_state(basketball_recipe)
    state["selected_clips"] = clips
    state["video_path"] = None
    update = render_node(state)

    assert update["output_path"]
    assert update["output_path"].endswith(".manifest.json")
    assert update["human_gate"] == GATE_APPROVE_SHARE
    assert update["needs_human"] is True


def test_summarize_node_never_raises_and_sets_summary(basketball_recipe):
    state = _base_state(basketball_recipe)
    state["selected_clips"] = [
        Clip(start=0.0, end=5.0, moment_type="made_basket", score=0.9, subject_present=True),
    ]
    state["proposed_duration"] = 5.0
    update = summarize_node(state)

    assert update["summary"]
    assert "mock" in update["notes"][-1]


def test_deliver_node_notes_shared_state(basketball_recipe):
    state = _base_state(basketball_recipe)
    state["shared"] = True
    update = deliver_node(state)
    assert "shared" in update["notes"][-1]


def test_route_after_select(basketball_recipe):
    underfilled_state = _base_state(basketball_recipe, mode="mock")
    underfilled_state["human_gate"] = GATE_UNDERFILLED
    assert route_after_select(underfilled_state) == "underfilled"

    ready_state = _base_state(basketball_recipe, mode="mock")
    ready_state["human_gate"] = GATE_APPROVE_CLIPS
    assert route_after_select(ready_state) == "approve"


# --------------------------------------------------------------------------- #
#  Compiled graph / run_pipeline
# --------------------------------------------------------------------------- #


def test_build_graph_compiles():
    app = build_graph()
    assert hasattr(app, "invoke")
    assert hasattr(app, "get_state")


def test_run_pipeline_completes_end_to_end_in_mock_mode(tmp_path, monkeypatch, basketball_recipe):
    monkeypatch.setenv("HYPEREEL_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("HYPEREEL_MEMORY_PATH", str(tmp_path / "memory.json"))

    result = run_pipeline("nonexistent_local.mp4", basketball_recipe, auto_approve=True)

    assert result.recipe_id == basketball_recipe.id
    # The mock pipeline always finds *some* candidates; either they filled the
    # budget (non-empty clips) or the run is explicitly flagged as underfilled.
    has_clips = bool(result.clips)
    flagged_underfilled = any("underfilled" in n for n in result.notes)
    assert has_clips or flagged_underfilled
    assert result.output_path  # a manifest path is fine offline
    assert result.output_path.endswith(".manifest.json")
    assert result.summary  # the summarize node always writes a recap

    # When clips were produced, every clip must satisfy the recipe's invariants.
    sel = basketball_recipe.selection
    starts = [c.start for c in result.clips]
    assert starts == sorted(starts), "clips must be in ascending chronological order"
    for clip in result.clips:
        assert 0.0 <= clip.start < clip.end
        assert sel.min_clip <= clip.duration <= sel.max_clip, "clip length must be shaped to the budget window"
        assert clip.score >= sel.min_score
        # individual audience + a real subject_selector -> subject must be present
        assert clip.subject_present is True
    # No overlaps between adjacent clips.
    for a, b in zip(result.clips, result.clips[1:]):
        assert a.end <= b.start, "selected clips must not overlap"
    # Reported total duration matches the sum of the kept clips.
    assert result.total_duration == pytest.approx(sum(c.duration for c in result.clips))


def test_run_pipeline_hitl_manual_resume_through_both_gates(tmp_path, monkeypatch, basketball_recipe):
    """Drive the *real* compiled graph the way the UI does: run to the first
    ``interrupt_before`` gate, approve via ``update_state``, resume with
    ``invoke(None, ...)``, and repeat for the second gate."""
    monkeypatch.setenv("HYPEREEL_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("HYPEREEL_MEMORY_PATH", str(tmp_path / "memory.json"))

    app = build_graph()
    config = {"configurable": {"thread_id": "manual-resume"}}
    initial = new_state("nonexistent_local.mp4", basketball_recipe)

    # 1) Runs until it pauses before the first human gate (approve_clips).
    app.invoke(initial, config)
    snap = app.get_state(config)
    assert snap.next == ("approve_clips",), "graph must pause before render for clip approval"
    assert snap.values.get("output_path") is None, "nothing is rendered before gate 1 is approved"

    # 2) Human approves the clip list; resume runs render + summarize, then
    #    pauses before the second gate (approve_share).
    app.update_state(config, {"approved": True, "needs_human": False})
    app.invoke(None, config)
    snap = app.get_state(config)
    assert snap.next == ("approve_share",), "graph must pause before sharing"
    assert snap.values.get("output_path"), "render must have produced an output path by gate 2"
    assert snap.values.get("summary"), "summary must exist by gate 2"

    # 3) Human approves the share; resume runs deliver and reaches the end.
    app.update_state(config, {"shared": True, "needs_human": False})
    app.invoke(None, config)
    snap = app.get_state(config)
    assert not snap.next, "graph must reach END after both gates are approved"
    assert snap.values.get("shared") is True


def test_run_pipeline_quality_based_recipe_completes_end_to_end(tmp_path, monkeypatch):
    """The quality_based path (architecture walkthrough) must run through the
    same pipeline as the event_based flagship — proving the recipe generalizes."""
    from pathlib import Path

    from hypereel.recipe import load_recipe

    repo_root = Path(__file__).resolve().parents[1]
    recipe = load_recipe(repo_root / "recipes" / "architecture_walkthrough.yaml")

    monkeypatch.setenv("HYPEREEL_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("HYPEREEL_MEMORY_PATH", str(tmp_path / "memory.json"))

    result = run_pipeline("nonexistent_local.mp4", recipe, auto_approve=True)

    assert result.recipe_id == recipe.id
    assert result.output_path and result.output_path.endswith(".manifest.json")
    has_clips = bool(result.clips)
    flagged_underfilled = any("underfilled" in n for n in result.notes)
    assert has_clips or flagged_underfilled
    # Order matches whatever the recipe declared (this one uses best_first).
    if recipe.selection.ordering == "best_first":
        scores = [c.score for c in result.clips]
        assert scores == sorted(scores, reverse=True)
    else:
        starts = [c.start for c in result.clips]
        assert starts == sorted(starts)


def test_run_pipeline_judge_revises_once_then_converges(tmp_path, monkeypatch, basketball_recipe):
    """A judge that always votes 'revise' must still terminate: the loop is
    bounded by _MAX_REVISIONS, so the compiled graph selects -> judge -> select
    -> judge(accept) -> gates -> END without hanging."""
    monkeypatch.setenv("HYPEREEL_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("HYPEREEL_MEMORY_PATH", str(tmp_path / "memory.json"))

    verdict = '{"verdict": "revise", "action": "require_confirmed_moments", "feedback": "padding"}'
    monkeypatch.setattr(nodes_mod, "get_llm_provider", lambda settings: _FakeLLM(verdict))

    result = run_pipeline("nonexistent_local.mp4", basketball_recipe, auto_approve=True)

    assert result.recipe_id == basketball_recipe.id
    assert result.output_path  # completed through render
    # Exactly one revision was requested (bounded), and the override was applied.
    revision_notes = [n for n in result.notes if "requesting revision" in n]
    assert len(revision_notes) == 1
    assert any("judge overrides applied" in n for n in result.notes)


def test_run_pipeline_stops_at_first_gate_when_not_auto_approved(tmp_path, monkeypatch, basketball_recipe):
    monkeypatch.setenv("HYPEREEL_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("HYPEREEL_MEMORY_PATH", str(tmp_path / "memory.json"))

    result = run_pipeline(
        "nonexistent_local.mp4",
        basketball_recipe,
        auto_approve=False,
        thread_id="stop-at-gate",
    )

    assert result.recipe_id == basketball_recipe.id
    assert any("paused for human approval" in n for n in result.notes)
    # Nothing has been rendered yet — we stopped before approve_clips/render.
    assert result.output_path is None


def test_run_pipeline_never_raises_on_bad_settings_object(tmp_path, monkeypatch, basketball_recipe):
    # Nodes read settings via get_settings() internally, but keep the pipeline
    # offline/isolated regardless of the (broken) settings object passed in.
    monkeypatch.setenv("HYPEREEL_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("HYPEREEL_MEMORY_PATH", str(tmp_path / "memory.json"))

    # A settings object missing attributes should still degrade, not raise.
    class BrokenSettings:
        pass

    result = run_pipeline("nonexistent_local.mp4", basketball_recipe, settings=BrokenSettings())
    assert result.recipe_id == basketball_recipe.id
