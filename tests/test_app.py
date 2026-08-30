"""Tests for the Streamlit app's provider-mode indicator.

Only the non-Streamlit helper is exercised — importing ``hypereel.app`` does
sys.path surgery but pulls in no ``streamlit`` (that import is lazy, inside
``main``), so this stays offline. The autouse ``_force_mock_providers`` fixture
pins both providers to mock, so the helper must report mock/mock.
"""

from __future__ import annotations

from pathlib import Path

import hypereel.app as app
from hypereel.recipe import load_recipe

_RECIPES = Path(__file__).resolve().parents[1] / "recipes"
_GENERIC = load_recipe(_RECIPES / "basketball_generic.yaml")
_QUALITY = load_recipe(_RECIPES / "architecture_walkthrough.yaml")


def test_provider_mode_reports_mock_when_forced(monkeypatch):
    # conftest's autouse fixture already forces mock; be explicit anyway.
    monkeypatch.setenv("HYPEREEL_VISION_PROVIDER", "mock")
    monkeypatch.setenv("HYPEREEL_LLM_PROVIDER", "mock")
    assert app._provider_mode() == ("mock", "mock")


def test_provider_mode_never_raises(monkeypatch):
    # A bogus provider name must degrade to mock, not blow up the sidebar.
    monkeypatch.setenv("HYPEREEL_VISION_PROVIDER", "not_a_real_provider")
    monkeypatch.setenv("HYPEREEL_LLM_PROVIDER", "mock")
    vision, llm = app._provider_mode()
    assert (vision, llm) == ("mock", "mock")


# --------------------------------------------------------------------------- #
#  "Under the hood" pipeline diagram — pure HTML helpers, no Streamlit needed.
# --------------------------------------------------------------------------- #


def test_pipeline_node_order_matches_graph_nodes():
    # The diagram must map every real graph node exactly once, in edge order.
    flat = [key for _, _, nodes in app._PIPELINE for (key, *_rest) in nodes]
    assert flat == app._NODE_ORDER
    assert app._GATES.issubset(set(app._NODE_ORDER))


def test_every_node_declares_a_known_tool_kind():
    # Tool-calling is a rubric item: every node must name its tool/model kind.
    for _label, _accent, nodes in app._PIPELINE:
        for key, name, desc, tool_label, tool_kind in nodes:
            assert tool_kind in app._TOOL_KIND, f"{key}: unknown tool kind {tool_kind}"
            assert tool_label, f"{key}: empty tool label"
    # The model calls (vision + text LLM) must be represented — they're the
    # tool-calling the reviewers look for.
    kinds = {key: kind for _l, _a, nodes in app._PIPELINE for (key, _n, _d, _tl, kind) in nodes}
    assert kinds["classify"] == "model" and kinds["judge"] == "model" and kinds["summarize"] == "model"
    assert kinds["ingest"] == "tool" and kinds["render"] == "tool"


def test_pipeline_html_shows_tool_chips():
    html = app._pipeline_html(set(), "classify")
    assert "vision LLM" in html and "yt-dlp" in html and "ffmpeg" in html


def test_stage_progress_lights_the_right_gate():
    assert app._stage_progress("input") == (set(), None)
    done, active = app._stage_progress("gate1")
    assert active == "approve_clips" and "judge" in done and "render" not in done
    done, active = app._stage_progress("gate2")
    assert active == "approve_share" and "render" in done
    done, active = app._stage_progress("done")
    assert active is None and done == set(app._NODE_ORDER)


def test_pipeline_html_branch_only_with_state():
    # Structural view (no state) omits the routing branch...
    assert '<div class="hr-branch">' not in app._pipeline_html(set(), None)
    # ...and with state the taken branches light up.
    state = {"revision_count": 1, "judge_decision": "accept", "human_gate": "approve_clips"}
    html = app._pipeline_html(*app._stage_progress("gate1"), state)
    assert '<span class="hr-br revise on">' in html
    assert '<span class="hr-br accept on">' in html


def test_pipeline_html_branch_off_before_judge_runs():
    state = {"revision_count": 0, "judge_decision": ""}
    html = app._pipeline_html(set(), "classify", state)
    assert '<span class="hr-br revise off">' in html
    assert '<span class="hr-br accept off">' in html


def test_decisions_html_summarizes_the_run():
    state = {
        "active_signals": ["audio_peak", "vision_classify"],
        "candidates": [0] * 40, "scored_clips": [0] * 18, "selected_clips": [0] * 9,
        "proposed_duration": 132.0, "judge_verdict": {"score": 0.86},
        "judge_decision": "accept", "revision_count": 1, "mode": "live",
    }
    state["classifications"] = [0] * 40  # one vision call per window
    state["summary"] = "great reel"
    html = app._decisions_html(state)
    for token in ("FUNNEL", "40→18→9", "JUDGE", "0.86", "132s", "LIVE", "TOOL CALLS"):
        assert token in html
    # vision ×40, text LLM = judge (1 + 1 revision) + summary (1) = 3
    assert "👁 40" in html and "🧠 3" in html
    # Never raises on an empty/partial state.
    assert app._decisions_html({}) and app._decisions_html(None)


# --------------------------------------------------------------------------- #
#  Per-node timing chips + the "why 0 clips?" diagnosis funnel.
# --------------------------------------------------------------------------- #


def test_fmt_secs_switches_to_minutes():
    assert app._fmt_secs(2.44) == "2.4s"
    assert app._fmt_secs(0) == "0.0s"
    assert app._fmt_secs(185) == "3m 05s"


def test_pipeline_html_shows_timing_chip_for_completed_nodes():
    html = app._pipeline_html({"plan", "ingest"}, "propose", None, {"ingest": 92.0, "plan": 0.3})
    assert "⏱ 1m 32s" in html   # ingest, formatted as minutes
    assert "⏱ 0.3s" in html     # plan
    # A node with no timing gets no chip.
    assert html.count('class="el"') == 2


class _Cls:
    """Minimal stand-in for a Classification (attribute access path)."""

    def __init__(self, subject_present=False, moment_type=None, confidence=0.0, reason=""):
        self.subject_present = subject_present
        self.moment_type = moment_type
        self.confidence = confidence
        self.reason = reason


class _Sel:
    max_duration = 300.0
    min_score = 0.5
    min_clip = 4.0


class _SS:
    audience = "individual"
    type = "team_color"


class _Recipe:
    selection = _Sel()
    subject_selector = _SS()


def _ebe_like_state(**over):
    """A run state shaped like the EBE 0-clip run: 207 windows, 0 subject present."""
    state = {
        "recipe": _Recipe(),
        "candidates": [0] * 207,
        "classifications": [_Cls(subject_present=False) for _ in range(207)],
        "scored_clips": [],
        "selected_clips": [],
        "proposed_duration": 0.0,
        "human_gate": "underfilled",
        "max_duration": 5.0,
    }
    state.update(over)
    return state


def test_funnel_pins_the_subject_gate_as_the_collapse_point():
    f = app._funnel(_ebe_like_state())
    assert f["subject_on"] is True
    assert (f["n_cand"], f["n_cls"], f["n_present"], f["n_scored"], f["n_sel"]) == (207, 207, 0, 0, 0)


def test_diagnosis_blames_the_subject_filter_when_nothing_is_present():
    head, steps = app._diagnosis(_ebe_like_state())
    assert "did not confirm your subject" in head
    joined = " ".join(steps).lower()
    # Actionable, control-specific guidance — not "redo everything".
    assert "team" in joined and ("hd" in joined or "color" in joined)
    html = app._diagnosis_html(_ebe_like_state())
    assert '<div class="hr-fstep drop">' in html  # the subject-seen box is flagged red
    assert "subject seen" in html and "207" in html


def test_diagnosis_flags_a_tiny_budget_when_clips_qualified_but_none_fit():
    # Clips passed scoring, but the 5s override can't hold one 4s+lead clip.
    state = _ebe_like_state(
        classifications=[_Cls(subject_present=True, moment_type="made_basket") for _ in range(3)],
        scored_clips=[1, 2, 3], selected_clips=[],
    )
    head, steps = app._diagnosis(state)
    assert "budget" in head.lower()
    assert any("Max duration override" in s for s in steps)


def test_diagnosis_is_none_for_a_healthy_reel():
    state = _ebe_like_state(
        classifications=[_Cls(subject_present=True) for _ in range(10)],
        scored_clips=[1] * 8, selected_clips=[1] * 6, proposed_duration=260.0, max_duration=None,
        human_gate="approve_clips",
    )
    assert app._diagnosis(state) is None
    assert app._diagnosis_html(state) == ""


_NEBIUS_402 = (
    "nebius error: Error code: 402 - {'detail': 'Payment Required: You have "
    "exhausted your budget. Please add funds to continue using the API.'}"
)


def test_is_provider_error_detects_api_failures():
    assert app._is_provider_error(_NEBIUS_402) is True
    assert app._is_provider_error("groq error: 401 Unauthorized") is True
    assert app._is_provider_error("classification error") is True
    assert app._is_provider_error("made basket, subject in frame") is False
    assert app._is_provider_error("") is False
    assert app._is_provider_error(None) is False


def test_trim_collapses_and_caps():
    assert app._trim("  a   b\n c ") == "a b c"
    long = "x" * 300
    out = app._trim(long, 50)
    assert len(out) == 50 and out.endswith("…")


def test_funnel_counts_provider_errors():
    state = _ebe_like_state(
        classifications=[_Cls(subject_present=False, reason=_NEBIUS_402) for _ in range(207)]
    )
    f = app._funnel(state)
    assert f["n_errors"] == 207
    assert "402" in f["error_reason"]


def test_diagnosis_blames_provider_not_prompt_when_windows_errored():
    # Every window is an API error — the true cause is the provider, and it must
    # win over the "subject filter dropped everything" story.
    state = _ebe_like_state(
        classifications=[_Cls(subject_present=False, reason=_NEBIUS_402) for _ in range(207)]
    )
    head, steps = app._diagnosis(state)
    assert "provider" in head.lower()
    assert "did not confirm your subject" not in head  # not the prompt story
    joined = " ".join(steps).lower()
    assert "groq" in joined or "gemini" in joined       # points at the free fix
    assert "402" in joined or "payment" in joined        # surfaces the real error


def test_diagnosis_provider_error_survives_team_override():
    # Even with the subject filter OFF (audience=team → n_present unused), an
    # all-errored run must still be blamed on the provider.
    ss = _SS()
    ss.audience = "team"
    recipe = _Recipe()
    recipe.subject_selector = ss
    state = _ebe_like_state(
        recipe=recipe,
        classifications=[_Cls(reason=_NEBIUS_402) for _ in range(207)],
    )
    head, _ = app._diagnosis(state)
    assert "provider" in head.lower()


def test_diagnosis_ignores_a_few_stray_errors_on_a_healthy_reel():
    # A couple of transient errors amid a real reel must NOT hijack the diagnosis.
    cls = [_Cls(subject_present=True) for _ in range(10)] + [_Cls(reason=_NEBIUS_402)]
    state = _ebe_like_state(
        classifications=cls, scored_clips=[1] * 8, selected_clips=[1] * 6,
        proposed_duration=260.0, max_duration=None, human_gate="approve_clips",
    )
    assert app._diagnosis(state) is None


def test_diagnosis_handles_dict_classifications_and_empty_state():
    # Classifications may arrive as plain dicts — the accessor must handle both.
    state = _ebe_like_state(classifications=[{"subject_present": False} for _ in range(207)])
    assert app._funnel(state)["n_present"] == 0
    # Never raises on empty / None.
    assert app._funnel({}) and app._funnel(None) is not None


# --------------------------------------------------------------------------- #
#  Pre-flight prompt checker — validate the form before the (long) run.
# --------------------------------------------------------------------------- #

_GOOD_BRIEF = "Highlights for the BLACK and BLUE team; the opponent wears white. Use HD if available."


def _levels(checks):
    return {lvl for lvl, _ in checks}


def test_recipe_needs_subject_honors_audience_override():
    assert app._recipe_needs_subject(_GENERIC) is True          # individual + team_color
    assert app._recipe_needs_subject(_GENERIC, "team") is False  # override turns filter off
    assert app._recipe_needs_subject(_QUALITY) is False          # type == none
    assert app._recipe_needs_subject(None) is False


def test_preflight_blocks_when_generic_recipe_has_no_brief():
    checks = app._preflight(_GENERIC, "https://youtu.be/x", None, None, "nebius", "nebius")
    errors = [m for lvl, m in checks if lvl == "error"]
    assert errors and "needs a subject description" in errors[0]


def test_preflight_team_override_drops_the_subject_requirement():
    # With audience=team the one-team filter is off, so an empty brief is allowed.
    checks = app._preflight(_GENERIC, "https://youtu.be/x", None, None, "nebius", "nebius", "team")
    assert "error" not in _levels(checks)


def test_preflight_clean_run_has_no_blockers():
    checks = app._preflight(_GENERIC, "https://youtu.be/x", _GOOD_BRIEF, None, "nebius", "nebius")
    assert "error" not in _levels(checks)
    # A fully clean brief shouldn't warn about color/opponent/quality.
    warns = " ".join(m for lvl, m in checks if lvl == "warn").lower()
    assert "color" not in warns and "opponent" not in warns and "hd" not in warns


def test_preflight_warns_on_number_only_brief_and_missing_quality():
    checks = app._preflight(_GENERIC, "https://youtu.be/x", "capture player #23 and #4", None, "nebius", "nebius")
    warns = " ".join(m for lvl, m in checks if lvl == "warn").lower()
    assert "color" in warns          # no jersey color mentioned
    assert "hd" in warns             # no quality preference
    assert "error" not in _levels(checks)  # warnings don't block


def test_preflight_blocks_on_empty_source_and_tiny_budget():
    checks = app._preflight(_GENERIC, "  ", _GOOD_BRIEF, 5.0, "nebius", "nebius")
    errors = " ".join(m for lvl, m in checks if lvl == "error").lower()
    assert "source is empty" in errors
    assert "too small" in errors     # 5s < 2*min_clip

    # A recipe that failed to load is a single blocking error.
    assert app._preflight(None, "x", None, None, "mock", "mock") == [
        ("error", "Recipe failed to load — pick another recipe.")
    ]


def test_preflight_warns_when_vision_is_mock():
    checks = app._preflight(_GENERIC, "https://youtu.be/x", _GOOD_BRIEF, None, "mock", "mock")
    warns = " ".join(m for lvl, m in checks if lvl == "warn").lower()
    assert "mock" in warns
    assert "error" not in _levels(checks)  # mock is a warning, not a blocker


# --------------------------------------------------------------------------- #
#  Live provider health ping — one real call to catch a 402/bad key up front.
# --------------------------------------------------------------------------- #


class _FakeSt:
    """Minimal Streamlit stand-in exposing just ``session_state`` (a dict)."""

    def __init__(self):
        self.session_state = {}


def test_provider_health_skips_the_live_call_for_mock():
    # Both mock → healthy without ever touching the network/factory.
    ok, msg = app._provider_health("mock", "mock")
    assert ok is True and "mock" in msg.lower()


def test_humanize_provider_error_points_to_free_providers_on_budget():
    human = app._humanize_provider_error(_NEBIUS_402)
    low = human.lower()
    assert "groq" in low and "gemini" in low       # the free fix
    assert "budget" in low or "quota" in low


def test_humanize_provider_error_flags_auth():
    human = app._humanize_provider_error("groq error: 401 Unauthorized: invalid api key")
    assert "authentication" in human.lower()


def test_looks_like_budget_error():
    assert app._looks_like_budget_error(_NEBIUS_402) is True
    assert app._looks_like_budget_error("500 internal server error") is False


def test_cached_provider_health_pings_once_per_pair(monkeypatch):
    calls = []

    def fake_health(v, l):
        calls.append((v, l))
        return False, "nebius error: 402 Payment Required"

    monkeypatch.setattr(app, "_provider_health", fake_health)
    st = _FakeSt()

    ok1, msg1 = app._cached_provider_health(st, "nebius", "nebius")
    ok2, msg2 = app._cached_provider_health(st, "nebius", "nebius")
    assert (ok1, msg1) == (ok2, msg2) == (False, "nebius error: 402 Payment Required")
    assert calls == [("nebius", "nebius")]              # cached — pinged once

    # Changing the provider pair re-pings.
    app._cached_provider_health(st, "groq", "groq")
    assert calls == [("nebius", "nebius"), ("groq", "groq")]

    # Clearing the cache (the "Re-check" button) re-pings.
    st.session_state.pop("provider_health", None)
    app._cached_provider_health(st, "groq", "groq")
    assert calls[-1] == ("groq", "groq") and len(calls) == 3
