"""Tests for hypereel.select.selector — offline, MockVisionProvider only."""

from __future__ import annotations

import pytest

from hypereel.analyze.classifier import classify_candidates
from hypereel.config import Settings
from hypereel.models import CandidateWindow, Classification, Clip, SelectionPolicy, SignalConfig
from hypereel.providers.mock import MockVisionProvider
from hypereel.select.selector import (
    _apply_scoreboard,
    _confirm_moment,
    _scoreboard_cfg,
    _shape_window,
    score_candidates,
    select_clips,
)
from hypereel.signals.scoreboard import KEY_LIVE, KEY_SCORE_CHANGE, KEY_SUBJECT_SCORE_CHANGE


def _classify(candidates, recipe):
    return classify_candidates(candidates, recipe, MockVisionProvider(), Settings())


def test_score_candidates_filters_by_min_score_and_subject_presence(sample_candidates, basketball_recipe):
    classifications = _classify(sample_candidates, basketball_recipe)

    clips = score_candidates(sample_candidates, classifications, basketball_recipe)

    assert clips, "expected at least one clip to clear min_score"
    for clip in clips:
        assert clip.score >= basketball_recipe.selection.min_score
        # individual audience + a real subject_selector -> subject must be present
        assert clip.subject_present is True

    # sorted by score desc
    scores = [c.score for c in clips]
    assert scores == sorted(scores, reverse=True)


def test_score_candidates_team_audience_does_not_require_subject(sample_candidates, basketball_recipe):
    team_recipe = basketball_recipe.model_copy(deep=True)
    team_recipe.subject_selector.audience = "team"

    classifications = _classify(sample_candidates, team_recipe)
    clips = score_candidates(sample_candidates, classifications, team_recipe)

    assert clips
    assert any(c.subject_present is False for c in clips)
    for clip in clips:
        assert clip.score >= team_recipe.selection.min_score


def test_score_candidates_empty_input_returns_empty(basketball_recipe):
    assert score_candidates([], [], basketball_recipe) == []


def test_score_candidates_never_emits_unclassified_motion(basketball_recipe):
    recipe = basketball_recipe.model_copy(deep=True)
    recipe.subject_selector.audience = "team"
    recipe.selection.min_score = 0
    window = CandidateWindow(start=5, end=10, signal_scores={"motion_intensity": 1})
    classification = Classification(
        moment_type=None, subject_present=False, confidence=1, reason="no event"
    )
    assert score_candidates([window], [classification], recipe) == []


def test_select_clips_respects_time_budget(sample_candidates, basketball_recipe):
    classifications = _classify(sample_candidates, basketball_recipe)
    scored = score_candidates(sample_candidates, classifications, basketball_recipe)

    budget = 20.0
    selected = select_clips(scored, basketball_recipe, max_duration=budget)

    total = sum(c.duration for c in selected)
    assert total <= budget
    assert selected  # at least one clip should fit


def test_select_clips_chronological_ordering_is_ascending(sample_candidates, basketball_recipe):
    classifications = _classify(sample_candidates, basketball_recipe)
    scored = score_candidates(sample_candidates, classifications, basketball_recipe)

    # Generous budget so more than one clip is retained, to actually exercise ordering.
    selected = select_clips(scored, basketball_recipe, max_duration=1000.0)

    assert len(selected) >= 2
    starts = [c.start for c in selected]
    assert starts == sorted(starts)


def test_select_clips_empty_input_returns_empty(basketball_recipe):
    assert select_clips([], basketball_recipe) == []


# --------------------------------------------------------------------------- #
#  select_clips edge cases — driven with hand-built clips for determinism.
# --------------------------------------------------------------------------- #


def _clip(start, end, score):
    return Clip(start=start, end=end, moment_type="made_basket", score=score, subject_present=True)


def test_select_clips_dedup_overlap_drops_overlapping_clip(basketball_recipe):
    recipe = basketball_recipe.model_copy(deep=True)
    recipe.selection.dedup_overlap = True
    # Two overlapping windows; the lower-scoring one must be dropped.
    high = _clip(0.0, 8.0, 0.95)
    overlapping_low = _clip(4.0, 12.0, 0.80)  # overlaps `high`
    disjoint = _clip(20.0, 28.0, 0.70)

    selected = select_clips([overlapping_low, high, disjoint], recipe, max_duration=1000.0)

    starts = {c.start for c in selected}
    assert 0.0 in starts and 20.0 in starts
    assert 4.0 not in starts, "overlapping lower-scored clip should be deduped out"


def test_select_clips_best_first_ordering_is_score_descending(basketball_recipe):
    recipe = basketball_recipe.model_copy(deep=True)
    recipe.selection.ordering = "best_first"
    recipe.selection.dedup_overlap = False

    clips = [_clip(0.0, 6.0, 0.6), _clip(20.0, 26.0, 0.9), _clip(40.0, 46.0, 0.75)]
    selected = select_clips(clips, recipe, max_duration=1000.0)

    scores = [c.score for c in selected]
    assert scores == sorted(scores, reverse=True)


def test_select_clips_skips_over_budget_clip_but_keeps_smaller_ones(basketball_recipe):
    recipe = basketball_recipe.model_copy(deep=True)
    recipe.selection.dedup_overlap = False
    # Highest score is also the longest and won't fit; a smaller, lower-scored
    # clip further down the order still should. (skip-not-break)
    too_long = _clip(0.0, 12.0, 0.99)   # 12s
    fits = _clip(20.0, 26.0, 0.70)      # 6s

    selected = select_clips([too_long, fits], recipe, max_duration=8.0)

    starts = {c.start for c in selected}
    assert 20.0 in starts, "a smaller clip must still be selected after skipping the over-budget one"
    assert 0.0 not in starts, "the over-budget top clip must be skipped, not selected"
    assert sum(c.duration for c in selected) <= 8.0


# --------------------------------------------------------------------------- #
#  Clip must never run past the end of the source video (live-render bug).
# --------------------------------------------------------------------------- #


def test_score_candidates_clamps_clip_end_to_video_duration(basketball_recipe):
    # A buzzer-beater at the very end of a 120s video: lead_out would overshoot.
    window = CandidateWindow(
        start=115.0, end=120.0, signal_scores={"audio_peak": 0.9, "motion_intensity": 0.9}
    )
    classifications = _classify([window], basketball_recipe)

    clips = score_candidates([window], classifications, basketball_recipe, video_duration=120.0)

    assert clips, "the end-of-game highlight should still be selected"
    for clip in clips:
        assert clip.end <= 120.0, "clip must not run past the end of the source video"


def test_shape_window_pulls_start_back_to_preserve_min_clip():
    sel = SelectionPolicy(min_clip=4.0, max_clip=12.0, lead_in=2.0, lead_out=3.0)
    # Clamping the end to the video length would otherwise leave a <min_clip clip.
    window = CandidateWindow(start=119.0, end=120.0)

    start, end = _shape_window(window, sel, video_duration=120.0)

    assert end <= 120.0
    assert end - start >= sel.min_clip, "start should be pulled back to keep min_clip length"


def test_shape_window_no_clamp_without_video_duration():
    sel = SelectionPolicy(min_clip=4.0, max_clip=12.0, lead_in=2.0, lead_out=3.0)
    window = CandidateWindow(start=115.0, end=120.0)
    # No duration known -> no clamp, existing behavior preserved.
    _, end = _shape_window(window, sel, video_duration=None)
    assert end > 120.0


# --------------------------------------------------------------------------- #
#  Scoreboard game-state gate (dead-ball crush / made-basket boost)
# --------------------------------------------------------------------------- #


def _scoreboard_recipe(basketball_recipe):
    """basketball_recipe + a declared scoreboard signal (enables the gate)."""
    recipe = basketball_recipe.model_copy(deep=True)
    recipe.signals.append(
        SignalConfig(
            type="scoreboard",
            role="scorer",
            weight=0.0,
            params={"dead_ball_factor": 0.15, "made_basket_boost": 0.2},
        )
    )
    return recipe


def test_scoreboard_cfg_absent_without_signal(basketball_recipe):
    assert _scoreboard_cfg(basketball_recipe) is None


def test_scoreboard_cfg_present_with_signal(basketball_recipe):
    cfg = _scoreboard_cfg(_scoreboard_recipe(basketball_recipe))
    assert cfg is not None
    assert cfg["dead_ball_factor"] == 0.15
    assert cfg["made_basket_boost"] == 0.2
    # basketball_recipe has a made_basket moment -> that's the confirm label.
    assert cfg["confirm_moment"] == "made_basket"


def test_apply_scoreboard_crushes_frozen_clock(basketball_recipe):
    cfg = _scoreboard_cfg(_scoreboard_recipe(basketball_recipe))
    window = CandidateWindow(start=0.0, end=8.0, signal_scores={KEY_LIVE: 0.0})
    assert _apply_scoreboard(0.8, window, "made_basket", cfg) == pytest.approx(0.8 * 0.15)


def test_apply_scoreboard_leaves_live_window_untouched(basketball_recipe):
    cfg = _scoreboard_cfg(_scoreboard_recipe(basketball_recipe))
    window = CandidateWindow(start=0.0, end=8.0, signal_scores={KEY_LIVE: 1.0})
    assert _apply_scoreboard(0.8, window, "block", cfg) == pytest.approx(0.8)


def test_apply_scoreboard_boosts_confirmed_made_basket(basketball_recipe):
    cfg = _scoreboard_cfg(_scoreboard_recipe(basketball_recipe))
    window = CandidateWindow(start=0.0, end=8.0, signal_scores={KEY_LIVE: 1.0, KEY_SCORE_CHANGE: 1.0})
    assert _apply_scoreboard(0.6, window, "made_basket", cfg) == pytest.approx(0.8)
    # A non-scoring play (a block) near a score change is NOT boosted.
    assert _apply_scoreboard(0.6, window, "block", cfg) == pytest.approx(0.6)


def test_apply_scoreboard_noop_when_unread_or_no_cfg(basketball_recipe):
    window = CandidateWindow(start=0.0, end=8.0)  # no scoreboard keys read
    cfg = _scoreboard_cfg(_scoreboard_recipe(basketball_recipe))
    assert _apply_scoreboard(0.7, window, "made_basket", cfg) == pytest.approx(0.7)  # unread -> untouched
    assert _apply_scoreboard(0.7, window, "made_basket", None) == pytest.approx(0.7)  # no signal -> untouched


def test_confirm_moment_upgrades_unnamed_make_on_score_change(basketball_recipe):
    """Score change + subject present + vision returned null -> confirmed make."""
    cfg = _scoreboard_cfg(_scoreboard_recipe(basketball_recipe))
    window = CandidateWindow(start=0.0, end=8.0, signal_scores={KEY_SCORE_CHANGE: 1.0})
    unsure = Classification(moment_type=None, subject_present=True, confidence=0.4)
    assert _confirm_moment(window, unsure, cfg) == "made_basket"


def test_confirm_moment_credits_only_the_subject_team(basketball_recipe):
    """With per-row attribution, only the subject team's score change confirms a make."""
    cfg = _scoreboard_cfg(_scoreboard_recipe(basketball_recipe))
    unsure = Classification(moment_type=None, subject_present=True, confidence=0.4)
    # Subject team scored -> confirmed make.
    ours = CandidateWindow(
        start=0.0, end=8.0,
        signal_scores={KEY_SCORE_CHANGE: 1.0, KEY_SUBJECT_SCORE_CHANGE: 1.0},
    )
    assert _confirm_moment(ours, unsure, cfg) == "made_basket"
    # A score changed but it was the OPPONENT's box -> NOT credited to the subject.
    theirs = CandidateWindow(
        start=0.0, end=8.0,
        signal_scores={KEY_SCORE_CHANGE: 1.0, KEY_SUBJECT_SCORE_CHANGE: 0.0},
    )
    assert _confirm_moment(theirs, unsure, cfg) is None


def test_confirm_moment_does_not_upgrade_without_evidence(basketball_recipe):
    cfg = _scoreboard_cfg(_scoreboard_recipe(basketball_recipe))
    unsure = Classification(moment_type=None, subject_present=True, confidence=0.4)
    # No score change -> no upgrade.
    assert _confirm_moment(CandidateWindow(start=0.0, end=8.0), unsure, cfg) is None
    # Score change but subject absent -> no upgrade (don't credit an opponent make).
    absent = Classification(moment_type=None, subject_present=False, confidence=0.4)
    w = CandidateWindow(start=0.0, end=8.0, signal_scores={KEY_SCORE_CHANGE: 1.0})
    assert _confirm_moment(w, absent, cfg) is None
    # Vision already named a play -> that label stands, no override.
    named = Classification(moment_type="block", subject_present=True, confidence=0.8)
    assert _confirm_moment(w, named, cfg) == "block"
    # No scoreboard cfg -> passthrough.
    assert _confirm_moment(w, unsure, None) is None


def test_score_candidates_drops_dead_ball_window(basketball_recipe):
    """A high-signal window on a frozen clock is filtered out below min_score."""
    recipe = _scoreboard_recipe(basketball_recipe)
    # Two identical strong candidates; one is flagged dead-ball by the scoreboard.
    live = CandidateWindow(start=0.0, end=8.0, signal_scores={"audio_peak": 0.9, "motion_intensity": 0.9, KEY_LIVE: 1.0})
    dead = CandidateWindow(start=20.0, end=28.0, signal_scores={"audio_peak": 0.9, "motion_intensity": 0.9, KEY_LIVE: 0.0})
    strong = Classification(moment_type="made_basket", subject_present=True, confidence=0.9)
    clips = score_candidates([live, dead], [strong, strong], recipe)

    kept_starts = {c.start for c in clips}
    assert 0.0 in kept_starts, "the live window should survive"
    assert 20.0 not in kept_starts, "the dead-ball window must be crushed below min_score"
