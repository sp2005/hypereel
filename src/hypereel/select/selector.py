"""Score candidate windows and pick the final, budget-fitted set of clips.

Two-stage split mirrors the recipe's signal roles: :func:`score_candidates`
blends the cheap proposer signals with the vision model's confidence into a
single 0..1 score and shapes each window into a clip; :func:`select_clips`
then greedily fills the time budget and orders the result the way the
recipe's ``selection.ordering`` asks for. Both are pure and never raise —
selection is a place where the reel must always finish, even on a picky
recipe or a mostly-empty candidate list.
"""

from __future__ import annotations

from ..models import CandidateWindow, Classification, Clip, Recipe, SelectionPolicy
from ..signals.scoreboard import KEY_LIVE, KEY_SCORE_CHANGE, KEY_SUBJECT_SCORE_CHANGE

# moment_type name fragments that a scoreboard *score change* corroborates.
_SCORING_MOMENT_FRAGMENTS = ("basket", "three", "layup", "dunk", "jumper", "score", "field_goal")


def _scoreboard_cfg(recipe: Recipe) -> dict | None:
    """Return the scoreboard gate config if the recipe declares the signal, else None.

    Keeps the gate strictly opt-in and recipe-driven: recipes without a
    ``scoreboard`` signal (every non-sports recipe, and the test fixtures) get
    ``None`` and the scoring path is completely unchanged.
    """
    for sig in recipe.signals:
        if sig.type == "scoreboard" and sig.enabled:
            p = sig.params or {}
            return {
                "dead_ball_factor": float(p.get("dead_ball_factor", 0.15)),
                "made_basket_boost": float(p.get("made_basket_boost", 0.15)),
                # Which recipe moment_type a scoreboard-confirmed make is labelled
                # as. The scoreboard is the *objective* "a basket was scored"
                # signal, so when the vision model is too unsure to name a play
                # (common at low resolution) we still credit the make here.
                "confirm_moment": _confirm_moment_name(recipe),
            }
    return None


def _confirm_moment_name(recipe: Recipe) -> str | None:
    """The moment_type a scoreboard score-change should be labelled as, or None.

    Prefers an exact ``made_basket``; otherwise the first moment whose name reads
    like a scoring play. None means the recipe has no scoring moment to credit,
    so no upgrade happens.
    """
    names = [m.name for m in recipe.moment_types]
    if "made_basket" in names:
        return "made_basket"
    for n in names:
        if any(frag in n.lower() for frag in _SCORING_MOMENT_FRAGMENTS):
            return n
    return None


def _subject_scored(window: CandidateWindow) -> bool:
    """Did the SUBJECT team score in/near this window?

    Prefers the per-row subject signal when the scoreboard could attribute the
    change to a specific team's box (so an opponent basket is NOT counted). Falls
    back to the team-agnostic "some score changed" signal only when no per-row
    attribution is available (e.g. a single-score-box overlay).
    """
    sb = window.signal_scores
    subj = sb.get(KEY_SUBJECT_SCORE_CHANGE)
    if subj is not None:
        return subj >= 0.5
    return sb.get(KEY_SCORE_CHANGE, 0.0) >= 0.5


def _confirm_moment(
    window: CandidateWindow, classification: Classification, cfg: dict | None
) -> str | None:
    """Effective moment_type, upgraded from a scoreboard-confirmed make.

    When the vision model returned no moment_type but the scoreboard shows the
    *subject team's* score change in/near the window AND the subject is present,
    we trust the scoreboard: the make happened even if the model wouldn't commit
    to naming it. Otherwise the vision label stands.
    """
    if cfg is None or not cfg.get("confirm_moment"):
        return classification.moment_type
    if (
        classification.moment_type is None
        and classification.subject_present
        and _subject_scored(window)
    ):
        return cfg["confirm_moment"]
    return classification.moment_type


def _apply_scoreboard(
    score: float, window: CandidateWindow, moment_type: str | None, cfg: dict | None
) -> float:
    """Gate/boost a blended score with scoreboard game-state annotations.

    Frozen-clock (dead-ball / timeout) windows are crushed toward zero so they
    fall below ``min_score``; a confirmed score change boosts a scoring-type
    moment (``moment_type`` is the *effective* label, so a scoreboard-confirmed
    make gets the boost too). Windows the scoreboard couldn't read are untouched.
    """
    if cfg is None:
        return score
    sb = window.signal_scores
    live = sb.get(KEY_LIVE)
    if live is not None and live < 0.5:
        score *= cfg["dead_ball_factor"]
    if _subject_scored(window):
        mt = (moment_type or "").lower()
        if any(frag in mt for frag in _SCORING_MOMENT_FRAGMENTS):
            score = min(1.0, score + cfg["made_basket_boost"])
    return max(0.0, min(1.0, score))


def _combined_score(window: CandidateWindow, classification: Classification, recipe: Recipe) -> float:
    """Weighted blend of proposer signal_scores and scorer confidence, clamped to 0..1."""
    proposer_sigs = recipe.proposer_signals()
    scorer_sigs = recipe.scorer_signals()
    proposer_weight = sum(s.weight for s in proposer_sigs)
    scorer_weight = sum(s.weight for s in scorer_sigs)
    total_weight = proposer_weight + scorer_weight

    if total_weight <= 0:
        return max(0.0, min(1.0, classification.confidence))

    proposer_part = sum(window.signal_scores.get(s.type, 0.0) * s.weight for s in proposer_sigs)
    scorer_part = classification.confidence * scorer_weight
    score = (proposer_part + scorer_part) / total_weight
    return max(0.0, min(1.0, score))


def _shape_window(
    window: CandidateWindow,
    selection: SelectionPolicy,
    video_duration: float | None = None,
) -> tuple[float, float]:
    """Apply lead-in/lead-out, then clamp the resulting length to [min_clip, max_clip].

    The start (padded by lead_in, clamped >= 0) is treated as the anchor; if the
    padded window is too long or too short the *end* is adjusted, so a clip
    always keeps its lead-in.

    Finally, when the source ``video_duration`` is known, the clip is clamped to
    never run past the end of the video (a lead-out on an end-of-game
    buzzer-beater would otherwise point past the last frame — harmless on the
    manifest path, but it makes the live moviepy render raise and fall the whole
    reel back to a manifest). If clamping the end would make the clip shorter
    than ``min_clip``, the start is pulled back to preserve the length where the
    footage allows.
    """
    start = max(0.0, window.start - selection.lead_in)
    end = window.end + selection.lead_out
    length = end - start
    if length < selection.min_clip:
        end = start + selection.min_clip
    elif length > selection.max_clip:
        end = start + selection.max_clip

    if video_duration is not None and video_duration > 0 and end > video_duration:
        end = video_duration
        if end - start < selection.min_clip:
            start = max(0.0, end - selection.min_clip)
    return start, end


def score_candidates(
    candidates: list[CandidateWindow],
    classifications: list[Classification],
    recipe: Recipe,
    video_duration: float | None = None,
) -> list[Clip]:
    """Score+shape each candidate into a :class:`Clip`, filtered and sorted.

    A clip survives if its blended score clears ``recipe.selection.min_score``
    and, for an individual audience with a real subject selector, the vision
    model reported the subject present. Team audiences (and recipes with no
    subject selector) skip that second check. Result is sorted by score desc.

    ``video_duration`` (when known) is passed to :func:`_shape_window` so no
    clip runs past the end of the source video.
    """
    require_subject = (
        recipe.subject_selector.audience == "individual" and recipe.subject_selector.type != "none"
    )
    selection = recipe.selection
    sb_cfg = _scoreboard_cfg(recipe)

    clips: list[Clip] = []
    for window, classification in zip(candidates, classifications):
        try:
            # Effective moment_type may be upgraded from a scoreboard-confirmed
            # make when the vision model was too unsure to name the play.
            moment_type = _confirm_moment(window, classification, sb_cfg)
            # A high motion score is not semantic evidence. Never emit an
            # unclassified clip unless a configured grounding signal upgraded it.
            if moment_type is None:
                continue
            score = _combined_score(window, classification, recipe)
            # Ground the score in game state (dead-ball crush / made-basket boost)
            # *before* the min_score gate, so timeouts/sidelines are filtered out.
            score = _apply_scoreboard(score, window, moment_type, sb_cfg)
            if score < selection.min_score:
                continue
            if require_subject and not classification.subject_present:
                continue

            upgraded = moment_type is not None and moment_type != classification.moment_type
            reason = (
                "scoreboard-confirmed make (score changed nearby; subject present)"
                if upgraded
                else classification.reason
            )
            start, end = _shape_window(window, selection, video_duration)
            clips.append(
                Clip(
                    start=start,
                    end=end,
                    moment_type=moment_type,
                    score=round(score, 4),
                    reason=reason,
                    subject_present=classification.subject_present,
                )
            )
        except Exception:
            continue

    clips.sort(key=lambda c: c.score, reverse=True)
    return clips


def _overlaps(clip: Clip, others: list[Clip]) -> bool:
    return any(clip.start < o.end and o.start < clip.end for o in others)


def select_clips(
    scored: list[Clip],
    recipe: Recipe,
    max_duration: float | None = None,
    audience: str | None = None,
) -> list[Clip]:
    """Greedily fill the time budget with the highest-scoring clips, then order them.

    ``audience`` is accepted for API symmetry with the rest of the selection
    step (subject-presence filtering already happened in
    :func:`score_candidates`); it does not change behavior here.

    Clips are considered highest-score first; a clip that would blow the
    budget is skipped (not the whole run) so smaller, still-valuable clips
    lower in score order can still fit. ``selection.dedup_overlap`` drops any
    clip overlapping one already picked. The final set is then ordered per
    ``selection.ordering``: ``chronological``/``narrative_arc`` -> start asc,
    ``best_first`` -> score desc.
    """
    if not scored:
        return []

    selection = recipe.selection
    budget = selection.max_duration if max_duration is None else max_duration

    picked: list[Clip] = []
    used = 0.0
    for clip in sorted(scored, key=lambda c: c.score, reverse=True):
        if used + clip.duration > budget:
            continue
        if selection.dedup_overlap and _overlaps(clip, picked):
            continue
        picked.append(clip)
        used += clip.duration

    if selection.ordering == "best_first":
        picked.sort(key=lambda c: c.score, reverse=True)
    else:
        # 'chronological' and 'narrative_arc' (documented placeholder for now)
        # both present the final set in timeline order.
        picked.sort(key=lambda c: c.start)
    return picked
