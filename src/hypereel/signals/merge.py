"""Merge candidate windows from multiple signals into one deduped timeline.

Different signals (audio, motion, ...) each propose their own windows; this
module is where those proposals become one ranked timeline the rest of the
pipeline consumes. It is also the guaranteed fallback: `propose_candidates`
must always return something, even with no video and no media libraries
installed, so the graph can keep moving.
"""

from __future__ import annotations

from ..models import CandidateWindow, Recipe

_FALLBACK_DURATION = 120.0
_SYNTHETIC_STEP = 10.0


def merge_windows(
    window_lists: list[list[CandidateWindow]], *, min_gap: float = 1.0, max_len: float = 15.0
) -> list[CandidateWindow]:
    """Merge overlapping/near windows across signal lists.

    Windows within ``min_gap`` seconds of each other are merged into one, with
    ``signal_scores`` unioned (max per key). Anything longer than ``max_len``
    after merging is split back into evenly sized pieces. Result is sorted by
    start.
    """
    flat: list[CandidateWindow] = [w for lst in window_lists for w in lst]
    if not flat:
        return []

    flat.sort(key=lambda w: w.start)

    merged: list[CandidateWindow] = []
    current = flat[0]
    for w in flat[1:]:
        if w.start <= current.end + min_gap:
            scores = dict(current.signal_scores)
            for key, value in w.signal_scores.items():
                scores[key] = max(scores.get(key, value), value)
            current = CandidateWindow(
                start=current.start,
                end=max(current.end, w.end),
                signal_scores=scores,
            )
        else:
            merged.append(current)
            current = w
    merged.append(current)

    result: list[CandidateWindow] = []
    for w in merged:
        if w.duration <= max_len or max_len <= 0:
            result.append(w)
            continue
        n_pieces = max(1, int(w.duration / max_len) + (1 if w.duration % max_len else 0))
        piece_len = w.duration / n_pieces
        for i in range(n_pieces):
            piece_start = w.start + i * piece_len
            piece_end = min(w.start + (i + 1) * piece_len, w.end)
            if piece_end > piece_start:
                result.append(
                    CandidateWindow(
                        start=piece_start,
                        end=piece_end,
                        signal_scores=dict(w.signal_scores),
                    )
                )

    result.sort(key=lambda w: w.start)
    return result


def _median_ideal_len(recipe: Recipe) -> float:
    lens = sorted(mt.ideal_len for mt in recipe.moment_types) if recipe.moment_types else []
    if not lens:
        return 10.0
    return lens[len(lens) // 2]


def _synthetic_scores(recipe: Recipe) -> dict[str, float]:
    """Signal scores for a synthetic window.

    Crucially, these are keyed to the recipe's *actual* proposer signal types
    (e.g. ``audio_peak``, ``motion_intensity``) — not a generic ``"synthetic"``
    key — because the selector's blended score only sums the signal types the
    recipe declares. A synthetic window keyed on an unknown name would score ~0
    and be filtered out, leaving an empty reel in mock/offline mode. A moderate
    0.7 keeps synthetic candidates above a typical ``min_score`` once combined
    with the vision model's confidence, so the demo produces a populated reel.
    """
    proposer_types = [s.type for s in recipe.proposer_signals()]
    if not proposer_types:
        return {"synthetic": 0.7}
    return {t: 0.7 for t in proposer_types}


def _synthetic_windows(duration: float, recipe: Recipe) -> list[CandidateWindow]:
    """Deterministic, evenly-spaced fallback timeline (no signals available)."""
    clip_len = min(10.0, _median_ideal_len(recipe)) or 10.0
    scores = _synthetic_scores(recipe)
    windows: list[CandidateWindow] = []
    t = 0.0
    while t < duration:
        end = min(t + clip_len, duration)
        if end > t:
            windows.append(CandidateWindow(start=t, end=end, signal_scores=dict(scores)))
        t += _SYNTHETIC_STEP
    return windows


def propose_candidates(
    video_path: str | None, duration: float, recipe: Recipe, settings
) -> list[CandidateWindow]:
    """Run every proposer signal in ``recipe``, merge results, and guarantee output.

    This is the top-level entry the graph calls. It falls back to a
    deterministic synthetic timeline whenever real signals produce nothing
    (no video, mock mode, or missing libraries) so the downstream pipeline
    always has candidates to score and select from. Never raises.
    """
    from .audio import audio_peaks
    from .motion import motion_windows

    signal_fns = {
        "audio_peak": audio_peaks,
        "motion_intensity": motion_windows,
    }

    window_lists: list[list[CandidateWindow]] = []
    try:
        if video_path and duration and duration > 0:
            for sig in recipe.proposer_signals():
                fn = signal_fns.get(sig.type)
                if fn is None:
                    continue
                try:
                    windows = fn(video_path, duration, settings, sig.params or None)
                except Exception:
                    windows = []
                if windows:
                    window_lists.append(windows)
    except Exception:
        window_lists = []

    try:
        candidates = merge_windows(window_lists) if window_lists else []
    except Exception:
        candidates = []

    if not candidates:
        try:
            effective_duration = duration if duration and duration > 0 else _FALLBACK_DURATION
            candidates = _synthetic_windows(effective_duration, recipe)
        except Exception:
            candidates = _synthetic_windows(_FALLBACK_DURATION, recipe.__class__.model_construct())

    if not candidates:
        # Absolute last resort — guarantees a non-empty return no matter what.
        try:
            last_scores = _synthetic_scores(recipe)
        except Exception:
            last_scores = {"synthetic": 0.7}
        candidates = [CandidateWindow(start=0.0, end=10.0, signal_scores=last_scores)]

    candidates.sort(key=lambda w: w.start)
    return candidates
