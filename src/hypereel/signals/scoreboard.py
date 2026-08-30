"""Scoreboard-based game-state grounding (OCR-free).

At low resolution the vision model can't reliably tell *live basketball* from
the things that were polluting early reels — timeouts, huddles, players
standing around a dead ball, warm-ups, and kids shooting on the side/adjacent
baskets of a multi-court venue. The on-screen scoreboard overlay answers two
questions the frames alone can't, without any OCR:

  * **game_state** — is the game clock *advancing* (live play) or *frozen*
    (timeout / dead ball)?  Detected by diffing the clock-digit sub-region
    across a few samples: running digits change second-to-second, a frozen
    clock does not.
  * **score_change** — did a team's score change in/near the window (i.e. a
    basket was actually made)?  Same pixel-change trick on the score boxes.

Crucially this reads only the scoreboard's *own* clock box, never the camera
wall-clock burn-in (which ticks even during timeouts and would defeat the
whole point). Regions are expressed as fractions of the frame so they survive
a resolution change, and every default matches the AAU broadcast overlay this
was built against; a recipe can override them via the ``scoreboard`` signal's
``params``.

Everything degrades to "no opinion" (annotations simply absent) when cv2 or
the video is unavailable, so the rest of the pipeline is never affected.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..models import CandidateWindow

# Reserved ``signal_scores`` keys this module writes onto candidate windows.
# The leading underscore keeps them out of the proposer-signal blend in the
# selector (which only sums the recipe's *declared* signal types).
KEY_LIVE = "_sb_live"                 # 1.0 = clock advancing (live), 0.0 = frozen
KEY_SCORE_CHANGE = "_sb_score_change"  # 1.0 = *some* team's score changed near the window
# 1.0 = the SUBJECT team's own score box changed (they scored); 0.0 = it didn't
# (a score change belonged to the opponent). Only written when the recipe's
# scoreboard signal declares which row is the subject via ``subject_score_index``.
KEY_SUBJECT_SCORE_CHANGE = "_sb_subject_score_change"

# Defaults measured against the 640x360 AAU overlay ("Elite / Unl" scoreboard,
# bottom-left). Each region is (x1, y1, x2, y2) as a fraction of width/height.
_DEFAULT_CLOCK_REGION = (0.084, 0.827, 0.144, 0.884)
_DEFAULT_SCORE_REGIONS = (
    (0.052, 0.832, 0.088, 0.885),   # top team score box
    (0.052, 0.885, 0.088, 0.937),   # bottom team score box
)
# Detection metric: fraction of pixels whose intensity moved by more than
# ``change_delta`` between two samples of a region. A single digit ticking over
# only nudges a *mean* diff (a few pixels in a small box), but it flips a clear
# FRACTION of pixels — so this discriminates a live tick from a frozen clock far
# more reliably. Measured on this overlay: a frozen clock reads ~0.000, a
# one-second tick ~0.02-0.03, a two/three-point score change ~0.06-0.10.
_DEFAULT_CHANGE_DELTA = 40.0        # per-pixel intensity delta that "counts"
_DEFAULT_CLOCK_FRAC = 0.012         # >= this fraction changed -> clock advanced (live)
_DEFAULT_SCORE_FRAC = 0.03          # >= this fraction changed -> a score changed


def _crop_gray(frame, region: tuple[float, float, float, float]):
    """Return the grayscale crop of ``frame`` for a fractional region, or None."""
    import cv2  # local import: cv2 is optional

    h, w = frame.shape[:2]
    x1 = max(0, int(region[0] * w))
    y1 = max(0, int(region[1] * h))
    x2 = min(w, int(region[2] * w))
    y2 = min(h, int(region[3] * h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _read_region(cap, fps: float, t: float, region):
    """Seek to ``t`` seconds and return the grayscale crop of ``region``, or None."""
    import cv2  # local import

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return _crop_gray(frame, region)


def _max_consecutive_change_frac(crops: Sequence, delta: float) -> float:
    """Largest fraction-of-pixels-changed (by > ``delta``) between consecutive crops."""
    import numpy as np  # local import

    best = 0.0
    prev = None
    for c in crops:
        if c is None:
            continue
        if prev is not None and prev.shape == c.shape:
            diff = np.abs(c.astype("int16") - prev.astype("int16"))
            best = max(best, float(np.mean(diff > delta)))
        prev = c
    return best


def _clock_is_live(cap, fps, window, region, delta, frac, duration) -> Optional[bool]:
    """True if the clock digits advance across the window (live), False if frozen.

    Samples a >=4s span centered on the window so a running clock is guaranteed
    to cross at least one second boundary even for a short window. Returns None
    when nothing could be read (so the caller records no opinion).
    """
    center = (window.start + window.end) / 2.0
    span = max(window.duration, 4.0)
    lo = max(0.0, center - span / 2.0)
    hi = min(duration, center + span / 2.0) if duration > 0 else center + span / 2.0
    n = max(4, min(8, int(hi - lo) + 1))
    crops = []
    for i in range(n):
        t = lo + (hi - lo) * (i / (n - 1))
        crops.append(_read_region(cap, fps, t, region))
    if sum(c is not None for c in crops) < 2:
        return None
    return _max_consecutive_change_frac(crops, delta) >= frac


def _changed_score_regions(cap, fps, window, regions, delta, frac, duration):
    """Indices of the score boxes that change across [start-3, end+3].

    Returns a set of region indices that changed (empty = none changed), or None
    when nothing could be read. Per-region resolution is what lets the caller
    tell *which* team scored, so an opponent basket isn't credited to the subject.
    """
    lo = max(0.0, window.start - 3.0)
    hi = window.end + 3.0
    if duration > 0:
        hi = min(hi, duration)
    times = [lo, (lo + hi) / 2.0, hi]
    changed: set[int] = set()
    read_any = False
    for i, region in enumerate(regions):
        crops = [_read_region(cap, fps, t, region) for t in times]
        if sum(c is not None for c in crops) >= 2:
            read_any = True
            if _max_consecutive_change_frac(crops, delta) >= frac:
                changed.add(i)
    return changed if read_any else None


def annotate_candidates(
    video_path: Optional[str],
    candidates: list[CandidateWindow],
    params: Optional[dict] = None,
) -> tuple[list[CandidateWindow], dict]:
    """Annotate each candidate with scoreboard-derived game state, in place.

    Writes :data:`KEY_LIVE` and :data:`KEY_SCORE_CHANGE` into each window's
    ``signal_scores`` (absent when the scoreboard couldn't be read). Returns the
    same ``candidates`` list plus a small stats dict for logging. Never raises:
    on any failure it returns the candidates untouched with an ``ok=False`` stat.
    """
    stats = {"ok": False, "annotated": 0, "live": 0, "dead": 0,
             "score_changes": 0, "subject_score_changes": 0}
    if not video_path or not candidates:
        return candidates, stats

    try:
        import cv2  # noqa: F401  (availability probe)
        import numpy  # noqa: F401
    except ImportError:
        return candidates, stats

    params = params or {}
    clock_region = tuple(params.get("clock_region") or _DEFAULT_CLOCK_REGION)
    score_regions = [tuple(r) for r in (params.get("score_regions") or _DEFAULT_SCORE_REGIONS)]
    delta = float(params.get("change_delta", _DEFAULT_CHANGE_DELTA))
    clock_frac = float(params.get("clock_frac", _DEFAULT_CLOCK_FRAC))
    score_frac = float(params.get("score_frac", _DEFAULT_SCORE_FRAC))
    # Which score-box row belongs to the subject team (0 = top, 1 = bottom, ...).
    # When set, only that row's changes credit the subject with a made basket,
    # so opponent baskets are no longer confirmed as the subject's.
    raw_idx = params.get("subject_score_index")
    subject_idx = int(raw_idx) if raw_idx is not None else None

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return candidates, stats
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        duration = (frame_count / fps) if fps > 0 else 0.0
        if fps <= 0:
            cap.release()
            return candidates, stats

        for window in candidates:
            try:
                live = _clock_is_live(cap, fps, window, clock_region, delta, clock_frac, duration)
                changed = _changed_score_regions(cap, fps, window, score_regions, delta, score_frac, duration)
            except Exception:
                live, changed = None, None
            if live is not None:
                window.signal_scores[KEY_LIVE] = 1.0 if live else 0.0
                stats["annotated"] += 1
                stats["live" if live else "dead"] += 1
            if changed:  # non-empty set -> some team's score changed
                window.signal_scores[KEY_SCORE_CHANGE] = 1.0
                stats["score_changes"] += 1
            if changed is not None and subject_idx is not None:
                subject_scored = subject_idx in changed
                window.signal_scores[KEY_SUBJECT_SCORE_CHANGE] = 1.0 if subject_scored else 0.0
                if subject_scored:
                    stats["subject_score_changes"] += 1
        cap.release()
        stats["ok"] = True
        return candidates, stats
    except Exception:
        return candidates, stats
