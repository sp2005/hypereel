"""Motion-based highlight signal: high-motion spans via frame differencing.

Lazy-imports opencv; degrades to an empty candidate list when cv2 or the
video file is unavailable so a signal outage never breaks the pipeline.
"""

from __future__ import annotations

from ..models import CandidateWindow


def motion_windows(
    video_path: str, duration: float, settings, params: dict | None = None
) -> list[CandidateWindow]:
    """Detect high-motion spans by sampling frames and diffing consecutive ones.

    Returns [] whenever cv2 is missing, the file can't be opened, or nothing
    clears the motion threshold — never raises.
    """
    if duration <= 0:
        return []

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return []

    params = params or {}
    threshold = float(params.get("threshold", 0.5))
    lead_in = max(0.0, float(params.get("candidate_lead_in", 0.0)))
    lead_out = max(0.0, float(params.get("candidate_lead_out", 0.0)))
    sample_fps = max(1, int(getattr(settings, "motion_sample_fps", 2)))

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        native_fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
        step = max(1, round(native_fps / sample_fps))

        prev_gray = None
        scores: list[tuple[float, float]] = []  # (timestamp, raw diff score)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    score = float(np.mean(diff)) / 255.0
                    t = frame_idx / native_fps if native_fps else 0.0
                    scores.append((t, score))
                prev_gray = gray
            frame_idx += 1
        cap.release()

        if not scores:
            return []

        max_score = max(score for _, score in scores) or 1.0
        windows: list[CandidateWindow] = []
        in_motion = False
        start_t = 0.0
        peak = 0.0
        for t, score in scores:
            norm = score / max_score
            if norm >= threshold:
                if not in_motion:
                    in_motion = True
                    start_t = t
                    peak = norm
                else:
                    peak = max(peak, norm)
            elif in_motion:
                end_t = max(t, start_t + 1.0)
                windows.append(
                    CandidateWindow(
                        start=max(0.0, start_t - lead_in),
                        end=min(end_t + lead_out, duration),
                        signal_scores={"motion_intensity": peak},
                    )
                )
                in_motion = False
        if in_motion:
            windows.append(
                CandidateWindow(
                    start=max(0.0, start_t - lead_in),
                    end=min(start_t + 5.0 + lead_out, duration),
                    signal_scores={"motion_intensity": peak},
                )
            )
        return windows
    except Exception:
        return []
