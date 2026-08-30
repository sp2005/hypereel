"""Audio-based highlight signals: commentary detection and crowd/energy peaks.

Both functions are best-effort and degrade to a safe empty/False result when
librosa or the audio track itself is unavailable — a missing library must
shrink the candidate set, never crash the pipeline.
"""

from __future__ import annotations

from ..models import CandidateWindow


def detect_commentary(video_path: str) -> bool:
    """Best-effort check for speech/commentary on the audio track."""
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return False

    try:
        y, sr = librosa.load(video_path, sr=None, mono=True)
        if y.size == 0:
            return False
        # Speech has frequent zero-crossings relative to pure crowd noise/music;
        # a cheap heuristic, good enough to gate the `enabled_if` condition.
        zcr = librosa.feature.zero_crossing_rate(y)
        return bool(np.mean(zcr) > 0.02)
    except Exception:
        return False


def audio_peaks(
    video_path: str, duration: float, settings, params: dict | None = None
) -> list[CandidateWindow]:
    """Detect loud crowd/energy peaks and turn them into candidate windows.

    Returns [] whenever librosa is missing, the file can't be read, or no
    peaks clear the threshold — never raises.
    """
    if duration <= 0:
        return []

    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return []

    params = params or {}
    threshold = float(params.get("threshold", 0.6))
    min_window = float(params.get("min_window", 1.0))

    try:
        y, sr = librosa.load(video_path, sr=None, mono=True)
        if y.size == 0:
            return []

        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        if rms.size == 0:
            return []
        peak_rms = float(rms.max())
        norm = rms / (peak_rms or 1.0)
        times = librosa.frames_to_time(range(len(norm)), sr=sr, hop_length=hop)

        windows: list[CandidateWindow] = []
        in_peak = False
        start_t = 0.0
        peak_score = 0.0
        for t, score in zip(times, norm):
            t = float(t)
            score = float(score)
            if score >= threshold:
                if not in_peak:
                    in_peak = True
                    start_t = t
                    peak_score = score
                else:
                    peak_score = max(peak_score, score)
            elif in_peak:
                end_t = max(t, start_t + min_window)
                windows.append(
                    CandidateWindow(
                        start=start_t,
                        end=min(end_t, duration),
                        signal_scores={"audio_peak": peak_score},
                    )
                )
                in_peak = False
        if in_peak:
            windows.append(
                CandidateWindow(
                    start=start_t,
                    end=min(start_t + min_window, duration),
                    signal_scores={"audio_peak": peak_score},
                )
            )
        return windows
    except Exception:
        return []
