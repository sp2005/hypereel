"""Offline tests for hypereel.signals — audio/motion detectors and merge/propose.

No network, no assumption that librosa/opencv are installed: the detectors
must degrade to [] gracefully, and propose_candidates must always synthesize
a fallback timeline.
"""

from __future__ import annotations

from hypereel.config import get_settings
from hypereel.models import CandidateWindow
from hypereel.signals.audio import audio_peaks, detect_commentary
from hypereel.signals.merge import merge_windows, propose_candidates
from hypereel.signals.motion import motion_windows


def test_audio_peaks_missing_file_returns_empty_list():
    settings = get_settings()
    assert audio_peaks("/nonexistent/video.mp4", 60.0, settings) == []


def test_audio_peaks_zero_duration_returns_empty_list():
    settings = get_settings()
    assert audio_peaks("/nonexistent/video.mp4", 0.0, settings) == []


def test_detect_commentary_missing_file_returns_false():
    assert detect_commentary("/nonexistent/video.mp4") is False


def test_motion_windows_missing_file_returns_empty_list():
    settings = get_settings()
    assert motion_windows("/nonexistent/video.mp4", 60.0, settings) == []


def test_motion_windows_zero_duration_returns_empty_list():
    settings = get_settings()
    assert motion_windows("/nonexistent/video.mp4", 0.0, settings) == []


def test_merge_windows_merges_overlapping_and_unions_scores():
    a = CandidateWindow(start=0.0, end=5.0, signal_scores={"audio_peak": 0.7})
    b = CandidateWindow(start=4.0, end=9.0, signal_scores={"motion_intensity": 0.6})

    merged = merge_windows([[a], [b]])

    assert len(merged) == 1
    window = merged[0]
    assert window.start == 0.0
    assert window.end == 9.0
    assert window.signal_scores == {"audio_peak": 0.7, "motion_intensity": 0.6}


def test_merge_windows_keeps_far_apart_windows_separate():
    a = CandidateWindow(start=0.0, end=5.0, signal_scores={"audio_peak": 0.7})
    b = CandidateWindow(start=20.0, end=25.0, signal_scores={"motion_intensity": 0.6})

    merged = merge_windows([[a], [b]], min_gap=1.0)

    assert len(merged) == 2
    assert [w.start for w in merged] == sorted(w.start for w in merged)


def test_merge_windows_splits_windows_longer_than_max_len():
    long_window = CandidateWindow(start=0.0, end=40.0, signal_scores={"audio_peak": 0.9})

    merged = merge_windows([[long_window]], max_len=15.0)

    assert all(w.duration <= 15.0 + 1e-6 for w in merged)
    assert [w.start for w in merged] == sorted(w.start for w in merged)
    assert merged[0].start == 0.0
    assert merged[-1].end == 40.0


def test_merge_windows_empty_input_returns_empty_list():
    assert merge_windows([[], []]) == []


def test_propose_candidates_always_returns_nonempty_list(basketball_recipe):
    settings = get_settings()

    candidates = propose_candidates(None, 120.0, basketball_recipe, settings)

    assert len(candidates) > 0
    starts = [w.start for w in candidates]
    assert starts == sorted(starts)
    assert all(isinstance(w, CandidateWindow) for w in candidates)


def test_propose_candidates_never_raises_with_zero_duration_and_no_video(basketball_recipe):
    settings = get_settings()

    candidates = propose_candidates(None, 0.0, basketball_recipe, settings)

    assert len(candidates) > 0


def test_propose_candidates_with_bogus_video_path_still_returns_nonempty(basketball_recipe):
    settings = get_settings()

    candidates = propose_candidates("/nonexistent/video.mp4", 120.0, basketball_recipe, settings)

    assert len(candidates) > 0
    starts = [w.start for w in candidates]
    assert starts == sorted(starts)
