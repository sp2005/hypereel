"""Tests for hypereel.analyze.classifier — offline, MockVisionProvider only."""

from __future__ import annotations

from hypereel.analyze.classifier import classify_candidates, extract_frames
from hypereel.config import Settings
from hypereel.providers.mock import MockVisionProvider


def test_extract_frames_returns_empty_without_video(sample_candidates):
    # No video_path -> no cv2 call, no filesystem writes; must never raise.
    paths = extract_frames(None, sample_candidates[0], 3, "/tmp/hypereel-does-not-matter")
    assert paths == []


def test_classify_candidates_aligns_and_applies_mock_miss_pattern(sample_candidates, basketball_recipe):
    settings = Settings()
    provider = MockVisionProvider()

    results = classify_candidates(sample_candidates, basketball_recipe, provider, settings)

    assert len(results) == len(sample_candidates) == 12

    for i, classification in enumerate(results):
        if i % 3 == 2:
            # MockVisionProvider's deliberate miss pattern.
            assert classification.moment_type is None
            assert classification.confidence < 0.5
        else:
            assert classification.moment_type is not None
            assert classification.confidence > 0.0


def test_classify_candidates_empty_input_returns_empty(basketball_recipe):
    settings = Settings()
    provider = MockVisionProvider()
    assert classify_candidates([], basketball_recipe, provider, settings) == []
