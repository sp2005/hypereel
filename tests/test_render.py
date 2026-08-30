"""Tests for the render module's manifest fallback.

moviepy is not installed in this environment, so these exercise the offline
path that the whole pipeline (and grader) relies on: render_reel must never
raise, and must degrade to a JSON manifest describing the cut list.
"""

from __future__ import annotations

import json
from pathlib import Path

from hypereel.config import get_settings
from hypereel.models import Clip
from hypereel.render.renderer import can_render, render_reel


def _clips() -> list[Clip]:
    return [
        Clip(start=0.0, end=5.0, moment_type="made_basket", score=0.9, reason="scored", subject_present=True),
        Clip(start=10.0, end=16.0, moment_type="block", score=0.7, reason="blocked shot", subject_present=False),
    ]


def test_can_render_returns_bool():
    assert isinstance(can_render(), bool)


def test_render_reel_falls_back_to_manifest_when_no_video(tmp_path, basketball_recipe):
    out_path = tmp_path / "reel.mp4"
    clips = _clips()

    result = render_reel(clips, None, basketball_recipe, str(out_path), get_settings())

    manifest_path = tmp_path / "reel.manifest.json"
    assert manifest_path.exists()
    assert result.output_path == str(manifest_path)
    assert result.total_duration == sum(c.duration for c in clips)
    assert result.recipe_id == basketball_recipe.id
    assert result.notes  # explains render was simulated

    data = json.loads(manifest_path.read_text())
    assert data["recipe_id"] == basketball_recipe.id
    assert data["clip_count"] == len(clips)
    assert len(data["clips"]) == len(clips)
    assert data["clips"][0]["moment_type"] == "made_basket"
    assert data["clips"][0]["duration"] == clips[0].duration


def test_render_reel_never_raises_with_missing_video_path(tmp_path, basketball_recipe):
    out_path = tmp_path / "reel.mp4"
    clips = _clips()

    # A nonexistent source video should still degrade to the manifest, not raise.
    result = render_reel(clips, "/does/not/exist.mp4", basketball_recipe, str(out_path), get_settings())

    assert result.output_path is not None
    assert Path(result.output_path).exists()


def test_render_reel_with_no_clips_still_returns_result(tmp_path, basketball_recipe):
    out_path = tmp_path / "empty.mp4"

    result = render_reel([], None, basketball_recipe, str(out_path), get_settings())

    assert result.total_duration == 0.0
    assert result.output_path is not None
