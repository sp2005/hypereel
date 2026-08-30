"""Offline tests for hypereel.ingest.source_resolver — no network involved."""

from __future__ import annotations

import pytest

from hypereel.config import get_settings
from hypereel.ingest.source_resolver import (
    IngestError,
    _base_ydl_opts,
    _hd_unlock_ydl_opts,
    is_url,
    parse_video_quality,
    resolve_source,
)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("http://example.com/video.mp4", True),
        ("https://youtube.com/watch?v=abc123", True),
        ("https://youtu.be/abc123", True),
        ("/local/path/video.mp4", False),
        ("video.mp4", False),
        ("ftp://example.com/video.mp4", False),
        ("", False),
    ],
)
def test_is_url(source, expected):
    assert is_url(source) is expected


def test_resolve_source_missing_local_path_raises():
    settings = get_settings()
    with pytest.raises(IngestError):
        resolve_source("/nonexistent/path/to/video.mp4", settings)


def test_resolve_source_existing_local_path_returns_it(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not a real video, just placeholder bytes")
    settings = get_settings()

    path, duration = resolve_source(str(video), settings)

    assert path == str(video)
    assert isinstance(duration, float)
    assert duration >= 0.0


def test_resolve_source_empty_string_raises():
    settings = get_settings()
    with pytest.raises(IngestError):
        resolve_source("", settings)


# --------------------------------------------------------------------------- #
#  Prompt-driven output quality
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Prefer #23; use HD if available", 1080),
        ("capture the black/red team in full HD", 1080),
        ("high quality please", 1080),
        ("just a quick 360p preview", 360),
        ("keep it low-res for a fast preview", 360),
        ("give me 720 if you can", 720),
        ("the team in black jerseys with red trim", None),  # no quality intent
        ("", None),
        (None, None),
    ],
)
def test_parse_video_quality(text, expected):
    assert parse_video_quality(text) == expected


def test_parse_video_quality_specific_resolution_beats_generic_hd():
    # "720p" and a bare "hd" in the same sentence -> the explicit 720 wins.
    assert parse_video_quality("HD is fine but cap at 720p to keep it small") == 720


def test_hd_unlock_opts_layer_sabr_workaround_on_base():
    base = _base_ydl_opts(__import__("pathlib").Path("/tmp/dl"), 1080)
    hd = _hd_unlock_ydl_opts(base)
    # The SABR-defeating extras are present...
    assert hd["cookiesfrombrowser"] == ("chrome", None, None, None)
    assert hd["js_runtimes"] == {"deno": {"path": None}}
    assert hd["remote_components"] == ["ejs:github"]
    # ...and the base is untouched (no cookies/js on the plain fallback profile).
    assert "cookiesfrombrowser" not in base
    assert "js_runtimes" not in base


def test_base_opts_format_respects_max_height():
    base = _base_ydl_opts(__import__("pathlib").Path("/tmp/dl"), 720)
    assert "height<=720" in base["format"]
    default = _base_ydl_opts(__import__("pathlib").Path("/tmp/dl"), None)
    assert "height<=1080" in default["format"]
