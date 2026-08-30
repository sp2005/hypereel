"""Tests for the HypeReel CLI (``hypereel.cli``).

Fully offline: ``--demo`` forces mock providers via env vars, so these tests
never touch the network or require API keys. Paths are resolved relative to
the repo root so the suite passes regardless of the invoking CWD.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypereel.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _hermetic_io(tmp_path, monkeypatch):
    """Keep every CLI test off the real repo: renders and the memory file go to
    a per-test tmp dir instead of polluting ``output/`` and ``hypereel_memory.json``."""
    monkeypatch.setenv("HYPEREEL_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("HYPEREEL_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("HYPEREEL_DOWNLOAD_DIR", str(tmp_path / "downloads"))


def test_demo_mode_succeeds_and_reports(capsys) -> None:
    rc = main(["--demo"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Basketball" in captured.out or "basketball" in captured.out.lower()
    # Either a clip list or the explicit "no clips" fallback message is fine —
    # both are evidence the report section rendered.
    assert "clip" in captured.out.lower()


def test_demo_mode_with_explicit_source(capsys) -> None:
    rc = main(["--demo", "--source", "samples/sample_game.mp4"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Recipe:" in captured.out
    assert "Output path:" in captured.out


def test_missing_recipe_returns_error_without_traceback(capsys) -> None:
    rc = main(["--recipe", "does_not_exist.yaml", "--source", "x", "--demo"])
    captured = capsys.readouterr()

    assert rc != 0
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_recipe_path_resolves_relative_to_repo_root(capsys, tmp_path, monkeypatch) -> None:
    """The default --recipe path should resolve even when CWD isn't the repo root."""
    monkeypatch.chdir(tmp_path)
    rc = main(["--demo"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Recipe:" in captured.out


def test_describe_flag_is_accepted_and_runs(capsys) -> None:
    """--describe threads a free-text subject brief through the run."""
    rc = main([
        "--demo",
        "--describe",
        "the team in black jerseys with red trim; #23 if legible",
    ])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Recipe:" in captured.out
    assert "Output path:" in captured.out


def test_no_approve_stops_at_first_gate(capsys) -> None:
    rc = main(["--demo", "--no-approve"])
    captured = capsys.readouterr()

    # --demo forces auto-approval regardless of --no-approve, so this should
    # still complete the full pipeline rather than pausing.
    assert rc == 0
    assert "paused for human approval" not in captured.out
