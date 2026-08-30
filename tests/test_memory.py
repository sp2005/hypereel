"""Tests for the JSON-backed MemoryStore (pure stdlib, no heavy deps)."""

from __future__ import annotations

from hypereel.memory.store import MemoryStore
from hypereel.models import Clip


def _clip(moment_type: str) -> Clip:
    return Clip(start=0.0, end=5.0, moment_type=moment_type, score=0.8)


def test_load_profile_defaults_to_empty_dict(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    assert store.load_profile() == {}


def test_save_and_load_profile_round_trips(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    profile = {"favorite_player": 23, "team_color": "red"}

    store.save_profile(profile)

    assert store.load_profile() == profile


def test_record_feedback_and_learned_preferences(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    recipe_id = "bball_test"

    # Round 1: made_basket kept twice, block dropped once.
    store.record_feedback(
        recipe_id,
        kept=[_clip("made_basket"), _clip("made_basket")],
        dropped=[_clip("block")],
    )
    # Round 2: block dropped again (now clearly disliked), steal_break kept.
    store.record_feedback(
        recipe_id,
        kept=[_clip("steal_break")],
        dropped=[_clip("block")],
    )

    prefs = store.learned_preferences(recipe_id)

    assert prefs["preferred_moment_types"] == ["made_basket", "steal_break"]
    assert prefs["disliked_moment_types"] == ["block"]


def test_learned_preferences_empty_when_no_feedback(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    assert store.learned_preferences("nonexistent_recipe") == {}


def test_corrupt_memory_file_does_not_crash(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("garbage")
    store = MemoryStore(str(path))

    assert store.load_profile() == {}
