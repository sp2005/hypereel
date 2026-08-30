"""Tests that the shipped recipe YAML files parse into valid Recipe objects.

Both recipes are exercised: the flagship event_based basketball recipe and
the quality_based architecture stub that proves the schema generalizes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypereel.recipe import RecipeError, load_recipe, parse_recipe

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES_DIR = REPO_ROOT / "recipes"


def test_basketball_player_recipe_loads() -> None:
    recipe = load_recipe(RECIPES_DIR / "basketball_player.yaml")

    assert recipe.highlight_model == "event_based"
    assert len(recipe.moment_types) >= 3
    assert recipe.subject_selector.value == 23
    assert recipe.proposer_signals()
    assert recipe.scorer_signals()


def test_generic_recipe_is_prompt_driven_and_filters_one_team() -> None:
    """The generic recipe must stay reusable AND still filter to one team.

    Single-team filtering hinges on the selector's require_subject rule
    (audience == "individual" and type != "none"). If a future edit relaxes
    either — or hardcodes a color/number/scoreboard region — the recipe stops
    being generic or stops excluding the opponent, so guard both here.
    """
    recipe = load_recipe(RECIPES_DIR / "basketball_generic.yaml")

    ss = recipe.subject_selector
    # require_subject would be True in select/selector.score_candidates ->
    # opponent (subject_present=false) plays get dropped.
    assert ss.audience == "individual"
    assert ss.type != "none"
    # Prompt-driven: no baked-in per-video brief; the runtime prompt supplies it.
    assert not ss.description
    # No scoreboard signal — its pixel regions are per-broadcast, not generic.
    assert all(sig.type != "scoreboard" for sig in recipe.signals)
    assert recipe.proposer_signals()
    assert recipe.scorer_signals()


def test_architecture_walkthrough_recipe_loads() -> None:
    recipe = load_recipe(RECIPES_DIR / "architecture_walkthrough.yaml")

    assert recipe.highlight_model == "quality_based"
    assert recipe.scoring_rubric
    assert len(recipe.scoring_rubric) > 0


# --------------------------------------------------------------------------- #
#  Error paths — a bad policy file must become a clean RecipeError, not a
#  raw stack trace (the first link in the "tool failure -> graceful" story).
# --------------------------------------------------------------------------- #


def test_missing_recipe_file_raises_recipe_error(tmp_path) -> None:
    with pytest.raises(RecipeError, match="not found"):
        load_recipe(tmp_path / "nope.yaml")


def test_unparseable_yaml_raises_recipe_error(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("this: : : not valid yaml\n  - [unbalanced")
    with pytest.raises(RecipeError):
        load_recipe(bad)


def test_non_mapping_recipe_raises_recipe_error() -> None:
    with pytest.raises(RecipeError, match="must be a mapping"):
        parse_recipe(["not", "a", "mapping"])


def test_schema_invalid_recipe_raises_recipe_error() -> None:
    # Structurally a mapping, but missing required fields / wrong types.
    with pytest.raises(RecipeError, match="invalid recipe"):
        parse_recipe({"id": "x", "highlight_model": "not_a_valid_model"})
