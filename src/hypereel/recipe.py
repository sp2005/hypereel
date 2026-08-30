"""Load and validate recipe YAML files into :class:`~hypereel.models.Recipe`.

A malformed recipe becomes a clean, explained :class:`RecipeError` — never a
raw stack trace. This is the first line of the "tool failure -> graceful"
story: a bad policy file fails loudly and early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from .models import Recipe


class RecipeError(ValueError):
    """Raised when a recipe file is missing, unparseable, or invalid."""


def load_recipe(path: Union[str, Path]) -> Recipe:
    """Read a YAML recipe from ``path`` and validate it."""
    p = Path(path)
    if not p.exists():
        raise RecipeError(f"recipe file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise RecipeError(f"could not parse YAML in {p}: {exc}") from exc
    return parse_recipe(raw, source=str(p))


def parse_recipe(data: dict, source: str = "<dict>") -> Recipe:
    """Validate an already-loaded mapping into a :class:`Recipe`."""
    if not isinstance(data, dict):
        raise RecipeError(f"recipe {source} must be a mapping, got {type(data).__name__}")
    try:
        return Recipe.model_validate(data)
    except ValidationError as exc:
        raise RecipeError(f"invalid recipe {source}:\n{exc}") from exc
