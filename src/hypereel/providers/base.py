"""Provider abstractions.

Two thin interfaces let the rest of the system stay provider-agnostic:

* :class:`VisionProvider` — confirms/classifies (event_based) or scores
  (quality_based) sampled frames from a candidate window.
* :class:`LLMProvider` — plain text generation for the summary/report step
  (this is the call the graded submission routes through Nebius).

Concrete providers (gemini/groq/nebius) live in sibling modules and are built
lazily by :mod:`hypereel.providers.factory`, so importing this package never
requires any model SDK to be installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..models import Classification, Recipe


class VisionProvider(ABC):
    """Turns frames from a candidate window into a :class:`Classification`."""

    name: str = "base"

    @abstractmethod
    def classify_window(
        self,
        frame_paths: Sequence[str],
        recipe: Recipe,
        window_index: int = 0,
    ) -> Classification:
        """Judge one candidate window.

        Args:
            frame_paths: paths to a few sampled JPEG/PNG frames from the window.
            recipe: the active recipe (moment_types = the rubric, subject_selector
                = who/what to look for, scoring_rubric for quality domains).
            window_index: position of this window in the timeline (used by the
                mock provider to produce deterministic, varied output).

        Returns:
            A :class:`Classification`. Implementations must never raise on an
            empty/garbled model response — return a low-confidence, reasoned
            Classification instead so the pipeline degrades gracefully.
        """
        raise NotImplementedError


class LLMProvider(ABC):
    """Plain text generation (summaries, the final report, recipe scaffolding)."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        """Return generated text for ``prompt`` (never raises; returns '' on error)."""
        raise NotImplementedError
