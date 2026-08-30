"""Deterministic mock providers.

These let the ENTIRE pipeline run offline with no API key and no network — used
by the test suite and by ``--demo`` mode. Output is deterministic (seeded by
window index) so tests can assert on it.
"""

from __future__ import annotations

from typing import Sequence

from ..models import Classification, Recipe
from .base import LLMProvider, VisionProvider


class MockVisionProvider(VisionProvider):
    """Fakes vision classification without any model call.

    Strategy: every 3rd window is a "miss" (low confidence), the rest are hits
    cycling through the recipe's moment_types. The recipe's subject is reported
    present on ~2/3 of hits. Deterministic given ``window_index``.
    """

    name = "mock"

    def classify_window(
        self,
        frame_paths: Sequence[str],
        recipe: Recipe,
        window_index: int = 0,
    ) -> Classification:
        moment_names = [m.name for m in recipe.moment_types] or ["highlight"]

        # Every 3rd window is a deliberate miss so selection/threshold logic
        # and the "not enough highlights" self-correction path get exercised.
        if window_index % 3 == 2:
            return Classification(
                moment_type=None,
                subject_present=False,
                confidence=0.2,
                reason="mock: no clear highlight in this window",
            )

        moment = moment_names[window_index % len(moment_names)]
        subject_present = (window_index % 3) != 1  # ~2/3 of hits show the subject
        confidence = 0.6 + 0.1 * (window_index % 4)  # 0.6..0.9, varied
        return Classification(
            moment_type=moment,
            subject_present=subject_present,
            confidence=round(min(confidence, 0.95), 2),
            reason=f"mock: looks like a '{moment}' (window {window_index})",
        )


class MockLLMProvider(LLMProvider):
    """Returns a canned, structured-looking summary. No network."""

    name = "mock"

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        return (
            "## HypeReel Summary (mock)\n"
            "Assembled the strongest moments into a reel that fits the requested "
            "time budget. Clips were chosen for highlight strength and subject "
            "presence, then ordered per the recipe.\n\n"
            "_This text was produced by the offline MockLLMProvider; set "
            "HYPEREEL_LLM_PROVIDER=nebius (or gemini/groq) for a live summary._"
        )
