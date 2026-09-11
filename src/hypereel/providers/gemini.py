"""Gemini-backed vision/LLM providers.

Lazily imports ``google.generativeai`` so the package can be imported (and the
factory's fallback-to-mock path exercised) without the SDK installed.
"""

from __future__ import annotations

from typing import Sequence

from ..config import Settings
from ..observability import record_provider_failure
from ..models import Classification, Recipe
from .base import LLMProvider, VisionProvider
from ._util import build_classification_prompt, encode_frames_b64, parse_classification_json


class GeminiVisionProvider(VisionProvider):
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        import google.generativeai as genai  # noqa: F401  (lazy import; raises ImportError if absent)

        self._genai = genai
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)

    def classify_window(
        self,
        frame_paths: Sequence[str],
        recipe: Recipe,
        window_index: int = 0,
    ) -> Classification:
        try:
            prompt = build_classification_prompt(recipe)
            parts: list = [prompt]
            for raw in encode_frames_b64(frame_paths):
                parts.append({"mime_type": "image/jpeg", "data": raw})

            response = self._model.generate_content(parts)
            text = getattr(response, "text", "") or ""
            return parse_classification_json(text, recipe)
        except Exception as exc:
            record_provider_failure(exc)
            return Classification(
                moment_type=None,
                subject_present=False,
                confidence=0.0,
                reason=f"gemini error: {exc}",
            )


class GeminiLLMProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        import google.generativeai as genai

        self._genai = genai
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        try:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = self._model.generate_content(
                full_prompt,
                generation_config={"max_output_tokens": max_tokens},
            )
            return getattr(response, "text", "") or ""
        except Exception as exc:
            record_provider_failure(exc)
            return ""
