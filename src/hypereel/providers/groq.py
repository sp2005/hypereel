"""Groq-backed vision/LLM providers (OpenAI-style chat API).

Lazily imports ``groq`` so the package can be imported (and the factory's
fallback-to-mock path exercised) without the SDK installed.
"""

from __future__ import annotations

from typing import Sequence

from ..config import Settings
from ..observability import record_provider_failure
from ..models import Classification, Recipe
from .base import LLMProvider, VisionProvider
from ._util import (
    build_classification_prompt,
    encode_frames_b64,
    frame_to_data_uri,
    parse_classification_json,
)


class GroqVisionProvider(VisionProvider):
    name = "groq"

    def __init__(self, settings: Settings) -> None:
        from groq import Groq  # raises ImportError if SDK absent

        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_vision_model

    def classify_window(
        self,
        frame_paths: Sequence[str],
        recipe: Recipe,
        window_index: int = 0,
    ) -> Classification:
        try:
            prompt = build_classification_prompt(recipe)
            content: list[dict] = [{"type": "text", "text": prompt}]
            for raw in encode_frames_b64(frame_paths):
                content.append(
                    {"type": "image_url", "image_url": {"url": frame_to_data_uri(raw)}}
                )

            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": content}],
            )
            text = completion.choices[0].message.content or ""
            return parse_classification_json(text, recipe)
        except Exception as exc:
            record_provider_failure(exc)
            return Classification(
                moment_type=None,
                subject_present=False,
                confidence=0.0,
                reason=f"groq error: {exc}",
            )


class GroqLLMProvider(LLMProvider):
    name = "groq"

    def __init__(self, settings: Settings) -> None:
        from groq import Groq

        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.groq_vision_model

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            record_provider_failure(exc)
            return ""
