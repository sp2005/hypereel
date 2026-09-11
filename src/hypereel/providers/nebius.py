"""OpenAI-compatible vision/LLM providers, used for both Nebius and Fireworks.

Both services expose an OpenAI-compatible chat/completions API; the ``flavor``
argument picks which base_url/api_key/model triple from :class:`Settings` to
use. Lazily imports ``openai`` so the package can be imported (and the
factory's fallback-to-mock path exercised) without the SDK installed.
"""

from __future__ import annotations

from typing import Sequence

from ..config import Settings
from ..observability import (
    authorize_provider_call,
    record_provider_failure,
    record_provider_usage,
)
from ..models import Classification, Recipe
from .base import LLMProvider, VisionProvider
from ._util import (
    build_classification_prompt,
    build_verification_prompt,
    encode_frames_b64,
    frame_to_data_uri,
    parse_classification_json,
)


def _flavor_config(settings: Settings, flavor: str) -> tuple[str, str, str]:
    """Return (api_key, base_url, model) for the given flavor."""
    if flavor == "fireworks":
        return settings.fireworks_api_key, settings.fireworks_base_url, settings.fireworks_model
    return settings.nebius_api_key, settings.nebius_base_url, settings.nebius_model


class OpenAICompatVisionProvider(VisionProvider):
    def __init__(self, settings: Settings, flavor: str = "nebius") -> None:
        from openai import OpenAI  # raises ImportError if SDK absent

        self.name = flavor
        api_key, base_url, model = _flavor_config(settings, flavor)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def classify_window(
        self,
        frame_paths: Sequence[str],
        recipe: Recipe,
        window_index: int = 0,
    ) -> Classification:
        try:
            prompt = build_classification_prompt(recipe)
            content: list[dict] = [{"type": "text", "text": prompt}]
            encoded_frames = encode_frames_b64(frame_paths)
            for index, raw in enumerate(encoded_frames):
                if len(encoded_frames) >= 5 and index == 0:
                    label = "BEFORE CONTEXT (do not classify an event visible only here)"
                elif len(encoded_frames) >= 5 and index == len(encoded_frames) - 1:
                    label = "AFTER CONTEXT (do not classify an event visible only here)"
                else:
                    candidate_index = index if len(encoded_frames) >= 5 else index + 1
                    label = f"CANDIDATE ACTION FRAME {candidate_index}"
                content.append({"type": "text", "text": label})
                content.append(
                    {"type": "image_url", "image_url": {"url": frame_to_data_uri(raw)}}
                )

            authorize_provider_call(
                provider=self.name, model=self._model, operation="vision_classification"
            )
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
            )
            record_provider_usage(completion)
            text = completion.choices[0].message.content or ""
            return parse_classification_json(text, recipe)
        except Exception as exc:
            record_provider_failure(exc)
            return Classification(
                moment_type=None,
                subject_present=False,
                confidence=0.0,
                reason=f"{self.name} error: {exc}",
            )

    def verify_window(
        self,
        frame_paths: Sequence[str],
        recipe: Recipe,
        proposed: Classification,
        window_index: int = 0,
    ) -> Classification:
        """Verify a positive verdict against event-specific lookalikes."""
        if proposed.moment_type is None:
            return proposed
        try:
            content: list[dict] = [
                {"type": "text", "text": build_verification_prompt(recipe, proposed)}
            ]
            for index, raw in enumerate(encode_frames_b64(frame_paths), start=1):
                content.append({"type": "text", "text": f"CANDIDATE ACTION FRAME {index}"})
                content.append(
                    {"type": "image_url", "image_url": {"url": frame_to_data_uri(raw)}}
                )
            authorize_provider_call(
                provider=self.name, model=self._model, operation="vision_verification"
            )
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
            )
            record_provider_usage(completion)
            return parse_classification_json(completion.choices[0].message.content or "", recipe)
        except Exception as exc:
            record_provider_failure(exc)
            return Classification(
                moment_type=None,
                subject_present=False,
                confidence=0.0,
                reason=f"{self.name} verification error: {exc}",
            )


class OpenAICompatLLMProvider(LLMProvider):
    def __init__(self, settings: Settings, flavor: str = "nebius") -> None:
        from openai import OpenAI

        self.name = flavor
        api_key, base_url, model = _flavor_config(settings, flavor)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            authorize_provider_call(
                provider=self.name, model=self._model, operation="quality_judge"
            )
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            record_provider_usage(completion)
            return completion.choices[0].message.content or ""
        except Exception as exc:
            record_provider_failure(exc)
            return ""
