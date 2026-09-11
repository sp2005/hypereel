"""Transparent provider adapters for opt-in, graph-scoped operation spans."""

from __future__ import annotations

from typing import Sequence

from ..models import Classification, Recipe
from ..observability import trace_provider_call
from .base import LLMProvider, VisionProvider


class TracedVisionProvider(VisionProvider):
    def __init__(self, provider: VisionProvider, requested: str):
        self._provider = provider
        self._requested = requested

    @property
    def name(self) -> str:
        return self._provider.name

    def classify_window(
        self, frame_paths: Sequence[str], recipe: Recipe, window_index: int = 0,
    ) -> Classification:
        return trace_provider_call(
            lambda: self._provider.classify_window(frame_paths, recipe, window_index=window_index),
            name=f"provider.vision.{self.name}",
            metadata={
                "provider_operation": "classify_window",
                "actual_provider": self.name,
                "requested_provider": self._requested,
                "provider_fallback": self.name != self._requested,
                "provider_mock": self.name == "mock",
                "candidate_index": window_index,
                "frame_count": len(frame_paths),
            },
        )


class TracedLLMProvider(LLMProvider):
    def __init__(self, provider: LLMProvider, requested: str):
        self._provider = provider
        self._requested = requested

    @property
    def name(self) -> str:
        return self._provider.name

    def generate(self, prompt: str, *, system: str = "", max_tokens: int = 800) -> str:
        return trace_provider_call(
            lambda: self._provider.generate(prompt, system=system, max_tokens=max_tokens),
            name=f"provider.llm.{self.name}",
            metadata={
                "provider_operation": "generate",
                "actual_provider": self.name,
                "requested_provider": self._requested,
                "provider_fallback": self.name != self._requested,
                "provider_mock": self.name == "mock",
            },
        )
