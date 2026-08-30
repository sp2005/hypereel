"""Lazily construct providers from :class:`~hypereel.config.Settings`.

Heavy model SDKs are imported *inside* the branch that needs them, so a repo
with only the mock provider configured never has to install gemini/groq/openai.
If a real provider is requested but its key/SDK is missing, we log a note and
fall back to the mock provider so the pipeline still completes (graceful
degradation is a design goal — see the design doc).
"""

from __future__ import annotations

import logging

from ..config import Settings, get_settings
from .base import LLMProvider, VisionProvider
from .mock import MockLLMProvider, MockVisionProvider

_log = logging.getLogger("hypereel.providers")


def get_vision_provider(settings: Settings | None = None) -> VisionProvider:
    settings = settings or get_settings()
    provider = (settings.vision_provider or "mock").lower()

    if provider == "mock":
        return MockVisionProvider()

    key = settings.key_for(provider)
    if not key:
        _log.warning("no API key for vision provider '%s'; using mock.", provider)
        return MockVisionProvider()

    try:
        if provider == "gemini":
            from .gemini import GeminiVisionProvider

            return GeminiVisionProvider(settings)
        if provider == "groq":
            from .groq import GroqVisionProvider

            return GroqVisionProvider(settings)
        if provider in ("nebius", "fireworks"):
            from .nebius import OpenAICompatVisionProvider

            return OpenAICompatVisionProvider(settings, flavor=provider)
    except Exception as exc:  # SDK missing or init failure -> degrade
        _log.warning("could not init vision provider '%s' (%s); using mock.", provider, exc)
        return MockVisionProvider()

    _log.warning("unknown vision provider '%s'; using mock.", provider)
    return MockVisionProvider()


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    provider = (settings.llm_provider or "mock").lower()

    if provider == "mock":
        return MockLLMProvider()

    key = settings.key_for(provider)
    if not key:
        _log.warning("no API key for LLM provider '%s'; using mock.", provider)
        return MockLLMProvider()

    try:
        if provider == "gemini":
            from .gemini import GeminiLLMProvider

            return GeminiLLMProvider(settings)
        if provider == "groq":
            from .groq import GroqLLMProvider

            return GroqLLMProvider(settings)
        if provider in ("nebius", "fireworks"):
            from .nebius import OpenAICompatLLMProvider

            return OpenAICompatLLMProvider(settings, flavor=provider)
    except Exception as exc:
        _log.warning("could not init LLM provider '%s' (%s); using mock.", provider, exc)
        return MockLLMProvider()

    _log.warning("unknown LLM provider '%s'; using mock.", provider)
    return MockLLMProvider()
