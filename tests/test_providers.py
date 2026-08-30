"""Tests for the live provider modules (gemini/groq/nebius).

Entirely offline: no network calls, no API keys, and the model SDKs
(google-generativeai, groq, openai) are assumed absent from the test
environment. These tests confirm:

  1. Importing each provider module never fails (no SDK import at module
     scope — only inside __init__).
  2. Constructing a provider without its SDK installed raises ImportError,
     which is exactly what lets ``providers.factory`` fall back to mock.
  3. The tolerant JSON-parsing helper correctly maps a fenced JSON blob to a
     Classification.
"""

from __future__ import annotations

import importlib

import pytest

from hypereel.config import Settings
from hypereel.providers._util import build_classification_prompt, parse_classification_json
from hypereel.providers.factory import get_llm_provider, get_vision_provider
from hypereel.providers.mock import MockLLMProvider, MockVisionProvider


def _sdk_installed(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def test_gemini_module_imports() -> None:
    module = importlib.import_module("hypereel.providers.gemini")
    assert hasattr(module, "GeminiVisionProvider")
    assert hasattr(module, "GeminiLLMProvider")


def test_groq_module_imports() -> None:
    module = importlib.import_module("hypereel.providers.groq")
    assert hasattr(module, "GroqVisionProvider")
    assert hasattr(module, "GroqLLMProvider")


def test_nebius_module_imports() -> None:
    module = importlib.import_module("hypereel.providers.nebius")
    assert hasattr(module, "OpenAICompatVisionProvider")
    assert hasattr(module, "OpenAICompatLLMProvider")


def _settings() -> Settings:
    return Settings(
        gemini_api_key="fake",
        groq_api_key="fake",
        nebius_api_key="fake",
        fireworks_api_key="fake",
    )


def test_gemini_construction_without_sdk_raises_importerror() -> None:
    if _sdk_installed("google.generativeai"):
        pytest.skip("google-generativeai is installed in this environment")
    from hypereel.providers.gemini import GeminiVisionProvider, GeminiLLMProvider

    with pytest.raises(ImportError):
        GeminiVisionProvider(_settings())
    with pytest.raises(ImportError):
        GeminiLLMProvider(_settings())


def test_groq_construction_without_sdk_raises_importerror() -> None:
    if _sdk_installed("groq"):
        pytest.skip("groq is installed in this environment")
    from hypereel.providers.groq import GroqVisionProvider, GroqLLMProvider

    with pytest.raises(ImportError):
        GroqVisionProvider(_settings())
    with pytest.raises(ImportError):
        GroqLLMProvider(_settings())


def test_nebius_construction_without_sdk_raises_importerror() -> None:
    if _sdk_installed("openai"):
        pytest.skip("openai is installed in this environment")
    from hypereel.providers.nebius import OpenAICompatVisionProvider, OpenAICompatLLMProvider

    with pytest.raises(ImportError):
        OpenAICompatVisionProvider(_settings(), flavor="nebius")
    with pytest.raises(ImportError):
        OpenAICompatLLMProvider(_settings(), flavor="fireworks")


def test_parse_classification_json_handles_fenced_block(basketball_recipe) -> None:
    text = (
        '```json\n'
        '{"moment_type":"made_basket","subject_present":true,'
        '"confidence":0.8,"reason":"x"}\n'
        '```'
    )
    result = parse_classification_json(text, basketball_recipe)
    assert result.moment_type == "made_basket"
    assert result.subject_present is True
    assert result.confidence == 0.8
    assert result.reason == "x"


def test_parse_classification_json_rejects_unknown_moment_type(basketball_recipe) -> None:
    text = '{"moment_type":"not_in_rubric","subject_present":false,"confidence":0.5,"reason":"y"}'
    result = parse_classification_json(text, basketball_recipe)
    assert result.moment_type is None


def test_parse_classification_json_handles_garbage(basketball_recipe) -> None:
    result = parse_classification_json("not json at all", basketball_recipe)
    assert result.moment_type is None
    assert result.confidence == 0.0


# --------------------------------------------------------------------------- #
#  Classification prompt: the free-text subject brief leads "who to look for"
# --------------------------------------------------------------------------- #


def test_prompt_leads_with_free_text_subject_brief(basketball_recipe) -> None:
    """A subject_selector.description is placed verbatim in the prompt and leads."""
    recipe = basketball_recipe.model_copy(deep=True)
    brief = "the team in BLACK jerseys with RED trim (UNL on the scoreboard); #23 if legible"
    recipe.subject_selector.description = brief

    prompt = build_classification_prompt(recipe)

    assert brief in prompt
    assert "Who/what to look for:" in prompt
    # Structured fields still ride along as supporting cues.
    assert "Additional structured cues" in prompt
    assert "team_color=red" in prompt


def test_prompt_without_brief_falls_back_to_structured_selector(basketball_recipe) -> None:
    prompt = build_classification_prompt(basketball_recipe)
    assert "Who/what to look for: type=jersey_number" in prompt
    assert "Additional structured cues" not in prompt


def test_prompt_brief_works_without_structured_selector() -> None:
    """A pure free-text brief (subject type 'none') still drives the prompt."""
    from hypereel.models import MomentType, Recipe, SubjectSelector

    recipe = Recipe(
        id="r",
        name="n",
        domain="basketball",
        moment_types=[MomentType(name="made_basket", description="scores")],
        subject_selector=SubjectSelector(type="none", description="the black/red team"),
    )
    prompt = build_classification_prompt(recipe)
    assert "Who/what to look for: the black/red team" in prompt
    assert "No specific subject to track" not in prompt
    assert "Additional structured cues" not in prompt


# --------------------------------------------------------------------------- #
#  Factory: graceful fall-back to mock (a core design goal)
# --------------------------------------------------------------------------- #


def test_factory_defaults_to_mock() -> None:
    settings = Settings()  # vision_provider/llm_provider default to "mock"
    assert isinstance(get_vision_provider(settings), MockVisionProvider)
    assert isinstance(get_llm_provider(settings), MockLLMProvider)


def test_factory_falls_back_to_mock_when_key_missing() -> None:
    # A real provider is requested but no API key is configured -> mock, no raise.
    settings = Settings(vision_provider="gemini", llm_provider="nebius")
    assert not settings.gemini_api_key and not settings.nebius_api_key
    assert isinstance(get_vision_provider(settings), MockVisionProvider)
    assert isinstance(get_llm_provider(settings), MockLLMProvider)


def test_factory_falls_back_to_mock_when_sdk_missing() -> None:
    # Key is present, but if the SDK isn't installed, construction fails inside
    # the factory and it degrades to mock instead of propagating the ImportError.
    if _sdk_installed("google.generativeai"):
        pytest.skip("google-generativeai is installed; SDK-missing path can't be exercised")
    settings = Settings(vision_provider="gemini", gemini_api_key="fake")
    assert isinstance(get_vision_provider(settings), MockVisionProvider)


def test_factory_unknown_provider_falls_back_to_mock() -> None:
    settings = Settings(vision_provider="not_a_provider", llm_provider="not_a_provider")
    assert isinstance(get_vision_provider(settings), MockVisionProvider)
    assert isinstance(get_llm_provider(settings), MockLLMProvider)
