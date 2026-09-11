"""Provider adapters preserve identity of results, arguments, and exceptions."""

from unittest.mock import Mock

import pytest

from hypereel import observability
from hypereel.config import Settings
from hypereel.models import Classification
from hypereel.providers.base import LLMProvider, VisionProvider
from hypereel.providers.factory import get_llm_provider, get_vision_provider
from hypereel.providers.mock import MockLLMProvider, MockVisionProvider
from hypereel.providers.traced import TracedLLMProvider, TracedVisionProvider


@pytest.mark.parametrize("fail_completion", [False, True])
def test_vision_delegates_exactly_once(monkeypatch, basketball_recipe, fail_completion):
    span = Mock()
    if fail_completion:
        span.on_chain_end.side_effect = RuntimeError("trace upload failed")
    start = Mock(return_value=span)
    monkeypatch.setattr(observability, "_start_provider_span", start)
    provider = Mock(spec=VisionProvider)
    provider.name = "gemini"
    result = Classification(moment_type="made_basket", confidence=0.8)
    provider.classify_window.return_value = result
    frames = ["private-frame.jpg"]
    wrapped = TracedVisionProvider(provider, "gemini")
    assert wrapped.classify_window(frames, basketball_recipe, 7) is result
    provider.classify_window.assert_called_once_with(frames, basketball_recipe, window_index=7)
    assert wrapped.name == provider.name
    metadata = start.call_args.args[1]
    assert metadata["candidate_index"] == 7
    assert metadata["frame_count"] == 1
    assert metadata["provider_fallback"] is False
    assert "private-frame.jpg" not in repr(start.call_args)
    span.on_chain_end.assert_called_once_with({})


@pytest.mark.parametrize("output", ["generated text", ""])
def test_llm_preserves_arguments_and_empty_responses(monkeypatch, output):
    span = Mock()
    span.on_chain_end.side_effect = RuntimeError("trace upload failed")
    start = Mock(return_value=span)
    monkeypatch.setattr(observability, "_start_provider_span", start)
    provider = Mock(spec=LLMProvider)
    provider.name = "nebius"
    provider.generate.return_value = output
    wrapped = TracedLLMProvider(provider, "nebius")
    assert wrapped.generate("private prompt", system="private system", max_tokens=42) is output
    provider.generate.assert_called_once_with("private prompt", system="private system", max_tokens=42)
    assert "private prompt" not in repr(start.call_args)
    assert "private system" not in repr(start.call_args)


def test_exception_preserved_when_error_tracing_fails(monkeypatch):
    span = Mock()
    span.on_chain_error.side_effect = RuntimeError("trace failed")
    monkeypatch.setattr(observability, "_start_provider_span", lambda *args: span)
    provider = Mock(spec=LLMProvider)
    provider.name = "custom"
    original = ValueError("private provider error")
    provider.generate.side_effect = original
    with pytest.raises(ValueError) as caught:
        TracedLLMProvider(provider, "custom").generate("prompt")
    assert caught.value is original
    provider.generate.assert_called_once()
    assert str(span.on_chain_error.call_args.args[0]) == "ValueError"


def test_span_start_failure_never_retries_call(monkeypatch):
    monkeypatch.setattr("langgraph.config.get_config", Mock(side_effect=RuntimeError("no config")))
    call = Mock(return_value="result")
    assert observability.trace_provider_call(call, name="test", metadata={}) == "result"
    call.assert_called_once()


@pytest.mark.parametrize("enabled,key", [(False, "key"), (True, "")])
def test_factory_without_tracing_returns_original_types(enabled, key):
    settings = Settings(tracing_enabled=enabled, langsmith_api_key=key)
    assert type(get_vision_provider(settings)) is MockVisionProvider
    assert type(get_llm_provider(settings)) is MockLLMProvider


@pytest.mark.parametrize("vendor", ["gemini", "groq", "nebius"])
@pytest.mark.parametrize("kind", ["vision", "llm"])
def test_caught_provider_failure_keeps_fallback_and_resets_context(
    monkeypatch, basketball_recipe, vendor, kind,
):
    from hypereel.providers.gemini import GeminiVisionProvider, GeminiLLMProvider
    from hypereel.providers.groq import GroqVisionProvider, GroqLLMProvider
    from hypereel.providers.nebius import OpenAICompatVisionProvider, OpenAICompatLLMProvider

    classes = {
        "gemini": (GeminiVisionProvider, GeminiLLMProvider),
        "groq": (GroqVisionProvider, GroqLLMProvider),
        "nebius": (OpenAICompatVisionProvider, OpenAICompatLLMProvider),
    }
    cls = classes[vendor][0 if kind == "vision" else 1]
    provider = cls.__new__(cls)  # SDK-free stub with the real implementation
    provider.name = vendor
    if vendor == "gemini":
        provider._model = Mock()
        request = provider._model.generate_content
    else:
        provider._model = "test-model"
        provider._client = Mock()
        request = provider._client.chat.completions.create
    request.side_effect = TimeoutError("secret upstream body")
    span = Mock()
    monkeypatch.setattr(observability, "_start_provider_span", lambda *args: span)
    if kind == "vision":
        wrapped = TracedVisionProvider(provider, vendor)
        result = wrapped.classify_window([], basketball_recipe)
        assert result.confidence == 0
        assert result.reason == f"{vendor} error: secret upstream body"
    else:
        wrapped = TracedLLMProvider(provider, vendor)
        assert wrapped.generate("prompt") == ""
    request.assert_called_once()
    assert str(span.on_chain_error.call_args.args[0]) == "caught provider failure: TimeoutError"
    span.on_chain_end.assert_not_called()
    # A subsequent successful call must not inherit the previous failure.
    span.reset_mock()
    assert observability.trace_provider_call(lambda: "ok", name="next", metadata={}) == "ok"
    span.on_chain_end.assert_called_once_with({})
    span.on_chain_error.assert_not_called()
