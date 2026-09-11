"""Offline tracing integration tests: use the real tracer with a fake transport."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langsmith import Client

from hypereel import observability
from hypereel.config import Settings, get_settings
from hypereel.graph.build import build_graph, run_pipeline
from hypereel.graph.state import new_state


@pytest.fixture
def tracing_settings(tmp_path):
    return Settings(
        tracing_enabled=True, langsmith_api_key="test-key",
        output_dir=str(tmp_path / "output"), download_dir=str(tmp_path / "downloads"),
        memory_path=str(tmp_path / "memory.json"),
    )


@pytest.fixture
def transport(monkeypatch):
    client = Mock(spec=Client)
    constructor = Mock(return_value=client)
    monkeypatch.setattr("langsmith.Client", constructor)
    return client, constructor


def test_configuration_is_opt_in(monkeypatch):
    assert get_settings().tracing_enabled is False
    monkeypatch.setenv("HYPEREEL_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-secret")
    monkeypatch.setenv("LANGSMITH_PROJECT", "team-project")
    settings = get_settings()
    assert settings.tracing_enabled
    assert settings.langsmith_project == "team-project"
    assert "test-secret" not in repr(settings)


@pytest.mark.parametrize("enabled,key", [(False, "test-key"), (True, "")])
def test_disabled_or_missing_key_does_not_initialize(monkeypatch, enabled, key):
    factory = Mock(side_effect=AssertionError("must not initialize"))
    monkeypatch.setattr(observability, "_make_tracer", factory)
    config = {"configurable": {"thread_id": "unchanged"}}
    assert observability.with_tracing(
        config, Settings(tracing_enabled=enabled, langsmith_api_key=key),
        recipe_id="test", entrypoint="test",
    ) is config
    factory.assert_not_called()


@pytest.mark.parametrize("error", [ImportError("missing SDK"), RuntimeError("init failed")])
def test_initialization_failure_is_a_noop(monkeypatch, tracing_settings, error):
    monkeypatch.setattr(observability, "_make_tracer", Mock(side_effect=error))
    config = {"configurable": {"thread_id": "unchanged"}}
    assert observability.with_tracing(
        config, tracing_settings, recipe_id="test", entrypoint="test",
    ) is config


def test_config_preserves_caller_fields_and_hides_payloads(transport, tracing_settings):
    _, constructor = transport
    config = {"configurable": {"thread_id": "original"}, "tags": ["caller"],
              "metadata": {"custom": "value"}, "callbacks": []}
    traced = observability.with_tracing(
        config, tracing_settings, recipe_id="basketball", entrypoint="test",
    )
    assert traced["configurable"] == config["configurable"]
    assert config["callbacks"] == []
    assert config["tags"] == ["caller"]
    assert config["metadata"] == {"custom": "value"}
    assert traced["metadata"]["custom"] == "value"
    assert traced["metadata"]["recipe_id"] == "basketball"
    assert traced["callbacks"][0].raise_error is False
    assert constructor.call_args.kwargs["hide_inputs"] is True
    assert constructor.call_args.kwargs["hide_outputs"] is True
    second = observability.with_tracing(
        config, tracing_settings, recipe_id="basketball", entrypoint="test",
    )
    assert traced["metadata"]["hypereel_execution_id"] != second["metadata"]["hypereel_execution_id"]


@pytest.mark.parametrize("upload_fails", [False, True])
def test_runner_results_and_call_counts_unchanged(
    transport, tracing_settings, basketball_recipe, mocker, upload_fails,
):
    from hypereel.providers.mock import MockVisionProvider, MockLLMProvider

    client, _ = transport
    if upload_fails:
        client.create_run.side_effect = RuntimeError("offline")
        client.update_run.side_effect = RuntimeError("offline")
    vision = mocker.spy(MockVisionProvider, "classify_window")
    llm = mocker.spy(MockLLMProvider, "generate")
    baseline = run_pipeline("demo://test", basketball_recipe,
                            settings=replace(tracing_settings, tracing_enabled=False))
    counts = (vision.call_count, llm.call_count)
    traced = run_pipeline("demo://test", basketball_recipe, settings=tracing_settings)
    assert traced.model_dump() == baseline.model_dump()
    assert (vision.call_count, llm.call_count) == tuple(n * 2 for n in counts)
    assert client.create_run.called


@pytest.mark.parametrize("streaming", [False, True])
def test_native_traces_include_nodes_and_preserve_gates(
    transport, tracing_settings, basketball_recipe, streaming,
):
    client, _ = transport
    graph = build_graph(settings=tracing_settings)
    config = observability.with_tracing(
        {"configurable": {"thread_id": "fixed"}}, tracing_settings,
        recipe_id=basketball_recipe.id, entrypoint="test",
    )

    def advance(value):
        if streaming:
            list(graph.stream(value, config, stream_mode="updates"))
        else:
            graph.invoke(value, config)

    advance(new_state("demo://test", basketball_recipe))
    assert graph.get_state(config).next == ("approve_clips",)
    graph.update_state(config, {"approved": True, "needs_human": False})
    advance(None)
    assert graph.get_state(config).next == ("approve_share",)
    graph.update_state(config, {"shared": True, "needs_human": False})
    advance(None)
    assert not graph.get_state(config).next
    starts = [call.kwargs for call in client.create_run.call_args_list]
    names = {run["name"] for run in starts}
    assert {
        "plan", "ingest", "propose", "scoreboard", "classify", "select", "judge",
        "approve_clips", "render", "summarize", "approve_share", "deliver",
    } <= names
    endings = {str(call.kwargs["run_id"]): call.kwargs for call in client.update_run.call_args_list}
    for run in starts:
        assert endings[str(run["id"])]["end_time"] >= run["start_time"]
    execution_id = config["metadata"]["hypereel_execution_id"]
    assert all(run["extra"]["metadata"]["hypereel_execution_id"] == execution_id for run in starts)


def test_streamlit_start_retains_tracing_config(
    monkeypatch, transport, tracing_settings, basketball_recipe,
):
    import hypereel.app as ui

    monkeypatch.setattr(ui, "get_settings", lambda: tracing_settings)
    monkeypatch.setattr(ui, "build_graph", lambda: build_graph(settings=tracing_settings))
    st = SimpleNamespace(session_state=SimpleNamespace())
    ui._start_run(st, basketball_recipe, "demo://test", None, None)
    config = st.session_state.config
    assert config["metadata"]["hypereel_entrypoint"] == "streamlit"
    assert st.session_state.app.get_state(config).next == ("approve_clips",)
    assert transport[0].create_run.called


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_spans_are_children_and_identify_mock_fallback(
    transport, tracing_settings, basketball_recipe, streaming,
):
    client, constructor = transport
    settings = replace(tracing_settings, vision_provider="gemini", llm_provider="nebius")
    graph = build_graph(settings=settings)
    config = observability.with_tracing(
        {"configurable": {"thread_id": "provider-spans"}}, settings,
        recipe_id=basketball_recipe.id, entrypoint="test",
    )
    initial = new_state("demo://test", basketball_recipe)
    if streaming:
        list(graph.stream(initial, config, stream_mode="updates"))
    else:
        graph.invoke(initial, config)
    snapshot = graph.get_state(config)
    runs = [call.kwargs for call in client.create_run.call_args_list]
    by_id = {str(run["id"]): run for run in runs}
    vision = [run for run in runs if run["name"] == "provider.vision.mock"]
    llm = [run for run in runs if run["name"] == "provider.llm.mock"]
    assert len(vision) == len(snapshot.values["candidates"])
    assert len(llm) == 1  # judge; summary is after the approval gate
    for run in vision + llm:
        parent = by_id[str(run["parent_run_id"])]
        assert parent["name"] == ("classify" if run in vision else "judge")
        metadata = run["extra"]["metadata"]
        assert metadata["actual_provider"] == "mock"
        assert metadata["requested_provider"] == ("gemini" if run in vision else "nebius")
        assert metadata["provider_fallback"] is True
        assert metadata["provider_mock"] is True
        assert run["inputs"] == {}
    assert [run["extra"]["metadata"]["candidate_index"] for run in vision] == list(range(len(vision)))
    assert all(run["extra"]["metadata"]["frame_count"] == 0 for run in vision)
    constructor.assert_called_once()  # shared client; no client per provider call


def test_provider_health_calls_outside_graph_do_not_trace(
    transport, tracing_settings, basketball_recipe,
):
    from hypereel.providers.factory import get_vision_provider, get_llm_provider

    assert get_vision_provider(tracing_settings).classify_window([], basketball_recipe)
    assert get_llm_provider(tracing_settings).generate("health check")
    transport[1].assert_not_called()
    transport[0].create_run.assert_not_called()


def test_revision_executions_are_each_traced(
    transport, tracing_settings, basketball_recipe, monkeypatch,
):
    from hypereel.providers.mock import MockLLMProvider

    monkeypatch.setattr(MockLLMProvider, "generate", lambda *args, **kwargs:
                        '{"verdict":"revise","action":"broaden"}')
    run_pipeline("demo://test", basketball_recipe, settings=tracing_settings)
    runs = [call.kwargs for call in transport[0].create_run.call_args_list]
    assert sum(run["name"] == "select" for run in runs) == 2
    assert sum(run["name"] == "judge" for run in runs) == 2
    assert sum(run["name"] == "provider.llm.mock" for run in runs) == 3


def test_node_exception_is_traced_without_changing_result(
    transport, tracing_settings, basketball_recipe, monkeypatch,
):
    import importlib

    build = importlib.import_module("hypereel.graph.build")
    monkeypatch.setattr(build, "plan_node", Mock(side_effect=ValueError("test node failure")))
    baseline = run_pipeline("demo://test", basketball_recipe,
                            settings=replace(tracing_settings, tracing_enabled=False))
    traced = run_pipeline("demo://test", basketball_recipe, settings=tracing_settings)
    assert traced.model_dump() == baseline.model_dump()
    starts = [call.kwargs for call in transport[0].create_run.call_args_list]
    plan = next(run for run in starts if run["name"] == "plan")
    ending = next(call.kwargs for call in transport[0].update_run.call_args_list
                  if str(call.kwargs["run_id"]) == str(plan["id"]))
    assert "ValueError" in ending["error"]
    assert ending["end_time"] >= plan["start_time"]


def test_caught_live_provider_errors_are_llm_error_spans(
    transport, tracing_settings, basketball_recipe, monkeypatch,
):
    from hypereel.providers import factory
    from hypereel.providers.nebius import OpenAICompatLLMProvider, OpenAICompatVisionProvider

    def fake_provider(cls):
        provider = cls.__new__(cls)
        provider.name = "nebius"
        provider._model = "test-model"
        provider._client = Mock()
        provider._client.chat.completions.create.side_effect = TimeoutError("private API details")
        return provider

    monkeypatch.setattr(factory, "_get_vision_provider", lambda settings: fake_provider(OpenAICompatVisionProvider))
    monkeypatch.setattr(factory, "_get_llm_provider", lambda settings: fake_provider(OpenAICompatLLMProvider))
    baseline = run_pipeline("demo://test", basketball_recipe,
                            settings=replace(tracing_settings, tracing_enabled=False))
    traced = run_pipeline("demo://test", basketball_recipe, settings=tracing_settings)
    assert traced.model_dump() == baseline.model_dump()
    starts = [call.kwargs for call in transport[0].create_run.call_args_list]
    model_runs = [run for run in starts if run["name"].startswith("provider.")]
    assert {run["name"] for run in model_runs} == {"provider.vision.nebius", "provider.llm.nebius"}
    endings = {str(call.kwargs["run_id"]): call.kwargs for call in transport[0].update_run.call_args_list}
    for run in model_runs:
        assert run["run_type"] == "llm"
        error = endings[str(run["id"])]["error"]
        assert "TimeoutError" in error
        assert "private API details" not in error
