"""Post-metric judge tests use fake providers and a fake LangSmith transport."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json

import pytest
from langsmith import Client

from hypereel.config import Settings
from hypereel.evaluation import judge as module
from hypereel.evaluation.dataset import load_dataset
from hypereel.evaluation.runner import run_selection, run_pipeline_evaluation
from hypereel.evaluation.report import write_report, append_iteration_history, _judge_markdown
from hypereel.evaluation.cli import main
from hypereel.providers.base import LLMProvider
from hypereel.providers.traced import TracedLLMProvider
from hypereel.recipe import load_recipe

ROOT = Path(__file__).resolve().parents[4]
SMOKE = ROOT / 'evals/datasets/smoke.jsonl'
VALID = dict(relevance=.8, coverage=None, coherence=.7, diversity=.6, overall_score=.75,
             reasoning='Based only on the edit list; coverage evidence is incomplete.',
             recommendations=['Review action boundaries.'])


class FakeProvider(LLMProvider):
    name = 'test'
    def __init__(self, response=None):
        self.response = json.dumps(VALID) if response is None else response
        self.calls = []
    def generate(self, prompt, *, system='', max_tokens=800):
        self.calls.append((prompt, system, max_tokens))
        return self.response


@pytest.fixture
def context():
    case = load_dataset(SMOKE)[0]
    case.exhaustive = False
    recipe = load_recipe(SMOKE.parent / case.recipe_path)
    result = dict(status='success', clips=[], metrics={'candidate_recall': .5},
                  budget_seconds=24, trace_execution_id='pipeline-parent')
    settings = Settings(provider_spend_ledger_path='')
    return result, recipe, case, settings


def assess(context):
    return module.judge_reel(*context, evaluation_run_id='evaluation-id')


@pytest.mark.parametrize('patch', [
    {'relevance': 1.1}, {'relevance': -1}, {'relevance': True}, {'relevance': '0.5'},
    {'overall_score': float('nan')}, {'overall_score': float('inf')},
    {'reasoning': ''}, {'recommendations': 'do better'}, {'unexpected': 1},
])
def test_schema_rejects_invalid_outputs(patch):
    with pytest.raises(module.JudgeResponseError):
        module.parse_assessment(json.dumps({**VALID, **patch}))


@pytest.mark.parametrize('raw', ['{}', 'not json', '[]', 'prefix ' + json.dumps(VALID),
                                '{"relevance":0,"relevance":1}'])
def test_parser_rejects_missing_malformed_or_duplicate_keys(raw):
    with pytest.raises(module.JudgeResponseError):
        module.parse_assessment(raw)


def test_valid_fenced_json_and_abstention():
    assert module.parse_assessment('```json\n' + json.dumps(VALID) + '\n```').coverage is None
    assert set(module.parse_assessment(json.dumps(VALID)).model_dump()) == set(VALID)


def test_judge_is_read_only_and_uses_post_metric_evidence(context, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: provider)
    before = deepcopy(context[0])
    result = assess(context)
    assert result['status'] == 'success'
    assert result['assessment'] == VALID
    assert context[0] == before
    assert len(provider.calls) == 1
    prompt, system, _ = provider.calls[0]
    assert '"candidate_recall": 0.5' in prompt
    assert 'NOT video or audio' in system
    assert 'untrusted data' in system
    assert result['basis'] == 'metadata_only'


@pytest.mark.parametrize('raw,status', [('', 'unavailable'), ('bad json', 'invalid_response')])
def test_provider_response_failures_do_not_become_scores(context, monkeypatch, raw, status):
    provider = FakeProvider(raw)
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: provider)
    result = assess(context)
    assert result['status'] == status
    assert result['assessment'] is None
    assert len(provider.calls) == 1


def test_coverage_requires_evidence(context, monkeypatch):
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: FakeProvider(json.dumps({**VALID, 'coverage': 1})))
    assert assess(context)['status'] == 'invalid_response'
    context[2].exhaustive = True
    assert assess(context)['assessment']['coverage'] == 1


def test_mock_and_failed_cases_skip_without_generation(context, monkeypatch):
    provider = FakeProvider()
    provider.name = 'mock'
    factory = Mock(return_value=provider)
    monkeypatch.setattr(module, 'get_llm_provider', factory)
    assert assess(context)['status'] == 'skipped'
    assert not provider.calls
    factory.reset_mock()
    context[0]['status'] = 'failed'
    assert assess(context)['status'] == 'skipped'
    factory.assert_not_called()


def test_request_budget_is_respected(context, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: provider)
    context[0]['provider_attempted_calls'] = 2
    context[3].max_provider_calls = 2
    assert assess(context)['status'] == 'unavailable'
    assert not provider.calls


@pytest.mark.parametrize('upload_fails', [False, True])
def test_traces_include_judge_and_nested_provider(context, monkeypatch, upload_fails):
    client = Mock(spec=Client)
    if upload_fails:
        client.create_run.side_effect = RuntimeError('transport unavailable')
        client.update_run.side_effect = RuntimeError('transport unavailable')
    monkeypatch.setattr('langsmith.Client', Mock(return_value=client))
    settings = context[3]
    settings.tracing_enabled = True
    settings.langsmith_api_key = 'fake-key'
    settings.llm_provider = 'groq'
    settings.groq_vision_model = 'judge-model-v1'
    provider = FakeProvider()
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: TracedLLMProvider(provider, 'test'))
    result = assess(context)
    assert result['status'] == 'success'
    assert len(provider.calls) == 1
    runs = [c.kwargs for c in client.create_run.call_args_list]
    root = next(r for r in runs if r['name'] == 'evaluation.llm_judge')
    child = next(r for r in runs if r['name'] == 'provider.llm.test')
    assert str(child['parent_run_id']) == str(root['id'])
    assert root['extra']['metadata']['evaluation_run_id'] == 'evaluation-id'
    assert root['extra']['metadata']['pipeline_trace_execution_id'] == 'pipeline-parent'
    assert root['extra']['metadata']['judge_configured_model'] == result['configured_model'] == 'judge-model-v1'


def test_runner_opt_in_preserves_metrics_and_reports_results(tmp_path, monkeypatch):
    provider = FakeProvider()
    factory = Mock(return_value=provider)
    monkeypatch.setattr(module, 'get_llm_provider', factory)
    settings = Settings(provider_spend_ledger_path='')
    baseline = run_selection(SMOKE, settings=settings)
    factory.assert_not_called()
    judged = run_selection(SMOKE, settings=settings, judge=True)
    assert len(provider.calls) == 3
    for old, new in zip(baseline['cases'], judged['cases']):
        assert old['metrics'] == new['metrics']
        assert old['clips'] == new['clips']
        assert old['status'] == new['status']
        assert new['llm_judge']['status'] == 'success'
    write_report(judged, tmp_path / 'report')
    summary = (tmp_path / 'report/summary.md').read_text()
    assert '## LLM-as-a-Judge' in summary
    assert '0.7500' in summary
    assert VALID['reasoning'] in summary
    assert VALID['recommendations'][0] in summary
    saved = json.loads((tmp_path / 'report/report.json').read_text())
    assert saved['cases'][0]['llm_judge']['assessment'] == VALID
    append_iteration_history(judged, tmp_path / 'history.jsonl', 'judge test')
    assert json.loads((tmp_path / 'history.jsonl').read_text())['cases'][0]['llm_judge']['status'] == 'success'


def test_pipeline_judge_runs_after_graph_and_is_not_production_critic(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: provider)
    example = ROOT / 'src/hypereel/evaluation/examples/pipeline_smoke.jsonl'
    settings = Settings(provider_spend_ledger_path='')
    baseline = run_pipeline_evaluation(example, settings=settings)
    judged = run_pipeline_evaluation(example, settings=settings, judge=True)
    assert len(provider.calls) == 1
    assert judged['cases'][0]['metrics'] == baseline['cases'][0]['metrics']
    assert judged['cases'][0]['artifact_status'] == 'paused_before_render'
    assert judged['cases'][0]['judge_verdict'] == baseline['cases'][0]['judge_verdict']
    assert judged['cases'][0]['llm_judge']['status'] == 'success'


def test_judge_spend_cannot_exhaust_later_pipeline_cases(tmp_path, monkeypatch):
    from hypereel.observability import (
        authorize_provider_call, provider_budget_scope, record_provider_usage,
    )

    dataset = tmp_path / 'pipeline.jsonl'
    dataset.write_text('\n'.join(json.dumps(dict(
        case_id=str(i), synthetic=True, source='demo://test',
        recipe_path=str(ROOT / 'recipes/basketball_player.yaml'),
    )) for i in range(2)))
    executions = []

    def spend():
        authorize_provider_call(provider='test', model='test', operation='test')
        record_provider_usage(SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=40000, completion_tokens=0,
        )))

    def pipeline(case, recipe, settings, dataset_dir):
        with provider_budget_scope(settings) as usage:
            spend()  # $0.40 per case; both fit within the $1 shared cap.
        executions.append(case.case_id)
        return dict(status='success', clips=[], metrics={'clip_count': 0},
                    budget_seconds=30, provider_attempted_calls=usage['attempted_calls'],
                    estimated_provider_spend_usd=usage['estimated_spend_usd'])

    class BudgetedProvider(FakeProvider):
        def generate(self, prompt, **kwargs):
            executions.append('judge')
            spend()
            return super().generate(prompt, **kwargs)

    monkeypatch.setattr('hypereel.evaluation.pipeline.evaluate_pipeline_case', pipeline)
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: BudgetedProvider())
    reports = []
    for enabled in (False, True):
        executions.clear()
        settings = Settings(max_provider_spend_usd=1, provider_call_reserve_usd=.25,
                            provider_spend_ledger_path=str(tmp_path / f'{enabled}.json'))
        reports.append(run_pipeline_evaluation(dataset, settings=settings, judge=enabled))

    baseline, judged = reports
    assert executions == ['0', '1', 'judge', 'judge']
    assert judged['operational_success_rate'] == baseline['operational_success_rate'] == 1
    for old, new in zip(baseline['cases'], judged['cases']):
        assert new['status'] == old['status'] == 'success'
        assert new['metrics'] == old['metrics']
        assert new['clips'] == old['clips']
        assert new['llm_judge']['status'] == 'unavailable'
    assert json.loads((tmp_path / 'True.json').read_text())['estimated_spend_usd'] == .8


def test_cli_judge_flag_and_error_report(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: FakeProvider('malformed'))
    assert main(['run', '--dataset', str(SMOKE), '--judge', '--output', str(tmp_path / 'report')]) == 0
    summary = (tmp_path / 'report/summary.md').read_text()
    assert r'invalid\_response' in summary
    assert 'N/A' in summary
    assert 'failed JSON/schema validation' in summary


def test_provider_exception_is_sanitized_and_never_retried(context, monkeypatch):
    provider = FakeProvider()
    provider.generate = Mock(side_effect=RuntimeError('private upstream body'))
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: provider)
    result = assess(context)
    assert result['status'] == 'unavailable'
    assert result['assessment'] is None
    assert 'private upstream body' not in json.dumps(result)
    provider.generate.assert_called_once()


def test_no_ledger_spend_cap_includes_pipeline_spend(context, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: provider)
    context[0]['estimated_provider_spend_usd'] = 1
    context[3].max_provider_spend_usd = 1
    assert assess(context)['status'] == 'unavailable'
    assert not provider.calls


def test_invalid_response_is_traced_as_judge_failure(context, monkeypatch):
    client = Mock(spec=Client)
    monkeypatch.setattr('langsmith.Client', Mock(return_value=client))
    context[3].tracing_enabled = True
    context[3].langsmith_api_key = 'fake-key'
    provider = FakeProvider('invalid')
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: TracedLLMProvider(provider, 'test'))
    assert assess(context)['status'] == 'invalid_response'
    root = next(c.kwargs for c in client.create_run.call_args_list
                if c.kwargs['name'] == 'evaluation.llm_judge')
    ending = next(c.kwargs for c in client.update_run.call_args_list
                  if str(c.kwargs['run_id']) == str(root['id']))
    assert 'JudgeResponseError' in ending['error']
    assert ending['end_time'] >= root['start_time']
    assert len(provider.calls) == 1


def test_judge_report_escapes_untrusted_markdown():
    text = '![image](https://example.invalid/pixel) [link](https://example.invalid)'
    assessment = {**VALID, 'reasoning': text,
                  'recommendations': ['\\![image](url) | `code`\n<img src="url">']}
    lines = _judge_markdown([{'case_id': 'case', 'llm_judge': {
        'status': 'success', 'assessment': assessment,
    }}])
    row = lines[-1]
    assert text not in row
    assert r'\!\[image\]\(https://example.invalid/pixel\)' in row
    assert r'\[link\]\(https://example.invalid\)' in row
    assert r'\\\!\[image\]\(url\) \| \`code\`' in row
    assert '<img' not in row
    assert '&lt;img' in row
    assert '\n' not in row
    assert assessment['reasoning'] == text


@pytest.mark.parametrize('provider_name', ['gemini', 'groq'])
def test_uninstrumented_usage_is_unavailable(context, monkeypatch, provider_name):
    provider = FakeProvider()
    provider.name = provider_name
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: provider)
    result = assess(context)
    assert result['status'] == 'success'
    assert len(provider.calls) == 1
    assert result['usage_accounting_available'] is False
    assert result['provider_attempted_calls'] is None
    assert result['estimated_provider_spend_usd'] is None
    assert result['provider_usage'] is None


@pytest.mark.parametrize('fail', [False, True])
def test_instrumented_usage_preserves_known_counts_and_cost(context, monkeypatch, fail):
    from hypereel.observability import authorize_provider_call, record_provider_usage

    class InstrumentedProvider(FakeProvider):
        def generate(self, prompt, **kwargs):
            authorize_provider_call(provider='test', model='test', operation='judge')
            if fail:
                raise RuntimeError('provider failed before usage was returned')
            record_provider_usage(SimpleNamespace(usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=20,
            )))
            return super().generate(prompt, **kwargs)

    monkeypatch.setattr(module, 'get_llm_provider', lambda _: InstrumentedProvider())
    result = assess(context)
    assert result['status'] == ('unavailable' if fail else 'success')
    assert result['provider_attempted_calls'] == 1
    assert len(result['provider_usage']) == 1
    assert result['usage_accounting_available'] is (not fail)
    if fail:
        assert result['estimated_provider_spend_usd'] is None
    else:
        assert result['estimated_provider_spend_usd'] == pytest.approx(.0016)


@pytest.mark.parametrize('provider_name,model_field', [
    ('gemini', 'gemini_model'), ('groq', 'groq_vision_model'),
    ('nebius', 'nebius_model'), ('fireworks', 'fireworks_model'),
])
def test_configured_model_provenance(context, monkeypatch, provider_name, model_field):
    context[3].llm_provider = provider_name
    provider = FakeProvider()
    provider.name = provider_name
    monkeypatch.setattr(module, 'get_llm_provider', lambda _: provider)
    for model in ('model-v1', 'model-v2'):
        setattr(context[3], model_field, model)
        result = assess(context)
        assert result['configured_model'] == model
    provider.name = 'mock'
    result = assess(context)
    assert result['status'] == 'skipped'
    assert result['actual_provider'] == 'mock'
    assert result['configured_model'] == 'model-v2'
