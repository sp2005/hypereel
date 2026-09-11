"""Offline contracts, metric edge cases, and real CLI integration."""
import json
from pathlib import Path

import pytest

from hypereel.models import Clip
from hypereel.evaluation.cli import main
from hypereel.evaluation.dataset import load_dataset
from hypereel.evaluation.metrics import (
    match_events, pipeline_diagnostic_metrics, selection_metrics,
)
from hypereel.models import CandidateWindow, Classification
from hypereel.evaluation.report import aggregate, append_iteration_history
from hypereel.evaluation.runner import run_selection
from hypereel.evaluation.schemas import ReferenceEvent
from hypereel.recipe import load_recipe
from hypereel.select.selector import score_candidates, select_clips

SMOKE = Path(__file__).resolve().parents[1] / 'evals/datasets/smoke.jsonl'


def event(key, t):
    return ReferenceEvent(event_id=key, event_time=t, action_start=t-1,
                          action_end=t+1, moment_type='made_basket')


def test_matching_uses_maximum_one_to_one_assignment():
    clips = [Clip(start=0, end=10, moment_type='made_basket'),
             Clip(start=0, end=5, moment_type='made_basket')]
    events = [event('early', 3), event('late', 8)]
    assert match_events(clips, events) == [(0, 1), (1, 0)]
    assert len(match_events(clips, [event('single', 3)])) == 1
    assert match_events([Clip(start=0, end=5, moment_type='block')], events) == []


def test_metrics_overlap_completeness_and_budget():
    clips = [Clip(start=0, end=10, moment_type='made_basket'),
             Clip(start=5, end=15, moment_type='made_basket')]
    metrics, matches = selection_metrics(clips, budget=19, video_duration=20,
                                        events=[event('e', 3)], exhaustive=True)
    assert metrics['selected_duration_seconds'] == 20
    assert metrics['budget_compliance'] is False
    assert metrics['overlap_rate'] == .25
    assert metrics['relevant_clip_precision'] == .5
    assert metrics['selected_event_recall'] == 1
    assert metrics['selection_f1'] == pytest.approx(2 / 3)
    assert metrics['action_completeness'] == 1
    assert len(matches) == 1


def test_extended_detection_temporal_and_confusion_metrics():
    refs = [event('correct', 5), event('missed', 15)]
    clips = [Clip(start=4, end=6, moment_type='made_basket'),
             Clip(start=14, end=16, moment_type='block'),
             Clip(start=18, end=20, moment_type='made_basket')]
    metrics, _ = selection_metrics(
        clips, budget=30, video_duration=60, events=refs, exhaustive=True,
    )
    assert metrics['micro_f1'] == pytest.approx(0.4)
    assert metrics['false_positives_per_video_minute'] == 2
    assert metrics['mean_temporal_iou'] == 1
    assert metrics['event_recall_at_iou_0_5'] == .5
    assert metrics['confusion_matrix']['made_basket']['block'] == 1
    assert metrics['confusion_matrix']['__none__']['made_basket'] == 1


def test_pipeline_diagnostics_measure_negatives_calibration_and_funnel():
    refs = [event('event', 5)]
    candidates = [CandidateWindow(start=4, end=6), CandidateWindow(start=20, end=22)]
    classifications = [
        Classification(moment_type='made_basket', confidence=.9),
        Classification(moment_type=None, confidence=.8),
    ]
    metrics = pipeline_diagnostic_metrics(
        candidates, classifications, refs, exhaustive=True, selected_match_count=1,
    )
    assert metrics['candidate_classification_accuracy'] == 1
    assert metrics['negative_window_specificity'] == 1
    assert metrics['classification_event_recall'] == 1
    assert metrics['selection_survival_rate'] == 1
    assert metrics['candidate_event_average_precision'] == 1


def test_missing_labels_and_zero_denominators_are_na():
    metrics, _ = selection_metrics([], budget=0, video_duration=20)
    assert metrics['budget_compliance'] is True
    for name in ('budget_utilization', 'overlap_rate', 'relevant_clip_precision',
                 'selected_event_recall', 'selection_f1',
                 'action_completeness', 'matched_event_count'):
        assert metrics[name] is None
    metrics, _ = selection_metrics([Clip(start=0, end=10)], budget=10, video_duration=20,
                                  events=[], exhaustive=True)
    assert metrics['relevant_clip_precision'] == 0
    metrics, _ = selection_metrics([Clip(start=0, end=10)], budget=10, video_duration=20,
                                  events=[], exhaustive=False)
    assert metrics['relevant_clip_precision'] is None


@pytest.mark.parametrize('mutation', [
    {'classifications': []}, {'video_duration': -1}, {'max_duration': -1},
    {'schema_version': 2}, {'video_duration': float('nan')},
    {'candidates': [{'start': 10, 'end': 5}]},
])
def test_invalid_datasets_have_line_number(tmp_path, mutation):
    row = json.loads(SMOKE.read_text().splitlines()[0])
    row.update(mutation)
    path = tmp_path / 'invalid.jsonl'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='invalid.jsonl:1:'):
        load_dataset(path)


def test_duplicate_ids_and_empty_dataset_rejected(tmp_path):
    path = tmp_path / 'data.jsonl'
    row = SMOKE.read_text().splitlines()[0]
    path.write_text(row + '\n' + row)
    with pytest.raises(ValueError, match='duplicate case_id'):
        load_dataset(path)
    path.write_text('\n')
    with pytest.raises(ValueError, match='empty'):
        load_dataset(path)


def test_runner_matches_production_and_does_not_mutate_inputs():
    cases = load_dataset(SMOKE)
    before = [c.model_dump() for c in cases]
    report = run_selection(SMOKE)
    assert report['failed_count'] == 0
    assert report['case_count'] == 3
    for case, result in zip(cases, report['cases']):
        recipe = load_recipe(SMOKE.parent / case.recipe_path)
        expected = select_clips(score_candidates(case.candidates, case.classifications, recipe,
                                                video_duration=case.video_duration),
                                recipe, max_duration=case.max_duration)
        assert result['clips'] == [c.model_dump(mode='json') for c in expected]
        assert result['elapsed_seconds'] >= 0
        assert len(result['recipe_sha256']) == 64
    assert before == [c.model_dump() for c in cases]
    assert report['cases'][0]['metrics']['clip_count'] == 2
    assert report['cases'][2]['metrics']['clip_count'] == 0


def test_runner_records_failure_and_continues(tmp_path):
    rows = [json.loads(line) for line in SMOKE.read_text().splitlines()][:2]
    rows[0]['recipe_path'] = 'missing.yaml'
    rows[1]['recipe_path'] = str((SMOKE.parent / rows[1]['recipe_path']).resolve())
    path = tmp_path / 'cases.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    assert main(['run', '--dataset', str(path), '--output', str(tmp_path / 'report')]) == 1
    report = json.loads((tmp_path / 'report/report.json').read_text())
    assert report['failed_count'] == 1
    assert report['cases'][0]['status'] == 'failed'
    assert report['cases'][1]['status'] == 'success'


def test_cli_writes_reports_without_models_or_memory(tmp_path, monkeypatch):
    import hypereel.providers.factory as factory
    import hypereel.memory.store as memory
    def unexpected(*args, **kwargs):
        raise AssertionError('evaluation must not invoke models or memory')
    monkeypatch.setattr(factory, 'get_vision_provider', unexpected)
    monkeypatch.setattr(factory, 'get_llm_provider', unexpected)
    monkeypatch.setattr(memory.MemoryStore, 'record_feedback', unexpected)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'report'
    args = ['run', '--mode', 'selection', '--dataset', str(SMOKE), '--output', str(output)]
    assert main(args) == 0
    original = (output / 'report.json').read_text()
    report = json.loads(original)
    assert report['aggregates']['synthetic']['case_count'] == 3
    assert report['aggregates']['non_synthetic']['case_count'] == 0
    assert (output / 'summary.md').exists()
    assert len((output / 'cases.jsonl').read_text().splitlines()) == 3
    assert main(args) == 2  # no accidental overwrite
    assert (output / 'report.json').read_text() == original
    assert not (tmp_path / 'hypereel_memory.json').exists()


def test_aggregate_excludes_na_and_separates_groups():
    cases = [dict(synthetic=True, status='success', metrics={'precision': .5}),
             dict(synthetic=True, status='success', metrics={'precision': None}),
             dict(synthetic=False, status='success', metrics={'precision': 1})]
    result = aggregate(cases)
    assert result['synthetic']['metrics']['precision'] == {'macro_mean': .5, 'applicable_cases': 1}
    assert result['non_synthetic']['metrics']['precision']['macro_mean'] == 1


def test_iteration_history_is_append_only(tmp_path):
    history = tmp_path / 'history.jsonl'
    report = run_selection(SMOKE)
    append_iteration_history(report, history, 'baseline')
    append_iteration_history(report, history, 'repeatability check')
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [row['change_note'] for row in rows] == ['baseline', 'repeatability check']
    assert all(row['dataset_sha256'] == report['dataset_sha256'] for row in rows)
