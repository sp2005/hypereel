"""Candidate recall and operational health, without production changes."""
import json
from pathlib import Path

import pytest

from hypereel.config import Settings
from hypereel.models import CandidateWindow
from hypereel.evaluation.metrics import candidate_recall, operational_success_rate
from hypereel.evaluation.schemas import ReferenceEvent
from hypereel.evaluation.report import aggregate, write_report
from hypereel.evaluation.runner import run_selection, run_pipeline_evaluation

ROOT = Path(__file__).resolve().parents[4]


def event(key, t):
    return ReferenceEvent(event_id=key, event_time=t, action_start=t,
                          action_end=t+1, moment_type='block')


def test_recall_uses_all_candidates_and_counts_each_event_once():
    windows = [CandidateWindow(start=0, end=10), CandidateWindow(start=0, end=10),
               CandidateWindow(start=5, end=15)]
    assert candidate_recall(windows, [event('a', 1), event('b', 12), event('c', 20)]) == pytest.approx(2/3)


def test_recall_boundaries_are_start_inclusive_end_exclusive():
    assert candidate_recall([CandidateWindow(start=5, end=10)], [event('start', 5), event('end', 10)]) == .5


@pytest.mark.parametrize('references', [None, []])
def test_recall_without_reference_events_is_na(references):
    assert candidate_recall([CandidateWindow(start=0, end=10)], references) is None


def test_recall_without_candidates_is_zero_when_events_exist():
    assert candidate_recall([], [event('a', 1)]) == 0


@pytest.mark.parametrize('statuses,expected', [
    ([], None), (['success'], 1), (['failed', 'degraded'], 0),
    (['success', 'failed', 'degraded'], 1/3),
])
def test_operational_rate_includes_failures_and_degradation(statuses, expected):
    assert operational_success_rate([{'status': s} for s in statuses]) == expected


def test_selection_recall_is_before_filtering_and_accepts_partial_annotations(tmp_path):
    row = json.loads((ROOT / 'evals/datasets/smoke.jsonl').read_text().splitlines()[0])
    row['recipe_path'] = str(ROOT / 'recipes/basketball_player.yaml')
    row['max_duration'] = 0  # no final clips; candidate recall must remain 1
    row['exhaustive'] = False
    path = tmp_path / 'selection.jsonl'
    path.write_text(json.dumps(row))
    report = run_selection(path)
    assert report['cases'][0]['metrics']['clip_count'] == 0
    assert report['cases'][0]['metrics']['candidate_recall'] == 1
    assert report['cases'][0]['metrics']['relevant_clip_precision'] is None
    assert report['operational_success_rate'] == 1
    assert report['metric_version'] == 'selection-v4'


def test_pipeline_recall_and_degraded_suppression(tmp_path):
    base = dict(schema_version=1, case_id='synthetic', synthetic=True,
                recipe_path=str(ROOT / 'recipes/basketball_player.yaml'),
                source='demo://test', max_duration=24,
                reference_events=[event('a', 1).model_dump()])
    degraded = dict(base, case_id='missing', synthetic=False, source='missing.mp4')
    failed = dict(base, case_id='failed', recipe_path='missing.yaml')
    path = tmp_path / 'pipeline.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in (base, degraded, failed)))
    report = run_pipeline_evaluation(path, settings=Settings())
    assert report['cases'][0]['metrics']['candidate_recall'] == 1
    assert report['cases'][1]['metrics']['candidate_recall'] is None
    assert report['operational_success_rate'] == pytest.approx(1/3)
    write_report(report, tmp_path / 'report')
    saved = json.loads((tmp_path / 'report/report.json').read_text())
    assert saved['operational_success_rate'] == pytest.approx(1/3)
    assert saved['aggregates']['synthetic']['operational_success_rate'] == .5
    assert saved['aggregates']['non_synthetic']['operational_success_rate'] == 0
    summary = (tmp_path / 'report/summary.md').read_text()
    assert 'Operational success rate: **0.3333** (1/3 attempted cases)' in summary
    assert '| candidate_recall | 1.0000 | 1 |' in summary
    assert 'Candidate recall | Elapsed seconds' in summary
    assert 'N/A' in summary


def test_aggregation_does_not_condition_health_on_quality_availability():
    cases = [dict(synthetic=True, status='success', metrics={'candidate_recall': None}),
             dict(synthetic=True, status='degraded', metrics={'candidate_recall': 1}),
             dict(synthetic=True, status='failed', metrics={})]
    groups = aggregate(cases)
    assert groups['synthetic']['operational_success_rate'] == pytest.approx(1/3)
    assert groups['synthetic']['metrics']['candidate_recall'] == {'macro_mean': None, 'applicable_cases': 0}
    assert groups['non_synthetic']['operational_success_rate'] is None
