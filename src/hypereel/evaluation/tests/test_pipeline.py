"""Graph integration tests that never call live providers or render."""
import json
from pathlib import Path

from hypereel.config import Settings
from hypereel.evaluation.runner import run_pipeline_evaluation
from hypereel.evaluation.report import write_report

ROOT = Path(__file__).resolve().parents[4]


def dataset(tmp_path, **overrides):
    row = dict(case_id='graph-demo', synthetic=True,
               recipe_path=str(ROOT / 'recipes/basketball_player.yaml'),
               source='demo://test', max_duration=24)
    row.update(overrides)
    path = tmp_path / 'pipeline.jsonl'
    path.write_text(json.dumps(row) + '\n')
    return path


def test_graph_reaches_gate_without_render_or_feedback(tmp_path, monkeypatch):
    from hypereel.graph import nodes
    from hypereel.memory.store import MemoryStore
    from hypereel.graph.build import build_graph
    from hypereel.graph.state import new_state
    from hypereel.recipe import load_recipe

    def forbidden(*args, **kwargs):
        raise AssertionError('render or memory must not run')
    monkeypatch.setattr(nodes, 'render_reel', forbidden)
    monkeypatch.setattr(MemoryStore, 'record_feedback', forbidden)
    settings = Settings(memory_path=str(tmp_path / 'normal-memory.json'))
    path = dataset(tmp_path)
    report = run_pipeline_evaluation(path, settings=settings)
    case = report['cases'][0]
    assert report['failed_count'] == 0
    assert case['status'] == 'success'
    assert case['stop_node'] == 'approve_clips'
    assert case['artifact_status'] == 'paused_before_render'
    assert case['candidate_count'] > 0
    assert case['actual_vision_provider'] == 'mock'
    assert not Path(settings.memory_path).exists()
    recipe = load_recipe(ROOT / 'recipes/basketball_player.yaml')
    graph = build_graph(settings=settings)
    config = {'configurable': {'thread_id': 'baseline'}}
    graph.invoke(new_state('demo://test', recipe, max_duration=24), config)
    expected = graph.get_state(config).values['selected_clips']
    assert case['clips'] == [c.model_dump(mode='json') for c in expected]
    write_report(report, tmp_path / 'report')
    assert 'pipeline evaluation' in (tmp_path / 'report/summary.md').read_text()


def test_missing_real_source_is_degraded_not_success(tmp_path):
    path = dataset(tmp_path, synthetic=False, source='missing.mp4')
    report = run_pipeline_evaluation(path, settings=Settings())
    assert report['degraded_count'] == 1
    case = report['cases'][0]
    assert case['status'] == 'degraded'
    assert case['metrics']['relevant_clip_precision'] is None
    assert case['pipeline_errors']


def test_case_workspace_is_isolated_and_removed(tmp_path, monkeypatch):
    import hypereel.evaluation.pipeline as adapter
    original = adapter.build_graph
    seen = []
    def capture(*, settings):
        seen.append(settings)
        return original(settings=settings)
    monkeypatch.setattr(adapter, 'build_graph', capture)
    settings = Settings(download_dir='normal-downloads', output_dir='normal-output')
    run_pipeline_evaluation(dataset(tmp_path), settings=settings)
    assert settings.download_dir == 'normal-downloads'
    assert seen[0].download_dir != settings.download_dir
    assert not Path(seen[0].download_dir).parent.exists()
