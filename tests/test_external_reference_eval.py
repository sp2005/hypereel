"""Integrity and quality-metric tests for the externally labeled benchmark."""
import json
import hashlib
from collections import Counter
from pathlib import Path

import pytest

from hypereel.evaluation.dataset import load_dataset
from hypereel.evaluation.metrics import selection_metrics
from hypereel.evaluation.schemas import PipelineCase
from hypereel.models import Clip


DATASETS = Path(__file__).resolve().parents[1] / "evals" / "datasets"
STEM = "externally-labeled-east-bay-elite-11u-2025-26-vs-spartans-2026-03-21-71c9f6"
RAW = DATASETS / f"{STEM}.raw.json"
PIPELINE = DATASETS / f"{STEM}.pipeline.jsonl"
SECOND_STEM = "externally-labeled-unlimited-vs-campus-2026-03-14-404790"
SECOND_RAW = DATASETS / f"{SECOND_STEM}.raw.json"
COMBINED = DATASETS / "externally-labeled-real-video-benchmark.pipeline.jsonl"
GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden"
HOLDOUT = Path(__file__).resolve().parents[1] / "evals" / "holdout"


def test_raw_reference_feed_is_complete_and_ordered():
    data = json.loads(RAW.read_text())
    events = data["events"]
    assert data["event_count"] == len(events) == 244
    assert [event["sequence"] for event in events] == list(range(1, 245))
    assert all(event["timestamp_seconds"] >= 0 for event in events)
    assert max(event["timestamp_seconds"] for event in events) == 2803
    assert Counter(event["event_type"] for event in events) == {
        "2PT": 75, "TOV": 63, "STL": 28, "OR": 26, "FT": 22,
        "DR": 18, "AST": 7, "BLK": 3, "3PT": 2,
    }


def test_pipeline_reference_cases_validate_and_preserve_label_policy():
    player, team = load_dataset(PIPELINE, PipelineCase)
    assert player.case_id.endswith("lucky-23")
    assert player.exhaustive is False
    assert Counter(e.moment_type for e in player.reference_events or []) == {"made_basket": 1}

    assert team.exhaustive is True
    assert Counter(e.moment_type for e in team.reference_events or []) == {
        "made_basket": 20, "steal": 28, "block": 3,
    }
    assert all(e.action_start <= e.event_time <= e.action_end
               for e in team.reference_events or [])


def test_second_lighting_variant_and_combined_benchmark_are_complete():
    raw = json.loads(SECOND_RAW.read_text())
    assert raw["event_count"] == 218
    assert Counter(e["event_type"] for e in raw["events"]) == {
        "2PT": 52, "3PT": 38, "TOV": 32, "DR": 29, "FT": 17,
        "OR": 17, "STL": 17, "AST": 14, "BLK": 2,
    }
    first, second = load_dataset(COMBINED, PipelineCase)
    assert first.exhaustive and second.exhaustive
    assert len(first.reference_events or []) == 51
    assert Counter(e.moment_type for e in second.reference_events or []) == {
        "made_basket": 21, "three_pointer": 9, "steal": 17, "block": 2,
    }


def test_quality_metrics_reward_correct_output_and_penalize_errors():
    team = load_dataset(PIPELINE, PipelineCase)[1]
    references = team.reference_events or []
    event = references[0]
    correct = Clip(start=event.action_start, end=event.action_end,
                   moment_type=event.moment_type)
    false_positive = Clip(start=0, end=5, moment_type="block")
    metrics, _ = selection_metrics(
        [correct, false_positive], budget=600, video_duration=2823,
        events=references, exhaustive=True,
    )
    assert metrics["relevant_clip_precision"] == pytest.approx(0.5)
    assert metrics["selected_event_recall"] == pytest.approx(1 / 51)
    assert metrics["selection_f1"] == pytest.approx(2 * .5 * (1 / 51) / (.5 + (1 / 51)))
    assert metrics["action_completeness"] == 1


def test_wrong_label_and_truncated_action_do_not_receive_full_credit():
    event = load_dataset(PIPELINE, PipelineCase)[1].reference_events[0]
    wrong = Clip(start=event.action_start, end=event.action_end, moment_type="block")
    metrics, _ = selection_metrics(
        [wrong], budget=600, video_duration=2823, events=[event], exhaustive=True,
    )
    assert metrics["matched_event_count"] == 0
    assert metrics["selected_event_recall"] == 0
    assert metrics["relevant_clip_precision"] == 0

    truncated = Clip(start=event.event_time - 1, end=event.event_time + 1,
                     moment_type=event.moment_type)
    metrics, _ = selection_metrics(
        [truncated], budget=600, video_duration=2823, events=[event], exhaustive=True,
    )
    assert metrics["matched_event_count"] == 1
    assert metrics["action_completeness"] == 0


def test_versioned_golden_dataset_manifest_and_event_ledger_are_consistent():
    manifest = json.loads((GOLDEN / "manifest.json").read_text())
    ledger_path = GOLDEN / "golden_events.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]

    assert manifest["dataset_id"] == "hypereel-external-gold"
    assert manifest["version"] == "1.0.0"
    assert manifest["split"] == "development"
    assert manifest["event_count"] == len(rows) == 462
    assert manifest["eligible_event_count"] == sum(
        row["eligible_highlight"] for row in rows
    ) == 100
    assert manifest["negative_event_count"] == sum(
        not row["eligible_highlight"] for row in rows
    ) == 362
    assert manifest["event_ledger_sha256"] == hashlib.sha256(
        ledger_path.read_bytes()
    ).hexdigest()
    assert len({row["gold_id"] for row in rows}) == len(rows)
    assert {row["game_id"] for row in rows} == {
        "east-bay-elite-vs-spartans",
        "unlimited-vs-campus",
    }
    assert all(row["split"] == "development" for row in rows)
    assert all(row["event_label_status"] == "source-annotated" for row in rows)
    assert all(row["annotation_source"] == "external" for row in rows)
    assert all(row["boundary_status"] == "provisional" for row in rows)

    cases = load_dataset(GOLDEN / "cases" / "development.pipeline.jsonl", PipelineCase)
    assert len(cases) == 2
    assert all(case.exhaustive for case in cases)
    assert sum(len(case.reference_events or []) for case in cases) == 100


def test_final_holdout_is_complete_and_isolated_from_development():
    manifest = json.loads((HOLDOUT / "manifest.json").read_text())
    rows = [json.loads(line) for line in
            (HOLDOUT / "golden_events.jsonl").read_text().splitlines()]
    case, = load_dataset(HOLDOUT / "cases" / "final.pipeline.jsonl", PipelineCase)
    development_cases = load_dataset(
        GOLDEN / "cases" / "development.pipeline.jsonl", PipelineCase
    )

    assert manifest["split"] == "final_holdout"
    assert manifest["development_use_permitted"] is False
    assert manifest["event_count"] == len(rows) == 194
    assert manifest["eligible_event_count"] == 35
    assert manifest["negative_event_count"] == 159
    assert manifest["eligible_label_counts"] == {
        "block": 5,
        "made_basket": 10,
        "steal": 12,
        "three_pointer": 8,
    }
    assert all(row["split"] == "final_holdout" for row in rows)
    assert len(case.reference_events or []) == 35
    assert case.exhaustive is True
    assert case.case_id not in {item.case_id for item in development_cases}
    assert case.source not in {item.source for item in development_cases}
