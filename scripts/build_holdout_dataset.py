"""Build the sealed final holdout from its externally labeled source export."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "evals" / "holdout"
SOURCE = HOLDOUT / "source" / (
    "externally-labeled-east-bay-elite-13u-vs-ptown-wu-2026-05-16-d13c27.raw.json"
)
GAME_ID = "east-bay-elite-13u-vs-ptown-wu"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_label(event: dict) -> str | None:
    if event["event_type"] == "2PT" and event["outcome"] == "made":
        return "made_basket"
    if event["event_type"] == "3PT" and event["outcome"] == "made":
        return "three_pointer"
    if event["event_type"] == "STL":
        return "steal"
    if event["event_type"] == "BLK":
        return "block"
    return None


def main() -> None:
    raw = json.loads(SOURCE.read_text())
    rows = []
    references = []
    for event in raw["events"]:
        label = expected_label(event)
        timestamp = event["timestamp_seconds"]
        row = {
            "gold_id": f"{GAME_ID}:{event['event_id']}",
            "dataset_version": "external-final-holdout-v1",
            "split": "final_holdout",
            "game_id": GAME_ID,
            "video_url": raw["video_url"],
            "event_time": timestamp,
            "provisional_action_start": max(0, timestamp - 5),
            "provisional_action_end": timestamp + 4,
            "source_event_type": event["event_type"],
            "source_outcome": event["outcome"],
            "player_name": event["player_name"],
            "jersey_number": event["jersey_number"],
            "source_jersey_label": event["source_jersey_label"],
            "team": event["team"],
            "eligible_highlight": label is not None,
            "expected_moment_type": label,
            "annotation_source": "external",
            "event_label_status": "source-annotated",
            "boundary_status": "provisional",
        }
        rows.append(row)
        if label:
            references.append({
                "event_id": event["event_id"],
                "event_time": timestamp,
                "action_start": max(0, timestamp - 5),
                "action_end": timestamp + 4,
                "moment_type": label,
            })

    ledger = HOLDOUT / "golden_events.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    case = {
        "schema_version": 1,
        "case_id": "external-final-holdout-east-bay-elite-13u-vs-ptown-wu",
        "synthetic": False,
        "recipe_path": "../../recipes/basketball_team_evaluation.yaml",
        "source": raw["video_url"],
        "max_duration": 600,
        "audience": "team",
        "subject_description": "Evaluate eligible events for both teams.",
        "reference_events": references,
        "exhaustive": True,
    }
    case_path = HOLDOUT / "cases" / "final.pipeline.jsonl"
    case_path.write_text(json.dumps(case) + "\n")

    label_counts = Counter(ref["moment_type"] for ref in references)
    manifest = {
        "dataset_id": "hypereel-external-final-holdout",
        "version": "1.0.0",
        "split": "final_holdout",
        "development_use_permitted": False,
        "created_from": "externally labeled game events",
        "taxonomy_recipe": "../../recipes/basketball_team_evaluation.yaml",
        "game_id": GAME_ID,
        "video_url": raw["video_url"],
        "event_count": len(rows),
        "eligible_event_count": len(references),
        "negative_event_count": len(rows) - len(references),
        "eligible_label_counts": dict(sorted(label_counts.items())),
        "source_event_counts": dict(sorted(Counter(
            event["event_type"] for event in raw["events"]
        ).items())),
        "source_sha256": sha256(SOURCE),
        "event_ledger_sha256": sha256(ledger),
        "pipeline_case_sha256": sha256(case_path),
        "limitations": [
            "External point labels have not yet received independent human verification.",
            "Action boundaries are provisional event_time -5s/+4s windows.",
            "This dataset must not be inspected or used during development or tuning.",
        ],
    }
    (HOLDOUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
