"""Build the versioned HypeReel golden dataset from external labels."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evals" / "datasets"
GOLDEN = ROOT / "evals" / "golden"

GAMES = [
    {
        "id": "east-bay-elite-vs-spartans",
        "slug": "east-bay-elite-11u-2025-26-vs-spartans-2026-03-21-71c9f6",
        "lighting_slice": "baseline-gym",
    },
    {
        "id": "unlimited-vs-campus",
        "slug": "unlimited-vs-campus-2026-03-14-404790",
        "lighting_slice": "alternate-lighting",
    },
]


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
    source_dir = GOLDEN / "source"
    cases_dir = GOLDEN / "cases"
    source_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    manifest_games = []
    combined_cases = []
    for game in GAMES:
        slug = game["slug"]
        raw_src = SOURCE / f"externally-labeled-{slug}.raw.json"
        case_src = SOURCE / f"externally-labeled-{slug}.pipeline.jsonl"
        raw = json.loads(raw_src.read_text())
        cases = [json.loads(line) for line in case_src.read_text().splitlines() if line.strip()]
        team_case = next(case for case in cases if case["exhaustive"])

        raw_dest = source_dir / f"{game['id']}.external-labels.json"
        raw_dest.write_text(json.dumps(raw, indent=2) + "\n")
        case_dest = cases_dir / f"{game['id']}.pipeline.jsonl"
        case_dest.write_text(json.dumps(team_case) + "\n")
        combined_cases.append(team_case)

        counts = Counter()
        eligible = 0
        for event in raw["events"]:
            label = expected_label(event)
            eligible += label is not None
            counts[event["event_type"]] += 1
            t = event["timestamp_seconds"]
            all_rows.append({
                "gold_id": f"{game['id']}:{event['event_id']}",
                "dataset_version": "external-gold-v1",
                "split": "development",
                "game_id": game["id"],
                "video_url": raw["video_url"],
                "lighting_slice": game["lighting_slice"],
                "event_time": t,
                "provisional_action_start": max(0, t - 5),
                "provisional_action_end": t + 4,
                "source_event_type": event["event_type"],
                "source_outcome": event["outcome"],
                "player_name": event["player_name"],
                "jersey_number": event["jersey_number"],
                "team": event["team"],
                "eligible_highlight": label is not None,
                "expected_moment_type": label,
                "annotation_source": "external",
                "event_label_status": "source-annotated",
                "boundary_status": "provisional",
            })
        manifest_games.append({
            "game_id": game["id"],
            "lighting_slice": game["lighting_slice"],
            "raw_event_count": len(raw["events"]),
            "eligible_event_count": eligible,
            "source_event_counts": dict(sorted(counts.items())),
            "source_sha256": sha256(raw_src),
            "video_url": raw["video_url"],
        })

    ledger = GOLDEN / "golden_events.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in all_rows))
    combined = cases_dir / "development.pipeline.jsonl"
    combined.write_text("".join(json.dumps(case) + "\n" for case in combined_cases))
    manifest = {
        "dataset_id": "hypereel-external-gold",
        "version": "1.0.0",
        "split": "development",
        "created_from": "externally labeled game events",
        "taxonomy_recipe": "../../recipes/basketball_team_evaluation.yaml",
        "event_count": len(all_rows),
        "eligible_event_count": sum(r["eligible_highlight"] for r in all_rows),
        "negative_event_count": sum(not r["eligible_highlight"] for r in all_rows),
        "event_ledger_sha256": sha256(ledger),
        "games": manifest_games,
        "limitations": [
            "External point labels have not yet received independent human verification.",
            "Action boundaries are provisional event_time -5s/+4s windows.",
            "The third-game holdout is intentionally absent from this development dataset.",
        ],
    }
    (GOLDEN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
