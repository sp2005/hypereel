# Real-video reference benchmark

This benchmark converts complete externally labeled event feeds for two
games into repeatable ground truth for HypeReel. The Unlimited vs. Campus game
has materially different lighting, providing a basic cross-venue robustness test.
It is intended to measure output quality independently of human approval.

## Files

- `datasets/externally-labeled-...raw.json` preserves all 244 source annotations, including
  player, jersey, team, event type, outcome, and absolute video timestamp.
- `datasets/externally-labeled-...pipeline.jsonl` contains two validated pipeline cases:
  a partial player-specific case for Lucky N (#23), and an exhaustive team-event
  case containing 20 made two-point baskets, 28 steals, and 3 blocks.
- `datasets/externally-labeled-real-video-benchmark.pipeline.jsonl` is the canonical combined
  benchmark: 2 games, 462 raw annotations, and 100 exhaustive eligible events.
- `recipes/basketball_team_evaluation.yaml` is an evaluation-only recipe without
  player-color or scoreboard assumptions.

Missed shots, free throws, turnovers, assists, and rebounds remain in the raw
reference file as negative evidence. They are not eligible highlights under the
team evaluation recipe, so selecting them lowers precision in the exhaustive case.

## Run against the real pipeline

This downloads the source video and may make paid model calls according to `.env`:

```bash
python -m hypereel.evaluation run --mode pipeline \
  --dataset evals/datasets/externally-labeled-real-video-benchmark.pipeline.jsonl \
  --output evals/results/external-baseline
```

The report separates pipeline health from quality and records candidate recall,
selected-event recall, relevant-clip precision, selection F1, action completeness,
overlap, boundary validity, duration, and budget compliance.

## Interpretation

- Candidate recall measures whether proposal found annotated events before filtering.
- Selected-event recall measures how many eligible events reach the final reel.
- Relevant-clip precision measures how many selected clips match eligible events.
- Selection F1 balances final precision and recall.
- Action completeness measures whether matched clips contain the provisional action window.

The external labels supply point timestamps, not action boundaries. The benchmark therefore
uses `[event_time - 5s, event_time + 4s]` provisional windows. Event matching is
grounded in the source timestamp; action-completeness results should be treated as
provisional until a reviewer verifies the start/end boundaries.

The player case is non-exhaustive because an external `STL` label does not prove the current
recipe's narrower `steal_break` definition. The team case uses an explicit `steal`
label and is exhaustive for its declared event types.

## Regression policy

Store the first successful real run as the baseline. Subsequent changes should not:

- reduce candidate recall, selected-event recall, precision, or F1;
- increase invalid boundaries or overlap;
- change a successful run to degraded or failed;
- silently compare reports with different dataset, recipe, or metric hashes.

The automated tests validate dataset integrity and prove that false positives,
missed events, wrong labels, and truncated clips affect the expected metrics.
