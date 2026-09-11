# Offline selection evaluation

This first increment replays frozen candidate windows and classifications through
`score_candidates()` and `select_clips()`. It does not invoke the graph, models,
media processing, the quality judge, rendering, feedback storage, or LangSmith.
Existing production code is unchanged.

## Run

From the repository root, after installing the core package:

```bash
python -m hypereel.evaluation.cli run --mode selection --dataset evals/datasets/smoke.jsonl
```

The command prints the generated report path under `evals/results/<run-id>/`.
An explicit new or empty output directory is also supported:

```bash
python -m hypereel.evaluation.cli run --dataset evals/datasets/smoke.jsonl --output evals/results/my-baseline
```

An existing nonempty directory is never overwritten. Exit codes: 0 means all
cases executed, 1 means one or more cases failed, and 2 means invalid input or
report-output failure. Metric values do not yet trigger regression exit codes.
`classification`, `pipeline`, `render`, `compare`, and `--config` are not implemented.

## Dataset

JSONL has one `SelectionCase` per line. See `datasets/smoke.jsonl` for runnable
examples. Recipe paths are relative to the dataset file, not the working directory.

Required fields:

- `case_id`: unique identifier.
- `synthetic`: distinguishes synthetic fixtures from non-synthetic cases.
- `recipe_path`, `video_duration`: recipe and source duration in seconds.
- `candidates`: production `CandidateWindow` objects with start, end, signal_scores.
- `classifications`: production `Classification` objects, aligned one-to-one with candidates.

Optional fields:

- `schema_version`: currently 1.
- `max_duration`: nonnegative budget override; zero means select no clips.
- `audience`: individual/team override, applied before scoring.
- `reference_events`: eligible, target-subject events, each containing `event_id`,
  `event_time`, `action_start`, `action_end`, and `moment_type`.
- `exhaustive`: true only when every eligible event throughout the source has been
  annotated. Defaults false. Without exhaustive annotations, unmatched predictions
  are unjudged and precision is N/A.

All timestamps use the referenced source timeline. `reference_events: null` means
unannotated; `[]` with `exhaustive: true` means there are no eligible events.
For real benchmarks, label only moments eligible under the recipe and subject
brief; split tuning/test data by source game. A non-synthetic flag by itself does
not prove human review or annotation quality. The shipped cases are synthetic
framework checks, not evidence of real-video accuracy.

## Metrics (selection-v1)

- Clip count and total selected seconds.
- Budget compliance (with a 1e-9 second floating-point tolerance).
- Budget utilization (N/A for zero budget; descriptive, not quality).
- Invalid boundary count: start < 0, end <= start, or end > video duration.
- Overlap rate: `(sum of clip durations - interval union duration) / sum durations`.
- Relevant-clip precision: matched clips / selected clips, only with exhaustive labels.
- Action completeness: matched clips containing the entire reference action / matches.
- Matched and reference event counts; distinct selected moment-type count.

Matching requires an equal moment label and `clip.start <= event_time < clip.end`.
Maximum-cardinality bipartite matching prevents a clip or event from counting twice.
Stable input order breaks ties; matching does not optimize for action completeness.
Undefined ratios are JSON null / report N/A, never zero. Selection recall is not
used as a headline quality metric because the reel budget intentionally excludes
some events. Subject recognition, candidate recall, rendered quality, quarter
coverage, and live provider latency are outside this first increment.

## Reports

- `report.json`: full results, hashes, code revision/dirty flag, and aggregates.
- `cases.jsonl`: per-case selected clips, matches, errors, and metrics.
- `summary.md`: human-readable group metrics and case outcomes.

Aggregates are unweighted per-case macro means and show the applicable-case count.
Synthetic/non-synthetic groups are separate; failed cases remain visible and are
excluded from metric means. Per-case elapsed time covers recipe loading, selection,
and metric computation; it is not model or video-processing latency.

Dataset bytes and recipe files are SHA-256 hashed. Compare results only when
inputs and metric version are compatible. A dirty code revision is explicitly
reported; archive your changes or commit before treating results as a reproducible
baseline. Repeated runs should yield the same clips and metrics; timestamps and
elapsed time will differ. LangSmith experiment uploads and comparison reports
remain future increments.
