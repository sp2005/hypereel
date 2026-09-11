# HypeReel evaluation package

Selection mode evaluates fixed candidate windows and classifications using the
existing scoring and selection functions, without models or graph execution.
Pipeline mode executes the existing graph through the first approval interrupt.
Neither mode renders, shares, or writes user feedback memory.

## Run

From the repository root after installing HypeReel:

```bash
python -m hypereel.evaluation run --mode selection --dataset evals/datasets/smoke.jsonl
```

The existing `python -m hypereel.evaluation.cli` entrypoint also works. Use
`--output <directory>` to choose a new or empty report directory. By default,
reports go to a unique directory under `evals/results/`.

## Modules

| Module | Responsibility |
|---|---|
| `schemas.py` | Versioned replay and reference-event validation |
| `dataset.py` | JSONL loading, duplicate IDs, line-numbered input errors |
| `metrics.py` | One-to-one event matching and deterministic clip metrics |
| `runner.py` | Selection replay, timing, hashes, per-case failure collection |
| `report.py` | JSON/JSONL/Markdown output and grouped aggregates |
| `cli.py` | Argument validation and exit codes |

## Data contract

Each JSONL record requires `case_id`, `synthetic`, `recipe_path`,
`video_duration`, `candidates`, and an equally sized `classifications` list.
Recipe paths resolve relative to the dataset. Candidates and classifications
use the production Pydantic contracts; evaluation validates finite timestamps,
source bounds, and score/confidence ranges without changing those contracts.

Optional fields are `schema_version` (1), `max_duration`, `audience`,
`reference_events`, and `exhaustive`. An event contains `event_id`, `event_time`,
`action_start`, `action_end`, and `moment_type`. Reference events must be eligible
under the recipe and target subject. Only set `exhaustive` when every eligible
event in the source has been annotated.

## Results and limits

Reports contain selected clips, event matches, per-case errors, execution time,
dataset/recipe hashes, and code revision/working-tree status. Metrics cover budget
compliance/utilization, boundaries, overlap, relevant-clip precision, action
completeness, and moment-type diversity. Event matching requires equal labels
and `clip.start <= event_time < clip.end`, with maximum-cardinality one-to-one
assignment. Missing annotations and undefined ratios produce null/N/A.

Synthetic and non-synthetic cases have separate macro averages with applicable
case counts. Non-synthetic does not imply reviewed labels. Failed cases are
retained in reports and excluded from averages. Timings describe local replay,
not model latency. These metrics do not measure full-pipeline or rendered-video
quality, and no regression pass thresholds are configured yet.

Exit codes: 0 for completed cases, 1 if any case failed, 2 for input/output errors.
No existing nonempty output directory is overwritten.


## Pipeline integration

An additional `--mode pipeline` runs the unmodified `build_graph()` through
`plan → ingest → propose → scoreboard → classify → select → judge`, including
its existing revision loop, and stops before `approve_clips`. It never resumes
an approval gate. A fresh graph/checkpointer and temporary workspace isolate
each case; downloads/frames in that workspace are removed afterward.

Offline demo from the repository root:

```bash
HYPEREEL_VISION_PROVIDER=mock HYPEREEL_LLM_PROVIDER=mock HYPEREEL_TRACING_ENABLED=false LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false python -m hypereel.evaluation run --mode pipeline --dataset src/hypereel/evaluation/examples/pipeline_smoke.jsonl
```

For real evaluation, provide a pipeline JSONL dataset. Required fields are
`case_id`, `synthetic`, `recipe_path`, and `source` (local path or a URL accepted
by ingestion). Local source and recipe paths are relative to the dataset file.
Optional fields are `max_duration` (>0), `audience`, `subject_description`,
`max_candidates` (>0), `reference_events`, and `exhaustive`. Do not supply frozen
candidates/classifications or video_duration; the graph computes them.

Pipeline mode uses environment provider settings and can make paid model calls
or download sources. Existing opt-in graph tracing is reused when enabled;
this is tracing, not a LangSmith evaluation-experiment upload. Selection mode
remains offline regardless of provider settings.

Reports retain graph candidates, classifications, selected clips, judge verdict,
revision count, requested providers, actual vision provider, notes, pipeline
errors, and the trace execution ID when present. Elapsed time includes execution
through the first gate. Render quality and summaries are not evaluated.

A non-synthetic case is degraded when ingestion fails, vision falls back to mock,
or a vision result explicitly reports a provider error. Degraded cases remain
visible, are excluded from aggregate scores, suppress reference-label metrics,
and cause exit code 1. Synthetic demo sources intentionally exercise the
pipeline's existing synthetic-ingestion fallback. Other internal fallbacks that
the graph does not expose cannot be reliably distinguished; notes and judge
verdicts remain available for review. Missing recipe files or graph exceptions
are case failures, and subsequent cases continue.

The package API also exports `run_pipeline_evaluation(dataset_path, settings=None)`.
To run the package-local integration tests:

```bash
python -m pytest tests src/hypereel/evaluation/tests
```

## Candidate recall, final quality, and operational success (selection-v3)

Both selection replay and pipeline evaluation report `candidate_recall`:

```text
annotated events with a timestamp inside at least one candidate / annotated events
```

Containment is `candidate.start <= event_time < candidate.end`. Predicted labels,
classification confidence, subject filtering, and the final clip budget do not
participate. Each annotated event counts once even when multiple windows cover it.
No candidates with nonempty references gives 0; missing or empty references gives
null/N/A. Non-exhaustive annotations give recall on the annotated subset only.
Degraded pipeline cases suppress candidate recall, consistent with other
reference-based metrics. Group reports show the macro average across successful,
applicable cases, and each case's recall appears in the Markdown table.

`operational_success_rate` is reported at the overall and synthetic/non-synthetic
group levels:

```text
cases with status success / all attempted cases
```

Failed and degraded cases stay in the denominator. Successful cases count even if
the selected reel is empty or labels are missing. Empty groups give null/N/A.
The Markdown summary shows both rate and success/attempt counts. Dataset validation
errors occur before case execution and are not attempted cases. This measures
execution health under the runner's existing status detection, not AI correctness.
No status detection or production pipeline behavior was changed.

The report's metric version is now `selection-v3`. Final selection also reports
`selected_event_recall` and, when annotations are exhaustive, `selection_f1`.
JSON reports expose
per-case `metrics.candidate_recall`, overall `operational_success_rate`, and
`aggregates.<group>.operational_success_rate`.

The repository's real-video HoopIQ benchmark and interpretation rules are
documented in `evals/REFERENCE_BENCHMARK.md`.
