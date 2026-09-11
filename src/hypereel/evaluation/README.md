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

## Candidate recall, final quality, and operational success (selection-v6)

Both selection replay and pipeline evaluation report `candidate_recall`:

```text
annotated action intervals overlapping at least one candidate / annotated events
```

Overlap is `candidate.start < action_end and action_start < candidate.end`. Predicted labels,
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

The report's metric version is now `selection-v6`. Final selection also reports
`selected_event_recall` and, when annotations are exhaustive, `selection_f1`.
JSON reports expose
per-case `metrics.candidate_recall`, overall `operational_success_rate`, and
`aggregates.<group>.operational_success_rate`.

The repository's real-video HoopIQ benchmark and interpretation rules are
documented in `evals/REFERENCE_BENCHMARK.md`.

## Optional post-evaluation LLM-as-a-Judge

Add `--judge` to either evaluation mode to request a separate LLM assessment
**after every case's pipeline execution and deterministic metrics have completed**:

```bash
python -m hypereel.evaluation run --mode selection --dataset evals/datasets/smoke.jsonl --judge
```

For a pipeline run, keep the existing `--change-note` requirement:

```bash
python -m hypereel.evaluation run --mode pipeline --dataset YOUR_DATASET.jsonl --change-note "add advisory judge" --judge
```

Configure the existing `HYPEREEL_LLM_PROVIDER` and matching API key in `.env`.
Calls can incur API cost. No new provider or SDK is introduced. The default is
**off**, so existing commands and the production graph behave as before. Mock
providers (including fallback to mock) skip judging rather than invent scores.
Failed/degraded evaluation cases are also skipped. An empty successful reel can
be assessed to explain what is missing. The shipped smoke dataset is synthetic
and cannot demonstrate real-video quality, even with a live judge.

The judge receives recipe intent, the selected edit list, computed metrics,
available reference events, and evaluation scope. It does not receive source
media, source URLs, API keys, or the production judge's verdict. Recipe and clip
strings are explicitly treated as untrusted evidence, not instructions.

**This is a metadata-only judgment.** The existing LLM abstraction accepts text
and pipeline evaluation stops before rendering. Coherence means ordering and
continuity inferable from the edit list, not verified audiovisual editing.
Actual rendered-video quality requires a future multimodal evaluation interface.

Successful output is stored as `cases[i].llm_judge.assessment`:

```json
{
  "relevance": 0.8,
  "coverage": null,
  "coherence": 0.7,
  "diversity": 0.6,
  "overall_score": 0.75,
  "reasoning": "Assessment based on clip metadata; complete coverage is unknown.",
  "recommendations": ["Review the boundaries of the selected plays."]
}
```

All seven fields are required. Scores are finite numbers in [0,1] or null when
unsupported; coverage must be null without exhaustive reference evidence.
Reasoning is a bounded string and recommendations is a bounded array of strings.
JSON/code-fenced JSON is accepted, while missing/extra fields, duplicate keys,
non-numeric scores, invalid ranges, and malformed JSON are rejected. There is no
repair retry or fabricated fallback score.

The envelope records `success`, `skipped`, `invalid_response`, or `unavailable`,
plus rubric version, prompt hash, provider, elapsed time, trace correlation, and
provider usage where exposed by existing adapters. Judge scores never enter
metric aggregates or alter selected clips, operational status, or exit codes.
Judge issues are visible in CLI output and Markdown; they do not erase a valid
deterministic evaluation. Existing `elapsed_seconds` excludes this new judge;
`llm_judge.elapsed_seconds` measures it separately.

Markdown adds score and reasoning/recommendation tables. JSON, JSONL, and
append-only iteration history retain the structured judge envelope separately
from the production `judge_verdict`. Estimated judge spend is likewise separate
from the original pipeline spend. The existing budget scope/ledger is reused,
and consumed pipeline call allowances are deducted before judging. Cost tracking
and enforcement depend on the existing provider adapter's budget instrumentation;
this PR does not broaden that instrumentation to other providers.

`configured_model` records the selected model setting, not a provider-confirmed
model version; traces include it as `judge_configured_model`. Missing usage
accounting is represented by `usage_accounting_available: false` and null values,
not zero cost. If a failed call exposes only an attempt record, that record and
count are retained while estimated spend remains null. Judge text is escaped in
Markdown reports; structured JSON retains the original assessment strings.

Enable the existing `HYPEREEL_TRACING_ENABLED`, `LANGSMITH_API_KEY`, and
`LANGSMITH_PROJECT` settings for a root `evaluation.llm_judge` trace and nested
provider trace. Correlation metadata contains the evaluation run/case IDs and
prior pipeline trace execution ID. This is a separate post-processing trace, not
an extra production graph node. Existing input/output hiding is retained; use the
local report for scores and reasoning. Upload failures do not retry model calls.

Python callers can use `run_selection(path, judge=True, settings=settings)` or
`run_pipeline_evaluation(path, settings=settings, judge=True)`.

Before using these advisory scores as release gates, validate the rubric against
human reviews and measure agreement; self-review by the generation model is not
independent evidence of correctness.
