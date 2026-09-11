# HypeReel

A recipe-driven, agentic highlight-reel builder built on LangGraph.

> Full design document: [`design/HypeReel-Design.html`](design/HypeReel-Design.html)

> Demo: [Watch the HypeReel demo](https://drive.google.com/file/d/1c_W8egamBQFVbjVJV_sDs186M_oqpM1u/view?usp=sharing)

**The one-liner:** My agent helps a player or coach turn a full 60-minute game recording into a share-ready highlight reel matching a plain-language brief (one player, a whole team, only threes, no steals) in a Streamlit web app, replacing the 3–4 hours of manual scrubbing and clip-cutting in a video editor it takes today. It ingests the video, detects and scoreboard-confirms the right plays, and critiques and re-cuts its own reel on its own using ~6 tools (yt-dlp, frame sampling, scoreboard CV, a vision model, an LLM judge, and ffmpeg), hands off to a human to approve the clip list before it renders and again before it shares, and I'll know it works when a user gets a watchable, single-team reel in under 30 minutes that they'd actually post 8 times out of 10.

## What it does

The flagship demo is an AAU basketball game: point the agent at a full game recording and a recipe describing one player (jersey number + team color), and it comes back with a ≤5-minute reel of that player's made baskets, threes, blocks, and steal-to-fast-break sequences — spread across all four quarters, not clustered in one segment.

The core abstraction is the **recipe** — a declarative YAML policy that describes what counts as a highlight in a given domain, so the same pipeline generalizes well beyond basketball:

- **`event_based`** recipes (e.g. basketball) look for discrete events — a vision model classifies each candidate window against a short list of named `moment_types` (`made_basket`, `three_pointer`, `block`, `steal_break`, ...), each with a natural-language rubric.
- **`quality_based`** recipes (e.g. a real-estate walkthrough, see `recipes/architecture_walkthrough.yaml`) have no discrete events to detect — instead the vision model scores sliding windows against a composition/lighting/clarity rubric, and the sustained highest-scoring segments become clips.

Everything else — which signals propose candidates, how clips are scored and budget-fitted, transitions, aspect ratio, guardrails, acceptance criteria — is also part of the recipe, not hardcoded.

## Architecture

HypeReel is a LangGraph state machine, not a one-shot LLM call: cheap local signals **propose** candidate windows, a vision model **confirms/scores** them, deterministic code **selects** the final budget-fitted clip set, and a human **gates** the two write-ish actions (render and share).

```
plan → ingest → propose → classify → select
                                        │
                                        ▼
                         [ HITL Gate 1: approve clips ] ──▶ render → summarize
                                                                        │
                                                                        ▼
                                                  [ HITL Gate 2: approve & share ] ──▶ deliver
```

- **plan** — picks a strategy (`event_based` / `quality_based`) and the active proposer signals from the recipe.
- **ingest** — resolves the source (YouTube via `yt-dlp`, or a local path), with graceful fallback if it fails.
- **propose** — cheap, local signals (audio-peak, motion-intensity, scene-stability) generate candidate time windows.
- **classify** — a vision provider judges each candidate against the recipe's moment types / rubric.
- **select** — scores and greedily budget-fits candidates into a final clip list, ordered per the recipe.
- **approve_clips** *(HITL Gate 1)* — a human reviews/edits the proposed clip list before anything is rendered.
- **render** — cuts and concatenates the approved clips (or writes a manifest if media libs/ffmpeg aren't available).
- **summarize** — an LLM writes a short natural-language recap of the reel.
- **approve_share** *(HITL Gate 2)* — a human approves before any share/publish step runs.
- **deliver** — terminal node; marks the reel as shared (or ready-but-unshared).

See the flow diagram and full rationale in [`design/HypeReel-Design.html`](design/HypeReel-Design.html) (sections 2–3).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate

# Core install — enough for the offline demo AND the full test suite.
# This installs the `hypereel` package (editable) so `python -m hypereel.cli` works.
pip install -e .

# Optional: add the heavy media libs, live-model SDKs, and the web UI.
#   pip install -e ".[all]"      # everything
#   pip install -e ".[media]"    # yt-dlp, opencv, librosa, moviepy (real video in/out)
#   pip install -e ".[providers]" .[ui]   # live Gemini/Groq/Nebius, or Streamlit

cp .env.example .env              # fill in API keys, or leave everything as `mock`

# run the offline demo (no keys, no network, no heavy libs needed):
python -m hypereel.cli --demo

# or with a real video (needs the .[media] extra, plus a provider or mock):
python -m hypereel.cli --recipe recipes/basketball_player.yaml --source "<YouTube URL or local path>"

# web UI (needs the .[ui] extra):
streamlit run src/hypereel/app.py
```

> The package lives under `src/`. `pip install -e .` puts it on your path; without it, prefix commands with `PYTHONPATH=src` (e.g. `PYTHONPATH=src python -m hypereel.cli --demo`). Tests configure this automatically.

### CLI flags

| Flag | Description |
| --- | --- |
| `--recipe PATH` | Recipe YAML to use (default: `recipes/basketball_player.yaml`) |
| `--source STR` | YouTube URL or local video path |
| `--demo` | Run fully offline with mock providers (no keys/network); forces auto-approval through both gates |
| `--max-duration FLOAT` | Override the recipe's reel duration budget, in seconds |
| `--audience {individual,team}` | Override the recipe's audience |
| `--no-approve` | Stop at the first human-in-the-loop gate instead of auto-approving |
| `--out PATH` | Informational: where you intend the rendered reel to end up |

## Enabling real models

By default everything runs against deterministic **mock** providers — no API key, no network call, fully reproducible. To use a real vision/LLM provider, set in `.env`:

```
HYPEREEL_VISION_PROVIDER=mock|gemini|groq|nebius|fireworks
HYPEREEL_LLM_PROVIDER=mock|gemini|groq|nebius|fireworks
```

along with the matching API key (`GEMINI_API_KEY`, `GROQ_API_KEY`, `NEBIUS_API_KEY`, or `FIREWORKS_API_KEY`) and the provider extra (`pip install -e ".[providers]"`). Nebius and Fireworks use OpenAI-compatible endpoints; see `.env.example` for the full list of knobs (base URLs, model names, sampling settings).

For the evaluation's optional LLM-as-a-Judge, choose one text provider in `.env`:

| Provider | Selection | API key | Model setting |
| --- | --- | --- | --- |
| Gemini | `HYPEREEL_LLM_PROVIDER=gemini` | `GEMINI_API_KEY` | `GEMINI_MODEL` |
| Groq | `HYPEREEL_LLM_PROVIDER=groq` | `GROQ_API_KEY` | `GROQ_VISION_MODEL` (also used for text calls) |
| Nebius | `HYPEREEL_LLM_PROVIDER=nebius` | `NEBIUS_API_KEY` | `NEBIUS_MODEL` |

Set the matching key and a model available to your account. Vision selection is
independent; real pipeline evaluation also needs a configured vision provider.

**The graded submission routes at least one model call through Nebius Token Factory** — set both `HYPEREEL_VISION_PROVIDER=nebius` and `HYPEREEL_LLM_PROVIDER=nebius` with `NEBIUS_API_KEY` filled in.

Every provider path degrades gracefully: if an SDK isn't installed or a key is missing/invalid, HypeReel falls back to the mock provider rather than crashing the pipeline.

> **Note:** `.env` is only read if `python-dotenv` is installed (it's in the core deps, so a normal `pip install -e .` covers it). If you install by some other means and skip it, export the variables in your shell instead — otherwise a missing `.env` is silently ignored and everything falls back to `mock`.

## Optional LangSmith tracing

Install `pip install -e ".[telemetry]"` (also included in `.[all]`) and set:

```dotenv
HYPEREEL_TRACING_ENABLED=true
LANGSMITH_API_KEY=<your key>
LANGSMITH_PROJECT=hypereel-dev
# Optional: set LANGSMITH_ENDPOINT for your LangSmith region/deployment.
```

Then run the existing CLI or Streamlit commands. Graph invocations and their
nodes appear in the configured project, including timing and judge revisions.
Approval resumes produce separate invocation traces grouped by the same
`hypereel_execution_id` metadata value. Checkpoint thread IDs remain unchanged.
The `runner` entrypoint covers the CLI and direct `run_pipeline()` calls;
`streamlit` identifies web runs. Provider metadata describes the *requested*
configuration, not a guarantee that a live provider succeeded.

This integration uses an explicit callback. Leave `LANGSMITH_TRACING` and legacy
`LANGCHAIN_TRACING_V2` unset or false; enabling them independently activates the
framework's global tracing outside this opt-in layer. Graph inputs and outputs
are hidden by this layer's client, so videos, subject descriptions, and complete
state payloads are not uploaded as inputs/outputs. Trace names, metadata, timing,
and native exception messages are still recorded. Do not put secrets in metadata.

Tracing defaults off. Missing credentials/SDKs or initialization errors leave
execution unchanged; callback/upload errors are best-effort and never retry the
pipeline. Uploads use the SDK's background queue, so abrupt process termination
can lose pending traces. No network or account is needed for the normal tests.
Each factory-created vision/text provider also records a child operation span
under its graph node (`provider.vision.<name>` or `provider.llm.<name>`). These
reuse the graph client and execution ID, and record actual/requested provider,
fallback-to-mock status, and vision candidate index/frame count. Prompts, frame
paths, frame contents, and response text are omitted from these spans.
Calls outside an instrumented graph or evaluation judge (including UI health checks) create no
provider spans. Disabled tracing returns the original provider implementations.
Live vision/text operations are recorded with LangSmith's `llm` run type; mock
operations remain `chain` spans. Each executed node and provider call has start
and end timestamps, from which LangSmith displays execution time. Repeated
`select`/`judge` executions each get their own span. Approval nodes execute and
are traced only after the corresponding interrupt is resumed.

Unhandled node exceptions are captured by the native graph tracer. API exceptions
caught by Gemini, Groq, Nebius, and Fireworks are also recorded as provider-span
errors, using only the exception class name, while preserving the existing
empty-text/low-confidence fallback returns. A graph node may therefore complete
successfully even when its child model call failed. An empty response or a
low-confidence classification alone is not treated as an API failure. Ingest and
render fallbacks continue to be described in the existing application notes;
they are not reclassified as thrown node exceptions.

Provider spans do not expose token/cost accounting. Local evaluation reports
include usage and estimated spend where provider adapters supply them; this
instrumentation is not available for every provider. Execution time includes
provider work, not the human wait between approval resumes.

For programmatic callers using `build_graph()` directly, pass your invocation
config through `hypereel.observability.with_tracing(config, settings,
recipe_id=recipe.id, entrypoint="your-entrypoint")` once, and reuse it for resumes.

## Running tests

```bash
pip install -e ".[dev]"   # if you haven't already
pytest
```

All tests are offline and mock-based — no API keys, no network, and no heavy media libs required (pytest is configured to find the package under `src/`). The suite is green end-to-end: unit tests per module, provider-fallback tests, CLI integration tests, and full end-to-end pipeline + human-in-the-loop resume tests.

## Project structure

```
src/hypereel/
  models.py       Pydantic contracts: the recipe schema + runtime objects (candidates, clips, ReelResult)
  config.py       Settings loaded from env / .env
  recipe.py       load_recipe(path) -> Recipe, with RecipeError on anything malformed
  providers/      Vision/LLM provider ABCs, mock/gemini/groq/nebius implementations, and a lazy factory
  ingest/         Source resolver (yt-dlp / local path, with retries)
  signals/        Cheap proposer signals: audio-peak (librosa), motion-intensity (opencv), window merge
  analyze/        Frame sampling + vision-model classification of candidate windows
  select/         Scoring and knapsack-style selection to the time budget, with ordering + coverage
  render/         Clip cut/concat/overlay (moviepy/ffmpeg), with a manifest fallback when unavailable
  memory/         Persistent user profile + learned accept/reject preferences
  graph/          LangGraph state (state.py), node functions (nodes.py), and graph assembly + runner (build.py)
  evaluation/     Deterministic metrics, optional LLM-as-a-Judge, runners, and local reports
  app.py          Streamlit UI
  cli.py          Command-line entry point (`python -m hypereel.cli`)
recipes/          basketball_player.yaml (flagship) · architecture_walkthrough.yaml (generalization stub)
tests/            Unit + integration tests, all offline/mock-based
design/           HypeReel-Design.html — the full design document
```

## Human-in-the-loop

HypeReel treats **reads as autonomous and writes as gated**:

- **Gate 1 — approve the clip list.** Everything up to this point (ingest, propose, classify, select) is read-only analysis of the source video. Before anything is rendered, a human reviews the proposed clips and can drop any that don't belong.
- **Gate 2 — approve & share.** Rendering produces a local file — still not a write to anywhere external. Before any share/publish step runs, a human explicitly approves it. This is the one gate that guards an outward-facing action.

There is also a self-correction path: if the selected clips underfill the recipe's time budget, the pipeline still surfaces the (shorter) proposed list at Gate 1 with a clear warning, rather than silently shipping an incomplete reel or failing outright.

## Design notes

- **Free/offline by default.** The full pipeline — ingest, signal detection, classification, selection, render, summarize — runs with zero API keys via deterministic mock providers, which is also what keeps the test suite fully offline.
- **Heavy libraries are optional and lazily imported.** `yt-dlp`, `opencv-python`, `librosa`, and `moviepy` are only imported inside the functions that need them, so the core package, the CLI, and the test suite all import and run fine even when none of them are installed; the render step falls back to writing a JSON manifest instead of a video when `moviepy`/`ffmpeg` aren't available.
- **Python 3.14.** `streamlit` is imported lazily inside `app.py`'s `main()` for the same reason — the module (and everything that imports it) stays importable even without it installed.

## Evaluation

The evaluation package supports offline **selection replay** from fixed candidates
and classifications, and **pipeline evaluation** through the first approval gate.
Neither mode renders, shares, or writes user feedback memory.

- **Deterministic metrics** measure budget compliance/utilization, clip boundaries,
  overlap, candidate recall, reference-event precision/recall, action completeness,
  and diversity. Operational success rate measures successful cases out of all
  attempted cases. Metrics requiring unavailable evidence are reported as N/A.
- **LLM-as-a-Judge** is an optional review requested with `--judge`, after each
  case's metrics are computed. It returns validated JSON with `relevance`,
  `coverage`, `coherence`, `diversity`, `overall_score`, `reasoning`, and
  `recommendations`. Scores are in [0,1] or null when unsupported; coverage requires
  exhaustive reference annotations. These advisory results do not change selected
  clips, deterministic aggregates, case status, or exit codes.

The evaluation judge reviews recipe intent, clip metadata, metrics, and available
reference events—not rendered video or audio. It is separate from the production
graph's revision judge. Failed/degraded cases are skipped. The mock provider
(including fallback to mock) also skips judging: its canned responses cannot
provide a meaningful quality assessment, so no scores are fabricated.

### Running evaluations

Run from the repository root after installing HypeReel:

```bash
# Offline selection replay; no model calls or LangSmith uploads.
python -m hypereel.evaluation run --mode selection --dataset evals/datasets/smoke.jsonl

# Add an advisory judge using the real LLM configured in .env (may incur API cost).
python -m hypereel.evaluation run --mode selection --dataset evals/datasets/smoke.jsonl --judge

# Evaluate a real pipeline dataset and add the post-metrics judge.
# Replace YOUR_DATASET.jsonl with a pipeline dataset; --change-note is required.
python -m hypereel.evaluation run --mode pipeline --dataset YOUR_DATASET.jsonl --change-note "baseline with advisory judge" --judge
```

Configure a provider as described in [Enabling real models](#enabling-real-models).
Pipeline runs may download sources and call vision/LLM providers; real media
processing needs `pip install -e ".[media]"`. The smoke dataset is synthetic and
does not establish real-video quality, even with a live judge.

Reports are written to a unique directory under `evals/results/`: `summary.md`,
`report.json`, and `cases.jsonl`. Use `--output PATH` for a new or empty directory.
With `--judge`, reports include assessment status, scores, reasoning, and
recommendations; invalid responses and unavailable assessments remain visible.
Pipeline runs also append iteration history to `evals/iterations/history.jsonl`.

### LangSmith integration

Use the settings and telemetry extra in [Optional LangSmith tracing](#optional-langsmith-tracing).
Pipeline evaluation reuses graph/node/provider tracing. With `--judge`, a separate
`evaluation.llm_judge` trace contains the provider call, timing, and failures,
correlated by evaluation run/case IDs and the pipeline execution ID when present.
Inputs and outputs remain hidden; read scores and reasoning in the local reports.
Tracing is opt-in and does not upload LangSmith datasets or evaluation experiments.

See [`src/hypereel/evaluation/README.md`](src/hypereel/evaluation/README.md) for
dataset formats, metric definitions, judge limitations, and supported options.
