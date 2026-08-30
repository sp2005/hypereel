# HypeReel — Implementation Notes

Living technical reference for how HypeReel is actually built. The
[`README.md`](README.md) is the user-facing overview; this file is the
under-the-hood detail for maintainers. Keep it in sync with the code, and log
every notable change in [`changelog.md`](changelog.md).

Last updated: 2026-08-30.

---

## 1. High-level shape

HypeReel is a **LangGraph state machine** (not a one-shot LLM call). Cheap local
signals *propose* candidate windows, a vision model *classifies* them, a
scoreboard reader *grounds* them in game state, deterministic code *selects* the
budget-fitted set, an LLM-as-judge *critiques* the reel, and a human *gates* the
two write-ish actions.

```
plan → ingest → propose → scoreboard → classify → select → judge
      → (judge: revise → select ↺  |  accept → approve_clips)
      → approve_clips [HUMAN GATE 1] → render → summarize
      → approve_share [HUMAN GATE 2] → deliver → END
```

Compiled with `interrupt_before=["approve_clips", "approve_share"]` and an
`InMemorySaver` checkpointer, so a run pauses at each gate and resumes on the
same `thread_id`. See `src/hypereel/graph/build.py`.

---

## 2. Package layout

```
src/hypereel/
  models.py          Pydantic contracts: Recipe schema + runtime objects
  config.py          Settings from env / .env (get_settings())
  recipe.py          load_recipe(path) -> Recipe, RecipeError on malformed input
  providers/         Vision/LLM ABCs + mock/gemini/groq/nebius impls + lazy factory
  ingest/            source_resolver.py (yt-dlp / local path, HD-unlock, retries)
  signals/           audio.py, motion.py, scoreboard.py, merge.py (proposer signals)
  analyze/           classifier.py (frame sampling + vision classification)
  select/            selector.py (scoring, budget knapsack, moment confirmation)
  render/            renderer.py (moviepy/ffmpeg cut+concat, manifest fallback)
  memory/            store.py (persistent user profile + learned accept/reject)
  graph/             state.py (ReelState), nodes.py (node fns), build.py (assembly+runner)
  app.py             Streamlit UI (pipeline visualizer, diagnosis, pre-flight)
  cli.py             `python -m hypereel.cli`
recipes/             *.yaml recipe policies
scripts/             probe_vision.py (live one-window probe), verify_nebius.py
tests/               offline/mock-based unit + integration tests
```

---

## 3. The recipe abstraction

A recipe is a declarative YAML policy (validated by the Pydantic `Recipe` model
in `models.py`). It carries everything domain-specific so the pipeline
generalizes:

- **strategy** — `event_based` (discrete moments, e.g. basketball) or
  `quality_based` (sliding-window composition scoring, e.g. a walkthrough).
- **signals** — proposer/scorer signal configs with `weight`, `role`,
  optional `enabled_if` (e.g. `commentary_present`) and per-signal params.
- **moment_types** — named events with natural-language rubrics for the vision model.
- **subject_selector** — `type` (`jersey_number` | `team_color` | `none`),
  `value`, `team_color`, free-text `description`, and `audience`
  (`individual` | `team`).
- **selection** — `max_duration`, `min_clip`/`max_clip`, `min_score`, ordering, coverage.
- **acceptance** — criteria the LLM judge scores the assembled reel against.

Shipped recipes:
- `basketball_generic.yaml` (`basketball_generic_v1`) — **prompt-driven**, no
  hardcoded colors/numbers/scoreboard; all tuning via the runtime brief.
- `basketball_player23_blackred.yaml` — per-video tuned recipe for the UNL game
  (calibrated scoreboard regions + per-team attribution).
- `basketball_player.yaml`, `basketball_team_dark.yaml`,
  `basketball_game_highlights.yaml`, `architecture_walkthrough.yaml`.

**Generic-recipe invariant:** for single-team filtering to work,
`subject_selector.type: team_color` (NOT `none`) + `audience: individual` so the
selector's `require_subject` fires and drops opponent (`subject_present=False`)
plays. Guarded by `test_generic_recipe_is_prompt_driven_and_filters_one_team`.

---

## 4. Graph nodes (`graph/nodes.py`)

- **plan** — chooses strategy family + active proposer signals from the recipe.
- **ingest** — `resolve_source(...)` (yt-dlp / local, HD-unlock profile first
  then plain fallback); detects commentary to gate transcript signals; always
  yields a `video_duration > 0` fallback so the demo proceeds.
- **propose** — runs proposer signals (`propose_candidates`) into one merged
  candidate timeline. Applies `enabled_if` conditions. **Quick-test cap:** if
  `state["max_candidates"]` is set (>0), keeps only the first N windows by start
  time (`sorted(...key=start)[:cap]`) and notes it.
- **scoreboard** — OCR-free scoreboard reader (`signals/scoreboard.py`). Detects
  clock live/frozen and per-region score changes; pass-through when a recipe
  declares no `scoreboard` signal.
- **classify** — samples frames per window and calls the vision provider
  (`analyze/classifier.py`). Sets `mode` = `mock` | `live`.
- **select** — `select/selector.py`. Blends signal + classification scores;
  a window must clear `score >= min_score` AND (if `require_subject`)
  `subject_present`. `_confirm_moment()` upgrades `moment_type=null` →
  `made_basket` when a subject score-change is present (scoreboard-confirmed
  make). Greedy budget-fit to `max_duration`; sets `human_gate` to
  `approve_clips` or `underfilled`.
- **judge** — LLM-as-judge critic. Scores the reel against `acceptance`, and can
  bounce back to `select` **once** (`_MAX_REVISIONS = 1`) with corrective
  overrides. The revision is *smart*: if the reel is too thin
  (`_MIN_REEL_CLIPS = 3`, `_MIN_REEL_SECONDS = 20`) or tightening would leave
  `< 3` confirmed clips, it BROADENS instead of shrinking (never tightens into a
  corner). `route_after_judge` reads `judge_decision`.
- **approve_clips / approve_share** — pass-through nodes; graph interrupts
  *before* each.
- **render** — `render/renderer.py`; moviepy/ffmpeg cut+concat with a JSON
  manifest fallback when media libs are absent.
- **summarize** — LLM writes a short recap.
- **deliver** — terminal; marks shared / ready-but-unshared.

---

## 5. State (`graph/state.py`)

`ReelState` is a single `TypedDict(total=False)` threaded through every node.
Field groups: inputs, ingest, detection, selection/render, judge, control/HITL,
bookkeeping. `new_state(source, recipe, **overrides)` builds defaults and:

- bakes a non-empty `subject_description` override into a deep copy of
  `recipe.subject_selector.description` (so the vision prompt uses the brief
  without mutating the caller's recipe);
- carries runtime overrides: `max_duration`, `audience`, `max_candidates`.

Human-gate constants: `GATE_APPROVE_CLIPS`, `GATE_APPROVE_SHARE`,
`GATE_UNDERFILLED`.

---

## 6. Runtime overrides (how a user steers a run)

Three overrides funnel through `new_state` / `run_pipeline`, exposed in both the
CLI and the Streamlit sidebar:

| Override | CLI flag | Effect |
| --- | --- | --- |
| Free-text subject brief | `--describe TEXT` | Baked into `subject_selector.description`; leads the vision classification prompt. Also carries resolution intent via `parse_video_quality` ("use HD" → 1080, "720", "360"/"fast preview"). |
| Audience | `--audience {individual,team}` | `_effective_recipe` bakes it onto `subject_selector.audience`. `individual` turns the one-team subject filter ON (drops opponent); `team` turns it OFF. |
| Quick-test cap | `--max-candidates N` | Classify only the first N candidate windows (chronological). Unset/0 = unbounded. A fast dry-run knob. |

---

## 7. Providers (`providers/`)

- ABCs in `base.py`; lazy `factory.py` (`get_vision_provider`, `get_llm_provider`)
  reads `HYPEREEL_VISION_PROVIDER` / `HYPEREEL_LLM_PROVIDER`.
- **mock** — deterministic, offline, no keys; default and what keeps the test
  suite hermetic.
- **nebius** — OpenAI-compatible; `OpenAICompatVisionProvider`. Single
  `settings.nebius_model` (`Qwen/Qwen2.5-VL-72B-Instruct`, a vision model) for
  both vision and text. The graded submission routes ≥1 call through Nebius.
- **groq** — OpenAI-style, `meta-llama/llama-4-scout-17b-16e-instruct`
  (vision+text, free tier). Recommended free fallback.
- **gemini** — `gemini-2.0-flash` via the DEPRECATED `google.generativeai` SDK
  (works, warns).

**Graceful degradation:** every provider path falls back to mock if the SDK is
missing or a key is absent/invalid, rather than crashing.

**Error surfacing:** `classify_window` catches ALL exceptions and returns
`Classification(subject_present=False, confidence=0.0, reason=f"{name} error: {exc}")`;
`.generate` returns `""` on exception. This is the hook the app uses to detect a
dead provider (a provider error lives in every classification's `.reason`).

---

## 8. Streamlit app (`app.py`)

- **Provider badge** — 🟢 LIVE / 🟡 PARTIAL / ⚪️ MOCK by calling the factory
  (ground truth). Footgun it surfaces: `.env` may set LLM=nebius but not VISION,
  so vision silently runs on mock unless BOTH are set.
- **Pre-flight checker** — `_preflight(...)` validates the form live (offline,
  deterministic) before submit. ERROR disables "Find highlights"; WARN flags
  likely-fail settings. Checks: empty source, missing subject brief when the
  recipe needs one, no color/exclusion/quality word in the brief, `max_duration
  < 2*min_clip`, mock vision, and provider health.
- **Live provider ping** — `_provider_health(vision, llm)` makes ONE real
  `classify_window` call with a real 64×64 image (`_health_frame_path`) and reads
  `.reason`. Cached per provider-pair in `st.session_state["provider_health"]`;
  a "Re-check providers" button clears it. CRITICAL: the ping MUST send a real
  image — a budget-exhausted account still answers cheap text-only calls but
  402s on image calls, so a frameless probe reads falsely healthy.
- **Pipeline visualizer** — collapsible expander renders the real LangGraph
  nodes as status cards (done/running/awaiting/queued) with per-node timings,
  tool-call chips (👁 vision-per-window, 🧠 text-LLM), the Judge routing branch
  (↻ revise / ✓ accept), and a per-run state strip (signals / funnel / judge /
  subject filter / reel / mode). Live via `app.stream(..., stream_mode="updates")`.
- **"Why 0 clips?" diagnosis** — `_funnel(state)` computes the
  candidate→classified→(subject seen)→passed→in-reel drop chain (also counts
  `n_errors`/`error_reason`); `_diagnosis` pins the collapsing gate. A
  **provider-failure branch trumps prompt-level causes** when `n_errors >=
  n_cls//2` and the run actually collapsed (points at groq/gemini +
  `scripts/probe_vision.py`).
- **Run reset + stop** — `_run_app` calls `_reset(st)` on a new "Find highlights"
  so a fresh run never shows the previous run's panel; a sidebar "Reset / clear
  panel" button is always available.

**Architectural constraint (important):** the LangGraph run executes
SYNCHRONOUSLY inside the Streamlit script thread (`_stream_pipeline` loops
`app.stream(...)`). A custom in-app "Stop" button CANNOT work mid-run —
Streamlit can't process a click while the script is blocked. The only mid-run
abort is Streamlit's ⏹ toolbar Stop; after that, click Reset.

---

## 9. Signals & scoreboard (`signals/`)

- **audio.py** — audio-peak (librosa), **motion.py** — motion-intensity (opencv),
  **merge.py** — window merge into one candidate timeline.
- **scoreboard.py** — OCR-free reader. `_changed_score_regions` returns the set
  of score-box indices that changed; writes `KEY_SUBJECT_SCORE_CHANGE`
  (`_sb_subject_score_change`) when the recipe's `subject_score_index` matches.
  `select/selector._subject_scored()` prefers that per-row signal (falls back to
  any-score-change only when no index is set), so an opponent make is not
  confirmed as ours. Region tuning caveat: tighten regions onto the DIGITS only
  or the possession underline produces phantom changes.

---

## 10. Testing

- All tests are offline/mock-based. `tests/conftest.py` has an autouse fixture
  forcing `HYPEREEL_*_PROVIDER=mock` so the repo `.env`'s live settings never
  leak into tests.
- Coverage: per-module unit tests, provider-fallback, CLI integration, full
  end-to-end pipeline + HITL resume, app helpers (funnel/diagnosis/preflight/
  provider-health), and graph node behavior.
- Current suite: **164 passed, 4 skipped** (as of 2026-08-30).

Run: `pip install -e ".[dev]" && pytest`.

---

## 11. Live-run playbook (the two AAU games)

Two distinct games / teams — do not conflate:

| Team | Jersey | Opponent | Source | Recipe |
| --- | --- | --- | --- | --- |
| **UNL** | Black + Red | White/Teal | `downloads/WWon1VSj06w_hd.mp4` (1276×718) | `basketball_player23_blackred.yaml` (tuned) or generic |
| **EBE** | Black + Blue | Yellow | `downloads/QDhlscjhIVQ.mp4` (1920×1078) | `basketball_generic.yaml` + prompt |

Recommended live run (both providers set so vision is truly live):

```bash
env HYPEREEL_VISION_PROVIDER=nebius HYPEREEL_LLM_PROVIDER=nebius \
  streamlit run src/hypereel/app.py
```

Use `audience=individual` for a single-team reel (filters the opponent out).
`--max-candidates`/"limit to first N" ≥ 40 for a meaningful dry run — a tiny cap
samples only the tip-off (all dead-ball windows).

**Root-cause lesson (2026-08-30):** a run that produced 0 clips was NOT a prompt
problem — the Nebius account had hit HTTP 402 (budget exhausted), so every vision
call errored to `confidence=0.0`. This drove the pre-flight live ping and the
provider-error diagnosis branch so the failure can never again masquerade as a
bad prompt.
