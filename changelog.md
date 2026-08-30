# Changelog

All notable changes to HypeReel. Newest first. Timestamps are local (IST).
Implementation detail lives in [`implementation-notes.md`](implementation-notes.md).

Format: each entry is `### YYYY-MM-DD HH:MM TZ — summary`, followed by
Added / Changed / Fixed / Verified bullets and the files touched.

---

### 2026-08-30 20:55 IST — Design doc: one-page overview infographic

**Added** `design/overview.png` as an "at a glance" figure at the top of
`design/HypeReel-Design.html` (after the header chips, before §1): one-liner,
9-step high-level workflow, the two HITL approval gates, the ~6 read/write tools,
the web-app UX, and the pluggable recipe library.

Used the regenerated version with the **"Multi-agent-built" chip removed** (it
had reintroduced the single-agent/multi-agent over-claim). Added a muted caption
noting the recipe library shows example domains for pluggability; v1 ships
basketball + an architecture-walkthrough stub (per §11 Scope).

Files: `design/HypeReel-Design.html`, `design/overview.png`.

---

### 2026-08-30 20:44 IST — Design doc: accurate architecture image + memory wording fix

**Changed**
- Replaced the §3 Mermaid `flowchart TD` with a static layered image
  (`design/architecture.png`) that matches the code: single LangGraph agent
  (shared `ReelState`), the real node flow (`plan → ingest → propose →
  scoreboard → classify → select → judge → approve clips → render → summarize →
  approve share → deliver`) with the `judge → select` revise ×1 loop and both
  HITL gates, an InMemorySaver checkpointer + local JSON profile store, a
  swappable model layer (Nebius/Qwen2.5-VL default; Groq/Gemini/mock fallbacks),
  the ~6-tool layer, and external services (YouTube, Nebius).
- Removed the now-unused mermaid.js CDN loader + init script; updated the footer
  note (diagram is a static image, not mermaid-rendered).

**Fixed** memory wording to match the code (was "JSON / SQLite" as if
implemented): §2 Memory row and §6 cards now say **in-memory checkpointer** for
per-run `ReelState` + **local JSON file** for the persistent profile, with SQLite
framed as a drop-in swap.

**Why:** an earlier ChatGPT-generated diagram (fed the design doc) drew a false
4-agent architecture (Planner/Detection/Vision/Editor) with Llama/Mistral +
JSON/SQLite labels. Regenerated from a corrected prompt that pins the
single-agent runtime and the real stack; verified accurate before embedding.

Files: `design/HypeReel-Design.html`, `design/architecture.png`.

---

### 2026-08-30 16:05 IST — Documentation: implementation notes + changelog

**Added**
- `implementation-notes.md` — living technical reference (graph shape, nodes,
  state, recipes, providers, app, signals, testing, live-run playbook).
- `changelog.md` — this file.

Files: `implementation-notes.md`, `changelog.md`.

---

### 2026-08-30 ~15:xx IST — Live end-to-end validation (EBE game)

**Verified** the full pipeline against `downloads/QDhlscjhIVQ.mp4` (Team EBE)
with `basketball_generic.yaml` + a black/blue prompt, live on Nebius:
- Quick-test cap truncates correctly (`kept the first N of 207 windows`).
- Nebius answers image calls again — **no HTTP 402** (post top-up).
- Vision detection fires with correct color reasons
  (`player in black jersey ... makes a basket`, `subject_present=True`).
- Judge revision loop works: 40-window run first selected 19 padded clips
  (judge 0.2, `require_confirmed_moments`) → re-selected 5 confirmed makes
  (judge 0.7).

**Lesson captured:** `--audience team` disables the subject filter, so an
opponent basket slipped into the reel; use `--audience individual` for a
single-team reel. A tiny cap (e.g. 5) samples only the tip-off (all dead-ball),
so use ≥ 40 for a meaningful dry run.

No code change — validation run only.

---

### 2026-08-30 — Quick-test cap: classify only the first N candidate windows

**Added**
- `ReelState.max_candidates: Optional[int]` (None/0 = unbounded), defaulted in
  `new_state` from overrides.
- `propose_node` caps candidates after proposing:
  `sorted(candidates, key=lambda w: w.start)[:cap]` when `cap > 0`, with a note.
- `run_pipeline(max_candidates=...)` param.
- Streamlit sidebar `number_input` ("Quick test: limit to first N candidate
  windows (0 = all)"); folded into pre-flight as a WARN when set.
- CLI `--max-candidates N` for parity.
- Tests: `test_new_state_carries_max_candidates`,
  `test_propose_node_caps_to_first_n_windows_chronologically`,
  `test_propose_node_cap_unset_or_zero_is_unbounded`.

**Verified** suite 164 passed / 4 skipped.

Files: `graph/state.py`, `graph/nodes.py`, `graph/build.py`, `app.py`, `cli.py`,
`tests/test_graph.py`.

---

### 2026-08-30 — Run reset + stop UX

**Added**
- `_run_app` calls `_reset(st)` as soon as a new "Find highlights" is clicked
  (before drawing the diagram), so a new run never shows the previous run's
  clips/diagnosis/timings.
- Sidebar "Reset / clear panel" button (always available) + Stop guidance caption.

**Documented constraint:** the LangGraph run executes synchronously in the
Streamlit script thread, so a custom in-app Stop cannot work mid-run — only
Streamlit's ⏹ toolbar Stop aborts. A true cancel would need a background thread
(not done; overkill for the demo).

Files: `app.py`.

---

### 2026-08-30 — Silent-provider-failure fixes

**Added**
- **Live provider ping** — `_provider_health(vision, llm)` makes one real
  `classify_window` call with a real 64×64 image (`_health_frame_path`) and reads
  `.reason`. Cached per provider-pair; "Re-check providers" button clears it.
  Folded into pre-flight as a hard blocker; humanized by
  `_humanize_provider_error` (402/budget → switch to groq/gemini; 401 → check key).
- **Provider-error diagnosis branch** — `_funnel` counts `n_errors`/
  `error_reason`; `_diagnosis` returns a provider-failure headline that TRUMPS
  prompt/subject-filter causes when `n_errors >= n_cls//2` and the run collapsed.
- Shared helpers `_trim`, `_is_provider_error`, `_looks_like_budget_error`.

**Why:** a dead provider previously masqueraded as a bad prompt (badge showed
LIVE, all nodes ran green, diagnosis blamed the prompt).

Files: `app.py`, `tests/test_app.py`, `scripts/probe_vision.py`.

---

### 2026-08-30 — Root cause: EBE 0-clip run (NOT the prompt)

**Diagnosed** via a live one-window probe (`scripts/probe_vision.py`): every
Nebius vision call returned `402 - Payment Required: You have exhausted your
budget.` and the text LLM returned `""`. All 207 vision calls errored →
`confidence=0.0` → 0 clips regardless of prompt or audience. Resolution had
already been ruled out (source is 1920×1078). Fixed by topping up Nebius; drove
the pre-flight ping + provider-error diagnosis above.

Files: `scripts/probe_vision.py` (added).

---

### 2026-08-30 — Audience override now actually works (correctness fix)

**Fixed** the runtime `audience` override was previously a no-op for filtering.
Now `_effective_recipe(recipe, overrides, audience=...)` bakes the override onto
`subject_selector.audience` (deep copy) and `select_node` passes
`audience=state.get("audience")`, so `team` genuinely turns the one-team subject
filter OFF. Guarded by
`test_select_node_team_audience_override_disables_subject_filter`.

Files: `graph/nodes.py`, `app.py`, `tests/`.

---

### 2026-08-30 — Pre-flight prompt checker + submission gating

**Added** `_preflight(...)` validating the sidebar form live (offline,
deterministic) before submit. ERROR disables "Find highlights"; WARN flags
likely-fail settings. Recipe now loaded in the sidebar so the "Describe" label
can flip to "required for this recipe".

Files: `app.py`, `tests/test_app.py`.

---

### 2026-08-30 — Per-node timing + "why 0 clips?" diagnosis

**Added**
- Per-node wall-clock timing in `_stream_pipeline` (⏱ chip per completed box).
- `_funnel(state)` drop-chain (candidate → classified → subject seen → passed →
  in reel) and `_diagnosis`/`_diagnosis_html` pinning the gate that collapsed to 0.

Files: `app.py`, `tests/test_app.py`.

---

### 2026-08-30 — "Under the hood" pipeline visualizer

**Added** a collapsible Streamlit expander rendering the real LangGraph nodes as
status cards (done/running/awaiting/queued), the Judge routing branch
(revise ↺ / accept ✓), a per-run state strip, and per-node tool-call chips
(👁 vision-per-window, 🧠 text-LLM). Live updates via
`app.stream(..., stream_mode="updates")`, falling back to blocking `invoke`.

Files: `app.py`, `tests/test_app.py`.

---

### 2026-08-30 — Generic prompt-driven recipe

**Added** `recipes/basketball_generic.yaml` (`basketball_generic_v1`) — no
hardcoded colors/numbers/opponent/scoreboard; all fine-tuning via the runtime
brief. Invariant: `subject_selector.type: team_color` + `audience: individual`
so opponent filtering works. Guarded by
`test_generic_recipe_is_prompt_driven_and_filters_one_team`.

Files: `recipes/basketball_generic.yaml`, `tests/test_recipes.py`.

---

### 2026-08-30 — Free-text subject brief + HD-from-URL + provider badge

**Added**
- Free-text subject brief (`SubjectSelector.description`) surfaced as the leading
  block of the classification prompt; supplied via `new_state(subject_description=)`,
  exposed as a sidebar `text_area` and CLI `--describe`.
- `parse_video_quality(text)` maps quality words in the brief to `max_height`.
- HD-from-URL in `ingest/source_resolver.py` (`resolve_source(max_height=...)`
  tries an HD-unlock ydl-opts profile first, then a plain fallback).
- Streamlit provider badge (🟢 LIVE / 🟡 PARTIAL / ⚪️ MOCK).

Files: `models.py`, `providers/_util.py`, `ingest/source_resolver.py`,
`graph/nodes.py`, `graph/state.py`, `app.py`, `cli.py`.

---

### 2026-08-30 — Scoreboard grounding + LLM-as-judge critic

**Added**
- `signals/scoreboard.py` — OCR-free clock/score reader with per-team
  attribution (`_changed_score_regions`, `subject_score_index`).
- `select/selector._confirm_moment()` — upgrades `moment_type=null` →
  `made_basket` on a scoreboard-confirmed subject score change.
- **Judge node** — LLM-as-judge scores the reel against acceptance criteria and
  can bounce to `select` once (`_MAX_REVISIONS = 1`) with *smart* corrective
  overrides (broadens when thin instead of tightening into a corner).

Files: `graph/nodes.py`, `graph/build.py`, `signals/scoreboard.py`,
`select/selector.py`.

---

### (earlier) — Initial HypeReel build

LangGraph state machine (plan → ingest → propose → classify → select →
approve_clips → render → summarize → approve_share → deliver), recipe schema,
mock/nebius/groq/gemini providers with graceful fallback, CLI + Streamlit UI,
offline mock-based test suite. See `README.md` and `design/HypeReel-Design.html`.
