# HypeReel — Week 3 Project Documentation

**Mastering Agentic AI Certification · Week 3: Build Your AI Agent**

- **Builder:** Shivani
- **Track:** Track 2 — Code-heavy with LangChain + LangGraph
- **Use case:** Bring-your-own (a recipe-driven highlight-reel builder)
- **GitHub:** https://github.com/shivanivarambally/hypereel
- **Demo video (≤5 min):** https://drive.google.com/file/d/1c_W8egamBQFVbjVJV_sDs186M_oqpM1u/view?usp=sharing
- **Design document (visual):** `design/HypeReel-Design.html` in the repo

---

## 1. Project overview

HypeReel turns a full game recording into a share-ready highlight reel that matches
a plain-language brief. The flagship demo is AAU basketball: a player or coach points
it at a 60-minute game, describes what they want in ordinary language (one player, a
whole team, only threes, no steals), and gets back a short reel of the plays that
matter, without the 3 to 4 hours of scrubbing and clip-cutting a video editor takes
today.

The piece I would reuse is the **recipe**. A recipe is a declarative YAML policy that
says what counts as a highlight in a given domain, so the same pipeline generalizes
past basketball. A recipe can be **event-based** (basketball: a vision model classifies
each candidate window against named moment types like made basket, three pointer, block,
steal break) or **quality-based** (a real-estate walkthrough: the vision model scores
sliding windows against a composition, lighting, and clarity rubric, and the sustained
highest-scoring segments become the clips).

Under the hood, HypeReel is a controllable, stateful, flow-based application with cycles
and human interrupts, built on LangGraph. It is a **single agent**: one graph, one shared
state object. The orchestration is deterministic on purpose. I added agentic behavior
only where it earns its cost, in two places: an LLM-as-judge that scores the assembled
reel and can send it back for one revision, and two human-in-the-loop gates that hold
every write action until a person approves.

## 2. The one-liner

My agent helps a player or coach turn a full 60-minute game recording into a share-ready
highlight reel matching a plain-language brief (one player, a whole team, only threes,
no steals) in a Streamlit web app, replacing the 3–4 hours of manual scrubbing and
clip-cutting in a video editor it takes today. It ingests the video, detects and
scoreboard-confirms the right plays, and critiques and re-cuts its own reel on its own
using ~6 tools (yt-dlp, frame sampling, scoreboard CV, a vision model, an LLM judge, and
ffmpeg), hands off to a human to approve the clip list before it renders and again before
it shares, and I'll know it works when a user gets a watchable, single-team reel in under
30 minutes that they'd actually post 8 times out of 10.

## 3. The agent framework

| Field | Fill-in |
| --- | --- |
| **Agent goal** | Turn a full game recording plus a plain-language brief into a time-boxed, share-ready highlight reel for one team or player. |
| **Where people use it** | A Streamlit web app (chat-like input plus a review panel). A CLI covers scripted and demo runs. |
| **Steps, in order** | 1) plan strategy from the recipe; 2) ingest and resolve the source (yt-dlp or local); 3) propose candidate windows from cheap local signals; 4) read the scoreboard; 5) classify each window with a vision model; 6) select clips to the time budget; 7) the LLM judge scores the reel and may revise once; 8) **human approves the clip list**; 9) render; 10) summarize; 11) **human approves the share**; 12) deliver. |
| **What it can do (tools)** | ~6 tools. `source resolver` (yt-dlp/local) — READ; `audio-peak detector` — READ; `motion detector` — READ; `scoreboard reader` (OCR-free CV) — READ; `vision classifier` (VLM) — READ; `LLM judge` — READ/reason; `renderer` (ffmpeg/moviepy, writes a *local* file) — WRITE; `share/upload` (behind a gate, connector stubbed) — WRITE. |
| **What it remembers** | **Session:** LangGraph `ReelState` plus an in-memory checkpointer — candidates, classifications, selected clips, approval flags. **Persistent:** a local JSON file — user profile (jersey number, team colors, default length, platform) and preferences learned from accept/reject. |
| **What it should never do** | Never share or upload without explicit approval. Never fabricate a moment label it is not confident about. Never include recipe-listed exclusions (timeouts, bench, shaky pans). Never center non-subject minors. |
| **Human-in-the-loop** | Gate 1: approve, reorder, or drop the proposed clip list before anything renders. Gate 2: approve before any share or publish. Both are implemented as LangGraph `interrupt_before` stops, so the run pauses and resumes on the same thread. |
| **What happens when something breaks** | Download fails: retry twice with backoff, then ask for another link or local file. Vision provider errors: degrade to audio+motion scoring, or fall back to the mock provider. Provider out of budget (a real 402 I hit): a pre-flight ping blocks the run with a clear message instead of silently returning zero clips. Subject never found: suggest switching to team-level. Budget underfilled: surface the shorter list at Gate 1 with a warning rather than shipping silently. |
| **How I know it worked** | End-to-end task completion, not single-shot accuracy: a user gets a watchable, single-team reel in under 30 minutes that they would actually post 8 times out of 10. |

## 4. Architecture

```
plan → ingest → propose → scoreboard → classify → select → judge
      → (judge: revise → select ↺  |  accept → approve_clips)
      → approve_clips [HUMAN GATE 1] → render → summarize
      → approve_share [HUMAN GATE 2] → deliver → END
```

Compiled with `interrupt_before=["approve_clips", "approve_share"]` and an in-memory
checkpointer, so the graph pauses at each gate and resumes on the same `thread_id`.
Cheap local signals **propose**, a vision model **confirms**, a scoreboard reader
**grounds**, deterministic code **selects**, the LLM judge **critiques**, and a human
**gates** the two write-ish actions.

Providers sit behind thin interfaces (`VisionProvider`, `LLMProvider`) built lazily by a
factory that reads two env vars, so the model layer is swappable: **Nebius Token Factory
running Qwen2.5-VL-72B** by default (the graded build routes model calls through Nebius),
with **Groq (Llama-4-Scout)**, **Gemini**, and a deterministic offline **mock** as
fallbacks. A missing key or SDK degrades to mock rather than crashing.

_(The full layered architecture image and per-section rationale are in
`design/HypeReel-Design.html`.)_

## 5. Datasets used

Source footage was two public AAU basketball game uploads on YouTube, used as raw input:

| Team | Jerseys | Opponent | YouTube ID | Resolution |
| --- | --- | --- | --- | --- |
| EBE | Black + Blue | Yellow | `QDhlscjhIVQ` | 1920×1078 |
| UNL | Black + Red | White/Teal | `WWon1VSj06w` | ~1276×718 (HD-unlock) |

No labeled dataset or training was involved; the vision model classifies windows
zero-shot against the recipe's natural-language moment descriptions, and the scoreboard
reader is a deterministic CV routine. Rendered reels are kept private. The full test
suite is offline and mock-based (no network, no keys), so nothing in CI depends on this
footage.

## 6. Prompts used during demo & testing

- "Capture EBE — the team wearing BLACK and BLUE jerseys. Include their made baskets and best offensive plays (drives to the hoop, clean shots, the ball passing through the net). The OPPONENT wears YELLOW jerseys — exclude the opponent's plays entirely; do not include a basket scored by the yellow team."
- "Capture UNL — the team wearing BLACK jerseys with RED trim (labeled "UNL" on the scoreboard). Include their made baskets and best offensive plays (drives to the hoop, clean shots, the ball passing through the net). The OPPONENT wears WHITE/TEAL jerseys — exclude the opponent's plays entirely; do not include a basket scored by the white/teal team.

## 7. Iterations I tried

In rough order (see `changelog.md` in the repo for the dated log):

1. **Initial build** — the LangGraph state machine end to end, the recipe schema, the
   provider abstraction with graceful fallback, CLI and Streamlit UI, and an offline
   mock-based test suite.
2. **Scoreboard grounding + LLM-as-judge** — the OCR-free scoreboard reader with per-team
   score attribution, and the judge node with a single bounded, broadening revision.
3. **Pipeline visualizer + per-node timing + "why 0 clips?" diagnosis** — surfaced the
   real graph nodes, the judge routing branch, and a drop-chain that pins the gate where a
   run collapsed.
4. **Pre-flight checker + provider-error diagnosis** — validate the form before submit,
   and a live provider ping that blocks a dead or out-of-budget provider up front.
5. **Root-cause fix: the zero-clip run** — traced it to a Nebius HTTP 402 (budget
   exhausted), not a bad prompt. This is what drove the pre-flight ping and the
   provider-failure branch in the diagnosis.
6. **Audience-override correctness fix** — made `audience=team` genuinely turn the
   one-team subject filter off (it had been a no-op), so `individual` reliably drops the
   opponent.
7. **Generic prompt-driven recipe + quick-test cap** — moved all fine-tuning into the
   runtime brief, and added the first-N-windows cap for fast dry runs.
8. **Documentation and design assets** — implementation notes, changelog, and the design
   document with an accurate single-agent architecture diagram.

## 8. Learnings and observations

- **The first tool failure is where the build actually starts.** A run that returned zero
  clips looked like a prompt problem. The real cause was a provider budget error (HTTP
  402) that made every vision call fail silently. I stopped guessing at the prompt, wrote
  a one-window probe against the live model to confirm the cause, then made that failure
  loud with a pre-flight ping. I do not take the agent's first explanation as the answer;
  I verify it.
- **Be deliberate about reads vs writes.** Treating rendering a local file as autonomous
  but sharing as a gated write kept the human-in-the-loop boundary honest and matched how
  a coach would actually want to use this.
- **One subtle correctness bug taught the audience lesson.** `audience=team` disables the
  subject filter, so an opponent basket slipped into a "single-team" reel. Use
  `individual` for one team; the fix was making the override actually reach the selector.
- **Source quality bounds detection.** On lower-resolution footage the vision model is
  less certain, which is why scoreboard confirmation matters: it grounds "was this
  actually a made basket" in game state instead of pixels alone.
- **Deterministic orchestration with agentic behavior only where it earns its cost.** A
  single graph with a judge loop and two gates handled the variance without a multi-agent
  system. The recipe abstraction is what lets this extend to other domains later, when
  there is a reason to.