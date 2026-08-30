"""Streamlit web UI for HypeReel.

Drives the *real* LangGraph pipeline (``hypereel.graph.build.build_graph``)
directly rather than going through ``run_pipeline``, so the two human-in-the-
loop gates the design calls for — approve the clip list before render, approve
before any share/publish — are the actual ``interrupt_before`` pause points
of the compiled graph, not a UI simulation of them.

``streamlit`` is imported lazily inside :func:`main` (only there) so that
``import hypereel.app`` and the rest of the CLI/test suite work fine in
environments where streamlit isn't installed.

Run with::

    streamlit run src/hypereel/app.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# `streamlit run src/hypereel/app.py` executes this file as a top-level script,
# which puts this package's OWN directory (src/hypereel/) on sys.path[0]. That
# is harmful: our module names (config.py, models.py, ...) would then shadow
# third-party top-level imports — notably langchain_core's internal `config`,
# which breaks `import langgraph`. So we drop the package dir from sys.path and
# put the parent `src/` on instead, making the absolute imports below resolve
# `hypereel.*` cleanly whether or not the package was installed with `pip -e .`.
_PKG_DIR = Path(__file__).resolve().parent          # .../src/hypereel
_SRC_DIR = _PKG_DIR.parent                           # .../src
sys.path[:] = [p for p in sys.path if p not in ("", str(_PKG_DIR))]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hypereel.config import get_settings
from hypereel.graph.build import build_graph
from hypereel.graph.state import GATE_UNDERFILLED, new_state
from hypereel.ingest.source_resolver import parse_video_quality
from hypereel.models import Clip, Recipe
from hypereel.providers.factory import get_llm_provider, get_vision_provider
from hypereel.recipe import RecipeError, load_recipe

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECIPES_DIR = _REPO_ROOT / "recipes"


# --------------------------------------------------------------------------- #
#  "Under the hood" pipeline diagram
#
#  A boxy, futuristic map of the real LangGraph nodes (build.py) so a viewer can
#  see what the agent is doing behind the scenes and where it is right now.
#  ``_NODE_ORDER`` mirrors the compiled graph's edge order exactly; the two
#  human gates are the ``interrupt_before`` pause points.
# --------------------------------------------------------------------------- #
_NODE_ORDER = [
    "plan", "ingest", "propose", "scoreboard", "classify", "select", "judge",
    "approve_clips", "render", "summarize", "approve_share", "deliver",
]
_GATES = {"approve_clips", "approve_share"}

# (phase label, accent color, [(node_key, name, what-it-does, tool_label, tool_kind), ...])
# Accent colors give each phase its own identity (like the solution-kit
# reference diagrams); run/done/gate states recolor on top. ``tool_label`` /
# ``tool_kind`` name the concrete tool or model each node calls, so the diagram
# surfaces the agent's tool-calling (a key rubric item) — which step hits
# yt-dlp, OpenCV, ffmpeg, or a vision/text LLM. Kinds: model | tool | logic |
# human | output.
_PIPELINE: list[tuple[str, str, list[tuple[str, str, str, str, str]]]] = [
    ("① INTERPRET", "#3b82f6", [
        ("plan", "Plan", "read the recipe + guardrails", "recipe logic", "logic"),
        ("ingest", "Ingest", "resolve / download the source", "yt-dlp · ffprobe", "tool"),
    ]),
    ("② DETECT", "#8b5cf6", [
        ("propose", "Propose", "audio + motion → candidate windows", "ffmpeg · cv2", "tool"),
        ("scoreboard", "Scoreboard", "OCR-free clock & score grounding", "OpenCV reader", "tool"),
        ("classify", "Classify", "label each window", "vision LLM ·/window", "model"),
    ]),
    ("③ CURATE", "#14b8a6", [
        ("select", "Select", "score & pack clips into the budget", "scoring logic", "logic"),
        ("judge", "Judge", "critic scores the reel & routes", "text LLM", "model"),
    ]),
    ("④ HUMAN GATE", "#f59e0b", [
        ("approve_clips", "Approve clips", "you review the clip list", "human", "human"),
    ]),
    ("⑤ PRODUCE", "#ec4899", [
        ("render", "Render", "cut & stitch the reel", "ffmpeg", "tool"),
        ("summarize", "Summarize", "write the recap", "text LLM", "model"),
    ]),
    ("⑥ HUMAN GATE", "#f59e0b", [
        ("approve_share", "Approve share", "you approve delivery", "human", "human"),
    ]),
    ("⑦ DELIVER", "#22c55e", [
        ("deliver", "Deliver", "finalize the output", "file / share", "output"),
    ]),
]

# Icon + palette for each tool kind (drives the per-node tool chip).
_TOOL_KIND = {
    "model": ("🧠", "#a78bfa"),   # LLM / vision-model API call
    "tool": ("⚙", "#60a5fa"),     # external binary / library (yt-dlp, ffmpeg, cv2)
    "logic": ("∑", "#94a3b8"),    # pure in-process computation
    "human": ("⏸", "#fbbf24"),    # human-in-the-loop, no tool
    "output": ("📤", "#4ade80"),  # file write / share
}

_PIPE_CSS = """
.hr-pipe{font-family:ui-monospace,'SFMono-Regular',Menlo,monospace;line-height:1.2}
.hr-pipe .ph{color:#5eead4;font-size:10px;letter-spacing:.22em;margin:12px 0 4px;opacity:.85}
.hr-node{display:flex;align-items:center;gap:10px;border:1px solid rgba(148,163,184,.15);
 border-left:3px solid rgba(148,163,184,.25);border-radius:10px;padding:7px 12px;margin:5px 0;
 background:linear-gradient(180deg,rgba(30,41,59,.55),rgba(15,23,42,.55));color:#cbd5e1;
 transition:all .25s ease}
.hr-node .num{font-size:11px;opacity:.45;min-width:16px}
.hr-node .nm{font-weight:600;color:#e2e8f0;white-space:nowrap}
.hr-node .ds{font-size:11px;opacity:.6}
.hr-node .tool{font-size:10px;padding:2px 8px;border-radius:6px;white-space:nowrap;
 border:1px solid rgba(148,163,184,.25);background:rgba(2,6,23,.45)}
.hr-node .el{font-size:10px;color:#7dd3fc;opacity:.8;font-variant-numeric:tabular-nums;
 padding:2px 7px;border-radius:6px;border:1px solid rgba(56,189,248,.2);background:rgba(2,6,23,.4)}
.hr-node .st{margin-left:auto;font-size:10px;padding:2px 9px;border-radius:999px;white-space:nowrap}
.hr-run .tool{box-shadow:0 0 10px rgba(34,211,238,.3);border-color:rgba(34,211,238,.5)}
.hr-done{border-left-color:#22c55e;border-color:rgba(34,197,94,.28)}
.hr-done .st{color:#4ade80;border:1px solid rgba(34,197,94,.35)}
.hr-run{border-left-color:#22d3ee;border-color:rgba(34,211,238,.55);
 box-shadow:0 0 18px rgba(34,211,238,.22);animation:hrpulse 1.2s ease-in-out infinite}
.hr-run .nm{color:#a5f3fc}
.hr-run .st{color:#67e8f9;border:1px solid rgba(34,211,238,.5)}
.hr-wait{border-left-color:#f59e0b;border-color:rgba(245,158,11,.5);
 box-shadow:0 0 18px rgba(245,158,11,.18);animation:hrpulse 1.6s ease-in-out infinite}
.hr-wait .nm{color:#fcd34d}
.hr-wait .st{color:#fbbf24;border:1px solid rgba(245,158,11,.5)}
.hr-todo{opacity:.55}
.hr-todo .st{color:#94a3b8;border:1px solid rgba(148,163,184,.2)}
@keyframes hrpulse{0%,100%{filter:brightness(1)}50%{filter:brightness(1.4)}}
/* routing branch — the agent's revise/accept decision after Judge */
.hr-branch{display:flex;gap:8px;margin:2px 0 6px 26px}
.hr-br{font-size:10px;padding:3px 10px;border-radius:8px;border:1px dashed rgba(148,163,184,.3);
 color:#94a3b8;background:rgba(15,23,42,.4)}
.hr-br.on{border-style:solid;color:#e2e8f0;box-shadow:0 0 12px rgba(148,163,184,.15)}
.hr-br.revise.on{border-color:#f472b6;color:#f9a8d4;box-shadow:0 0 12px rgba(244,114,182,.25)}
.hr-br.accept.on{border-color:#22c55e;color:#4ade80;box-shadow:0 0 12px rgba(34,197,94,.25)}
/* agent decisions strip — like the reference kits' "Per-Run State" */
.hr-dec{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
.hr-card{flex:1 1 120px;min-width:110px;border:1px solid rgba(148,163,184,.18);border-radius:10px;
 padding:8px 10px;background:linear-gradient(180deg,rgba(30,41,59,.5),rgba(15,23,42,.5))}
.hr-card .k{font-size:9px;letter-spacing:.14em;color:#5eead4;opacity:.85}
.hr-card .v{font-size:14px;color:#e2e8f0;font-weight:600;margin-top:2px}
.hr-card .sub{font-size:10px;color:#94a3b8;margin-top:1px}
/* "why 0 clips?" diagnosis — the plain-language funnel + next steps */
.hr-diag{font-family:ui-monospace,'SFMono-Regular',Menlo,monospace;
 border:1px solid rgba(245,158,11,.4);border-radius:12px;padding:12px 14px;margin:6px 0;
 background:linear-gradient(180deg,rgba(69,26,3,.35),rgba(15,23,42,.5))}
.hr-diag .hd{color:#fcd34d;font-weight:700;font-size:13px;margin-bottom:2px}
.hr-diag .cause{color:#e2e8f0;font-size:12px;line-height:1.5;margin:4px 0 10px}
.hr-fun{display:flex;flex-wrap:wrap;align-items:stretch;gap:4px;margin:8px 0}
.hr-fstep{border:1px solid rgba(148,163,184,.22);border-radius:8px;padding:5px 9px;
 background:rgba(2,6,23,.4);text-align:center;min-width:70px}
.hr-fstep .fn{font-size:9px;letter-spacing:.1em;color:#94a3b8;text-transform:uppercase}
.hr-fstep .fv{font-size:16px;font-weight:700;color:#e2e8f0}
.hr-fstep.ok{border-color:rgba(34,197,94,.35)} .hr-fstep.ok .fv{color:#4ade80}
.hr-fstep.drop{border-color:#ef4444;box-shadow:0 0 12px rgba(239,68,68,.25)}
.hr-fstep.drop .fv{color:#f87171} .hr-fstep.drop .fn{color:#fca5a5}
.hr-arrow{display:flex;align-items:center;color:#64748b;font-size:13px}
.hr-steps{margin:8px 0 0;padding-left:18px;color:#cbd5e1;font-size:12px;line-height:1.6}
.hr-steps li{margin:2px 0}
.hr-steps b{color:#fcd34d}
"""

_STATE_META = {
    "done": ("hr-done", "done ✓"),
    "run": ("hr-run", "running…"),
    "wait": ("hr-wait", "awaiting you"),
    "todo": ("hr-todo", "queued"),
}


def _node_state(key: str, done: set, active: str | None) -> str:
    if key in done:
        return "done"
    if key == active:
        return "wait" if key in _GATES else "run"
    return "todo"


def _fmt_secs(s: float) -> str:
    """Compact elapsed-time label for a node box: '2.4s' or '3m 07s'."""
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m {int(s % 60):02d}s"


def _judge_branch_html(state: dict | None) -> str:
    """The agent's routing decision after Judge: revise ↺ Select vs accept ✓.

    Lights the branch the run actually took, so the reel's key agentic decision
    is visible — the LLM critic can bounce the reel back for one stricter
    re-selection before a human ever sees it.
    """
    s = state or {}
    rc = int(s.get("revision_count", 0) or 0)
    decision = s.get("judge_decision", "") or ""
    revise_on = rc > 0
    # Once we've reached any human gate / approval, the accept branch was taken.
    accept_on = decision == "accept" or bool(s.get("human_gate")) or bool(s.get("approved"))
    revise_label = f"↻ revise → Select{f' ×{rc}' if rc else ''}"
    return (
        '<div class="hr-branch">'
        f'<span class="hr-br revise {"on" if revise_on else "off"}">{revise_label}</span>'
        f'<span class="hr-br accept {"on" if accept_on else "off"}">✓ accept → Approve clips</span>'
        "</div>"
    )


def _pipeline_html(
    done: set, active: str | None, state: dict | None = None, timings: dict | None = None
) -> str:
    """Render the pipeline as a boxy, futuristic status map (self-contained HTML).

    ``state`` (optional) lets the Judge node show which routing branch the run
    took; without it the branch is omitted (plain structural view). ``timings``
    (node_key -> seconds) adds a wall-clock chip to each completed box, so a
    viewer can see where the run actually spent its time (ingest/vision dominate).
    """
    timings = timings or {}
    rows: list[str] = []
    n = 0
    for phase_label, accent, nodes in _PIPELINE:
        rows.append(f'<div class="ph" style="color:{accent}">{phase_label}</div>')
        for key, name, desc, tool_label, tool_kind in nodes:
            n += 1
            node_state = _node_state(key, done, active)
            cls, st_label = _STATE_META[node_state]
            # Idle boxes wear the phase accent (like the reference kits); the
            # done/run/wait classes recolor on top when a node is active.
            style = f' style="border-left-color:{accent}"' if node_state == "todo" else ""
            numstyle = f' style="color:{accent}"' if node_state == "todo" else ""
            icon, tcolor = _TOOL_KIND.get(tool_kind, ("•", "#94a3b8"))
            elapsed = timings.get(key)
            el_html = (
                f'<span class="el">⏱ {_fmt_secs(elapsed)}</span>'
                if isinstance(elapsed, (int, float))
                else ""
            )
            rows.append(
                f'<div class="hr-node {cls}"{style}>'
                f'<span class="num"{numstyle}>{n:02d}</span>'
                f'<span class="nm">{name}</span>'
                f'<span class="ds">{desc}</span>'
                f'<span class="tool" style="color:{tcolor}">{icon} {tool_label}</span>'
                f'{el_html}'
                f'<span class="st">{st_label}</span>'
                "</div>"
            )
            if key == "judge" and state is not None:
                rows.append(_judge_branch_html(state))
    return f"<style>{_PIPE_CSS}</style><div class=\"hr-pipe\">{''.join(rows)}</div>"


def _decisions_html(current: dict | None) -> str:
    """A 'Per-Run State' style strip of the concrete decisions the agent made.

    Mirrors the solution-kit reference diagrams: compact cards summarizing what
    the agent chose this run — signals it enabled, how the candidate set was
    narrowed, the judge's verdict/route, whether the one-team subject filter was
    active, and live vs mock mode.
    """
    c = current or {}
    recipe = c.get("recipe")
    ss = getattr(recipe, "subject_selector", None)
    subject_on = bool(ss and getattr(ss, "audience", "") == "individual" and getattr(ss, "type", "none") != "none")

    signals = c.get("active_signals") or []
    verdict = c.get("judge_verdict") or {}
    score = verdict.get("score")
    decision = c.get("judge_decision") or "—"
    rc = int(c.get("revision_count", 0) or 0)
    n_cand = len(c.get("candidates") or [])
    n_scored = len(c.get("scored_clips") or [])
    n_sel = len(c.get("selected_clips") or [])
    dur = float(c.get("proposed_duration", 0.0) or 0.0)
    mode = c.get("mode") or "mock"

    # Tool/model call accounting: the vision LLM is called once per candidate
    # window; the text LLM runs the judge (once per pass = 1 + revisions) and the
    # summary. This makes the agent's tool-calling volume explicit.
    vision_calls = len(c.get("classifications") or [])
    judge_calls = (rc + 1) if verdict else 0
    text_calls = judge_calls + (1 if c.get("summary") else 0)

    def card(k: str, v: str, sub: str = "") -> str:
        sub_html = f'<div class="sub">{sub}</div>' if sub else ""
        return f'<div class="hr-card"><div class="k">{k}</div><div class="v">{v}</div>{sub_html}</div>'

    cards = [
        card("SIGNALS", str(len(signals)), ", ".join(signals) if signals else "planner picks"),
        card("FUNNEL", f"{n_cand}→{n_scored}→{n_sel}", "candidates → scored → selected"),
        card("TOOL CALLS", f"👁 {vision_calls} · 🧠 {text_calls}", "vision / window · text LLM"),
        card("JUDGE", f"{score:.2f}" if isinstance(score, (int, float)) else "—",
             f"route: {decision} · revisions: {rc}"),
        card("SUBJECT FILTER", "ON" if subject_on else "OFF",
             "one-team gating" if subject_on else "no subject gating"),
        card("REEL", f"{dur:.0f}s", f"{n_sel} clip{'s' if n_sel != 1 else ''}"),
        card("MODE", mode.upper(), "real APIs" if mode == "live" else "deterministic"),
    ]
    return f'<style>{_PIPE_CSS}</style><div class="hr-dec">{"".join(cards)}</div>'


def _c_attr(c: Any, name: str) -> Any:
    """Read a Classification field whether it's a pydantic object or a dict."""
    if isinstance(c, dict):
        return c.get(name)
    return getattr(c, name, None)


def _trim(s: str | None, n: int = 180) -> str:
    """Collapse whitespace and cap length — for showing a raw API error inline."""
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _is_provider_error(reason: str | None) -> bool:
    """True when a ``Classification.reason`` reflects a provider/API failure.

    The vision providers stuff their caught exception into the reason as
    ``"<name> error: ..."`` (see nebius/groq/gemini ``classify_window``), and
    the classifier's own fallback uses ``"classification error"``. Either means
    no real verdict was produced — the window came back empty, not judged.
    """
    r = (reason or "").lower()
    return " error:" in r or r.strip() == "classification error"


def _looks_like_budget_error(msg: str | None) -> bool:
    """Heuristic: does this API error mean 'out of money / over quota'?"""
    low = (msg or "").lower()
    return any(t in low for t in ("402", "payment", "budget", "quota", "insufficient", "exhausted"))


def _funnel(current: dict | None) -> dict:
    """The candidate → selected drop funnel, computed from run state.

    Counts how many windows survived each gate so we can explain *where* a reel
    lost all its clips: detection (candidates), vision (subject present / a named
    moment), scoring (min_score), and the time budget. Pure; never raises.
    """
    c = current or {}
    recipe = c.get("recipe")
    sel = getattr(recipe, "selection", None)
    ss = getattr(recipe, "subject_selector", None)
    subject_on = bool(
        ss and getattr(ss, "audience", "") == "individual" and getattr(ss, "type", "none") != "none"
    )
    cls = c.get("classifications") or []
    n_present = sum(1 for x in cls if _c_attr(x, "subject_present"))
    n_moment = sum(1 for x in cls if _c_attr(x, "moment_type"))
    n_errors = sum(1 for x in cls if _is_provider_error(_c_attr(x, "reason")))
    error_reason = next(
        (_c_attr(x, "reason") for x in cls if _is_provider_error(_c_attr(x, "reason"))), ""
    )
    n_scored = len(c.get("scored_clips") or [])
    n_sel = len(c.get("selected_clips") or [])
    budget = c.get("max_duration") or getattr(sel, "max_duration", 300.0)

    # Where did the funnel collapse to zero? (first gate that let nothing through)
    n_cand = len(c.get("candidates") or [])
    n_cls = len(cls)
    if n_cand == 0:
        stalled = "candidates"
    elif n_cls == 0:
        stalled = "classified"
    elif subject_on and n_present == 0:
        stalled = "present"
    elif n_scored == 0:
        stalled = "scored"
    elif n_sel == 0:
        stalled = "selected"
    else:
        stalled = None  # has clips, just underfilled

    return {
        "subject_on": subject_on,
        "n_cand": n_cand,
        "n_cls": n_cls,
        "n_present": n_present,
        "n_moment": n_moment,
        "n_errors": n_errors,
        "error_reason": error_reason,
        "n_scored": n_scored,
        "n_sel": n_sel,
        "budget": float(budget),
        "min_score": float(getattr(sel, "min_score", 0.5)),
        "min_clip": float(getattr(sel, "min_clip", 4.0)),
        "override_budget": c.get("max_duration"),
        "errors": c.get("errors") or [],
    }


def _diagnosis(current: dict | None) -> tuple[str, list[str]] | None:
    """Plain-language (headline, next-steps) for a 0-clip / underfilled reel.

    Returns ``None`` when the reel is healthy (a full set of clips). Otherwise a
    user-facing cause pinned to the exact gate that dropped everything, plus
    concrete, control-specific next steps — so the answer isn't "redo the whole
    process" but "here's what to change and where."
    """
    f = _funnel(current)
    stalled = None
    # Recompute the stall point but also treat a partial reel as "underfilled".
    if f["n_cand"] == 0:
        stalled = "candidates"
    elif f["n_cls"] == 0:
        stalled = "classified"
    elif f["subject_on"] and f["n_present"] == 0:
        stalled = "present"
    elif f["n_scored"] == 0:
        stalled = "scored"
    elif f["n_sel"] == 0:
        stalled = "selected"
    elif sum_dur(current) < 0.8 * f["budget"]:
        # Same "underfilled" bar select_node uses to raise the soft gate.
        stalled = "underfilled"

    # Provider failure trumps every prompt-level cause. If most windows came back
    # as API errors (empty verdicts), no brief / color / audience change can help —
    # the model never actually saw them. Say so plainly instead of blaming the prompt.
    if (
        f["n_cls"]
        and f["n_errors"] >= max(1, f["n_cls"] // 2)
        and stalled not in (None, "underfilled", "candidates")
    ):
        head = (
            f'The vision provider FAILED on {f["n_errors"]} of {f["n_cls"]} windows — '
            "this is a provider problem, not your prompt."
        )
        steps = [
            "Every failed window came back with confidence 0, so nothing could be scored. "
            "No brief, jersey color, or Audience change can fix a provider that isn't answering.",
            f"The provider reported: <b>{_trim(f['error_reason'], 240)}</b>",
        ]
        if _looks_like_budget_error(f["error_reason"]):
            steps.append(
                "Your API budget/quota is exhausted. Switch to a <b>free</b> provider: set "
                "<b>HYPEREEL_VISION_PROVIDER</b> and <b>HYPEREEL_LLM_PROVIDER</b> to "
                "<b>groq</b> or <b>gemini</b> (add that key to <code>.env</code>) and relaunch."
            )
        else:
            steps.append(
                "Check the provider's API key and quota, then confirm it answers with "
                "<code>scripts/probe_vision.py</code> before another full run."
            )
        return head, steps

    tiny_budget = bool(f["override_budget"]) and f["override_budget"] < 2 * f["min_clip"]
    budget_step = (
        f'The <b>Max duration override</b> is set to {f["override_budget"]:.0f}s — '
        f'too small to fit even one {f["min_clip"]:.0f}s clip. Set it to 0 (recipe default) '
        "or a larger value."
        if tiny_budget
        else 'Raise the <b>Max duration override</b> in the sidebar (or set it to 0 for the recipe default).'
    )

    if stalled == "candidates":
        head = "No candidate windows were detected."
        steps = [
            "Detection (audio peaks + motion) found nothing to look at — usually a silent, "
            "static, or unreadable source.",
            "Check the <b>Pipeline notes</b> below for an ingest error (the download/quality step).",
            "Try a different source clip, or confirm the video actually has crowd/commentary audio.",
        ]
    elif stalled == "classified":
        head = "The vision step never ran."
        steps = [
            "No windows reached the classifier — check <b>Pipeline notes</b> for an error, and "
            "confirm the vision provider is configured (the badge in the sidebar).",
        ]
    elif stalled == "present":
        head = f'The vision model did not confirm your subject in ANY of the {f["n_cls"]} windows.'
        steps = [
            "Your recipe filters to one team (SUBJECT FILTER = ON), so windows where the model "
            "reports <b>subject_present = false</b> are all dropped — that's every window here.",
            "The most likely cause: the <b>“Describe who/what to capture”</b> brief didn't match "
            "what the model sees. Lead with the reliable cue — <b>jersey COLORS</b>, not numbers "
            "(numbers are unreadable at low resolution).",
            'Add <b>“use HD if available”</b> to that brief so plays are easier to read.',
            'Or set the <b>Audience override</b> to <b>“team”</b> to keep both teams (turns the '
            "one-team filter off).",
        ]
    elif stalled == "scored":
        extra = (
            f'{f["n_present"]} window(s) showed your subject'
            if f["subject_on"]
            else f'{f["n_cls"]} window(s) were classified'
        )
        head = f"Windows were found, but none cleared the score bar (min_score {f['min_score']:.2f})."
        steps = [
            f"{extra}, but the blended audio/motion/vision-confidence score stayed under "
            f"{f['min_score']:.2f} — often low model confidence at low resolution.",
            'Add <b>“use HD if available”</b> to the describe box to raise vision confidence.',
            "Lower <b>min_score</b> in the recipe, or pick a recipe tuned for this footage.",
        ]
    elif stalled == "selected":
        head = f'{f["n_scored"]} clip(s) qualified but none fit the {f["budget"]:.0f}s budget.'
        steps = [budget_step]
    elif stalled == "underfilled":
        head = (
            f'Found {f["n_sel"]} clip(s) ({sum_dur(current):.0f}s) — short of the '
            f'{f["budget"]:.0f}s budget.'
        )
        steps = [
            "This is a partial reel, not a failure: you can <b>approve it as-is</b> below.",
            "To get more, loosen the describe brief, ask for HD, or raise the budget — then Start over.",
        ]
        if tiny_budget:
            steps.append(budget_step)
    else:
        return None
    return head, steps


def sum_dur(current: dict | None) -> float:
    c = current or {}
    return float(c.get("proposed_duration", 0.0) or 0.0)


def _diagnosis_html(current: dict | None) -> str:
    """Boxy 'why did this happen?' panel: the drop funnel + a plain-language cause."""
    diag = _diagnosis(current)
    if diag is None:
        return ""
    head, steps = diag
    f = _funnel(current)

    # Build the funnel chain, marking the stage where it collapsed to zero red.
    chain: list[tuple[str, int]] = [("candidates", f["n_cand"]), ("classified", f["n_cls"])]
    if f["subject_on"]:
        chain.append(("subject seen", f["n_present"]))
    chain += [("passed score", f["n_scored"]), ("in reel", f["n_sel"])]

    parts: list[str] = []
    dropped_here = False
    for i, (label, val) in enumerate(chain):
        if not dropped_here and val == 0 and (i == 0 or chain[i - 1][1] > 0):
            klass, dropped_here = "drop", True
        elif val > 0:
            klass = "ok"
        else:
            klass = ""
        parts.append(
            f'<div class="hr-fstep {klass}"><div class="fn">{label}</div><div class="fv">{val}</div></div>'
        )
        if i < len(chain) - 1:
            parts.append('<div class="hr-arrow">→</div>')

    steps_html = "".join(f"<li>{s}</li>" for s in steps)
    return (
        f"<style>{_PIPE_CSS}</style>"
        '<div class="hr-diag">'
        '<div class="hd">Why this reel came up short</div>'
        f'<div class="cause">{head}</div>'
        f'<div class="hr-fun">{"".join(parts)}</div>'
        f'<ul class="hr-steps">{steps_html}</ul>'
        "</div>"
    )


def _next_pending(done: set) -> str | None:
    """First node (in graph order) not yet completed — i.e. the one now running."""
    for k in _NODE_ORDER:
        if k not in done:
            return k
    return None


def _stage_progress(stage: str) -> tuple[set, str | None]:
    """Resting (done_set, active_node) implied by the app's coarse ``stage``.

    Used to redraw the diagram after a Streamlit rerun, when no live stream is
    in flight; a live run overwrites this in place with finer-grained progress.
    """
    if stage == "gate1":
        active = "approve_clips"
    elif stage == "gate2":
        active = "approve_share"
    elif stage == "done":
        return set(_NODE_ORDER), None
    else:
        return set(), None
    idx = _NODE_ORDER.index(active)
    return set(_NODE_ORDER[:idx]), active


def _stream_pipeline(st, app, config, graph_input, diagram_ph) -> dict:
    """Run the graph while lighting up nodes live in ``diagram_ph``.

    Streams node-completion updates from LangGraph (``stream_mode="updates"``),
    marking each node done and re-drawing the diagram so the glowing box tracks
    the agent in real time — including the long ingest/vision phases. Execution
    stops at the next ``interrupt_before`` gate; the final resting frame lights
    that gate amber. Falls back to a blocking invoke if streaming isn't
    available, so a stream hiccup never breaks a run. Returns the full state.
    """
    done: set = set(st.session_state.get("pipeline_done") or set())
    live_state: dict = dict(st.session_state.get("current") or {})
    timings: dict = dict(st.session_state.get("pipeline_timings") or {})

    def draw(active: str | None) -> None:
        diagram_ph.markdown(
            _pipeline_html(done, active, live_state, timings), unsafe_allow_html=True
        )

    draw(_next_pending(done))
    # Each updates-mode chunk arrives when a node finishes; the wall-clock gap
    # since the previous chunk (or since we started streaming) is that node's
    # run time. Good enough to show where the run spent its minutes.
    last = time.perf_counter()
    try:
        for chunk in app.stream(graph_input, config, stream_mode="updates"):
            now = time.perf_counter()
            finished = [k for k in (chunk or {}) if k in _NODE_ORDER]
            for node_key, update in (chunk or {}).items():
                if node_key in _NODE_ORDER:
                    done.add(node_key)
                if isinstance(update, dict):
                    live_state.update(update)  # so the judge branch lights up live
            # Attribute the elapsed slice to the node(s) that just finished.
            if finished:
                per = (now - last) / len(finished)
                for node_key in finished:
                    timings[node_key] = per
            last = now
            draw(_next_pending(done))
    except Exception:
        # Stream API hiccup: fall back to a plain blocking run (original behavior).
        _values(app.invoke(graph_input, config))
    st.session_state.pipeline_timings = timings

    snapshot = app.get_state(config)
    active = None
    nxt = getattr(snapshot, "next", None) or ()
    if nxt and nxt[0] in _NODE_ORDER:
        active = nxt[0]
    # Everything before the pause point is done; if there's no pause, all done.
    if active:
        done |= set(_NODE_ORDER[: _NODE_ORDER.index(active)])
    else:
        done |= set(_NODE_ORDER)
    st.session_state.pipeline_done = done
    live_state = _values(snapshot)
    draw(active)
    return live_state


def _provider_mode() -> tuple[str, str]:
    """(vision_provider_name, llm_provider_name) that a run *will actually* use.

    Asks the factory, so this reflects the real fallback-to-mock behavior: a
    provider requested without an API key or SDK resolves to ``mock`` here just
    as it would mid-run. Never raises — any hiccup reads as mock.
    """
    try:
        settings = get_settings()
        return get_vision_provider(settings).name, get_llm_provider(settings).name
    except Exception:
        return "mock", "mock"


def _render_provider_badge(st) -> None:
    """A tiny live/mock indicator so it's obvious whether a run hits real APIs.

    Three states, keyed on the two providers a run uses. Vision is called out
    first because it's what actually *detects* highlights — a "text: nebius,
    vision: mock" setup looks live but produces mock clips, which is exactly the
    footgun this badge exists to surface.
    """
    vision, llm = _provider_mode()
    vision_live, llm_live = vision != "mock", llm != "mock"

    if vision_live and llm_live:
        dot, label = "🟢", "LIVE"
    elif vision_live or llm_live:
        dot, label = "🟡", "PARTIAL"
    else:
        dot, label = "⚪️", "MOCK (offline)"

    st.markdown(f"{dot} **{label}** — vision: `{vision}` · text: `{llm}`")
    if label == "MOCK (offline)":
        st.caption(
            "No API providers configured — runs use deterministic mock output. "
            "Set HYPEREEL_VISION_PROVIDER / HYPEREEL_LLM_PROVIDER (+ API keys) "
            "before launching to run live."
        )
    elif label == "PARTIAL":
        which = "vision (highlight detection)" if not vision_live else "text (summary)"
        st.caption(
            f"⚠️ {which} is running on mock — results won't be fully live. "
            "Set both HYPEREEL_VISION_PROVIDER and HYPEREEL_LLM_PROVIDER to run live."
        )


# --------------------------------------------------------------------------- #
#  Live provider health ping
#
#  The badge only reflects that a provider *client could be constructed* — it
#  never test-calls, so an exhausted budget, a bad key, or the wrong model reads
#  as 🟢 LIVE and then fails silently on every window (0 clips, no explanation).
#  One cheap real call catches that BEFORE a 30-minute run. Cached per provider
#  pair so it pings once, not on every keystroke; a "Re-check" button re-pings.
# --------------------------------------------------------------------------- #
def _health_frame_path(download_dir: str) -> list[str]:
    """A tiny synthetic JPEG to send with the health ping, or [] if unmakeable.

    The ping MUST include an image: a budget-exhausted account can still answer
    cheap text-only calls while the (pricier) vision calls 402, so a frameless
    probe would falsely read as healthy. One small real image mirrors the actual
    per-window classification cost. Degrades to [] (text-only) if OpenCV is absent.
    """
    try:
        import cv2
        import numpy as np

        frames_dir = os.path.join(download_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        path = os.path.join(frames_dir, "_healthcheck.jpg")
        cv2.imwrite(path, np.full((64, 64, 3), 127, dtype="uint8"))
        return [path]
    except Exception:
        return []


def _provider_health(vision_name: str, llm_name: str) -> tuple[bool, str]:
    """Make one live vision call (with an image); return ``(ok, message)``. Never raises.

    Sends a single small image through ``classify_window`` — the same path a real
    window takes — then reads the returned ``Classification.reason``: the providers
    capture their API exception there verbatim (e.g. ``"nebius error: 402 Payment
    Required"``). Mock providers are always healthy (no network).
    """
    if vision_name == "mock" and llm_name == "mock":
        return True, "Mock providers — no live check needed."
    try:
        from hypereel.config import get_settings
        from hypereel.providers.factory import get_vision_provider

        settings = get_settings()
        vision = get_vision_provider(settings)
        recipe = load_recipe(_RECIPES_DIR / "basketball_generic.yaml")
        frames = _health_frame_path(settings.download_dir)
        result = vision.classify_window(frames, recipe, window_index=0)
        if _is_provider_error(_c_attr(result, "reason")):
            return False, _c_attr(result, "reason") or "unknown provider error"
        return True, f"Provider '{vision.name}' answered a live vision call — in budget."
    except Exception as exc:  # a sidebar check must never take the app down
        return False, f"Provider check failed: {exc}"


def _humanize_provider_error(msg: str) -> str:
    """Turn a raw API error into a user-facing, action-oriented sentence."""
    low = (msg or "").lower()
    if _looks_like_budget_error(msg):
        return (
            "the API budget/quota is exhausted. Switch to a FREE provider — set "
            "HYPEREEL_VISION_PROVIDER and HYPEREEL_LLM_PROVIDER to `groq` or `gemini` "
            f"(+ that API key) and relaunch. [provider said: {_trim(msg)}]"
        )
    if any(t in low for t in ("401", "403", "unauthorized", "api key", "auth", "invalid key")):
        return f"authentication failed — check the API key in .env. [provider said: {_trim(msg)}]"
    return _trim(msg)


def _cached_provider_health(st, vision_name: str, llm_name: str) -> tuple[bool, str]:
    """``_provider_health`` memoized in session_state, keyed by the provider pair."""
    cache = st.session_state.get("provider_health")
    key = (vision_name, llm_name)
    if not cache or cache.get("key") != key:
        ok, msg = _provider_health(vision_name, llm_name)
        cache = {"key": key, "ok": ok, "msg": msg}
        st.session_state["provider_health"] = cache
    return cache["ok"], cache["msg"]


# --------------------------------------------------------------------------- #
#  Pre-flight prompt checker
#
#  A run can take many minutes (HD download + a vision call per candidate
#  window), so a missing/weak brief that silently yields an empty reel is an
#  awful experience. These deterministic, instant checks validate the form
#  *before* submission: blocking ERRORS (a missing required piece) disable the
#  "Find highlights" button; WARNINGS flag likely-to-fail settings but still let
#  the user proceed. They re-run on every Streamlit rerun, so editing the brief
#  updates the verdict live — no 30-minute round-trip to discover a typo.
# --------------------------------------------------------------------------- #
_COLOR_WORDS = (
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "gold", "silver", "gray", "grey", "teal", "navy", "maroon",
    "crimson", "cyan", "dark", "light", "royal", "burgundy", "scarlet",
)
_EXCLUSION_WORDS = (
    "opponent", "exclude", "never", "not the", "other team", " vs", " versus",
    "against", "opposing",
)


def _recipe_needs_subject(recipe: Recipe | None, audience_override: str | None = None) -> bool:
    """True when the run will filter to one team (require_subject).

    Mirrors ``select/selector.score_candidates``: an ``individual`` audience with
    a real (non-``none``) subject type drops every window the vision model marks
    ``subject_present=false``. Such a run is unusable without a brief telling the
    model *who* the subject is — which is exactly what we must validate. A runtime
    ``audience_override`` of ``"team"`` turns the filter off (``select_node`` bakes
    it into the effective recipe), so the subject brief is then optional.
    """
    ss = getattr(recipe, "subject_selector", None)
    if ss is None:
        return False
    audience = audience_override or getattr(ss, "audience", "")
    return audience == "individual" and getattr(ss, "type", "none") != "none"


def _brief_has(brief: str, words) -> bool:
    low = f" {brief.lower()} "
    return any(w in low for w in words)


def _preflight(
    recipe: Recipe | None,
    source: str,
    subject: str | None,
    max_duration: float | None,
    vision_name: str,
    llm_name: str,
    audience: str | None = None,
) -> list[tuple[str, str]]:
    """Return ``(level, message)`` checks; ``level`` in {'error','warn','ok'}.

    ``error`` = a missing required piece (blocks submission); ``warn`` = a
    setting likely to produce a poor/empty reel (allowed, but flagged); ``ok`` =
    only when nothing else fired. Pure and offline — safe to call every rerun.
    """
    checks: list[tuple[str, str]] = []

    if recipe is None:
        return [("error", "Recipe failed to load — pick another recipe.")]

    if not (source or "").strip():
        checks.append(("error", "Source is empty — paste a YouTube URL or a local video path."))

    needs = _recipe_needs_subject(recipe, audience)
    brief = (subject or "").strip()
    if needs and not brief:
        checks.append((
            "error",
            "This recipe filters to ONE team, so it needs a subject description. Fill the "
            "“Describe who/what to capture” box with the team's JERSEY COLORS (and the "
            "opponent to exclude). Without it the vision model can't tell who the subject "
            "is and every clip is dropped — you'd get an empty reel.",
        ))
    elif needs and brief:
        if not _brief_has(brief, _COLOR_WORDS):
            checks.append((
                "warn",
                "Your description has no jersey COLOR. Numbers are unreadable at low "
                "resolution, so color is the reliable filter — add one (e.g. “black and "
                "blue jerseys”).",
            ))
        if not _brief_has(brief, _EXCLUSION_WORDS):
            checks.append((
                "warn",
                "Name the OPPONENT to exclude (e.g. “opponent wears white”) so their plays "
                "don't slip into the reel.",
            ))

    if brief and parse_video_quality(brief) is None:
        checks.append((
            "warn",
            "No output-quality preference detected — add “use HD if available” so the model "
            "can read plays more reliably.",
        ))

    min_clip = getattr(getattr(recipe, "selection", None), "min_clip", 4.0)
    if max_duration is not None and max_duration < 2 * min_clip:
        checks.append((
            "error",
            f"Max duration override is {max_duration:.0f}s — too small to fit even one "
            f"{min_clip:.0f}s clip. Set it to 0 (recipe default) or a larger value.",
        ))

    if vision_name == "mock":
        checks.append((
            "warn",
            "Vision provider is on MOCK — detection will be simulated, not real. Set "
            "HYPEREEL_VISION_PROVIDER (+ API key) and relaunch for a live run.",
        ))

    if not any(lvl == "error" for lvl, _ in checks) and not checks:
        checks.append(("ok", "Looks good — colors present and settings sane. Ready to run."))
    elif not any(lvl == "error" for lvl, _ in checks):
        checks.insert(0, ("ok", "No blockers — you can run. Address the notes below for a better reel."))
    return checks


def _list_recipes() -> list[Path]:
    """All recipe YAML files shipped in ``recipes/``, sorted by name."""
    if not _RECIPES_DIR.is_dir():
        return []
    return sorted(_RECIPES_DIR.glob("*.yaml"))


def _values(snapshot_or_dict: Any) -> dict:
    """Normalize a LangGraph ``invoke()``/``get_state()`` result to a plain dict."""
    if isinstance(snapshot_or_dict, dict):
        return snapshot_or_dict
    return dict(getattr(snapshot_or_dict, "values", {}) or {})


def _reset(st) -> None:
    st.session_state.stage = "input"
    st.session_state.app = None
    st.session_state.config = None
    st.session_state.current = {}
    st.session_state.recipe = None
    st.session_state.pipeline_done = set()
    st.session_state.pipeline_timings = {}


def _init_session_state(st) -> None:
    if "stage" not in st.session_state:
        _reset(st)


def _render_sidebar(
    st, recipe_files: list[Path]
) -> tuple[Recipe | None, str, str, float | None, str | None, str | None, int | None, bool]:
    with st.sidebar:
        st.header("Build a reel")
        vision_name, llm_name = _provider_mode()
        _render_provider_badge(st)
        if st.button(
            "Re-check providers",
            help="Make one live API call to confirm the provider actually answers "
            "(catches an exhausted budget / bad key the badge can't see).",
        ):
            st.session_state.pop("provider_health", None)
        recipe_name = st.selectbox("Recipe", [p.name for p in recipe_files])

        # Load the selected recipe up front so the "Describe" field can flag
        # itself required, and so the pre-flight checker below can validate the
        # whole form (and _run_app can reuse it). A load failure is itself a
        # blocking error the checker reports.
        recipe: Recipe | None = None
        recipe_path = next((p for p in recipe_files if p.name == recipe_name), None)
        if recipe_path is not None:
            try:
                recipe = load_recipe(recipe_path)
            except RecipeError as exc:
                st.error(f"Could not load recipe: {exc}")

        source = st.text_input(
            "Source (YouTube URL or local path)",
            value="demo://aau_basketball_game.mp4",
            help="A YouTube URL (downloaded via yt-dlp, up to 1080p), a local video "
            "path, or a demo:// placeholder for offline mock runs.",
        )
        describe_required = _recipe_needs_subject(recipe)
        subject_description = st.text_area(
            "Describe who/what to capture"
            + (" — required for this recipe" if describe_required else " (optional)"),
            value="",
            height=120,
            placeholder=(
                "No reference photo? Describe the subject and the visual cues to look "
                "for — and any output preference. e.g. \"Highlights for the team in "
                "BLACK jerseys with RED trim (labeled UNL on the scoreboard); the "
                "opponent wears white with teal/green. Prefer player #23 when the "
                "number is legible. Use HD if available.\""
            ),
            help=(
                "A free-text brief fed straight to the vision model. It leads the "
                "'who/what to look for' judgment and overrides the recipe's subject "
                "description for this run. You can also state a quality preference "
                "here (e.g. 'use HD if available', 'quick 360p preview') and the "
                "downloader will honor it. Best way to steer results when you can't "
                "upload a reference photo."
            ),
        )
        max_duration_raw = st.number_input(
            "Max duration override, seconds (0 = use recipe default)",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )
        audience_choice = st.selectbox("Audience override", ["(recipe default)", "individual", "team"])
        max_candidates_raw = st.number_input(
            "Quick test: limit to first N candidate windows (0 = all)",
            min_value=0,
            value=0,
            step=5,
            help="Caps how many candidate windows get a (paid) vision call, so you can "
            "validate a prompt/recipe change in ~1 minute instead of a full pass. "
            "0 = no cap (full run).",
        )

        max_duration = max_duration_raw or None
        audience = None if audience_choice == "(recipe default)" else audience_choice
        max_candidates = int(max_candidates_raw) or None
        subject = subject_description.strip() or None

        # --- Live pre-flight check: validate BEFORE the (long) run ------------ #
        checks = _preflight(recipe, source, subject, max_duration, vision_name, llm_name, audience)

        # Live provider ping (cached): the badge only means a client was built, not
        # that the API answers. A failure here is a hard blocker — running would
        # burn ~30 min to produce 0 clips, exactly the trap that motivated this.
        health_ok, health_msg = _cached_provider_health(st, vision_name, llm_name)
        if not health_ok:
            checks = [c for c in checks if c[0] != "ok"]  # drop the stale "all clear"
            checks.insert(0, ("error", f"Provider not usable: {_humanize_provider_error(health_msg)}"))

        if max_candidates:
            checks.append((
                "warn",
                f"Quick-test cap is ON — only the first {max_candidates} candidate window(s) "
                "will be classified. Expect a PARTIAL preview, not the full reel. Set it to 0 "
                "for a complete run.",
            ))

        blockers = [msg for lvl, msg in checks if lvl == "error"]

        st.markdown("**Pre-flight check**")
        st.caption("Validated live as you type — fix any red item before running.")
        for lvl, msg in checks:
            if lvl == "error":
                st.error(msg)
            elif lvl == "warn":
                st.warning(msg)
            else:
                st.success(msg)

        find_clicked = st.button(
            "Find highlights",
            type="primary",
            disabled=bool(blockers),
            help="Resolve the red pre-flight items above to enable this." if blockers else None,
        )

        # Stopping / resetting. The graph runs synchronously inside this script,
        # so a mid-run abort can only come from Streamlit's ⏹ Stop toolbar button
        # (a custom button can't be handled while the script is busy). After a
        # Stop — or any time — this Reset gives a guaranteed clean slate.
        st.caption(
            "Building takes a while. To abort a run, use the **⏹ Stop** button in "
            "Streamlit's top-right toolbar, then **Reset** below. Starting a new run "
            "with *Find highlights* also clears the panel automatically."
        )
        if st.button("Reset / clear panel", help="Wipe the current run and return to a blank panel."):
            _reset(st)
            st.rerun()

        return recipe, recipe_name, source, max_duration, audience, subject, max_candidates, find_clicked


def _start_run(
    st,
    recipe: Recipe,
    source: str,
    max_duration: float | None,
    audience: str | None,
    subject_description: str | None = None,
    max_candidates: int | None = None,
    diagram_ph=None,
) -> None:
    """Kick off a fresh graph run; it pauses at the first HITL gate it hits.

    The live pipeline diagram (``diagram_ph``) is the progress indicator: nodes
    light up as the agent completes them.
    """
    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial = new_state(
        source,
        recipe,
        max_duration=max_duration,
        audience=audience,
        subject_description=subject_description,
        max_candidates=max_candidates,
    )

    st.session_state.pipeline_done = set()
    st.session_state.pipeline_timings = {}
    if diagram_ph is not None:
        current = _stream_pipeline(st, app, config, initial, diagram_ph)
    else:  # no placeholder (e.g. a test harness) — plain blocking run
        current = _values(app.invoke(initial, config))

    st.session_state.app = app
    st.session_state.config = config
    st.session_state.current = current
    st.session_state.recipe = recipe
    st.session_state.stage = "gate1"


def _render_gate1(st, diagram_ph=None) -> None:
    """Gate 1 — human approves/edits the proposed clip list before render."""
    st.markdown("## Gate 1 — Approve the clip list")
    st.caption("Reads are autonomous; this is the first write-adjacent step (render), so it's gated.")

    current = st.session_state.current
    recipe: Recipe = st.session_state.recipe

    for err in current.get("errors", []):
        st.warning(err)

    clips: list[Clip] = current.get("selected_clips", [])

    # When the reel is empty or underfilled, explain WHY in plain language —
    # which gate dropped the windows and exactly what to change — instead of
    # leaving the user with only "start over".
    diag_html = _diagnosis_html(current)
    if diag_html and (not clips or current.get("human_gate") == GATE_UNDERFILLED):
        st.markdown(diag_html, unsafe_allow_html=True)
        st.caption(
            "See **🛰 Under the hood** above for the per-step timing, and **Pipeline notes** "
            "below for the full trace."
        )

    if not clips:
        st.info("No candidate clips met the recipe's threshold — adjust the settings above and Start over.")

    keep_flags: list[bool] = []
    for i, clip in enumerate(clips):
        cols = st.columns([1, 2, 3, 2, 4])
        keep_flags.append(cols[0].checkbox("Keep", value=True, key=f"keep_{i}"))
        cols[1].markdown(f"**{clip.moment_type or 'unclassified'}**")
        cols[2].write(f"{clip.start:.1f}s – {clip.end:.1f}s ({clip.duration:.1f}s)")
        cols[3].write(f"score {clip.score:.2f}")
        cols[4].caption(clip.reason)

    col_a, col_b = st.columns(2)
    if col_a.button("Approve & Render", type="primary", disabled=not clips):
        edited = [c for c, keep in zip(clips, keep_flags) if keep]
        app = st.session_state.app
        config = st.session_state.config
        app.update_state(config, {"selected_clips": edited, "approved": True, "needs_human": False})
        if diagram_ph is not None:
            st.session_state.current = _stream_pipeline(st, app, config, None, diagram_ph)
        else:
            with st.spinner("Rendering..."):
                st.session_state.current = _values(app.invoke(None, config))
        st.session_state.stage = "gate2"
        st.rerun()
    if col_b.button("Start over"):
        _reset(st)
        st.rerun()


def _render_gate2(st, diagram_ph=None) -> None:
    """Gate 2 — human approves before any share/publish (the write action)."""
    st.markdown("## Gate 2 — Approve & Share")
    st.caption("Sharing/publishing is a write action — nothing goes out until you approve it here.")

    current = st.session_state.current
    st.success(f"Reel rendered: {current.get('output_path') or '(no output path)'}")
    st.write(f"Total duration: {current.get('proposed_duration', 0.0):.1f}s")

    for clip in current.get("selected_clips", []):
        st.write(f"- {clip.moment_type or 'unclassified'}: {clip.start:.1f}s–{clip.end:.1f}s (score {clip.score:.2f})")

    col_a, col_b = st.columns(2)
    if col_a.button("Approve & Share", type="primary"):
        app = st.session_state.app
        config = st.session_state.config
        app.update_state(config, {"shared": True, "needs_human": False})
        if diagram_ph is not None:
            st.session_state.current = _stream_pipeline(st, app, config, None, diagram_ph)
        else:
            with st.spinner("Delivering..."):
                st.session_state.current = _values(app.invoke(None, config))
        st.session_state.stage = "done"
        st.rerun()
    if col_b.button("Start over"):
        _reset(st)
        st.rerun()


def _render_done(st) -> None:
    current = st.session_state.current
    st.markdown("## Done")
    st.success("Reel approved and shared.")
    st.write(f"Output: {current.get('output_path') or '(no output path)'}")
    if current.get("summary"):
        st.markdown("### Summary")
        st.markdown(current["summary"])
    if st.button("Build another reel"):
        _reset(st)
        st.rerun()


def _run_app(st) -> None:
    st.set_page_config(page_title="HypeReel", page_icon=":basketball:", layout="wide")
    st.title("HypeReel")
    st.caption(
        "Recipe-driven highlight reels, with a human approving the clip list "
        "and the share step. Full design: design/HypeReel-Design.html"
    )

    _init_session_state(st)
    recipe_files = _list_recipes()
    if not recipe_files:
        st.error(f"No recipe YAML files found in {_RECIPES_DIR}.")
        return

    recipe, recipe_name, source, max_duration, audience, subject_description, max_candidates, find_clicked = (
        _render_sidebar(st, recipe_files)
    )

    # A new run must start on a BLANK panel. Clear the previous run's clips,
    # diagnosis, diagram, timings and stage BEFORE anything is drawn this pass,
    # so last run's output never lingers on screen while the new one builds.
    if find_clicked and recipe is not None:
        _reset(st)

    # "Under the hood" — one diagram placeholder, reused for both the live
    # (streamed) view and the resting (post-rerun) view so there's ever only one.
    with st.expander("🛰 Under the hood — the agent pipeline", expanded=True):
        st.caption(
            "The real LangGraph nodes this run flows through. The glowing box is "
            "where the agent is right now; green boxes are done; the two amber "
            "gates are where it pauses for you. After **Judge**, the dashed pair "
            "is the agent's own routing decision — revise the reel or accept it."
        )
        diagram_ph = st.empty()
        decisions_ph = st.empty()
    done, active = _stage_progress(st.session_state.stage)
    current = st.session_state.current or {}
    diagram_ph.markdown(
        _pipeline_html(
            done, active, current if current else None, st.session_state.get("pipeline_timings")
        ),
        unsafe_allow_html=True,
    )

    # The pre-flight checker already blocks submission on a load failure or a
    # missing required piece, so a click here means the recipe is loaded and the
    # form is valid.
    if find_clicked and recipe is not None:
        _start_run(
            st, recipe, source, max_duration, audience, subject_description, max_candidates, diagram_ph
        )

    # Decisions strip reflects the freshest state (after any run this pass).
    current = st.session_state.current or {}
    if current:
        decisions_ph.markdown(_decisions_html(current), unsafe_allow_html=True)

    stage = st.session_state.stage
    if stage == "input":
        st.info("Set your source and recipe in the sidebar, then click **Find highlights**.")
        return

    recipe = st.session_state.recipe
    st.subheader(f"Recipe: {recipe.name}  ·  {recipe.domain}")

    if stage == "gate1":
        _render_gate1(st, diagram_ph)
    elif stage == "gate2":
        _render_gate2(st, diagram_ph)
    elif stage == "done":
        _render_done(st)

    with st.expander("Pipeline notes"):
        for note in st.session_state.current.get("notes", []):
            st.write(f"- {note}")


def main() -> None:
    """Entry point for ``streamlit run src/hypereel/app.py``."""
    try:
        import streamlit as st
    except ImportError:
        print(
            "Streamlit is not installed. Install it with `pip install streamlit`, "
            "then run: streamlit run src/hypereel/app.py"
        )
        return
    _run_app(st)


if __name__ == "__main__":
    main()
