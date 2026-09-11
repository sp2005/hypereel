"""The LangGraph state object.

A single ``TypedDict`` carries everything through the pipeline. It is plain
``typing`` (no langgraph import) so it can be used and tested standalone.

Field groups:
  * inputs        — what the user asked for
  * ingest        — resolved local video + metadata
  * detection     — proposed candidates and their classifications
  * selection     — chosen clips + render output
  * control/HITL  — flags the conditional edges and human gates read/write
  * bookkeeping   — errors, notes, provider mode
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from ..models import CandidateWindow, Classification, Clip, Recipe


class ReelState(TypedDict, total=False):
    """The single session-state object threaded through every graph node.

    This is HypeReel's working memory for one run: each node reads the fields it
    needs and returns a partial update that LangGraph merges back in, and the
    checkpointer persists the whole thing so a run can pause at a human gate and
    resume later. ``total=False`` lets nodes contribute fields incrementally as
    the pipeline progresses. Fields are grouped by pipeline stage below;
    persistent, cross-run memory lives separately in :mod:`hypereel.memory`.
    """

    # ---- inputs ----
    source: str                 # YouTube URL or local path
    recipe: Recipe
    max_duration: Optional[float]   # user override of recipe.selection.max_duration
    audience: Optional[str]         # user override: "individual" | "team"
    max_candidates: Optional[int]   # quick-test cap: classify only the first N windows (None = all)
    candidate_sampling: str
    evaluation_start_seconds: Optional[float]
    evaluation_end_seconds: Optional[float]

    # ---- ingest ----
    video_path: Optional[str]
    video_duration: float
    has_commentary: bool            # detected -> gates transcript signal

    # ---- detection ----
    strategy: str                   # planner's chosen strategy family
    active_signals: list[str]       # signal types the planner enabled
    candidates: list[CandidateWindow]
    uncapped_candidates: list[CandidateWindow]  # proposer output before quick-test truncation
    classifications: list[Classification]   # aligned 1:1 with candidates

    # ---- selection / render ----
    scored_clips: list[Clip]        # all candidates that passed threshold, scored
    selected_clips: list[Clip]      # budget-fitted final set
    proposed_duration: float
    output_path: Optional[str]
    summary: str

    # ---- quality judge (LLM-as-judge critic + bounded revision loop) ----
    judge_verdict: dict[str, Any]       # last critic verdict (score/issues/feedback)
    judge_decision: str                 # "" | "revise" | "accept" (router reads this)
    revision_count: int                 # how many times the judge has forced a re-select
    selection_overrides: dict[str, Any] # knobs the judge writes for select_node to honor

    # ---- control flow / human-in-the-loop ----
    needs_human: bool
    human_gate: str                 # "" | "approve_clips" | "approve_share" | "underfilled"
    human_decision: dict[str, Any]  # payload the UI writes back
    approved: bool
    shared: bool

    # ---- bookkeeping ----
    errors: list[str]
    notes: list[str]
    mode: str                       # "mock" | "live"


# Human-gate identifiers (kept as constants so UI + graph agree).
GATE_APPROVE_CLIPS = "approve_clips"
GATE_APPROVE_SHARE = "approve_share"
GATE_UNDERFILLED = "underfilled"


def new_state(source: str, recipe: Recipe, **overrides: Any) -> ReelState:
    """Build a fresh state with safe defaults.

    A non-empty ``subject_description`` override is baked into a copy of the
    recipe's ``subject_selector`` here, up front, so the entire run — most
    importantly the vision classification prompt — uses the user's free-text
    brief of who/what to capture. This is the runtime equivalent of editing the
    recipe's ``subject_selector.description`` and is how the UI/CLI let a user
    describe the subject (and its visual cues) when there's no reference photo.
    """
    subject_description = overrides.get("subject_description")
    if subject_description and subject_description.strip():
        recipe = recipe.model_copy(deep=True)
        recipe.subject_selector.description = subject_description.strip()

    state: ReelState = {
        "source": source,
        "recipe": recipe,
        "max_duration": overrides.get("max_duration"),
        "audience": overrides.get("audience"),
        "max_candidates": overrides.get("max_candidates"),
        "candidate_sampling": overrides.get("candidate_sampling", "chronological"),
        "evaluation_start_seconds": overrides.get("evaluation_start_seconds"),
        "evaluation_end_seconds": overrides.get("evaluation_end_seconds"),
        "video_path": None,
        "video_duration": 0.0,
        "has_commentary": False,
        "strategy": "",
        "active_signals": [],
        "candidates": [],
        "uncapped_candidates": [],
        "classifications": [],
        "scored_clips": [],
        "selected_clips": [],
        "proposed_duration": 0.0,
        "output_path": None,
        "summary": "",
        "judge_verdict": {},
        "judge_decision": "",
        "revision_count": 0,
        "selection_overrides": {},
        "needs_human": False,
        "human_gate": "",
        "human_decision": {},
        "approved": False,
        "shared": False,
        "errors": [],
        "notes": [],
        "mode": overrides.get("mode", "mock"),
    }
    return state
