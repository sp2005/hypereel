"""Pydantic data models for HypeReel.

These are the shared contracts used across every module: the *recipe* schema
(the agent's declarative policy) and the *runtime* objects that flow through the
LangGraph pipeline (candidate windows, classifications, selected clips).

Keep this module dependency-light (pydantic + stdlib only) so it imports
everywhere without pulling in heavy media/model libraries.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- #
#  Recipe schema  (the declarative "what is a highlight" policy)
# --------------------------------------------------------------------------- #

HighlightModel = Literal["event_based", "quality_based", "hybrid"]
SignalRole = Literal["proposer", "scorer"]
SubjectType = Literal["jersey_number", "team_color", "face_photo", "feature_class", "none"]
Audience = Literal["individual", "team"]
Ordering = Literal["chronological", "best_first", "narrative_arc"]


class MomentType(BaseModel):
    """One class of highlight the agent looks for.

    ``description`` is the natural-language rubric the vision model uses to judge
    a candidate — it is the LLM's grading criteria, not just documentation.
    """

    name: str
    description: str
    weight: float = 1.0
    ideal_len: float = 8.0  # seconds


class RubricDimension(BaseModel):
    """A scoring axis used by ``quality_based`` recipes (e.g. lighting)."""

    dim: str
    weight: float = 1.0


class SignalConfig(BaseModel):
    """A detection signal and how it participates in the pipeline.

    ``role='proposer'`` signals propose candidate windows (cheap, local).
    ``role='scorer'`` signals rank/confirm candidates (e.g. the vision model).
    ``enabled_if`` gates a signal on a runtime condition string the planner
    evaluates (e.g. ``"commentary_present"``).
    """

    type: str
    role: SignalRole = "proposer"
    weight: float = 1.0
    enabled: bool = True
    enabled_if: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class SubjectSelector(BaseModel):
    """Who/what to center the reel on (the recipe's 'variables')."""

    type: SubjectType = "none"
    value: Any = None          # e.g. 23, or ["kitchen", "staircase"]
    team_color: Optional[str] = None
    audience: Audience = "individual"
    # Free-text brief of who/what to capture and the visual cues to look for
    # (jersey colors, a number, the scoreboard label for the team, etc.). This
    # is the natural-language description a user gives when there's no reference
    # photo to upload; it's fed to the vision model verbatim and leads the
    # "who/what to look for" section of the classification prompt. May be set in
    # the recipe YAML or supplied at runtime from the UI/CLI.
    description: Optional[str] = None


class SelectionPolicy(BaseModel):
    """How to choose and assemble clips to fit the time budget."""

    max_duration: float = 300.0   # seconds — the reel budget
    min_clip: float = 4.0
    max_clip: float = 12.0
    lead_in: float = 2.0
    lead_out: float = 3.0
    min_score: float = 0.5
    ordering: Ordering = "chronological"
    coverage: Optional[str] = None      # NL hint, e.g. "spread across 4 quarters"
    dedup_overlap: bool = True


class StyleConfig(BaseModel):
    """Stylistic / output preferences (the 'feel' and the delivery format)."""

    transitions: str = "hard_cut"       # hard_cut | crossfade | slow_crossfade
    audio: str = "keep_original"        # keep_original | replace_with_music | duck
    overlays: list[str] = Field(default_factory=list)
    format: str = "16:9"                # 16:9 | 9:16 | 1:1
    platform: str = "local"             # local | youtube | ...


class Guardrails(BaseModel):
    """Hard limits — maps to the framework's 'what should it never do'."""

    never_include: list[str] = Field(default_factory=list)
    privacy: Optional[str] = None
    grounding: Optional[str] = None


class Recipe(BaseModel):
    """The complete declarative policy the agent interprets.

    The same schema expresses very different domains; only ``highlight_model``
    (and ``scoring_rubric`` for quality domains) changes the strategy family.
    """

    id: str
    name: str
    domain: str
    highlight_model: HighlightModel = "event_based"
    description: str = ""

    moment_types: list[MomentType] = Field(default_factory=list)
    scoring_rubric: Optional[list[RubricDimension]] = None
    signals: list[SignalConfig] = Field(default_factory=list)
    subject_selector: SubjectSelector = Field(default_factory=SubjectSelector)
    selection: SelectionPolicy = Field(default_factory=SelectionPolicy)
    style: StyleConfig = Field(default_factory=StyleConfig)
    guardrails: Guardrails = Field(default_factory=Guardrails)
    acceptance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_quality_rubric(self) -> "Recipe":
        if self.highlight_model == "quality_based" and not self.scoring_rubric:
            raise ValueError(
                "quality_based recipes must define a `scoring_rubric` "
                "(the dimensions the vision model scores windows on)."
            )
        if not self.moment_types:
            raise ValueError("recipe must define at least one moment_type.")
        return self

    def proposer_signals(self) -> list[SignalConfig]:
        return [s for s in self.signals if s.enabled and s.role == "proposer"]

    def scorer_signals(self) -> list[SignalConfig]:
        return [s for s in self.signals if s.enabled and s.role == "scorer"]


# --------------------------------------------------------------------------- #
#  Runtime objects  (flow through the graph)
# --------------------------------------------------------------------------- #


class CandidateWindow(BaseModel):
    """A proposed time window that *might* contain a highlight."""

    start: float
    end: float
    signal_scores: dict[str, float] = Field(default_factory=dict)  # signal_type -> score

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Classification(BaseModel):
    """A vision model's verdict on one candidate window."""

    moment_type: Optional[str] = None    # matched MomentType.name, or None
    subject_present: bool = False        # is the recipe's subject in frame?
    confidence: float = 0.0              # 0..1
    reason: str = ""                     # short grounded explanation


class Clip(BaseModel):
    """A selected, budget-fitted clip destined for the reel."""

    start: float
    end: float
    moment_type: Optional[str] = None
    score: float = 0.0
    reason: str = ""
    subject_present: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ReelResult(BaseModel):
    """Final output metadata returned to the UI."""

    output_path: Optional[str] = None
    clips: list[Clip] = Field(default_factory=list)
    total_duration: float = 0.0
    recipe_id: str = ""
    audience: str = "individual"
    summary: str = ""            # natural-language recap from the LLM provider
    notes: list[str] = Field(default_factory=list)
