"""Versioned selection-replay inputs. Paths resolve relative to the dataset."""
from __future__ import annotations

from math import isfinite
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

from ..models import CandidateWindow, Classification


class ReferenceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    event_id: str
    event_time: float = Field(ge=0)
    action_start: float = Field(ge=0)
    action_end: float = Field(gt=0)
    moment_type: str

    @model_validator(mode="after")
    def valid_interval(self):
        if not self.action_start <= self.event_time <= self.action_end:
            raise ValueError("event_time must lie inside the action interval")
        if self.action_end <= self.action_start:
            raise ValueError("action interval must have positive duration")
        return self


class SelectionCase(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    case_id: str = Field(min_length=1)
    synthetic: bool
    recipe_path: str
    video_duration: float = Field(gt=0)
    max_duration: float | None = Field(default=None, ge=0)
    audience: Literal["individual", "team"] | None = None
    candidates: list[CandidateWindow]
    classifications: list[Classification]
    # None means unannotated; [] with exhaustive=True means no eligible events.
    reference_events: list[ReferenceEvent] | None = None
    exhaustive: bool = False

    @model_validator(mode="after")
    def validate_replay(self):
        if len(self.candidates) != len(self.classifications):
            raise ValueError("candidates and classifications must have equal lengths")
        for window in self.candidates:
            if not (isfinite(window.start) and isfinite(window.end)
                    and 0 <= window.start < window.end <= self.video_duration):
                raise ValueError("candidate intervals must be finite and within the source")
            if any(not isfinite(s) or not 0 <= s <= 1 for s in window.signal_scores.values()):
                raise ValueError("signal scores must be finite and in [0, 1]")
        if any(not isfinite(c.confidence) or not 0 <= c.confidence <= 1
               for c in self.classifications):
            raise ValueError("confidence must be finite and in [0, 1]")
        events = self.reference_events or []
        if len({e.event_id for e in events}) != len(events):
            raise ValueError("reference event IDs must be unique")
        if any(e.action_end > self.video_duration for e in events):
            raise ValueError("reference events must be within the source")
        if self.exhaustive and self.reference_events is None:
            raise ValueError("exhaustive annotations require reference_events")
        return self


class PipelineCase(BaseModel):
    """A source-driven case; predictions come from the existing graph."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    case_id: str = Field(min_length=1)
    synthetic: bool
    recipe_path: str
    source: str = Field(min_length=1)
    max_duration: float | None = Field(default=None, gt=0)
    audience: Literal["individual", "team"] | None = None
    subject_description: str | None = None
    max_candidates: int | None = Field(default=None, gt=0)
    candidate_sampling: Literal[
        "chronological", "spread", "reference_stratified"
    ] = "chronological"
    evaluation_start_seconds: float | None = Field(default=None, ge=0)
    evaluation_end_seconds: float | None = Field(default=None, gt=0)
    reference_events: list[ReferenceEvent] | None = None
    exhaustive: bool = False

    @model_validator(mode="after")
    def validate_references(self):
        events = self.reference_events or []
        if len({e.event_id for e in events}) != len(events):
            raise ValueError("reference event IDs must be unique")
        if self.exhaustive and self.reference_events is None:
            raise ValueError("exhaustive annotations require reference_events")
        if self.source.startswith("demo://") and not self.synthetic:
            raise ValueError("demo sources must be marked synthetic")
        if (self.evaluation_start_seconds is not None
                and self.evaluation_end_seconds is not None
                and self.evaluation_start_seconds >= self.evaluation_end_seconds):
            raise ValueError("evaluation_start_seconds must be before evaluation_end_seconds")
        return self
