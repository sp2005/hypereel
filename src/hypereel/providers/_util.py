"""Shared helpers for the live vision/LLM providers (gemini/groq/nebius).

Kept dependency-light (stdlib only) so importing it never requires a model
SDK. Providers import from here to avoid duplicating prompt-building,
base64-encoding, and tolerant-JSON-parsing logic.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Sequence

from ..models import Classification, Recipe


def build_classification_prompt(recipe: Recipe) -> str:
    """Build the rubric + subject prompt shared by every vision provider."""
    rubric_lines = "\n".join(
        f"- {m.name}: {m.description}" for m in recipe.moment_types
    ) or "- highlight: any notable moment"

    subject = recipe.subject_selector
    # A free-text brief (from the recipe or supplied at runtime) is the richest
    # "who/what to look for" signal — it leads, and any structured selector
    # fields are appended as supporting cues. This is what a user types when
    # there's no reference photo: e.g. "the team in black jerseys with red trim,
    # labeled UNL on the scoreboard; player #23 if you can make out the number."
    brief = (subject.description or "").strip()
    structured = []
    if subject.type != "none":
        structured.append(f"type={subject.type}")
        if subject.value is not None:
            structured.append(f"value={subject.value}")
        if subject.team_color:
            structured.append(f"team_color={subject.team_color}")

    if brief:
        subject_desc = (
            f"Who/what to look for: {brief}\n"
            "Set subject_present=true only when the subject described above is the "
            "one making the play in the foreground game; false otherwise."
        )
        if structured:
            subject_desc += "\nAdditional structured cues: " + ", ".join(structured)
    elif structured:
        subject_desc = "Who/what to look for: " + ", ".join(structured)
    else:
        subject_desc = "No specific subject to track; judge the general action."

    # Reject non-play footage. This is the direct fix for reels that captured
    # timeouts, huddles, and kids shooting on the side/adjacent baskets of a
    # multi-court venue: the model must return moment_type=null for anything
    # that isn't a live play in the primary foreground game.
    reject_lines = [
        "Judge ONLY the primary game in the foreground — the court with the "
        "referees and the two teams playing in formation. IGNORE any action on "
        "adjacent or background courts and anyone shooting on a side basket.",
        "Return moment_type=null (low confidence) if the frames show a timeout, "
        "huddle, or players standing around during a dead ball; warm-ups or "
        "shoot-around; substitutions or walking to the bench; or a camera "
        "pan/zoom with motion blur and no clear play.",
        "When you are not confident a real play from the rubric is happening, "
        "return null rather than guessing — a padded reel is worse than a short one.",
    ]
    guard = recipe.guardrails
    if guard.never_include:
        reject_lines.append("Never include: " + ", ".join(guard.never_include) + ".")
    if guard.grounding:
        reject_lines.append(f"Grounding rule: {guard.grounding}")
    reject_block = "\n".join(f"- {line}" for line in reject_lines)

    return (
        "You are judging a short video clip (sampled frames shown in order) for a "
        "highlight-reel agent.\n\n"
        "Moment types (the rubric):\n"
        f"{rubric_lines}\n\n"
        f"{subject_desc}\n\n"
        "Rejection rules (apply these first):\n"
        f"{reject_block}\n\n"
        "Look at the frames and decide whether this window contains one of the "
        "moment types above and whether the subject is visible.\n\n"
        "Respond with STRICT JSON only, no prose, no markdown fences, matching "
        "exactly this shape:\n"
        '{"moment_type": <string or null>, "subject_present": <true|false>, '
        '"confidence": <number 0..1>, "reason": <short string>}'
    )


def encode_frames_b64(frame_paths: Sequence[str]) -> list[bytes]:
    """Read frame files and return raw bytes for each (skips unreadable files)."""
    out: list[bytes] = []
    for path in frame_paths:
        try:
            with open(path, "rb") as f:
                out.append(f.read())
        except OSError:
            continue
    return out


def frame_to_data_uri(raw: bytes, mime: str = "image/jpeg") -> str:
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_classification_json(text: str, recipe: Recipe) -> Classification:
    """Tolerantly parse a model's JSON response into a Classification.

    Strips ```json fences, finds the first {...} block, and falls back to a
    low-confidence Classification on any parse failure. Never raises.
    """
    try:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip())
        cleaned = re.sub(r"```$", "", cleaned.strip())
        match = _JSON_OBJECT_RE.search(cleaned)
        if not match:
            return Classification(
                moment_type=None,
                subject_present=False,
                confidence=0.0,
                reason="no JSON object found in model response",
            )
        data = json.loads(match.group(0))

        moment_type = data.get("moment_type")
        if moment_type is not None:
            moment_type = str(moment_type)
            valid_names = {m.name for m in recipe.moment_types}
            if valid_names and moment_type not in valid_names:
                # Model hallucinated a moment type outside the rubric.
                moment_type = None

        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return Classification(
            moment_type=moment_type,
            subject_present=bool(data.get("subject_present", False)),
            confidence=confidence,
            reason=str(data.get("reason", ""))[:500],
        )
    except Exception as exc:  # never raise from a parser
        return Classification(
            moment_type=None,
            subject_present=False,
            confidence=0.0,
            reason=f"parse error: {exc}",
        )
