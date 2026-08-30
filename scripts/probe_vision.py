"""One-window live probe: is Nebius actually returning usable verdicts?

Extracts a few frames from the real video and calls the vision provider ONCE
plus a trivial text generate(), printing exactly what comes back (incl. the
error `reason` if the API is failing). No 30-min run required.
"""
from __future__ import annotations

import sys

from hypereel.config import get_settings
from hypereel.recipe import load_recipe
from hypereel.models import CandidateWindow
from hypereel.analyze.classifier import extract_frames
from hypereel.providers.factory import get_vision_provider, get_llm_provider

VIDEO = "downloads/QDhlscjhIVQ.mp4"
BRIEF = (
    "Highlights for team EBE — players in BLACK and BLUE jerseys ('EBE' on the "
    "scoreboard). The opponent wears yellow. Prefer the black/blue jersey as the "
    "reliable filter."
)

settings = get_settings()
print(f"vision provider env resolved -> {get_vision_provider(settings).name}")
print(f"llm    provider env resolved -> {get_llm_provider(settings).name}")
print(f"model = {settings.nebius_model}  base = {settings.nebius_base_url}")
print(f"frames_per_candidate = {settings.frames_per_candidate}")

recipe = load_recipe("recipes/basketball_generic.yaml")
recipe.subject_selector.description = BRIEF

vision = get_vision_provider(settings)
llm = get_llm_provider(settings)

# Sample three windows spread across the game (start / middle / late).
for label, start in [("early", 300.0), ("mid", 1200.0), ("late", 2000.0)]:
    win = CandidateWindow(start=start, end=start + 6.0, signal_scores={'audio_peak':0.7})
    frames = extract_frames(VIDEO, win, settings.frames_per_candidate, "downloads/frames")
    print(f"\n[{label} @ {start:.0f}s] extracted {len(frames)} frame(s)")
    c = vision.classify_window(frames, recipe, window_index=0)
    print(
        f"  subject_present={c.subject_present}  moment_type={c.moment_type!r}  "
        f"confidence={c.confidence}"
    )
    print(f"  reason={c.reason!r}")

print("\n[text LLM] generate('Reply with the single word OK.') ->")
out = llm.generate("Reply with the single word OK.", max_tokens=10)
print(f"  {out!r}")
sys.stdout.flush()
