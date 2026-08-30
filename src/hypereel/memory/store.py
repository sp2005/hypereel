"""Tiny JSON-backed persistent memory: user profile + learned clip preferences.

Deliberately pure stdlib (json/pathlib/os) -- this is process-local state that
must survive restarts without pulling in a database or any heavy dependency.
Every method degrades to an empty/default result rather than raising, since a
missing or corrupt memory file should never crash the pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..models import Clip


class MemoryStore:
    """Reads/writes a single JSON file holding the user profile + feedback counts.

    File shape::

        {
          "profile": {...arbitrary user prefs...},
          "feedback": {
            "<recipe_id>": {
              "kept": {"<moment_type>": <count>, ...},
              "dropped": {"<moment_type>": <count>, ...}
            }
          }
        }
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)  # created lazily on first save/record, not here

    def _read(self) -> dict[str, Any]:
        """Load the whole file, returning a fresh default shape on any problem."""
        default: dict[str, Any] = {"profile": {}, "feedback": {}}
        try:
            if not self.path.exists():
                return default
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict):
                return default
            raw.setdefault("profile", {})
            raw.setdefault("feedback", {})
            return raw
        except Exception:
            return default

    def _write(self, data: dict[str, Any]) -> None:
        """Write temp-file-then-replace so a crash mid-write never leaves a
        truncated/corrupt memory file behind. Persistence is best-effort."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, self.path)
        except Exception:
            pass  # never raise on persistence failure

    def load_profile(self) -> dict:
        """Return the saved user profile, or {} if missing/corrupt."""
        return self._read().get("profile", {}) or {}

    def save_profile(self, profile: dict) -> None:
        """Persist the user profile, leaving feedback history untouched."""
        data = self._read()
        data["profile"] = profile
        self._write(data)

    def record_feedback(self, recipe_id: str, kept: list[Clip], dropped: list[Clip]) -> None:
        """Accumulate kept/dropped moment_type counts for this recipe.

        Called after a human reviews the selected clips; this is the raw
        signal ``learned_preferences`` later turns into preferred/disliked
        moment types.
        """
        data = self._read()
        bucket = data["feedback"].setdefault(recipe_id, {"kept": {}, "dropped": {}})
        for clip in kept:
            mt = clip.moment_type or "unknown"
            bucket["kept"][mt] = bucket["kept"].get(mt, 0) + 1
        for clip in dropped:
            mt = clip.moment_type or "unknown"
            bucket["dropped"][mt] = bucket["dropped"].get(mt, 0) + 1
        self._write(data)

    def learned_preferences(self, recipe_id: str) -> dict:
        """Derive preferred/disliked moment types from accumulated feedback counts.

        A moment type is "preferred" once it's been kept more often than
        dropped; "disliked" once it's been dropped more often than kept.
        Ties and unseen types are omitted. Returns {} if there's no signal.
        """
        bucket = self._read()["feedback"].get(recipe_id)
        if not bucket:
            return {}
        kept, dropped = bucket.get("kept", {}), bucket.get("dropped", {})
        preferred, disliked = [], []
        for mt in set(kept) | set(dropped):
            k, d = kept.get(mt, 0), dropped.get(mt, 0)
            if k > d:
                preferred.append(mt)
            elif d > k:
                disliked.append(mt)
        if not preferred and not disliked:
            return {}
        return {
            "preferred_moment_types": sorted(preferred),
            "disliked_moment_types": sorted(disliked),
        }
