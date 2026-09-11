"""Evaluation adapter around the unmodified graph, stopping before rendering."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from math import isfinite

from ..config import Settings
from ..graph.build import build_graph
from ..graph.state import new_state
from ..observability import with_tracing
from .metrics import selection_metrics


def evaluate_pipeline_case(case, recipe, settings: Settings, dataset_dir: Path) -> dict:
    source = case.source
    if "://" not in source:
        source = str((dataset_dir / source).resolve())
    # Isolate frames/downloads/output/memory even if a future node writes files.
    with TemporaryDirectory(prefix="hypereel-eval-") as work:
        isolated = replace(settings, download_dir=str(Path(work) / "downloads"),
                           output_dir=str(Path(work) / "output"),
                           memory_path=str(Path(work) / "memory.json"))
        graph = build_graph(settings=isolated)
        config = with_tracing(
            {"configurable": {"thread_id": str(uuid4())}}, isolated,
            recipe_id=recipe.id, entrypoint="evaluation",
        )
        initial = new_state(source, recipe, max_duration=case.max_duration,
                            audience=case.audience, subject_description=case.subject_description,
                            max_candidates=case.max_candidates)
        graph.invoke(initial, config)
        snapshot = graph.get_state(config)
        if tuple(snapshot.next) != ("approve_clips",):
            raise ValueError(f"unexpected pipeline stop: {snapshot.next}")
        state = snapshot.values
        duration = state.get("video_duration", 0)
        budget = case.max_duration if case.max_duration is not None else recipe.selection.max_duration
        if not isfinite(duration) or duration <= 0 or not isfinite(budget) or budget <= 0:
            raise ValueError("pipeline duration and budget must be finite and positive")
        if any(e.action_end > duration for e in case.reference_events or []):
            raise ValueError("reference events exceed the resolved source duration")
        degraded = []
        if not case.synthetic:
            if not state.get("video_path"):
                degraded.append("source unavailable; synthetic timeline used")
            if state.get("mode") == "mock":
                degraded.append("mock vision predictions on a non-synthetic case")
            degraded.extend(state.get("errors", []))
            classifications = state.get("classifications", [])
            provider_name = state.get("mode", "")
            if any(c.reason.startswith(f"{provider_name} error:")
                   or c.reason == "classification error" for c in classifications):
                degraded.append("one or more vision calls failed")
        clips = state.get("selected_clips", [])
        metrics, matches = selection_metrics(
            clips, budget=budget, video_duration=duration,
            events=None if degraded else case.reference_events,
            exhaustive=case.exhaustive and not degraded,
        )
        return {
            "status": "degraded" if degraded else "success", "degradation_reasons": degraded,
            "budget_seconds": budget, "video_duration": duration,
            "artifact_status": "paused_before_render", "stop_node": "approve_clips",
            "candidate_count": len(state.get("candidates", [])),
            "scored_count": len(state.get("scored_clips", [])),
            "candidates": [c.model_dump(mode="json") for c in state.get("candidates", [])],
            "classifications": [c.model_dump(mode="json") for c in state.get("classifications", [])],
            "clips": [c.model_dump(mode="json") for c in clips], "metrics": metrics, "matches": matches,
            "revision_count": state.get("revision_count", 0),
            "judge_verdict": state.get("judge_verdict", {}),
            "notes": state.get("notes", []), "pipeline_errors": state.get("errors", []),
            "requested_vision_provider": settings.vision_provider,
            "requested_llm_provider": settings.llm_provider,
            "actual_vision_provider": state.get("mode"),
            "trace_execution_id": config.get("metadata", {}).get("hypereel_execution_id"),
        }
