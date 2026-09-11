"""Evaluation adapter around the unmodified graph, stopping before rendering."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from math import isfinite

from ..config import Settings
from ..graph.build import build_graph
from ..graph.state import new_state
from ..observability import provider_budget_scope, with_tracing
from .metrics import candidate_recall, selection_metrics


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
                            max_candidates=case.max_candidates,
                            candidate_sampling=case.candidate_sampling,
                            evaluation_start_seconds=case.evaluation_start_seconds,
                            evaluation_end_seconds=case.evaluation_end_seconds)
        with provider_budget_scope(isolated) as provider_budget:
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
        uncapped_candidates = state.get("uncapped_candidates", state.get("candidates", []))
        metrics["candidate_recall"] = candidate_recall(
            uncapped_candidates, None if degraded else case.reference_events,
        )
        candidates = state.get("candidates", [])
        horizon_start = case.evaluation_start_seconds or 0.0
        horizon_end = (case.evaluation_end_seconds
                       if case.evaluation_end_seconds is not None
                       else max((candidate.end for candidate in candidates), default=0.0))
        if case.max_candidates and not degraded:
            slice_events = [event for event in case.reference_events or []
                            if horizon_start <= event.event_time < horizon_end]
            slice_metrics, _ = selection_metrics(
                clips, budget=budget, video_duration=duration,
                events=slice_events, exhaustive=case.exhaustive,
            )
            metrics.update({f"processed_slice_{key}": value for key, value in
                            slice_metrics.items()})
            metrics["processed_slice_candidate_recall"] = candidate_recall(
                candidates, slice_events
            )
            metrics["processed_slice_reference_event_count"] = len(slice_events)
        classifications = state.get("classifications", [])
        valid_classifications = sum(
            not (c.reason.startswith("parse error:") or c.reason == "classification error")
            for c in classifications
        )
        metrics["classification_schema_pass_rate"] = (
            valid_classifications / len(classifications) if classifications else None
        )
        true_positives = (metrics.get("confusion_counts") or {}).get("tp", 0)
        metrics["cost_per_true_positive_usd"] = (
            provider_budget["estimated_spend_usd"] / true_positives
            if true_positives else None
        )
        return {
            "status": "degraded" if degraded else "success", "degradation_reasons": degraded,
            "budget_seconds": budget, "video_duration": duration,
            "artifact_status": "paused_before_render", "stop_node": "approve_clips",
            "candidate_count": len(state.get("candidates", [])),
            "uncapped_candidate_count": len(uncapped_candidates),
            "evaluation_horizon_end_seconds": horizon_end,
            "evaluation_horizon_start_seconds": horizon_start,
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
            "provider_usage": provider_budget["calls"],
            "provider_attempted_calls": provider_budget["attempted_calls"],
            "estimated_provider_spend_usd": provider_budget["estimated_spend_usd"],
            "estimated_provider_spend_before_run_usd": provider_budget[
                "estimated_spend_before_run_usd"
            ],
            "estimated_provider_cumulative_spend_usd": (
                provider_budget["estimated_spend_before_run_usd"]
                + provider_budget["estimated_spend_usd"]
            ),
            "provider_spend_cap_usd": provider_budget["max_spend_usd"],
        }
