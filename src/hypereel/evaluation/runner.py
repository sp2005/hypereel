"""Selection replay and source-driven evaluation around existing pipeline code."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import subprocess
from uuid import uuid4
from math import isfinite

from ..recipe import load_recipe
from ..select.selector import score_candidates, select_clips
from .dataset import load_dataset
from .metrics import candidate_recall, operational_success_rate, selection_metrics


def run_selection(dataset_path: str | Path, *, judge: bool = False, settings=None) -> dict:
    return _run(dataset_path, judge=judge, settings=settings)


def run_pipeline_evaluation(dataset_path: str | Path, settings=None, *, judge: bool = False) -> dict:
    """Evaluate graph predictions at its first human gate; never auto-approve."""
    from ..config import get_settings
    return _run(dataset_path, mode="pipeline", settings=settings or get_settings(), judge=judge)


def _run(dataset_path, *, mode="selection", settings=None, judge=False) -> dict:
    path = Path(dataset_path).resolve()
    evaluation_run_id = str(uuid4())
    if judge and settings is None:
        from ..config import get_settings
        settings = get_settings()
    if mode == "pipeline":
        from .schemas import PipelineCase
        cases = load_dataset(path, PipelineCase)
    else:
        cases = load_dataset(path)
    results = []
    pending_judges = []
    for case in cases:
        result = {"case_id": case.case_id, "synthetic": case.synthetic, "status": "failed",
                  "recipe_sha256": None, "metrics": {}, "matches": [], "clips": []}
        start = perf_counter()
        try:
            recipe_path = (path.parent / case.recipe_path).resolve()
            result["recipe_sha256"] = sha256(recipe_path.read_bytes()).hexdigest()
            recipe = load_recipe(recipe_path)
            if mode == "pipeline":
                from .pipeline import evaluate_pipeline_case
                result.update(evaluate_pipeline_case(case, recipe, settings, path.parent))
                continue
            # Mirror the normal first selection pass, without judge overrides.
            if case.audience:
                recipe = recipe.model_copy(deep=True)
                recipe.subject_selector.audience = case.audience
            budget = recipe.selection.max_duration if case.max_duration is None else case.max_duration
            if budget < 0 or not isfinite(budget):
                raise ValueError("recipe budget must be finite and nonnegative")
            scored = score_candidates(case.candidates, case.classifications, recipe,
                                      video_duration=case.video_duration)
            clips = select_clips(scored, recipe, max_duration=budget, audience=case.audience)
            metrics, matches = selection_metrics(
                clips, budget=budget, video_duration=case.video_duration,
                events=case.reference_events, exhaustive=case.exhaustive,
            )
            metrics["candidate_recall"] = candidate_recall(case.candidates, case.reference_events)
            result.update(status="success", budget_seconds=budget, scored_count=len(scored),
                          metrics=metrics, matches=matches,
                          clips=[c.model_dump(mode="json") for c in clips])
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            result["elapsed_seconds"] = perf_counter() - start
            if judge:
                if result["status"] == "success":
                    pending_judges.append((result, recipe, case))
                else:
                    result["llm_judge"] = {
                        "status": "skipped", "assessment": None, "basis": "metadata_only",
                        "error": "case failed or degraded; judge skipped",
                    }
            results.append(result)
    # Judge spending must not consume the budget needed by later pipeline cases.
    for result, recipe, case in pending_judges:
        from .judge import judge_reel
        result["llm_judge"] = judge_reel(
            result, recipe, case, settings, evaluation_run_id=evaluation_run_id,
        )
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True, timeout=2, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True, timeout=2, check=True,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {
        "schema_version": 1, "metric_version": "selection-v6", "mode": mode,
        "evaluation_run_id": evaluation_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(path), "dataset_sha256": sha256(path.read_bytes()).hexdigest(),
        "code_revision": revision, "working_tree_dirty": dirty,
        "case_count": len(results), "failed_count": sum(r["status"] == "failed" for r in results),
        "degraded_count": sum(r["status"] == "degraded" for r in results),
        "operational_success_rate": operational_success_rate(results),
        "cases": results,
    }
