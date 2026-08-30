"""Command-line entry point for HypeReel.

Loads a recipe, runs the LangGraph pipeline (``hypereel.graph.build.run_pipeline``),
and prints a readable report of the resulting reel. Designed to run fully
offline with ``--demo`` (mock providers, no API keys, no network) or against a
real source once ``HYPEREEL_VISION_PROVIDER`` / ``HYPEREEL_LLM_PROVIDER`` are
set to a live provider.

Never lets a raw traceback reach the user: recipe errors and pipeline errors
are caught and reported as short, friendly messages with a non-zero exit code.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import get_settings
from .models import ReelResult, Recipe
from .recipe import RecipeError, load_recipe

# Repo root, so relative recipe paths (the CLI default) resolve the same way
# no matter what directory the CLI is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RECIPE = "recipes/basketball_player.yaml"
_DEMO_SOURCE_PLACEHOLDER = "demo://aau_basketball_game.mp4"


def _resolve_path(path: str) -> Path:
    """Resolve ``path`` against the CWD first, falling back to the repo root."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    candidate = _REPO_ROOT / path
    return candidate if candidate.exists() else p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypereel",
        description="Turn a raw game recording into a share-ready highlight reel.",
    )
    parser.add_argument(
        "--recipe",
        default=_DEFAULT_RECIPE,
        help=f"path to a recipe YAML file (default: {_DEFAULT_RECIPE})",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="YouTube URL or local video path to build the reel from",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run fully offline with mock providers (no API keys, no network); "
        "forces auto-approval through both human gates",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="override the recipe's reel duration budget, in seconds",
    )
    parser.add_argument(
        "--audience",
        choices=["individual", "team"],
        default=None,
        help="override the recipe's audience (individual player vs. whole team)",
    )
    parser.add_argument(
        "--describe",
        dest="subject_description",
        default=None,
        metavar="TEXT",
        help="free-text description of who/what to capture and the visual cues to "
        "look for (jersey colors, a number, the scoreboard team label). Fed to "
        "the vision model and overrides the recipe's subject description for this "
        "run — the best way to steer results when you have no reference photo.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        metavar="N",
        help="quick-test cap: classify only the first N candidate windows "
        "(chronological). Unset or 0 = all windows. Handy for a fast dry run.",
    )
    parser.add_argument(
        "--no-approve",
        action="store_true",
        help="stop at the first human-in-the-loop gate instead of auto-approving",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="informational: where you intend the rendered reel to end up",
    )
    return parser


def _print_report(result: ReelResult, *, recipe: Recipe, mode: str) -> None:
    print(f"Recipe: {recipe.name} ({recipe.id})")
    print(f"Mode:   {mode}")
    # ReelResult doesn't carry the raw candidate count; report it if the
    # pipeline attached one (e.g. via a non-schema attribute), otherwise just
    # report what we know for certain: how many clips were selected.
    n_candidates = getattr(result, "n_candidates", None)
    if n_candidates is not None:
        print(f"Candidates considered: {n_candidates} -> selected: {len(result.clips)}")
    else:
        print(f"Selected: {len(result.clips)} clip(s)")
    print()
    if not result.clips:
        print("No clips were selected.")
    else:
        print("Selected clips:")
        for i, clip in enumerate(result.clips, start=1):
            print(
                f"  [{i}] {clip.start:6.1f}s - {clip.end:6.1f}s "
                f"({clip.duration:4.1f}s)  {clip.moment_type or 'unclassified':<14} "
                f"score={clip.score:.2f}  subject_present={clip.subject_present}"
            )
            if clip.reason:
                print(f"       reason: {clip.reason}")
    print()
    print(f"Total duration: {result.total_duration:.1f}s")
    print(f"Audience:       {result.audience}")
    print(f"Output path:    {result.output_path or '(none)'}")
    if result.summary:
        print()
        print("Summary:")
        print(result.summary)
    if result.notes:
        print()
        print("Notes:")
        for note in result.notes:
            print(f"  - {note}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    recipe_path = _resolve_path(args.recipe)
    try:
        recipe = load_recipe(recipe_path)
    except RecipeError as exc:
        print(f"error: could not load recipe {args.recipe!r}: {exc}", file=sys.stderr)
        return 2

    source = args.source or (_DEMO_SOURCE_PLACEHOLDER if args.demo else None)
    if not source:
        print("error: --source is required unless --demo is given", file=sys.stderr)
        return 2

    auto_approve = not args.no_approve or args.demo

    try:
        from .graph.build import run_pipeline
    except Exception as exc:  # pragma: no cover - defensive, shouldn't happen once wired
        print(f"error: pipeline is not available: {exc}", file=sys.stderr)
        return 1

    try:
        if args.demo:
            # The graph nodes each call get_settings() fresh from the
            # environment, so forcing mock mode has to happen via env vars
            # (mutating a Settings instance we pass in wouldn't be seen by
            # them) — this is what makes --demo work with zero API keys.
            os.environ["HYPEREEL_VISION_PROVIDER"] = "mock"
            os.environ["HYPEREEL_LLM_PROVIDER"] = "mock"
        settings = get_settings()
        result = run_pipeline(
            source,
            recipe,
            settings=settings,
            auto_approve=auto_approve,
            max_duration=args.max_duration,
            audience=args.audience,
            subject_description=args.subject_description,
            max_candidates=args.max_candidates,
            thread_id="cli-demo" if args.demo else "cli",
        )
    except Exception as exc:  # never dump a raw traceback to the user
        print(f"error: reel build failed: {exc}", file=sys.stderr)
        return 1

    if settings.vision_provider == "mock" and settings.llm_provider == "mock":
        mode = "mock"
    else:
        mode = f"live (vision={settings.vision_provider}, llm={settings.llm_provider})"
    _print_report(result, recipe=recipe, mode=mode)

    if args.out:
        print()
        print(f"(requested output location: {args.out})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
