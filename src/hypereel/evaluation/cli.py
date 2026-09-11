"""Evaluate selection replay or graph predictions before the render gate."""
import argparse
from datetime import datetime, timezone
from uuid import uuid4
import sys
from pathlib import Path

from .runner import run_selection, run_pipeline_evaluation
from .report import write_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="replay fixed selection inputs")
    run.add_argument("--mode", choices=["selection", "pipeline"], default="selection")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, help="new or empty report directory")
    args = parser.parse_args(argv)
    try:
        output = args.output or Path("evals/results") / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        )
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"output must be a new or empty directory: {output}")
        runner = run_pipeline_evaluation if args.mode == "pipeline" else run_selection
        report = runner(args.dataset)
        write_report(report, output)
    except (ValueError, OSError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    print(f"Evaluated {report['case_count']} case(s); failed: {report['failed_count']}")
    if report.get("degraded_count"):
        print(f"Degraded: {report['degraded_count']} (excluded from quality averages)")
    print(f"Report: {output / 'summary.md'}")
    return 1 if report["failed_count"] or report.get("degraded_count") else 0


if __name__ == "__main__":
    raise SystemExit(main())
