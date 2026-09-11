"""Offline selection evaluation, independent of production graph execution.

Public API::

    from hypereel.evaluation import run_selection, write_report
    report = run_selection("evals/datasets/smoke.jsonl")
    write_report(report, "evals/results/my-run")
"""

from .runner import run_selection, run_pipeline_evaluation
from .report import write_report

__all__ = ["run_selection", "run_pipeline_evaluation", "write_report"]
