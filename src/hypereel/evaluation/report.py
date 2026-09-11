"""Local reports with separate synthetic/labeled groups and explicit denominators."""
import json
from pathlib import Path
from .metrics import operational_success_rate


def aggregate(cases: list[dict]) -> dict:
    groups = {}
    for synthetic, name in ((True, "synthetic"), (False, "non_synthetic")):
        subset = [c for c in cases if c["synthetic"] == synthetic]
        keys = sorted({k for c in subset for k, v in c["metrics"].items()
                       if v is None or isinstance(v, (int, float, bool))})
        groups[name] = {
            "case_count": len(subset),
            "failed_count": sum(c["status"] == "failed" for c in subset),
            "degraded_count": sum(c["status"] == "degraded" for c in subset),
            "operational_success_rate": operational_success_rate(subset),
            "metrics": {},
        }
        for key in keys:
            values = [c["metrics"][key] for c in subset
                      if c["status"] == "success"
                      and isinstance(c["metrics"].get(key), (int, float, bool))]
            groups[name]["metrics"][key] = {
                "macro_mean": sum(values) / len(values) if values else None,
                "applicable_cases": len(values),
            }
    return groups


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _rate(value):
    return "N/A" if value is None else f"{value:.4f}"


def _health_summary(cases):
    successful = sum(c["status"] == "success" for c in cases)
    return (f"Operational success rate: **{_rate(operational_success_rate(cases))}** "
            f"({successful}/{len(cases)} attempted cases).")


def write_report(report: dict, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data = {**report, "aggregates": aggregate(report["cases"]),
            "operational_success_rate": operational_success_rate(report["cases"])}
    (output / "report.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (output / "cases.jsonl").write_text("".join(
        json.dumps(case, allow_nan=False) + "\n" for case in report["cases"]
    ))
    mode = report.get("mode", "selection")
    description = ("Graph predictions evaluated at clip approval; rendering/sharing were not executed."
                   if mode == "pipeline" else
                   "Replay uses fixed predictions. It does not measure live model accuracy or render quality.")
    lines = [f"# HypeReel {mode} evaluation", "",
             f"Cases: {report['case_count']}; failed: {report['failed_count']}.", "",
             f"Degraded: {report.get('degraded_count', 0)} (excluded from quality averages, included in operational success rate).", "",
             _health_summary(report["cases"]), "",
             f"Dataset SHA-256: `{report['dataset_sha256']}`", "",
             f"Code revision: `{report['code_revision']}`; dirty: `{report['working_tree_dirty']}`.", "",
             description, ""]
    for name, group in data["aggregates"].items():
        lines += [f"## {name}", "", f"Cases: {group['case_count']}; failed: {group['failed_count']}.", "",
                  _health_summary([c for c in report["cases"] if c["synthetic"] == (name == "synthetic")]), "",
                  "| Metric | Macro mean | Applicable cases |", "|---|---:|---:|"]
        for metric, value in group["metrics"].items():
            mean = "N/A" if value["macro_mean"] is None else f"{value['macro_mean']:.4f}"
            lines.append(f"| {metric} | {mean} | {value['applicable_cases']} |")
        lines.append("")
    lines += ["Candidate recall counts each annotated event once when its action interval overlaps any candidate window. "
              "Missing/empty references are N/A; partial labels measure only the annotated subset.", "",
              "## Cases", "", "| Case | Status | Clips | Duration | Candidate recall | Elapsed seconds | Error |",
              "|---|---|---:|---:|---:|---:|---|"]
    for case in report["cases"]:
        m = case["metrics"]
        lines.append(f"| {_cell(case['case_id'])} | {case['status']} | {m.get('clip_count', 'N/A')} | "
                     f"{m.get('selected_duration_seconds', 'N/A')} | {_rate(m.get('candidate_recall'))} | {case['elapsed_seconds']:.6f} | "
                     f"{_cell(case.get('error') or '; '.join(case.get('degradation_reasons', [])))} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return output


def append_iteration_history(report: dict, history_path: str | Path, change_note: str) -> Path:
    """Append a compact, machine-readable run record; never rewrite history."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "evaluation_run_id": report["evaluation_run_id"],
        "created_at": report["created_at"],
        "mode": report["mode"],
        "change_note": change_note,
        "dataset_sha256": report["dataset_sha256"],
        "code_revision": report["code_revision"],
        "working_tree_dirty": report["working_tree_dirty"],
        "metric_version": report["metric_version"],
        "cases": [{
            "case_id": case["case_id"],
            "status": case["status"],
            "metrics": case["metrics"],
            "provider_attempted_calls": case.get("provider_attempted_calls"),
            "estimated_provider_spend_usd": case.get("estimated_provider_spend_usd"),
            "estimated_provider_cumulative_spend_usd": case.get(
                "estimated_provider_cumulative_spend_usd"
            ),
        } for case in report["cases"]],
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    return path
