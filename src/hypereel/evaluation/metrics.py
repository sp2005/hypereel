"""Deterministic selection metrics. Undefined measurements are null, not zero."""
from ..models import CandidateWindow, Clip
from .schemas import ReferenceEvent


def candidate_recall(
    candidates: list[CandidateWindow], events: list[ReferenceEvent] | None,
) -> float | None:
    """Fraction of annotated events covered by any candidate, ignoring labels.

    Each event counts once even with overlapping/duplicate candidates. Intervals
    are start-inclusive/end-exclusive, as in event matching. Partial annotations
    measure recall on the annotated subset only. No references means N/A.
    """
    if not events:
        return None
    covered = sum(any(w.start < e.action_end and e.action_start < w.end
                      for w in candidates) for e in events)
    return covered / len(events)


def operational_success_rate(cases: list[dict]) -> float | None:
    """Successful cases / all attempted cases, including failed and degraded.

    This measures execution health, not clip quality. Empty successful reels
    count as successful executions. Dataset validation failures precede attempts.
    """
    if not cases:
        return None
    return sum(case["status"] == "success" for case in cases) / len(cases)


def match_events(clips: list[Clip], events: list[ReferenceEvent]) -> list[tuple[int, int]]:
    """Maximum-cardinality, one-to-one matching by label and action overlap.

    Stable input order breaks ties. Any positive overlap with the externally
    supplied action interval is eligible; temporal errors are reported separately.
    """
    edges = [[j for j, event in enumerate(events)
              if clip.moment_type == event.moment_type
              and clip.start < event.action_end
              and event.action_start < clip.end] for clip in clips]
    owners: dict[int, int] = {}

    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or assign(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(clips)):
        assign(i, set())
    return sorted((i, j) for j, i in owners.items())


def selection_metrics(clips: list[Clip], *, budget: float, video_duration: float,
                      events: list[ReferenceEvent] | None = None, exhaustive: bool = False):
    duration = sum(c.duration for c in clips)
    union = 0.0
    end = 0.0
    for clip in sorted(clips, key=lambda c: c.start):
        union += max(0.0, clip.end - max(end, clip.start))
        end = max(end, clip.end)
    matches = match_events(clips, events) if events is not None else []
    precision = (len(matches) / len(clips)
                 if exhaustive and events is not None and clips else None)
    recall = len(matches) / len(events) if events else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    complete = sum(clips[i].start <= events[j].action_start
                   and clips[i].end >= events[j].action_end for i, j in matches)
    matched_clip_indexes = {i for i, _ in matches}
    matched_event_indexes = {j for _, j in matches}
    moment_types = sorted({c.moment_type for c in clips if c.moment_type}
                          | {e.moment_type for e in events or []})
    per_type = {}
    for moment_type in moment_types:
        tp = sum(clips[i].moment_type == moment_type for i, _ in matches)
        fp = sum(i not in matched_clip_indexes and c.moment_type == moment_type
                 for i, c in enumerate(clips))
        fn = sum(j not in matched_event_indexes and e.moment_type == moment_type
                 for j, e in enumerate(events or []))
        type_precision = tp / (tp + fp) if tp + fp else None
        type_recall = tp / (tp + fn) if tp + fn else None
        type_f1 = (2 * type_precision * type_recall / (type_precision + type_recall)
                   if type_precision is not None and type_recall is not None
                   and type_precision + type_recall else None)
        per_type[moment_type] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": type_precision, "recall": type_recall, "f1": type_f1,
        }
    class_f1s = [row["f1"] for row in per_type.values() if row["f1"] is not None]
    class_recalls = [row["recall"] for row in per_type.values() if row["recall"] is not None]
    timing_offsets = [abs((clips[i].start + clips[i].end) / 2 - events[j].event_time)
                      for i, j in matches]
    boundary_errors = [
        (abs(clips[i].start - events[j].action_start)
         + abs(clips[i].end - events[j].action_end)) / 2
        for i, j in matches
    ]
    return {
        "clip_count": len(clips),
        "selected_duration_seconds": duration,
        "budget_compliance": duration <= budget + 1e-9,
        "budget_utilization": duration / budget if budget else None,
        "invalid_boundary_count": sum(not 0 <= c.start < c.end <= video_duration for c in clips),
        "overlap_rate": max(0.0, duration - union) / duration if duration else None,
        "matched_event_count": len(matches) if events is not None else None,
        "reference_event_count": len(events) if events is not None else None,
        "relevant_clip_precision": precision,
        "selected_event_recall": recall,
        "selection_f1": f1,
        "macro_f1": sum(class_f1s) / len(class_f1s) if class_f1s else None,
        "balanced_accuracy": (sum(class_recalls) / len(class_recalls)
                              if class_recalls else None),
        "mean_event_time_error_seconds": (sum(timing_offsets) / len(timing_offsets)
                                          if timing_offsets else None),
        "mean_boundary_error_seconds": (sum(boundary_errors) / len(boundary_errors)
                                        if boundary_errors else None),
        "action_completeness": complete / len(matches) if matches else None,
        "distinct_moment_types": len({c.moment_type for c in clips if c.moment_type}),
        "confusion_counts": {
            "tp": len(matches),
            "fp": len(clips) - len(matched_clip_indexes),
            "fn": len(events or []) - len(matched_event_indexes),
        } if events is not None else None,
        "per_moment_type": per_type if events is not None else None,
    }, [{"clip_index": i, "event_id": events[j].event_id} for i, j in matches]
