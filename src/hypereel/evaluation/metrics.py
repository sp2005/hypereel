"""Deterministic selection metrics. Undefined measurements are null, not zero."""
from ..models import CandidateWindow, Classification, Clip
from .schemas import ReferenceEvent


def interval_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return intersection / union if union > 0 else 0.0


def _maximum_matching(edges: list[list[int]]) -> list[tuple[int, int]]:
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

    for i in range(len(edges)):
        assign(i, set())
    return sorted((i, j) for j, i in owners.items())


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


def _average_precision(scores_and_labels: list[tuple[float, bool]]) -> float | None:
    positives = sum(label for _, label in scores_and_labels)
    if not positives:
        return None
    ranked = sorted(scores_and_labels, key=lambda row: row[0], reverse=True)
    hits = 0
    precisions = []
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / positives


def pipeline_diagnostic_metrics(
    candidates: list[CandidateWindow],
    classifications: list[Classification],
    events: list[ReferenceEvent] | None,
    *,
    exhaustive: bool,
    selected_match_count: int,
) -> dict:
    """Candidate-level and stage-funnel diagnostics for live pipeline evals."""
    if events is None:
        return {}
    rows = []
    for window, classification in zip(candidates, classifications):
        labels = {event.moment_type for event in events
                  if window.start < event.action_end and event.action_start < window.end}
        predicted = classification.moment_type
        actual_positive = bool(labels)
        predicted_positive = predicted is not None
        correct = predicted in labels if labels else predicted is None
        event_score = classification.confidence if predicted_positive else 0.0
        rows.append((actual_positive, predicted_positive, correct,
                     classification.confidence, event_score))

    negatives = [row for row in rows if not row[0]]
    true_negatives = sum(not row[1] for row in negatives)
    specificity = (true_negatives / len(negatives)
                   if exhaustive and negatives else None)
    accuracy = (sum(row[2] for row in rows) / len(rows) if rows else None)

    bins = [[] for _ in range(10)]
    for _, _, correct, confidence, _ in rows:
        bins[min(9, int(confidence * 10))].append((confidence, float(correct)))
    ece = sum(
        len(bucket) / len(rows)
        * abs(sum(c for c, _ in bucket) / len(bucket)
              - sum(y for _, y in bucket) / len(bucket))
        for bucket in bins if bucket
    ) if rows else None

    detected_events = sum(any(
        window.start < event.action_end and event.action_start < window.end
        and classification.moment_type == event.moment_type
        for window, classification in zip(candidates, classifications)
    ) for event in events)
    threshold_sweep = {}
    for threshold in (0.25, 0.5, 0.75, 0.9):
        tp = fp = fn = 0
        for actual, _, _, _, score in rows:
            predicted = score >= threshold
            tp += actual and predicted
            fp += (not actual) and predicted
            fn += actual and not predicted
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        threshold_sweep[str(threshold)] = {
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
        }

    return {
        "candidate_classification_accuracy": accuracy,
        "negative_window_specificity": specificity,
        "negative_window_count": len(negatives) if exhaustive else None,
        "classification_calibration_error": ece,
        "candidate_event_average_precision": _average_precision(
            [(row[4], row[0]) for row in rows]
        ),
        "classification_event_recall": detected_events / len(events) if events else None,
        "classification_detected_event_count": detected_events,
        "selection_survival_rate": (selected_match_count / detected_events
                                    if detected_events else None),
        "candidate_threshold_sweep": threshold_sweep,
    }


def match_events(clips: list[Clip], events: list[ReferenceEvent]) -> list[tuple[int, int]]:
    """Maximum-cardinality, one-to-one matching by label and action overlap.

    Stable input order breaks ties. Any positive overlap with the externally
    supplied action interval is eligible; temporal errors are reported separately.
    """
    edges = [[j for j, event in enumerate(events)
              if clip.moment_type == event.moment_type
              and clip.start < event.action_end
              and event.action_start < clip.end] for clip in clips]
    return _maximum_matching(edges)


def selection_metrics(clips: list[Clip], *, budget: float, video_duration: float,
                      events: list[ReferenceEvent] | None = None, exhaustive: bool = False,
                      observed_duration_seconds: float | None = None):
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
    temporal_matches = _maximum_matching([
        [j for j, event in enumerate(events or [])
         if clip.start < event.action_end and event.action_start < clip.end]
        for clip in clips
    ]) if events is not None else []
    temporal_clip_indexes = {i for i, _ in temporal_matches}
    temporal_event_indexes = {j for _, j in temporal_matches}
    confusion_matrix: dict[str, dict[str, int]] = {}
    for i, j in temporal_matches:
        actual = events[j].moment_type
        predicted = clips[i].moment_type or "__none__"
        confusion_matrix.setdefault(actual, {})[predicted] = (
            confusion_matrix.setdefault(actual, {}).get(predicted, 0) + 1
        )
    for j, event in enumerate(events or []):
        if j not in temporal_event_indexes:
            confusion_matrix.setdefault(event.moment_type, {})["__miss__"] = (
                confusion_matrix.setdefault(event.moment_type, {}).get("__miss__", 0) + 1
            )
    for i, clip in enumerate(clips):
        if i not in temporal_clip_indexes:
            predicted = clip.moment_type or "__none__"
            confusion_matrix.setdefault("__none__", {})[predicted] = (
                confusion_matrix.setdefault("__none__", {}).get(predicted, 0) + 1
            )
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
    matched_ious = [interval_iou(clips[i].start, clips[i].end,
                                 events[j].action_start, events[j].action_end)
                    for i, j in matches]
    iou_recalls = {}
    for threshold in (0.1, 0.3, 0.5):
        threshold_matches = _maximum_matching([
            [j for j, event in enumerate(events or [])
             if clip.moment_type == event.moment_type
             and interval_iou(clip.start, clip.end, event.action_start, event.action_end)
             >= threshold]
            for clip in clips
        ])
        iou_recalls[f"event_recall_at_iou_{str(threshold).replace('.', '_')}"] = (
            len(threshold_matches) / len(events) if events else None
        )
    duplicate_count = sum(max(0, sum(
        clip.moment_type == event.moment_type
        and clip.start < event.action_end and event.action_start < clip.end
        for clip in clips
    ) - 1) for event in events or [])
    false_positives = len(clips) - len(matched_clip_indexes)
    near_miss_count = sum(
        i not in matched_clip_indexes and any(
            clip.moment_type == event.moment_type
            and min(abs(clip.end - event.action_start), abs(event.action_end - clip.start)) <= 2
            for event in events or []
        ) for i, clip in enumerate(clips)
    )
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
        "micro_f1": f1,
        "macro_f1": sum(class_f1s) / len(class_f1s) if class_f1s else None,
        "balanced_accuracy": (sum(class_recalls) / len(class_recalls)
                              if class_recalls else None),
        "mean_event_time_error_seconds": (sum(timing_offsets) / len(timing_offsets)
                                          if timing_offsets else None),
        "mean_boundary_error_seconds": (sum(boundary_errors) / len(boundary_errors)
                                        if boundary_errors else None),
        "mean_temporal_iou": (sum(matched_ious) / len(matched_ious)
                              if matched_ious else None),
        **iou_recalls,
        "duplicate_event_count": duplicate_count if events is not None else None,
        "duplicate_event_rate": (duplicate_count / len(events) if events else None),
        "false_positives_per_video_minute": (
            false_positives / ((observed_duration_seconds or video_duration) / 60)
            if exhaustive and events is not None
            and (observed_duration_seconds or video_duration) > 0 else None
        ),
        "near_miss_count": near_miss_count if events is not None else None,
        "action_completeness": complete / len(matches) if matches else None,
        "distinct_moment_types": len({c.moment_type for c in clips if c.moment_type}),
        "confusion_counts": {
            "tp": len(matches),
            "fp": len(clips) - len(matched_clip_indexes),
            "fn": len(events or []) - len(matched_event_indexes),
        } if events is not None else None,
        "per_moment_type": per_type if events is not None else None,
        "confusion_matrix": confusion_matrix if events is not None else None,
    }, [{"clip_index": i, "event_id": events[j].event_id} for i, j in matches]
