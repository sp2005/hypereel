"""Deterministic selection metrics. Undefined measurements are null, not zero."""
from ..models import Clip
from .schemas import ReferenceEvent


def match_events(clips: list[Clip], events: list[ReferenceEvent]) -> list[tuple[int, int]]:
    """Maximum-cardinality, one-to-one matching by label and contained event time.

    Stable input order breaks ties. This version uses timestamp containment,
    not an adjustable IoU threshold. Matches are not optimized for completeness.
    """
    edges = [[j for j, event in enumerate(events)
              if clip.moment_type == event.moment_type
              and clip.start <= event.event_time < clip.end] for clip in clips]
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
    complete = sum(clips[i].start <= events[j].action_start
                   and clips[i].end >= events[j].action_end for i, j in matches)
    return {
        "clip_count": len(clips),
        "selected_duration_seconds": duration,
        "budget_compliance": duration <= budget + 1e-9,
        "budget_utilization": duration / budget if budget else None,
        "invalid_boundary_count": sum(not 0 <= c.start < c.end <= video_duration for c in clips),
        "overlap_rate": max(0.0, duration - union) / duration if duration else None,
        "matched_event_count": len(matches) if events is not None else None,
        "reference_event_count": len(events) if events is not None else None,
        "relevant_clip_precision": len(matches) / len(clips)
            if exhaustive and events is not None and clips else None,
        "action_completeness": complete / len(matches) if matches else None,
        "distinct_moment_types": len({c.moment_type for c in clips if c.moment_type}),
    }, [{"clip_index": i, "event_id": events[j].event_id} for i, j in matches]
