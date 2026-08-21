from __future__ import annotations

from statistics import fmean

from eclipse_trajectory.config import ProjectConfig
from eclipse_trajectory.schemas import CandidateEvent, FrameMetric, WindowRecord


def build_windows(
    session_id: str,
    duration_seconds: float,
    metrics: list[FrameMetric],
    config: ProjectConfig,
) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    start = 0.0
    index = 0
    while start < duration_seconds - 1e-9 or (index == 0 and duration_seconds == 0):
        end = min(duration_seconds, start + config.windows.duration_seconds)
        members = [m for m in metrics if start - 1e-9 <= m.timestamp_seconds <= end + 1e-9]
        strongest = max(members, key=lambda item: item.change_score) if members else None
        maximum = strongest.change_score if strongest is not None else 0.0
        mean = fmean(item.change_score for item in members) if members else 0.0
        candidate = maximum >= config.detection.change_threshold
        reasons = _reasons(strongest, config) if candidate and strongest is not None else []
        windows.append(
            WindowRecord(
                session_id=session_id,
                window_id=f"window_{index:06d}",
                start_time_seconds=round(start, 6),
                end_time_seconds=round(end, 6),
                max_change_score=maximum,
                mean_change_score=mean,
                candidate=candidate,
                detection_reasons=reasons,
                strongest_changed_region=(
                    strongest.changed_region if strongest is not None else None
                ),
            )
        )
        index += 1
        start = index * config.windows.stride_seconds
        if duration_seconds == 0:
            break
    return windows


def build_candidate_events(
    session_id: str,
    duration_seconds: float,
    metrics: list[FrameMetric],
    windows: list[WindowRecord],
    config: ProjectConfig,
) -> list[CandidateEvent]:
    peaks = [item for item in metrics if item.change_score >= config.detection.change_threshold]
    groups: list[list[FrameMetric]] = []
    for metric in peaks:
        if not groups:
            groups.append([metric])
            continue
        current = groups[-1]
        gap = metric.timestamp_seconds - current[-1].timestamp_seconds
        span = metric.timestamp_seconds - current[0].timestamp_seconds
        if (
            gap <= config.detection.event_merge_gap_seconds
            and span <= config.detection.event_max_duration_seconds
        ):
            current.append(metric)
        else:
            groups.append([metric])

    events: list[CandidateEvent] = []
    for index, group in enumerate(groups):
        strongest = max(group, key=lambda item: item.change_score)
        start = max(0.0, group[0].timestamp_seconds - config.detection.event_padding_seconds)
        end = min(
            duration_seconds, group[-1].timestamp_seconds + config.detection.event_padding_seconds
        )
        overlap_ids = [
            window.window_id
            for window in windows
            if window.start_time_seconds < end and window.end_time_seconds > start
        ]
        events.append(
            CandidateEvent(
                session_id=session_id,
                event_id=f"event_{index:06d}",
                start_time_seconds=round(start, 6),
                interaction_time_seconds=strongest.timestamp_seconds,
                end_time_seconds=round(end, 6),
                max_change_score=strongest.change_score,
                detection_reasons=_reasons(strongest, config),
                strongest_changed_region=strongest.changed_region,
                overlapping_window_ids=overlap_ids,
            )
        )
    return events


def representative_times_for_window(
    window: WindowRecord, metrics: list[FrameMetric]
) -> dict[str, float]:
    members = [
        item
        for item in metrics
        if window.start_time_seconds - 1e-9
        <= item.timestamp_seconds
        <= window.end_time_seconds + 1e-9
    ]
    interaction = (
        max(members, key=lambda item: item.change_score).timestamp_seconds
        if members
        else window.start_time_seconds
    )
    return {
        "before": window.start_time_seconds,
        "interaction": interaction,
        "after": window.end_time_seconds,
    }


def representative_times_for_event(event: CandidateEvent) -> dict[str, float]:
    return {
        "before": event.start_time_seconds,
        "interaction": event.interaction_time_seconds,
        "after": event.end_time_seconds,
    }


def _reasons(metric: FrameMetric, config: ProjectConfig) -> list[str]:
    reasons = ["global_visual_change"]
    if metric.changed_fraction >= 0.40:
        reasons.append("large_workspace_transition")
    elif metric.changed_region is not None and metric.changed_fraction <= 0.25:
        reasons.append("localized_visual_change")
    if metric.dhash_distance >= 12:
        reasons.append("perceptual_hash_change")
    if metric.change_score < config.detection.change_threshold:
        return []
    return reasons
