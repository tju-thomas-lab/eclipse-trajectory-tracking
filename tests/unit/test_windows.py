from __future__ import annotations

from eclipse_trajectory.config import ProjectConfig
from eclipse_trajectory.detection.events import build_candidate_events, build_windows
from eclipse_trajectory.schemas import FrameMetric


def metric(index: int, timestamp: float, score: float) -> FrameMetric:
    return FrameMetric(
        sample_index=index,
        timestamp_seconds=timestamp,
        mean_absolute_change=min(score, 1.0),
        changed_fraction=min(score, 1.0),
        structural_similarity_approx=max(0.0, 1.0 - score),
        dhash_distance=8,
        change_score=score,
    )


def test_rolling_windows_have_overlap_and_cover_tail() -> None:
    config = ProjectConfig()
    metrics = [metric(index, index * 0.5, 0.01) for index in range(21)]
    windows = build_windows("session_test", 10.0, metrics, config)

    assert [(item.start_time_seconds, item.end_time_seconds) for item in windows] == [
        (0.0, 5.0),
        (2.5, 7.5),
        (5.0, 10.0),
        (7.5, 10.0),
    ]


def test_continuous_changes_are_capped_into_multiple_events() -> None:
    config = ProjectConfig.model_validate(
        {"detection": {"change_threshold": 0.03, "event_max_duration_seconds": 2.0}}
    )
    metrics = [metric(index, index * 0.5, 0.1) for index in range(13)]
    windows = build_windows("session_test", 6.0, metrics, config)
    events = build_candidate_events("session_test", 6.0, metrics, windows, config)

    assert len(events) >= 2
    assert all(item.end_time_seconds - item.start_time_seconds <= 3.0 + 1e-6 for item in events)
    assert all(item.overlapping_window_ids for item in events)
