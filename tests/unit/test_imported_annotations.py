from __future__ import annotations

import json
from pathlib import Path

from eclipse_trajectory.models.imported_annotations import import_annotation_bundle
from eclipse_trajectory.schemas import CandidateEvent, EvidenceRef, WindowRecord
from eclipse_trajectory.synthesis.hierarchy import synthesize_hierarchy
from eclipse_trajectory.util import atomic_write_json, atomic_write_jsonl, read_jsonl


def _evidence(role: str, timestamp: float) -> dict[str, object]:
    return EvidenceRef(
        role=role,  # type: ignore[arg-type]
        frame_id=f"frame_{role}",
        path=f"evidence/{role}.jpg",
        sha256="0" * 64,
        requested_timestamp_seconds=timestamp,
        actual_timestamp_seconds=timestamp,
        media_type="image/jpeg",
    ).model_dump(mode="json")


def test_import_bundle_resolves_event_and_window_evidence(tmp_path: Path) -> None:
    session = tmp_path / "session_test"
    session.mkdir()
    event = CandidateEvent(
        session_id=session.name,
        event_id="event_000000",
        start_time_seconds=1.0,
        interaction_time_seconds=1.5,
        end_time_seconds=2.0,
        max_change_score=0.2,
        detection_reasons=["change"],
        strongest_changed_region=None,
        overlapping_window_ids=["window_000000"],
        evidence=[EvidenceRef.model_validate(_evidence("interaction", 1.5))],
    )
    window = WindowRecord(
        session_id=session.name,
        window_id="window_000000",
        start_time_seconds=0.0,
        end_time_seconds=5.0,
        max_change_score=0.2,
        mean_change_score=0.1,
        candidate=True,
        detection_reasons=["change"],
        strongest_changed_region=None,
        evidence=[
            EvidenceRef.model_validate(_evidence("before", 0.0)),
            EvidenceRef.model_validate(_evidence("after", 5.0)),
        ],
    )
    atomic_write_jsonl(session / "candidate_events.jsonl", [event])
    atomic_write_jsonl(session / "windows.jsonl", [window])
    atomic_write_json(
        session / "manifest.json",
        {"implemented_backends": ["deterministic"], "counts": {"actions": 1}},
    )
    bundle_path = tmp_path / "annotations.json"
    atomic_write_json(
        bundle_path,
        {
            "session_id": session.name,
            "generator_label": "test visual inference",
            "executive_summary": "A setting was changed.",
            "actions": [
                {
                    "annotation_id": "label_000000",
                    "start_time_seconds": 1.0,
                    "end_time_seconds": 2.0,
                    "source_refs": [
                        {
                            "record_type": "window",
                            "record_id": "window_000000",
                            "evidence_roles": ["before", "after"],
                        },
                        {
                            "record_type": "event",
                            "record_id": "event_000000",
                            "evidence_roles": ["interaction"],
                        },
                    ],
                    "output": {
                        "primitive_action_type": "select",
                        "semantic_action": "Changed a visible setting.",
                        "low_level_instruction": "Select the visible setting.",
                        "high_level_summary": "Configured the setting.",
                        "episode_label": "configuration",
                        "confidence": 0.9,
                        "inference_sources": ["visible_state_transition"],
                    },
                }
            ],
        },
    )

    actions = import_annotation_bundle(session, bundle_path)

    assert len(actions) == 1
    assert actions[0].source_event_id == "event_000000"
    assert actions[0].source_window_ids == ["window_000000"]
    assert [frame.role for frame in actions[0].evidence.frames] == [
        "before",
        "after",
        "interaction",
    ]
    assert json.loads((session / "session.json").read_text())["executive_summary"] == (
        "A setting was changed."
    )
    assert json.loads((session / "manifest.json").read_text())["active_backend"] == (
        "imported_local_vlm"
    )
    assert len(list(read_jsonl(session / "model_outputs.jsonl"))) == 1

    _, synthesized_again = synthesize_hierarchy(session)
    assert synthesized_again["executive_summary"] == "A setting was changed."
