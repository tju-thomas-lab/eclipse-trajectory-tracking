from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eclipse_trajectory import __version__
from eclipse_trajectory.config import ProjectConfig
from eclipse_trajectory.detection.events import (
    build_candidate_events,
    build_windows,
    representative_times_for_event,
    representative_times_for_window,
)
from eclipse_trajectory.detection.metrics import calculate_frame_metrics, read_frame_metrics
from eclipse_trajectory.export.evidence import as_evidence_ref, extract_evidence_frames
from eclipse_trajectory.export.timeline import write_timeline
from eclipse_trajectory.ingest.video import inspect_video
from eclipse_trajectory.models.local_vlm import LocalOpenAICompatibleBackend
from eclipse_trajectory.models.prompts import SYSTEM_PROMPT, event_prompt
from eclipse_trajectory.privacy.redact import suppress_likely_identifiers
from eclipse_trajectory.schemas import (
    ActionEvidence,
    ActionRecord,
    CandidateEvent,
    FrameMetric,
    PrimitiveAction,
    SemanticAction,
    VideoMetadata,
    VLMNormalizedOutput,
    WindowRecord,
)
from eclipse_trajectory.synthesis.hierarchy import synthesize_hierarchy
from eclipse_trajectory.util import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_hash,
    read_jsonl,
    stage_is_current,
    write_stage_marker,
)


def run_pipeline(video_path: Path, config: ProjectConfig) -> Path:
    metadata = inspect_video(video_path)
    session_id = f"session_{metadata.source_sha256[:12]}"
    session_dir = config.runtime.output_root.resolve() / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    signature_base = {
        "source_sha256": metadata.source_sha256,
        "config_sha256": config.content_hash,
        "software_version": __version__,
    }

    metrics_signature = canonical_hash({**signature_base, "stage": "metrics"})
    metrics_path = session_dir / "frame_metrics.parquet"
    if stage_is_current(session_dir, "metrics", metrics_signature) and not config.runtime.overwrite:
        metrics = read_frame_metrics(metrics_path)
    else:
        metrics = calculate_frame_metrics(video_path, metadata, config, metrics_path)
        write_stage_marker(session_dir, "metrics", metrics_signature, ["frame_metrics.parquet"])

    evidence_signature = canonical_hash({**signature_base, "stage": "evidence"})
    evidence_cached = (
        stage_is_current(session_dir, "evidence", evidence_signature)
        and not config.runtime.overwrite
    )
    if evidence_cached:
        windows = [
            WindowRecord.model_validate(item) for item in read_jsonl(session_dir / "windows.jsonl")
        ]
        events = [
            CandidateEvent.model_validate(item)
            for item in read_jsonl(session_dir / "candidate_events.jsonl")
        ]
        evidence_cached = _evidence_files_present(session_dir, windows, events)
    if not evidence_cached:
        windows = build_windows(session_id, metadata.duration_seconds, metrics, config)
        events = build_candidate_events(
            session_id, metadata.duration_seconds, metrics, windows, config
        )
        _attach_evidence(video_path, metadata, session_dir, config, metrics, windows, events)
        atomic_write_jsonl(session_dir / "windows.jsonl", windows)
        atomic_write_jsonl(session_dir / "candidate_events.jsonl", events)
        write_stage_marker(
            session_dir,
            "evidence",
            evidence_signature,
            ["windows.jsonl", "candidate_events.jsonl", "evidence_index.jsonl", "evidence"],
        )

    actions = deterministic_actions(events)
    atomic_write_jsonl(session_dir / "actions.jsonl", actions)
    _write_vlm_requests(session_dir, events, config)
    (session_dir / "review_corrections.jsonl").touch(exist_ok=True)
    synthesize_hierarchy(session_dir)
    write_timeline(session_dir, actions)
    _write_manifest(session_dir, metadata, config, windows, events, actions)
    return session_dir


def deterministic_actions(events: list[CandidateEvent]) -> list[ActionRecord]:
    actions: list[ActionRecord] = []
    for index, event in enumerate(events):
        regions = [event.strongest_changed_region] if event.strongest_changed_region else []
        actions.append(
            ActionRecord(
                session_id=event.session_id,
                action_id=f"action_{index:06d}",
                source_event_id=event.event_id,
                start_time_seconds=event.start_time_seconds,
                end_time_seconds=event.end_time_seconds,
                primitive_action=PrimitiveAction(type="unknown_action", confidence=None),
                semantic_action=SemanticAction(description=None, confidence=None),
                low_level_instruction=(
                    "Inspect the visible state change; the specific operation is not identified."
                ),
                high_level_summary=None,
                episode_label="unknown_episode",
                context_transition="unknown",
                state_before=None,
                state_after=None,
                evidence=ActionEvidence(
                    frames=event.evidence,
                    changed_regions=regions,
                    inference_sources=["frame_difference", "perceptual_hash"],
                ),
                field_sources={
                    "primitive_action.type": "deterministic_state_change_fallback",
                    "semantic_action": None,
                    "clinical_intent": None,
                },
                model_provenance={
                    "backend": "deterministic",
                    "software_version": __version__,
                    "model": None,
                },
                ontology_version="0.1.0",
            )
        )
    return actions


def run_local_inference(
    session_dir: Path,
    endpoint: str,
    model: str,
    redact_likely_identifiers: bool = True,
) -> list[ActionRecord]:
    session_dir = session_dir.resolve()
    events = [
        CandidateEvent.model_validate(item)
        for item in read_jsonl(session_dir / "candidate_events.jsonl")
    ]
    backend = LocalOpenAICompatibleBackend(endpoint, model, session_dir)
    run_id = canonical_hash(
        {
            "backend": "local_openai_compatible",
            "endpoint": endpoint.rstrip("/"),
            "model": model,
        }
    )[:16]
    output_path = session_dir / "model_outputs.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        completed = {
            str(item["event_id"]): item
            for item in read_jsonl(output_path)
            if item.get("run_id") == run_id
        }

    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for event in events:
            if event.event_id in completed:
                continue
            normalized = backend.infer(event)
            if redact_likely_identifiers:
                normalized = suppress_likely_identifiers(normalized)
            record = {
                "run_id": run_id,
                "event_id": event.event_id,
                "raw_response": backend.last_raw_text,
                "normalized_output": normalized.model_dump(mode="json"),
                "provenance": backend.provenance,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            completed[event.event_id] = record

    actions = [
        _model_action(
            index,
            event,
            VLMNormalizedOutput.model_validate(completed[event.event_id]["normalized_output"]),
            backend.provenance,
        )
        for index, event in enumerate(events)
    ]
    atomic_write_jsonl(session_dir / "actions.jsonl", actions)
    synthesize_hierarchy(session_dir)
    write_timeline(session_dir, actions)
    _mark_active_model(session_dir, backend.provenance, run_id)
    return actions


def _model_action(
    index: int,
    event: CandidateEvent,
    output: VLMNormalizedOutput,
    provenance: dict[str, object],
) -> ActionRecord:
    regions = [event.strongest_changed_region] if event.strongest_changed_region else []
    return ActionRecord(
        session_id=event.session_id,
        action_id=f"action_{index:06d}",
        source_event_id=event.event_id,
        start_time_seconds=event.start_time_seconds,
        end_time_seconds=event.end_time_seconds,
        primitive_action=PrimitiveAction(
            type=output.primitive_action_type,
            coordinates_normalized=output.coordinates_normalized,
            text=output.typed_text,
            key_or_shortcut=output.key_or_shortcut,
            confidence=output.confidence,
            alternatives=output.alternative_interpretations,
        ),
        semantic_action=SemanticAction(
            application=output.application,
            workspace=output.workspace,
            visible_dialog=output.visible_dialog,
            selected_object=output.selected_object,
            target=output.ui_target,
            description=output.semantic_action,
            confidence=output.confidence,
        ),
        low_level_instruction=output.low_level_instruction,
        high_level_summary=output.high_level_summary,
        episode_label=output.episode_label or "unknown_episode",
        context_transition=output.context_transition,
        state_before={"visible_value": output.visible_value_before},
        state_after={"visible_value": output.visible_value_after},
        evidence=ActionEvidence(
            frames=event.evidence,
            changed_regions=regions,
            inference_sources=output.inference_sources + ["local_vlm"],
        ),
        field_sources={
            "primitive_action.type": "local_vlm_inference",
            "semantic_action": "local_vlm_inference",
            "clinical_intent": None,
        },
        model_provenance=provenance,
        ontology_version="0.1.0",
    )


def _attach_evidence(
    video_path: Path,
    metadata: VideoMetadata,
    session_dir: Path,
    config: ProjectConfig,
    metrics: list[FrameMetric],
    windows: list[WindowRecord],
    events: list[CandidateEvent],
) -> None:
    window_times = [(item, representative_times_for_window(item, metrics)) for item in windows]
    event_times = [(item, representative_times_for_event(item)) for item in events]
    requested = [
        timestamp
        for _, role_times in [*window_times, *event_times]
        for timestamp in role_times.values()
    ]
    extracted = extract_evidence_frames(
        video_path, metadata, requested, session_dir, config.evidence
    )
    index_rows = []
    for target in sorted(extracted):
        frame = extracted[target]
        index_rows.append(
            {
                "requested_timestamp_seconds": target,
                "actual_timestamp_seconds": frame.actual_timestamp_seconds,
                "frame_id": frame.frame_id,
                "path": frame.path,
                "sha256": frame.sha256,
                "media_type": frame.media_type,
            }
        )
    atomic_write_jsonl(session_dir / "evidence_index.jsonl", index_rows)
    for window, role_times in window_times:
        window.evidence = [
            as_evidence_ref(role, timestamp, extracted[round(timestamp, 6)])
            for role, timestamp in role_times.items()
        ]
    for event, role_times in event_times:
        event.evidence = [
            as_evidence_ref(role, timestamp, extracted[round(timestamp, 6)])
            for role, timestamp in role_times.items()
        ]


def _evidence_files_present(
    session_dir: Path,
    windows: list[WindowRecord],
    events: list[CandidateEvent],
) -> bool:
    windows_present = all(
        (session_dir / evidence.path).is_file()
        for record in windows
        for evidence in record.evidence
    )
    events_present = all(
        (session_dir / evidence.path).is_file() for record in events for evidence in record.evidence
    )
    return windows_present and events_present


def _write_vlm_requests(
    session_dir: Path, events: list[CandidateEvent], config: ProjectConfig
) -> None:
    rows = []
    for event in events:
        rows.append(
            {
                "schema_version": "0.1.0",
                "request_id": f"vlm_{event.event_id}",
                "event_id": event.event_id,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": event_prompt(event),
                "image_paths_in_order": [item.path for item in event.evidence],
                "image_roles_in_order": [item.role for item in event.evidence],
                "privacy": {
                    "redact_likely_identifiers": config.privacy.redact_likely_identifiers,
                    "do_not_transcribe_patient_or_plan_identity": True,
                },
            }
        )
    atomic_write_jsonl(session_dir / "vlm_requests.jsonl", rows)


def _write_manifest(
    session_dir: Path,
    metadata: VideoMetadata,
    config: ProjectConfig,
    windows: list[WindowRecord],
    events: list[CandidateEvent],
    actions: list[ActionRecord],
) -> None:
    manifest = {
        "schema_version": "0.1.0",
        "session_id": session_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source": metadata.model_dump(mode="json"),
        "source_filename_stored": False,
        "config": config.model_dump(mode="json"),
        "config_sha256": config.content_hash,
        "software": {"name": "eclipse-trajectory-tracking", "version": __version__},
        "counts": {
            "windows": len(windows),
            "candidate_events": len(events),
            "actions": len(actions),
            "unique_evidence_images": len(list((session_dir / "evidence").glob("*.jpg"))),
        },
        "implemented_backends": ["deterministic", "local_openai_compatible"],
        "active_backend": "deterministic",
        "ontology_version": "0.1.0",
        "clinical_rationale_policy": "always_null_unless_future_explicit_evidence_schema",
    }
    atomic_write_json(session_dir / "manifest.json", manifest)


def _mark_active_model(session_dir: Path, provenance: dict[str, object], run_id: str) -> None:
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["active_backend"] = "local_openai_compatible"
    manifest["active_model_run_id"] = run_id
    manifest["active_model_provenance"] = provenance
    atomic_write_json(manifest_path, manifest)
