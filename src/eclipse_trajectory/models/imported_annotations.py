from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eclipse_trajectory import __version__
from eclipse_trajectory.export.timeline import write_timeline
from eclipse_trajectory.schemas import (
    ActionEvidence,
    ActionRecord,
    CandidateEvent,
    ChangedRegion,
    EvidenceRef,
    ImportedActionAnnotation,
    ImportedAnnotationBundle,
    PrimitiveAction,
    SemanticAction,
    WindowRecord,
)
from eclipse_trajectory.synthesis.hierarchy import synthesize_hierarchy
from eclipse_trajectory.util import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_hash,
    read_jsonl,
)


def import_annotation_bundle(session_dir: Path, bundle_path: Path) -> list[ActionRecord]:
    """Normalize an evidence-linked annotation bundle into the standard action artifacts."""
    session_dir = session_dir.resolve()
    bundle = ImportedAnnotationBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    if bundle.session_id != session_dir.name:
        raise ValueError(
            f"Annotation session_id {bundle.session_id!r} does not match {session_dir.name!r}"
        )

    events = {
        item.event_id: item
        for item in (
            CandidateEvent.model_validate(row)
            for row in read_jsonl(session_dir / "candidate_events.jsonl")
        )
    }
    windows = {
        item.window_id: item
        for item in (
            WindowRecord.model_validate(row) for row in read_jsonl(session_dir / "windows.jsonl")
        )
    }
    bundle_payload = bundle.model_dump(mode="json")
    run_id = canonical_hash(bundle_payload)[:16]
    actions = [
        _annotation_action(index, bundle, annotation, events, windows, run_id)
        for index, annotation in enumerate(bundle.actions)
    ]
    if actions != sorted(
        actions, key=lambda item: (item.start_time_seconds, item.end_time_seconds)
    ):
        raise ValueError("Imported annotations must be ordered by start and end time")

    provenance = _provenance(bundle.generator_label, run_id)
    model_outputs = [
        {
            "run_id": run_id,
            "annotation_id": annotation.annotation_id,
            "raw_response": None,
            "normalized_output": annotation.output.model_dump(mode="json"),
            "source_refs": [item.model_dump(mode="json") for item in annotation.source_refs],
            "provenance": provenance,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for annotation in bundle.actions
    ]
    atomic_write_jsonl(session_dir / "model_outputs.jsonl", model_outputs)
    atomic_write_jsonl(session_dir / "actions.jsonl", actions)
    atomic_write_json(
        session_dir / "synthesis_overrides.json",
        {
            "actions_signature": canonical_hash([item.model_dump(mode="json") for item in actions]),
            "executive_summary": bundle.executive_summary,
            "limitations": bundle.limitations,
            "generator_label": bundle.generator_label,
        },
    )
    synthesize_hierarchy(session_dir)
    write_timeline(session_dir, actions)
    _mark_active_import(session_dir, provenance, run_id, len(actions))
    return actions


def _annotation_action(
    index: int,
    bundle: ImportedAnnotationBundle,
    annotation: ImportedActionAnnotation,
    events: dict[str, CandidateEvent],
    windows: dict[str, WindowRecord],
    run_id: str,
) -> ActionRecord:
    if annotation.end_time_seconds < annotation.start_time_seconds:
        raise ValueError(f"{annotation.annotation_id}: end time precedes start time")

    frames: list[EvidenceRef] = []
    regions: list[ChangedRegion] = []
    source_event_id: str | None = None
    source_window_ids: list[str] = []
    seen_frames: set[tuple[str, str]] = set()
    inference_sources: list[str] = []

    for source in annotation.source_refs:
        if source.record_type == "event":
            try:
                record: CandidateEvent | WindowRecord = events[source.record_id]
            except KeyError as exc:
                raise ValueError(f"Unknown event reference: {source.record_id}") from exc
            if source_event_id is not None and source_event_id != source.record_id:
                raise ValueError(f"{annotation.annotation_id}: only one event source is supported")
            source_event_id = source.record_id
            inference_sources.append("candidate_event_evidence")
        else:
            try:
                record = windows[source.record_id]
            except KeyError as exc:
                raise ValueError(f"Unknown window reference: {source.record_id}") from exc
            source_window_ids.append(source.record_id)
            inference_sources.append("rolling_window_evidence")
        if record.strongest_changed_region and record.strongest_changed_region not in regions:
            regions.append(record.strongest_changed_region)
        for frame in record.evidence:
            key = (frame.frame_id, frame.role)
            if frame.role in source.evidence_roles and key not in seen_frames:
                frames.append(frame)
                seen_frames.add(key)

    if not frames:
        raise ValueError(f"{annotation.annotation_id}: source filters selected no evidence frames")
    output = annotation.output
    inference_sources.extend(output.inference_sources)
    inference_sources.append("imported_evidence_grounded_visual_inference")
    state_before = annotation.state_before
    state_after = annotation.state_after
    if state_before is None and output.visible_value_before is not None:
        state_before = {"visible_value": output.visible_value_before}
    if state_after is None and output.visible_value_after is not None:
        state_after = {"visible_value": output.visible_value_after}

    return ActionRecord(
        session_id=bundle.session_id,
        action_id=f"action_{index:06d}",
        source_event_id=source_event_id,
        source_window_ids=source_window_ids,
        start_time_seconds=annotation.start_time_seconds,
        end_time_seconds=annotation.end_time_seconds,
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
        state_before=state_before,
        state_after=state_after,
        evidence=ActionEvidence(
            frames=frames,
            changed_regions=regions,
            inference_sources=list(dict.fromkeys(inference_sources)),
        ),
        field_sources={
            "primitive_action.type": "imported_visual_inference",
            "semantic_action": "imported_visual_inference",
            "clinical_intent": None,
        },
        model_provenance=_provenance(bundle.generator_label, run_id),
        ontology_version="0.1.0",
    )


def _provenance(generator_label: str, run_id: str | None) -> dict[str, Any]:
    return {
        "backend": "imported_local_vlm",
        "generator_label": generator_label,
        "run_id": run_id,
        "software_version": __version__,
        "eclipse_performance": "unvalidated",
        "review_status": "automated_unreviewed",
    }


def _mark_active_import(
    session_dir: Path, provenance: dict[str, Any], run_id: str, action_count: int
) -> None:
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    implemented = list(manifest.get("implemented_backends", []))
    if "imported_local_vlm" not in implemented:
        implemented.append("imported_local_vlm")
    manifest["implemented_backends"] = implemented
    manifest["active_backend"] = "imported_local_vlm"
    manifest["active_model_run_id"] = run_id
    manifest["active_model_provenance"] = provenance
    manifest.setdefault("counts", {})["actions"] = action_count
    atomic_write_json(manifest_path, manifest)
