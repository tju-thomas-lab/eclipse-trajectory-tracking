from __future__ import annotations

import json

from eclipse_trajectory.schemas import CandidateEvent

SYSTEM_PROMPT = """You analyze silent screen-recording evidence from Varian Eclipse for retrospective
research. Report only operations visibly supported by the ordered frames. Distinguish observation
from inference. Never invent clicks, keystrokes, coordinates, values, UI targets, patient/plan
identity, or clinical rationale. If evidence is insufficient, use null and unknown_action. The three
images are before, interaction, and after frames for a visually detected candidate event. A patient
or plan context change must be reported only as anonymous context_transition=new; never transcribe
an identifier. Return one JSON object and no markdown."""


def event_prompt(event: CandidateEvent) -> str:
    contract = {
        "primitive_action_type": "one configured primitive action; unknown_action is valid",
        "coordinates_normalized": None,
        "typed_text": None,
        "key_or_shortcut": None,
        "ui_target": None,
        "application": None,
        "workspace": None,
        "visible_dialog": None,
        "selected_object": None,
        "semantic_action": None,
        "visible_value_before": None,
        "visible_value_after": None,
        "low_level_instruction": None,
        "high_level_summary": None,
        "episode_label": "configured ontology label or unknown_episode",
        "context_transition": "same, new, or unknown",
        "confidence": None,
        "alternative_interpretations": [],
        "evidence_roles": ["before", "interaction", "after"],
        "inference_sources": ["visual_state_change", "local_vlm"],
        "clinical_intent": None,
    }
    observed = {
        "event_id": event.event_id,
        "start_time_seconds": event.start_time_seconds,
        "interaction_time_seconds": event.interaction_time_seconds,
        "end_time_seconds": event.end_time_seconds,
        "detection_reasons": event.detection_reasons,
        "changed_region_normalized": (
            event.strongest_changed_region.model_dump(mode="json")
            if event.strongest_changed_region is not None
            else None
        ),
    }
    return (
        "Analyze this candidate event. The images following this text are in temporal order.\n"
        f"Measured event metadata: {json.dumps(observed, sort_keys=True)}\n"
        f"Return this JSON contract: {json.dumps(contract, sort_keys=True)}"
    )
