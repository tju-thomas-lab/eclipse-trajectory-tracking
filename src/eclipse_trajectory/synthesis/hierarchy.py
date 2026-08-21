from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eclipse_trajectory.schemas import ActionRecord
from eclipse_trajectory.util import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_hash,
    read_jsonl,
)


def synthesize_hierarchy(
    session_dir: Path,
    executive_summary_override: str | None = None,
    additional_limitations: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions = [
        ActionRecord.model_validate(item) for item in read_jsonl(session_dir / "actions.jsonl")
    ]
    override_path = session_dir / "synthesis_overrides.json"
    if executive_summary_override is None and override_path.exists():
        override = json.loads(override_path.read_text(encoding="utf-8"))
        action_signature = canonical_hash([item.model_dump(mode="json") for item in actions])
        if override.get("actions_signature") == action_signature:
            candidate_summary = override.get("executive_summary")
            if isinstance(candidate_summary, str):
                executive_summary_override = candidate_summary
            candidate_limitations = override.get("limitations")
            if additional_limitations is None and isinstance(candidate_limitations, list):
                additional_limitations = [
                    item for item in candidate_limitations if isinstance(item, str)
                ]
    episodes: list[dict[str, Any]] = []
    context_index = 1
    current: dict[str, Any] | None = None

    for action in actions:
        if action.context_transition == "new" and current is not None:
            context_index += 1
            current = None
        label = action.episode_label or "unknown_episode"
        if current is None or current["label"] != label:
            current = {
                "schema_version": "0.1.0",
                "session_id": action.session_id,
                "episode_id": f"episode_{len(episodes):06d}",
                "anonymous_context_id": f"context_{context_index:04d}",
                "label": label,
                "start_time_seconds": action.start_time_seconds,
                "end_time_seconds": action.end_time_seconds,
                "action_ids": [],
                "step_instructions": [],
                "summary": None,
                "evidence_action_ids": [],
                "unresolved_observations": [],
            }
            episodes.append(current)
        current["end_time_seconds"] = action.end_time_seconds
        current["action_ids"].append(action.action_id)
        if action.low_level_instruction:
            current["step_instructions"].append(action.low_level_instruction)
        if action.high_level_summary:
            prior = current["summary"]
            if prior is None:
                current["summary"] = action.high_level_summary
            elif action.high_level_summary not in prior:
                current["summary"] = f"{prior} {action.high_level_summary}"
            current["evidence_action_ids"].append(action.action_id)
        if action.primitive_action.type == "unknown_action":
            current["unresolved_observations"].append(action.action_id)

    supported_summaries = [item["summary"] for item in episodes if item["summary"]]
    session_id = actions[0].session_id if actions else session_dir.name
    session = {
        "schema_version": "0.1.0",
        "session_id": session_id,
        "anonymous_context_count": context_index if actions else 0,
        "episode_ids": [item["episode_id"] for item in episodes],
        "executive_summary": executive_summary_override
        or (" ".join(supported_summaries) if supported_summaries else None),
        "executive_summary_evidence_episode_ids": [
            item["episode_id"] for item in episodes if item["summary"]
        ],
        "limitations": [
            "Clinical rationale is not inferred from screen changes.",
            "An absent executive summary means no local model supplied evidence-grounded summaries.",
            "Automated output requires qualified review before it can be treated as ground truth.",
        ]
        + (additional_limitations or []),
    }
    atomic_write_jsonl(session_dir / "episodes.jsonl", episodes)
    atomic_write_json(session_dir / "session.json", session)
    return episodes, session
