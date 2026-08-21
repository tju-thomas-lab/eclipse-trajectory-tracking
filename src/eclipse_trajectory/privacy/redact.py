from __future__ import annotations

import re

from eclipse_trajectory.schemas import VLMNormalizedOutput

_LABELED_IDENTIFIER = re.compile(
    r"(?i)\b(patient\s*(?:name|id)|mrn|medical\s+record(?:\s+number)?)\s*[:#=]\s*[^,;.\n]+"
)
_IDENTIFIER_TARGET = re.compile(r"(?i)\b(patient[_ ]?(?:name|id)|mrn|medical[_ ]?record)\b")


def suppress_likely_identifiers(output: VLMNormalizedOutput) -> VLMNormalizedOutput:
    values = output.model_dump(mode="python")
    for field in (
        "ui_target",
        "application",
        "workspace",
        "visible_dialog",
        "selected_object",
        "semantic_action",
        "visible_value_before",
        "visible_value_after",
        "low_level_instruction",
        "high_level_summary",
    ):
        value = values.get(field)
        if isinstance(value, str):
            values[field] = _LABELED_IDENTIFIER.sub(r"\1: [REDACTED_IDENTIFIER]", value)
    target = values.get("ui_target")
    if isinstance(target, str) and _IDENTIFIER_TARGET.search(target):
        values["typed_text"] = None
        values["visible_value_before"] = None
        values["visible_value_after"] = None
    return VLMNormalizedOutput.model_validate(values)
