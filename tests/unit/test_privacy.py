from __future__ import annotations

from eclipse_trajectory.privacy.redact import suppress_likely_identifiers
from eclipse_trajectory.schemas import VLMNormalizedOutput


def test_identifier_target_suppresses_values() -> None:
    output = VLMNormalizedOutput(
        primitive_action_type="type_text",
        ui_target="patient_id",
        typed_text="synthetic-123",
        visible_value_after="synthetic-123",
        low_level_instruction="Set patient ID: synthetic-123",
    )
    redacted = suppress_likely_identifiers(output)
    assert redacted.typed_text is None
    assert redacted.visible_value_after is None
    assert "synthetic-123" not in (redacted.low_level_instruction or "")
