from __future__ import annotations

import pytest
from pydantic import ValidationError

from eclipse_trajectory.schemas import ChangedRegion, PrimitiveAction, VLMNormalizedOutput


def test_normalized_coordinates_and_confidence_are_bounded() -> None:
    ChangedRegion(x=0.1, y=0.2, width=0.5, height=0.4)
    PrimitiveAction(type="left_click", coordinates_normalized=(0.5, 0.5), confidence=0.8)
    with pytest.raises(ValidationError):
        ChangedRegion(x=-0.1, y=0.2, width=0.5, height=0.4)
    with pytest.raises(ValidationError):
        PrimitiveAction(type="left_click", confidence=1.2)
    with pytest.raises(ValidationError):
        PrimitiveAction(type="left_click", coordinates_normalized=(1.1, 0.5))


def test_clinical_intent_cannot_be_populated() -> None:
    with pytest.raises(ValidationError):
        VLMNormalizedOutput(
            primitive_action_type="unknown_action",
            clinical_intent="to improve a clinical result",  # type: ignore[arg-type]
        )
