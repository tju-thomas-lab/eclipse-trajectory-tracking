from __future__ import annotations

from typing import Protocol

from eclipse_trajectory.schemas import CandidateEvent, VLMNormalizedOutput


class ActionInferenceBackend(Protocol):
    @property
    def provenance(self) -> dict[str, object]: ...

    def infer(self, event: CandidateEvent) -> VLMNormalizedOutput: ...
