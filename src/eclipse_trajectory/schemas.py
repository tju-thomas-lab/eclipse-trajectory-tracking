from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1.0"
NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]
NormalizedPoint = tuple[NormalizedCoordinate, NormalizedCoordinate]

PrimitiveActionType = Literal[
    "move",
    "hover",
    "left_click",
    "double_click",
    "right_click",
    "click_then_type",
    "type_text",
    "press_key",
    "keyboard_shortcut",
    "scroll",
    "drag",
    "select",
    "open",
    "close",
    "confirm",
    "cancel",
    "wait",
    "unknown_action",
    "no_action",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoMetadata(StrictModel):
    source_sha256: str
    container_format: str | None
    duration_seconds: float
    size_bytes: int
    bit_rate: int | None
    video_stream_index: int
    codec: str
    width: int
    height: int
    average_frame_rate: float | None
    time_base: float
    start_time_seconds: float


class ChangedRegion(StrictModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


class FrameMetric(StrictModel):
    sample_index: int
    timestamp_seconds: float
    mean_absolute_change: float = Field(ge=0.0, le=1.0)
    changed_fraction: float = Field(ge=0.0, le=1.0)
    structural_similarity_approx: float = Field(ge=0.0, le=1.0)
    dhash_distance: int = Field(ge=0)
    change_score: float = Field(ge=0.0)
    changed_region: ChangedRegion | None = None


class EvidenceRef(StrictModel):
    role: Literal["before", "interaction", "after"]
    frame_id: str
    path: str
    sha256: str
    requested_timestamp_seconds: float
    actual_timestamp_seconds: float
    media_type: str


class WindowRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    window_id: str
    start_time_seconds: float
    end_time_seconds: float
    max_change_score: float
    mean_change_score: float
    candidate: bool
    detection_reasons: list[str]
    strongest_changed_region: ChangedRegion | None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class CandidateEvent(StrictModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    event_id: str
    start_time_seconds: float
    interaction_time_seconds: float
    end_time_seconds: float
    max_change_score: float
    detection_reasons: list[str]
    strongest_changed_region: ChangedRegion | None
    overlapping_window_ids: list[str]
    evidence: list[EvidenceRef] = Field(default_factory=list)


class PrimitiveAction(StrictModel):
    type: PrimitiveActionType
    coordinates_normalized: NormalizedPoint | None = None
    text: str | None = None
    key_or_shortcut: str | None = None
    drag_start_normalized: NormalizedPoint | None = None
    drag_end_normalized: NormalizedPoint | None = None
    scroll_direction: Literal["up", "down", "left", "right"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alternatives: list[str] = Field(default_factory=list)


class SemanticAction(StrictModel):
    application: str | None = None
    workspace: str | None = None
    visible_dialog: str | None = None
    selected_object: str | None = None
    target: str | None = None
    description: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ActionEvidence(StrictModel):
    frames: list[EvidenceRef]
    changed_regions: list[ChangedRegion]
    inference_sources: list[str]


class ActionRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    action_id: str
    source_event_id: str
    start_time_seconds: float
    end_time_seconds: float
    primitive_action: PrimitiveAction
    semantic_action: SemanticAction
    low_level_instruction: str | None
    high_level_summary: str | None = None
    episode_label: str | None = None
    context_transition: Literal["same", "new", "unknown"] = "unknown"
    state_before: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    clinical_intent: None = None
    evidence: ActionEvidence
    field_sources: dict[str, str | None]
    model_provenance: dict[str, Any]
    ontology_version: str
    review_status: Literal["unreviewed", "accepted", "rejected", "flagged"] = "unreviewed"


class VLMNormalizedOutput(StrictModel):
    primitive_action_type: PrimitiveActionType
    coordinates_normalized: NormalizedPoint | None = None
    typed_text: str | None = None
    key_or_shortcut: str | None = None
    ui_target: str | None = None
    application: str | None = None
    workspace: str | None = None
    visible_dialog: str | None = None
    selected_object: str | None = None
    semantic_action: str | None = None
    visible_value_before: str | None = None
    visible_value_after: str | None = None
    low_level_instruction: str | None = None
    high_level_summary: str | None = None
    episode_label: str | None = None
    context_transition: Literal["same", "new", "unknown"] = "unknown"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alternative_interpretations: list[str] = Field(default_factory=list)
    evidence_roles: list[Literal["before", "interaction", "after"]] = Field(default_factory=list)
    inference_sources: list[str] = Field(default_factory=list)
    clinical_intent: None = None


class StageMarker(StrictModel):
    stage: str
    signature: str
    completed_at_utc: str
    outputs: list[str]
