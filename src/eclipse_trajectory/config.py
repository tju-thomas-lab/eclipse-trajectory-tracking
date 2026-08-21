from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplingConfig(ConfigModel):
    frames_per_second: float = Field(default=2.0, gt=0.0, le=30.0)
    analysis_width: int = Field(default=384, ge=64, le=1920)


class WindowsConfig(ConfigModel):
    duration_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    stride_seconds: float = Field(default=2.5, gt=0.0, le=60.0)


class DetectionConfig(ConfigModel):
    change_threshold: float = Field(default=0.035, gt=0.0, le=1.0)
    pixel_threshold: int = Field(default=18, ge=1, le=255)
    event_merge_gap_seconds: float = Field(default=1.25, ge=0.0, le=30.0)
    event_padding_seconds: float = Field(default=0.5, ge=0.0, le=10.0)
    event_max_duration_seconds: float = Field(default=5.0, gt=0.0, le=120.0)


class EvidenceConfig(ConfigModel):
    max_dimension: int = Field(default=1280, ge=256, le=7680)
    image_format: Literal["jpeg"] = "jpeg"
    jpeg_quality: int = Field(default=84, ge=40, le=100)


class PrivacyConfig(ConfigModel):
    redact_likely_identifiers: bool = True
    log_visible_text: bool = False


class RuntimeConfig(ConfigModel):
    output_root: Path = Path("output")
    overwrite: bool = False


class NetworkConfig(ConfigModel):
    mode: Literal["offline", "local_only"] = "offline"
    allowed_inference_hosts: list[str] = Field(
        default_factory=lambda: ["localhost", "127.0.0.1", "::1"]
    )


class ProjectConfig(ConfigModel):
    version: str = "0.1.0"
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    windows: WindowsConfig = Field(default_factory=WindowsConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: Path) -> ProjectConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ProjectConfig.model_validate(raw)
