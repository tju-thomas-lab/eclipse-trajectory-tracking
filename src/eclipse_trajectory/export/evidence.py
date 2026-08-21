from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from eclipse_trajectory.config import EvidenceConfig
from eclipse_trajectory.ingest.video import iter_video_frames
from eclipse_trajectory.schemas import EvidenceRef, VideoMetadata


@dataclass(frozen=True)
class ExtractedFrame:
    frame_id: str
    path: str
    sha256: str
    actual_timestamp_seconds: float
    media_type: str


def extract_evidence_frames(
    video_path: Path,
    metadata: VideoMetadata,
    requested_timestamps: list[float],
    session_dir: Path,
    config: EvidenceConfig,
) -> dict[float, ExtractedFrame]:
    targets = sorted({round(max(0.0, item), 6) for item in requested_timestamps})
    if not targets:
        return {}
    evidence_dir = session_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[float, ExtractedFrame] = {}
    actual_cache: dict[float, ExtractedFrame] = {}
    target_index = 0
    previous_timestamp: float | None = None
    previous_rgb: NDArray[np.uint8] | None = None

    for timestamp, rgb in iter_video_frames(
        video_path, metadata.video_stream_index, metadata.start_time_seconds
    ):
        while target_index < len(targets) and targets[target_index] <= timestamp + 1e-9:
            target = targets[target_index]
            if (
                previous_timestamp is not None
                and previous_rgb is not None
                and abs(target - previous_timestamp) <= abs(timestamp - target)
            ):
                selected_timestamp, selected_rgb = previous_timestamp, previous_rgb
            else:
                selected_timestamp, selected_rgb = timestamp, rgb
            cache_key = round(selected_timestamp, 6)
            extracted = actual_cache.get(cache_key)
            if extracted is None:
                extracted = _store_image(selected_rgb, selected_timestamp, evidence_dir, config)
                actual_cache[cache_key] = extracted
            resolved[target] = extracted
            target_index += 1
        previous_timestamp, previous_rgb = timestamp, rgb
        if target_index >= len(targets):
            break

    if target_index < len(targets):
        if previous_timestamp is None or previous_rgb is None:
            raise ValueError("The video did not yield any decodable frames")
        cache_key = round(previous_timestamp, 6)
        extracted = actual_cache.get(cache_key)
        if extracted is None:
            extracted = _store_image(previous_rgb, previous_timestamp, evidence_dir, config)
        for target in targets[target_index:]:
            resolved[target] = extracted
    return resolved


def as_evidence_ref(role: str, requested: float, extracted: ExtractedFrame) -> EvidenceRef:
    if role not in {"before", "interaction", "after"}:
        raise ValueError(f"Unexpected evidence role: {role}")
    return EvidenceRef.model_validate(
        {
            "role": role,
            "frame_id": extracted.frame_id,
            "path": extracted.path,
            "sha256": extracted.sha256,
            "requested_timestamp_seconds": round(requested, 6),
            "actual_timestamp_seconds": round(extracted.actual_timestamp_seconds, 6),
            "media_type": extracted.media_type,
        }
    )


def _store_image(
    rgb: NDArray[np.uint8],
    timestamp: float,
    evidence_dir: Path,
    config: EvidenceConfig,
) -> ExtractedFrame:
    image = Image.fromarray(rgb)
    longest = max(image.size)
    if longest > config.max_dimension:
        scale = config.max_dimension / longest
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=config.jpeg_quality,
        optimize=True,
        progressive=True,
        subsampling=0,
    )
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    frame_id = f"frame_{digest[:16]}"
    name = f"{digest}.jpg"
    destination = evidence_dir / name
    if not destination.exists():
        temporary = destination.with_suffix(".jpg.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    return ExtractedFrame(
        frame_id=frame_id,
        path=(Path("evidence") / name).as_posix(),
        sha256=digest,
        actual_timestamp_seconds=timestamp,
        media_type="image/jpeg",
    )
