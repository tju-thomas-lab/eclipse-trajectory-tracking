from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from PIL import Image

from eclipse_trajectory.config import ProjectConfig
from eclipse_trajectory.ingest.video import iter_video_frames
from eclipse_trajectory.schemas import ChangedRegion, FrameMetric, VideoMetadata


def calculate_frame_metrics(
    video_path: Path,
    metadata: VideoMetadata,
    config: ProjectConfig,
    destination: Path,
) -> list[FrameMetric]:
    period = 1.0 / config.sampling.frames_per_second
    next_sample = 0.0
    previous: NDArray[np.uint8] | None = None
    previous_hash: NDArray[np.bool_] | None = None
    metrics: list[FrameMetric] = []

    for timestamp, rgb in iter_video_frames(
        video_path, metadata.video_stream_index, metadata.start_time_seconds
    ):
        if timestamp + 1e-6 < next_sample:
            continue
        while next_sample <= timestamp + 1e-6:
            next_sample += period

        gray = _analysis_gray(rgb, config.sampling.analysis_width)
        current_hash = _dhash_bits(gray)
        if previous is None or previous_hash is None:
            metric = FrameMetric(
                sample_index=0,
                timestamp_seconds=round(timestamp, 6),
                mean_absolute_change=0.0,
                changed_fraction=0.0,
                structural_similarity_approx=1.0,
                dhash_distance=0,
                change_score=0.0,
                changed_region=None,
            )
        else:
            absolute = np.abs(gray.astype(np.int16) - previous.astype(np.int16))
            mean_change = float(absolute.mean() / 255.0)
            mask = absolute >= config.detection.pixel_threshold
            changed_fraction = float(mask.mean())
            hash_distance = int(np.count_nonzero(current_hash != previous_hash))
            region = _bounding_region(mask)
            change_score = (
                0.55 * mean_change + 0.35 * changed_fraction + 0.10 * min(hash_distance / 32.0, 1.0)
            )
            metric = FrameMetric(
                sample_index=len(metrics),
                timestamp_seconds=round(timestamp, 6),
                mean_absolute_change=mean_change,
                changed_fraction=changed_fraction,
                structural_similarity_approx=max(0.0, 1.0 - mean_change),
                dhash_distance=hash_distance,
                change_score=change_score,
                changed_region=region,
            )
        metrics.append(metric)
        previous = gray
        previous_hash = current_hash

    _write_parquet(destination, metrics)
    return metrics


def read_frame_metrics(path: Path) -> list[FrameMetric]:
    rows = pq.read_table(path).to_pylist()
    result: list[FrameMetric] = []
    for row in rows:
        region_values = row.pop("changed_region")
        row["changed_region"] = (
            ChangedRegion.model_validate(region_values) if region_values is not None else None
        )
        result.append(FrameMetric.model_validate(row))
    return result


def _analysis_gray(rgb: NDArray[np.uint8], width: int) -> NDArray[np.uint8]:
    source = Image.fromarray(rgb)
    height = max(1, round(source.height * width / source.width))
    return np.asarray(source.resize((width, height), Image.Resampling.BILINEAR).convert("L"))


def _dhash_bits(gray: NDArray[np.uint8]) -> NDArray[np.bool_]:
    image = Image.fromarray(gray).resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(image)
    return values[:, 1:] > values[:, :-1]


def _bounding_region(mask: NDArray[np.bool_]) -> ChangedRegion | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return ChangedRegion(
        x=left / width,
        y=top / height,
        width=(right - left) / width,
        height=(bottom - top) / height,
    )


def _write_parquet(path: Path, metrics: list[FrameMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.model_dump(mode="json") for item in metrics]
    table = pa.Table.from_pylist(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)
