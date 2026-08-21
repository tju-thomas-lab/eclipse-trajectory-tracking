from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import av
import numpy as np
from av.video.stream import VideoStream
from numpy.typing import NDArray

from eclipse_trajectory.schemas import VideoMetadata
from eclipse_trajectory.util import sha256_file

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov"}


def inspect_video(path: Path) -> VideoMetadata:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported video extension {path.suffix!r}; expected {supported}")

    source_hash = sha256_file(path)
    with av.open(str(path), mode="r") as container:
        try:
            stream = cast(
                VideoStream, next(item for item in container.streams if item.type == "video")
            )
        except StopIteration as exc:
            raise ValueError(f"No video stream found in {path}") from exc

        if stream.time_base is None:
            raise ValueError(f"Video stream in {path} has no time base")
        time_base = float(stream.time_base)
        start_time = (
            float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        )
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            duration = _scan_duration(path, stream.index, start_time)

        average_rate = float(stream.average_rate) if stream.average_rate is not None else None
        container_format = container.format.name if container.format is not None else None
        codec_context = stream.codec_context
        codec = codec_context.name or "unknown"
        bit_rate = int(container.bit_rate) if container.bit_rate is not None else None

        return VideoMetadata(
            source_sha256=source_hash,
            container_format=container_format,
            duration_seconds=max(0.0, duration),
            size_bytes=path.stat().st_size,
            bit_rate=bit_rate,
            video_stream_index=stream.index,
            codec=codec,
            width=int(codec_context.width),
            height=int(codec_context.height),
            average_frame_rate=average_rate,
            time_base=time_base,
            start_time_seconds=start_time,
        )


def _scan_duration(path: Path, stream_index: int, start_time: float) -> float:
    last_timestamp = start_time
    with av.open(str(path), mode="r") as container:
        stream = cast(
            VideoStream, next(item for item in container.streams if item.index == stream_index)
        )
        if stream.time_base is None:
            raise ValueError("Video stream has no time base")
        for decoded in container.decode(stream):
            frame = decoded
            if frame.pts is not None:
                last_timestamp = float(frame.pts * stream.time_base)
    return max(0.0, last_timestamp - start_time)


def iter_video_frames(
    path: Path, stream_index: int, start_time_seconds: float
) -> Iterator[tuple[float, NDArray[np.uint8]]]:
    """Yield relative source timestamps and RGB frames using bounded memory."""
    with av.open(str(path), mode="r") as container:
        stream = cast(
            VideoStream, next(item for item in container.streams if item.index == stream_index)
        )
        if stream.time_base is None:
            raise ValueError("Video stream has no time base")
        stream.thread_type = "AUTO"
        for decoded in container.decode(stream):
            frame = decoded
            if frame.pts is None:
                continue
            absolute_timestamp = float(frame.pts * stream.time_base)
            relative_timestamp = max(0.0, absolute_timestamp - start_time_seconds)
            rgb = cast(NDArray[np.uint8], frame.to_ndarray(format="rgb24"))
            yield relative_timestamp, rgb
