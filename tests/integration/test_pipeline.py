from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np

from eclipse_trajectory.config import ProjectConfig
from eclipse_trajectory.pipeline import run_pipeline
from eclipse_trajectory.util import read_jsonl


def _synthetic_video(path: Path) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=10)
    stream.width = 320
    stream.height = 180
    stream.pix_fmt = "yuv420p"
    for frame_index in range(60):
        image = np.full((180, 320, 3), 32, dtype=np.uint8)
        if frame_index >= 20:
            image[30:100, 30:150] = (40, 170, 220)
        if frame_index >= 40:
            image[90:160, 160:300] = (210, 100, 40)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_synthetic_video_end_to_end_and_resume(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    _synthetic_video(video)
    config = ProjectConfig.model_validate(
        {
            "sampling": {"frames_per_second": 2.0, "analysis_width": 160},
            "detection": {"change_threshold": 0.01},
            "evidence": {"max_dimension": 320, "jpeg_quality": 75},
            "runtime": {"output_root": tmp_path / "output"},
        }
    )

    session_dir = run_pipeline(video, config)
    metrics_marker_before = (session_dir / "stages" / "metrics.json").read_text(encoding="utf-8")
    session_dir_again = run_pipeline(video, config)

    assert session_dir_again == session_dir
    assert (session_dir / "stages" / "metrics.json").read_text(
        encoding="utf-8"
    ) == metrics_marker_before
    assert (session_dir / "frame_metrics.parquet").stat().st_size > 0
    assert len(list(read_jsonl(session_dir / "windows.jsonl"))) == 3
    assert len(list(read_jsonl(session_dir / "candidate_events.jsonl"))) >= 2
    assert len(list(read_jsonl(session_dir / "actions.jsonl"))) >= 2
    assert len(list(read_jsonl(session_dir / "vlm_requests.jsonl"))) >= 2
    assert list((session_dir / "evidence").glob("*.jpg"))
    assert "Protected Health Information" in (session_dir / "timeline.html").read_text(
        encoding="utf-8"
    )
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_filename_stored"] is False
    assert manifest["active_backend"] == "deterministic"
