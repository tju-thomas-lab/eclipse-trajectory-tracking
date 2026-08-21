from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from eclipse_trajectory.models.local_vlm import LocalOpenAICompatibleBackend, _parse_json_object
from eclipse_trajectory.schemas import CandidateEvent, EvidenceRef


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8000/v1",
        "http://example.com/v1",
        "http://user:secret@localhost:8000/v1",
    ],
)
def test_backend_rejects_nonlocal_or_credentialed_endpoints(endpoint: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LocalOpenAICompatibleBackend(endpoint, "test-model", tmp_path)


def test_backend_accepts_loopback_and_parses_fenced_json(tmp_path: Path) -> None:
    LocalOpenAICompatibleBackend("http://127.0.0.1:8000/v1", "test-model", tmp_path)
    assert _parse_json_object('```json\n{"primitive_action_type":"unknown_action"}\n```') == {
        "primitive_action_type": "unknown_action"
    }


def test_backend_sends_images_to_local_openai_compatible_server(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "synthetic.jpg").write_bytes(b"synthetic-image-bytes")
    evidence = EvidenceRef(
        role="interaction",
        frame_id="frame_synthetic",
        path="evidence/synthetic.jpg",
        sha256="0" * 64,
        requested_timestamp_seconds=1.0,
        actual_timestamp_seconds=1.0,
        media_type="image/jpeg",
    )
    event = CandidateEvent(
        session_id="session_synthetic",
        event_id="event_000000",
        start_time_seconds=0.5,
        interaction_time_seconds=1.0,
        end_time_seconds=1.5,
        max_change_score=0.2,
        detection_reasons=["global_visual_change"],
        strongest_changed_region=None,
        overlapping_window_ids=["window_000000"],
        evidence=[evidence],
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            assert request["model"] == "synthetic-model"
            image_url = request["messages"][1]["content"][1]["image_url"]["url"]
            assert image_url.startswith("data:image/jpeg;base64,")
            response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"primitive_action_type": "unknown_action"})
                        }
                    }
                ]
            }
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = LocalOpenAICompatibleBackend(
            f"http://127.0.0.1:{server.server_port}/v1", "synthetic-model", tmp_path
        )
        result = backend.infer(event)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.primitive_action_type == "unknown_action"
