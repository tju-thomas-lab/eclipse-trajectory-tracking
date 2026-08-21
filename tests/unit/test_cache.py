from __future__ import annotations

from pathlib import Path

from eclipse_trajectory.util import stage_is_current, write_stage_marker


def test_stage_marker_requires_matching_signature_and_outputs(tmp_path: Path) -> None:
    (tmp_path / "artifact.jsonl").write_text("", encoding="utf-8")
    write_stage_marker(tmp_path, "test", "signature-a", ["artifact.jsonl"])
    assert stage_is_current(tmp_path, "test", "signature-a")
    assert not stage_is_current(tmp_path, "test", "signature-b")
    (tmp_path / "artifact.jsonl").unlink()
    assert not stage_is_current(tmp_path, "test", "signature-a")
