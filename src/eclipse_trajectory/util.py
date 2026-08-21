from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from eclipse_trajectory.schemas import StageMarker


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object in {path}")
                yield value


def write_stage_marker(session_dir: Path, stage: str, signature: str, outputs: list[str]) -> None:
    marker = StageMarker(
        stage=stage,
        signature=signature,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        outputs=outputs,
    )
    atomic_write_json(session_dir / "stages" / f"{stage}.json", marker)


def stage_is_current(session_dir: Path, stage: str, signature: str) -> bool:
    marker_path = session_dir / "stages" / f"{stage}.json"
    if not marker_path.exists():
        return False
    try:
        marker = StageMarker.model_validate_json(marker_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    return marker.signature == signature and all(
        (session_dir / item).exists() for item in marker.outputs
    )
