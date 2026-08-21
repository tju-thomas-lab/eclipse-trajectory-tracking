from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from eclipse_trajectory.models.prompts import SYSTEM_PROMPT, event_prompt
from eclipse_trajectory.schemas import CandidateEvent, VLMNormalizedOutput


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class LocalOpenAICompatibleBackend:
    def __init__(
        self, endpoint: str, model: str, session_dir: Path, timeout_seconds: float = 180.0
    ):
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Local VLM endpoint must be an http:// loopback address")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Credentials are not permitted in the local VLM endpoint URL")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.session_dir = session_dir
        self.timeout_seconds = timeout_seconds
        self._opener = urllib.request.build_opener(_NoRedirect())
        self.last_raw_text: str | None = None

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "backend": "local_openai_compatible",
            "model": self.model,
            "endpoint_host": urllib.parse.urlsplit(self.endpoint).hostname,
            "checkpoint_revision": None,
            "checkpoint_sha256": None,
            "license": None,
            "eclipse_performance": "unknown",
        }

    def infer(self, event: CandidateEvent) -> VLMNormalizedOutput:
        content: list[dict[str, Any]] = [{"type": "text", "text": event_prompt(event)}]
        for evidence in event.evidence:
            image_path = (self.session_dir / evidence.path).resolve()
            if self.session_dir.resolve() not in image_path.parents:
                raise ValueError("Evidence path escaped the session directory")
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{evidence.media_type};base64,{encoded}"},
                }
            )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Local VLM returned HTTP {exc.code}") from exc
        text = response_data["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("Local VLM response content was not text")
        self.last_raw_text = text
        return VLMNormalizedOutput.model_validate(_parse_json_object(text))


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Local VLM did not return a JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Local VLM response JSON was not an object")
    return value
