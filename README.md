# Eclipse Trajectory Tracking

Local-first tooling for turning long, silent Varian Eclipse screen recordings into an
evidence-linked dataset that can be interpreted by a **local** multimodal model.

> **Protected health information warning:** recordings, screenshots, model responses, and
> summaries may contain patient identifiers. Keep the input and output directories on approved
> encrypted storage. This project never uploads recordings and accepts inference connections only
> to loopback addresses, but it cannot make an unmanaged workstation compliant by itself.

This software is for retrospective research analysis. It does not control Eclipse, approve plans,
modify patient records, or interact with treatment-delivery systems.

## What works in the first release

- Streams MP4, MKV, and MOV with PyAV; the whole video is never loaded into memory.
- Preserves source timing and records codec, resolution, frame rate, duration, hashes, and config.
- Creates 5-second rolling windows at a configurable 2.5-second stride.
- Measures visual change at a configurable sampling rate and writes `frame_metrics.parquet`.
- Detects event boundaries independently of the rolling windows.
- Extracts before/interaction/after evidence as content-addressed JPEG files shared by all records.
- Emits deterministic `unknown_action` records rather than inventing user actions.
- Creates local-VLM request records and can call an OpenAI-compatible server on localhost only.
- Produces JSONL, a session manifest, synthesis placeholders, and a self-contained local timeline.
- Resumes completed stages when the video and configuration hashes match.

The deterministic baseline is an extraction and handoff pipeline, not an expert action recognizer.
Expert semantic labels and workflow summaries require a configured local vision-language model and
human validation before use as ground truth. Eclipse performance has not yet been measured.

## Setup

Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) are recommended. PyAV wheels provide the video
decoding libraries, so a separate system FFmpeg installation is not required.

```powershell
uv sync --extra dev
uv run eclipse-trajectory inspect 2026-08-03_13-57-29.mp4
uv run eclipse-trajectory run 2026-08-03_13-57-29.mp4 --config configs/offline.yaml
```

The run command prints only the generated session directory—never OCR or visible screen text.

## Output layout

```text
output/session_<video-hash>/
├── evidence/                  # content-addressed screenshots
├── stages/                    # resumability markers
├── actions.jsonl              # deterministic or normalized model actions
├── candidate_events.jsonl     # event boundaries independent of windows
├── episodes.jsonl             # episode synthesis output/placeholders
├── evidence_index.jsonl       # requested-to-actual timestamps and image hashes
├── frame_metrics.parquet      # dense, downscaled visual metrics
├── manifest.json              # provenance and preprocessing configuration
├── review_corrections.jsonl   # append-only reviewer layer
├── session.json               # session/executive synthesis output
├── timeline.html              # local evidence browser
├── vlm_requests.jsonl         # portable local-model work queue
└── windows.jsonl              # every overlapping 5-second model window
```

Images are stored once and referenced by relative path. `vlm_requests.jsonl` contains prompts,
timestamps, and image paths rather than duplicated base64 data. The inference command converts
each image to a data URL only in memory for a localhost request.

## Local multimodal inference

Start an OpenAI-compatible multimodal server bound to loopback, then run:

```powershell
uv run eclipse-trajectory infer-actions output/session_<hash> `
  --backend local-openai-compatible `
  --endpoint http://127.0.0.1:8000/v1 `
  --model YOUR_LOCAL_MODEL
uv run eclipse-trajectory synthesize output/session_<hash>
```

Only `localhost`, `127.0.0.1`, and `::1` endpoints are accepted. Raw responses are preserved in
`model_outputs.jsonl`; normalized actions remain separate so provenance is auditable. The baseline
prompt directs the model to return null/unknown when visual evidence is insufficient and to omit
patient/plan identifiers when privacy redaction is enabled.

Useful commands:

```text
eclipse-trajectory inspect INPUT_VIDEO
eclipse-trajectory preprocess INPUT_VIDEO --config configs/offline.yaml
eclipse-trajectory detect SESSION_DIRECTORY
eclipse-trajectory infer-actions SESSION_DIRECTORY --backend deterministic
eclipse-trajectory synthesize SESSION_DIRECTORY
eclipse-trajectory review SESSION_DIRECTORY
eclipse-trajectory export SESSION_DIRECTORY --format jsonl
eclipse-trajectory run INPUT_VIDEO --config configs/offline.yaml
```

`review` opens (or prints the path to) the local timeline. Reviewer edits are intentionally not yet
implemented in the static baseline; `review_corrections.jsonl` is reserved as an append-only layer.
See [architecture](docs/architecture.md), [data schema](docs/data_schema.md), [security](docs/security.md),
and [validation](docs/validation.md).

## Development

```powershell
uv run pytest
uv run ruff check .
uv run mypy src
```

