# Architecture and revised plan

## Decisions made after reviewing the initial brief

The fixed five-second interval is a **model context window**, not a semantic action boundary. The
pipeline records all five-second windows with overlap while separately grouping frame-change peaks
into candidate events. A later model may merge, split, or reject events without changing ingestion.

Long recordings are handled with two bounded-memory sequential passes:

1. Decode at a sparse analysis cadence, downscale, calculate metrics, and write Parquet.
2. Decode again and extract only the sorted evidence timestamps selected from the metrics.

The second pass avoids random seeks, which are error-prone around variable frame rates and long GOPs.
At most the previous and current decoded frame are retained. Dense metrics—not source frames—may be
loaded for window construction; two samples per second is only 7,200 rows per hour.

Evidence filenames use hashes, not patient names or source filenames. Adjacent rolling windows share
content-addressed images. The manifest links the source only by SHA-256 and technical metadata.

## Data flow

```text
local recording
  -> inspect + SHA-256
  -> sparse visual metrics (streaming decode)
  -> rolling windows + independent candidate events
  -> evidence timestamps
  -> content-addressed screenshots (streaming decode)
  -> deterministic unknown/no-action records
  -> portable VLM queue
  -> optional localhost multimodal inference
  -> overlap reconciliation / episodes / session summary
  -> review layer and exports
```

Stage markers contain the input hash, config hash, code version, and output inventory. A completed
stage is reused only when its signature matches. Atomic replacement is used for manifests and JSONL
files so interrupted writes do not masquerade as completed work.

## Local VLM contract

Each candidate event request contains its temporal bounds, the overlapping model-window IDs, three
ordered evidence images, measured changed regions, and the ontology version. The response is JSON
with observation, inference, and uncertainty fields kept distinct. The model is explicitly told not
to supply clinical rationale and not to infer identifiers. Raw output, normalized output, and future
reviewer/adjudication layers are separate.

The initial localhost backend is deliberately generic so a verified local Qwen-family VLM, vLLM,
llama.cpp, or another server can be swapped without changing the extraction records. It will not
connect to a non-loopback host. Model weights are never downloaded automatically.

## Scaling limits and follow-on work

Visual change is an intentionally conservative proposal mechanism. It cannot reliably identify
clicks, typed values, or Eclipse objects by itself. Cursor tracking, OCR-region differencing, and a
validated local GUI inverse-dynamics model are planned replaceable detectors. The static timeline is
a first inspection surface; correction editing and adjudication remain follow-on work.

For very long recordings, VLM work should be processed from `vlm_requests.jsonl` incrementally. A
rolling structured state summary should carry plan/patient context across requests while forcing a
new anonymous context when the visible workspace changes. Raw image history should not be repeatedly
sent. The synthesis stage should reconcile duplicate observations caused by overlap before producing
episode and executive summaries.

