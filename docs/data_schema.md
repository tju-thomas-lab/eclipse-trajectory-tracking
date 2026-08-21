# Data schema

All timestamps are seconds from the source video timeline. JSONL records include `schema_version`.
Unknown values are JSON `null` or an explicit `unknown_*` enum; absence is never silently replaced by
a guess.

## `windows.jsonl`

Every rolling context window has `window_id`, start/end time, aggregate/max change scores, candidate
flag, changed regions, and evidence references with requested and actual decoded timestamps.

## `candidate_events.jsonl`

An event groups temporally adjacent metric peaks. It references all rolling windows that overlap it,
plus the strongest changed region and before/interaction/after evidence. Event boundaries do not
pretend to be exact mouse-event boundaries.

## `actions.jsonl`

Each action has primitive and Eclipse-semantic objects, state before/after, instruction, confidence,
alternative interpretations, evidence, field-level inference sources, provenance, and review status.
`source_event_id` may be null for low-motion operations found only in rolling evidence;
`source_window_ids` preserves those sources explicitly.
The deterministic backend emits `unknown_action` for measured change and does not invent coordinates,
text, UI targets, or clinical intent.

## Model and review layers

- `model_outputs.jsonl`: raw local server text and request provenance.
- imported annotation bundle: a portable object containing evidence source references, normalized
  outputs, an executive summary, generator provenance, and explicit limitations.
- `synthesis_overrides.json`: imported summary/limitations bound to a hash of the exact active
  actions, preventing a stale expert summary from being reused after actions change.
- `actions.jsonl`: schema-normalized model or deterministic output.
- `review_corrections.jsonl`: append-only corrections; never overwrites a model record.
- Future `adjudicated_actions.jsonl`: accepted ground truth after explicit adjudication.

The model queue stores relative image paths. This is both smaller than embedded base64 and easy to
convert into messages for common local multimodal runtimes.
