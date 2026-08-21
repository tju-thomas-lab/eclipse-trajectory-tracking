# Security and privacy

- Runtime video processing is local and has no telemetry.
- The VLM client rejects non-loopback endpoints, redirects, and URLs containing credentials.
- Videos, model weights, outputs, and common video extensions are ignored by Git.
- Logs contain stage counts and technical metadata only; visible text and images are not printed.
- Derived output must still be handled as PHI because visual redaction is not performed in v0.1.0.
- Privacy mode asks a VLM to omit identifiers, but that is not a security boundary. Inspect and, when
  required, de-identify outputs before sharing them.
- No model is downloaded or loaded automatically. Prefer verified Safetensors and record license,
  revision, checksum, and whether custom code is required.
- The program performs retrospective analysis only and contains no UI-control capability.

For offline operation, install dependencies and model weights in a controlled setup phase, then use
`configs/offline.yaml` with networking disabled at the operating-system/container layer. Localhost
inference remains available inside that boundary.

