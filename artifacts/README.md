# Measured Artifacts

Formal hardware results are committed below this directory by experiment number. Each experiment root must contain:

- `resolved-config.json` with full 40-character Hugging Face commit SHAs;
- `manifest.json` created before the run;
- request-level JSON traces and derived summaries;
- server logs and engine-metric snapshots needed to explain failures;
- `audit.json` produced by `audit-artifacts` with `passed: true`;
- a report or a stable link to the versioned repository report.

Generate the report with `build-report`; do not manually copy benchmark values into Markdown. The command verifies that the audit passed and that the audited JSON set and hashes have not changed.

Synthetic and controlled-mock artifacts do not belong here. They remain test fixtures or Experiment 001 calibration evidence and must never be labeled as hardware measurements.

Large binary profiler traces and model checkpoints should be stored in versioned external object storage with checksums and immutable links; their compact exported statistics and metadata remain in this repository.
