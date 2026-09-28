# Versioned Reports

This directory contains reports generated from formal, audited experiment artifacts. A report is admissible only when `build-report` verifies a passing audit, the complete audited JSON file set, and every recorded SHA-256 digest.

Report versions describe the evidence snapshot, not the software package version. Increment the version whenever measurements, claims, limitations, or decisions change. Keep old reports when a new hardware, model revision, backend version, or workload makes the result materially different.

Reports must not contain hand-entered benchmark numbers. Use the evidence workflow in [`EVIDENCE.md`](../EVIDENCE.md) to resolve revisions, create a pre-run manifest, execute the experiment, audit its artifacts, and generate the report.
