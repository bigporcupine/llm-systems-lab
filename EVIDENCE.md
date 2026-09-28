# Evidence Workflow

This workflow separates an implemented experiment from an admissible benchmark claim. A formal result is publishable only after every step succeeds.

## 1. Resolve moving model names

Never run a formal matrix with `resolve-before-run` or a moving branch name:

```bash
python -m pip install huggingface_hub
python -m llm_systems_lab resolve-config \
  --config experiments/exp003_batching_saturation/config.json \
  --output artifacts/exp003/resolved-config.json
```

Review the resolved commit identifiers before spending accelerator time.

## 2. Create the pre-run manifest

```bash
python -m llm_systems_lab create-manifest \
  --experiment-dir experiments/exp003_batching_saturation \
  --resolved-config artifacts/exp003/resolved-config.json \
  --output artifacts/exp003/manifest.json
```

The manifest hashes the experiment definition and resolved configuration and captures the repository commit and runtime environment.

## 3. Execute without editing the definition

Run the documented command from the same Git commit. Preserve request-level traces, server logs, resolved configuration, manifest, engine metrics, failures, and any profiler output. A failed cell remains an artifact with its error; it is not silently deleted.

## 4. Audit artifacts

```bash
python -m llm_systems_lab audit-artifacts \
  --artifact-dir artifacts/exp003 \
  --output artifacts/exp003/audit.json
```

The command exits non-zero when evidence is missing. It rejects unresolved revision placeholders, synthetic data presented as measured, successful requests without TTFT, missing raw traces, missing Git/accelerator/backend/model metadata, and missing resolved-config manifests.

## 5. Build the versioned report

Write an English limitations file before generating the report. Then use the passing audit as a publication gate:

```bash
python -m llm_systems_lab build-report \
  --artifact-dir artifacts/exp003 \
  --audit artifacts/exp003/audit.json \
  --title "Experiment 003 — Batching and Saturation" \
  --version v1.0.0 \
  --claim "State the narrow, falsifiable conclusion supported by this run." \
  --limitations-file artifacts/exp003/limitations.md \
  --decision "Record the selected configuration and its guardrails." \
  --output reports/exp003-v1.0.0.md
```

The builder refuses failed or stale audits, changed artifact hashes, new unaudited JSON files, empty limitations, and malformed report versions. It extracts benchmark values directly from audited JSON and links every result row to its raw artifact. It never fabricates missing measurements. Commit the artifacts, audit, limitations, and versioned report together so the evidence links remain stable.
