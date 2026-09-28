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

## 5. Write the versioned report

Copy `REPORT_TEMPLATE.md`, link each table to audited artifacts, include failed cells and limitations, and state only the narrow claim supported by that model revision, engine version, hardware, and workload. Commit artifacts and report together so links remain stable.
