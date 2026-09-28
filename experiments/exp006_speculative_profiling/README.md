# Experiment 006: Speculative Decoding and Profiling

## Why this experiment exists

Speculative decoding adds draft-model work and verification overhead. It helps only when accepted tokens replace enough target decode steps. Average latency alone cannot explain a win or loss, so this experiment couples end-to-end measurements with acceptance and kernel-level profiles.

## Speculative-decoding method

- Baseline: target model without speculation.
- Candidates: one immutable draft model with 1, 3, 5, and 8 speculative tokens.
- Workloads deliberately cover high, typical, and low target/draft agreement.
- Hold target model, output length, prompts, GPU, precision, and sampling fixed.
- Record accepted, proposed, and rejected draft tokens; acceptance rate is `accepted / proposed`.
- Report TTFT, TPOT, E2E, output tokens/s, target forward passes avoided, and draft overhead.
- Run the same quality set as Experiment 004 to ensure output equivalence under deterministic decoding.

## Profiling method

Use the scripts in `scripts/profiling/` for one representative prefill-heavy cell and one decode-heavy cell. Capture a short, bounded range after warm-up. Classify time into attention, GEMM, normalization, sampling, host scheduling, and communication. Never publish an unbounded profiler trace containing unrelated processes.

```bash
bash scripts/profiling/nsys_capture.sh \
  artifacts/exp006/profiles/vllm-decode \
  python -m llm_systems_lab benchmark-online --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-1.7B --requests 8 --concurrency 1 --input-tokens 128 --output-tokens 256
```

The conclusion must state the observed break-even acceptance rate; it must not generalize one draft/target pair to all models.

## Run the endpoint comparison

Launch target-only and speculative servers on separate ports. If the engines export Prometheus metrics, include both metrics URLs so proposed and accepted token counters are snapshotted around the workload.

Use `scripts/start_vllm_speculative.sh` for the candidate. Its speculative configuration is passed as an explicit JSON argument because the accepted keys are version-specific; preserve that exact JSON and the vLLM version in the result manifest.

```bash
python -m llm_systems_lab experiment-006-speculative \
  --baseline-url http://127.0.0.1:8000/v1 \
  --speculative-url http://127.0.0.1:8001/v1 \
  --baseline-metrics-url http://127.0.0.1:8000/metrics \
  --speculative-metrics-url http://127.0.0.1:8001/metrics \
  --model Qwen/Qwen3-1.7B \
  --target-revision "$TARGET_REVISION" \
  --draft-model Qwen/Qwen3-0.6B \
  --draft-revision "$DRAFT_REVISION" \
  --baseline-backend-version "$BASELINE_BACKEND_VERSION" \
  --speculative-backend-version "$SPECULATIVE_BACKEND_VERSION" \
  --speculative-config "$SPECULATIVE_CONFIG_JSON" \
  --input-size-hints 128 1024 \
  --output-tokens 256 --concurrency 1 --measured-requests 30 \
  --repetitions 3 --output-dir artifacts/exp006/draft-qwen3-0.6b
```

The artifact records both immutable model revisions, the exact speculative configuration, raw Prometheus text before and after each endpoint workload, and the parsed counter deltas.
