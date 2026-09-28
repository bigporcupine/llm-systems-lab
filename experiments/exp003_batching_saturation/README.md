# Experiment 003: Batching and Saturation

## Why this experiment exists

An engine can improve aggregate throughput while making interactive latency unusable. A single concurrency number also cannot reveal where queueing begins, whether the GPU is under-filled, or which batching cap is responsible. This experiment finds the throughput/latency frontier and the highest safe operating point under an explicit p95 latency budget.

## Hypotheses

1. Throughput initially rises with concurrency because continuous batching improves GPU utilization.
2. Beyond a knee, queueing increases TTFT and p95 end-to-end latency faster than throughput improves.
3. `max_num_seqs` and `max_num_batched_tokens` move the knee and must be tuned against workload shape.

## Method

- Hold model revision, GPU, precision, prompt set, output length, engine version, and sampling fixed.
- Sweep prompt lengths, client concurrency, `max_num_seqs`, and `max_num_batched_tokens` from `config.json`.
- Restart the server for every engine configuration. Exclude startup and compilation from steady-state request metrics, but record startup separately.
- Warm up every cell, then collect three independent measured repetitions.
- Save every request trace. Report TTFT p50/p95, E2E p95, output tokens/s, request goodput, and bootstrap 95% confidence intervals.
- Preserve the automatically sampled `gpu-telemetry.json` time series so the saturation knee can be compared with GPU utilization, memory, temperature, and power.
- Select the saturation knee as the highest-throughput cell satisfying the configured p95 E2E budget.

## Run

Start a configured OpenAI-compatible backend, then run:

```bash
python -m llm_systems_lab experiment-matrix \
  --experiment exp003-batching-saturation \
  --variant vllm-seqs-16-tokens-4096 \
  --base-url http://127.0.0.1:8000/v1 \
  --backend-version "$BACKEND_VERSION" \
  --config experiments/exp003_batching_saturation/config.json \
  --output-dir artifacts/exp003/vllm-seqs-16-tokens-4096
```

Repeat for every server configuration. Do not compare cells captured on different GPU models as if they were one sweep.

## Decision produced

The report names the safe concurrency limit, the saturation knee, and the batching configuration used for later experiments. Empty `artifacts/` means “not executed,” never “no effect.”
