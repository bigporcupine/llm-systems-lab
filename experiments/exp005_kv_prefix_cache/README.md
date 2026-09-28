# Experiment 005: KV Cache and Prefix Cache

## Why this experiment exists

KV memory grows with active sequence length and concurrency, while prefix caching trades lookup/storage overhead for avoided prefill work. This experiment separates those mechanisms and finds the reuse level at which prefix caching becomes beneficial.

## Part A: KV-cache growth

- Run one active request at each sequence length and sample GPU memory before load, after load, after prefill, and during decode.
- Repeat at increasing concurrency with fixed sequence length.
- Compare measured growth with the theoretical estimate `2 × layers × hidden_size × bytes_per_element × cached_tokens` per sequence (adjusted for grouped-query attention when applicable).
- Report measured/theoretical deltas and explain allocator reservation, block rounding, and CUDA graph memory.

## Part B: Prefix-cache break-even

- Construct prompts with byte-identical shared prefixes and unique suffixes.
- Compare cache disabled vs enabled for each prefix length and reuse count.
- Restart between policies, warm both paths, and verify hit counters from engine metrics.
- Report TTFT, E2E, throughput, cache-hit ratio, memory overhead, and the first reuse count where enabled latency is no worse than disabled latency.

Run the KV-memory sweep while the configured server is otherwise idle. Model dimensions must come from the pinned model configuration:

```bash
python -m llm_systems_lab experiment-005-kv-memory \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-1.7B \
  --sequence-lengths 128 512 1024 2048 4096 8192 \
  --concurrency 1 4 8 --output-tokens 64 \
  --layers MODEL_LAYERS --kv-heads MODEL_KV_HEADS --head-dim MODEL_HEAD_DIM \
  --bytes-per-element 2 --output-dir artifacts/exp005/kv-memory
```

## Engine comparison

Run the same shared-prefix corpus on vLLM and SGLang. Do not compare their cache-hit counters directly unless their definitions match; compare user-visible latency and throughput first.

## Run the break-even matrix

Launch two otherwise identical servers with prefix caching disabled and enabled, then run:

```bash
python -m llm_systems_lab experiment-005-prefix-cache \
  --disabled-url http://127.0.0.1:8000/v1 \
  --enabled-url http://127.0.0.1:8001/v1 \
  --model Qwen/Qwen3-1.7B \
  --prefix-lengths 0 128 512 1024 2048 \
  --reuse-counts 1 2 4 8 16 32 \
  --repetitions 3 \
  --output-dir artifacts/exp005/vllm
```

The runner creates byte-identical shared prefixes, unique suffixes, request-level traces, policy-level aggregates, and one break-even result per prefix length.

Results belong in `artifacts/exp005/<engine>/<policy>/` with raw traces and engine metrics snapshots.
