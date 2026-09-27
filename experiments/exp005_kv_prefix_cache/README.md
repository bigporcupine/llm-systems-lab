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

## Engine comparison

Run the same shared-prefix corpus on vLLM and SGLang. Do not compare their cache-hit counters directly unless their definitions match; compare user-visible latency and throughput first.

Results belong in `artifacts/exp005/<engine>/<policy>/` with raw traces and engine metrics snapshots.
