# Experiment 001: Measuring LLM Inference Correctly

## Status

Completed locally against a controlled OpenAI-compatible streaming server. See [the measured results](RESULTS.md).

## Why this experiment comes first

The purpose of Experiment 001 is simple:

> Calibrate the ruler before using it to measure a real LLM system.

Later experiments compare Transformers, vLLM, quantization methods, batching policies, KV-cache behavior, and other optimizations. A claim such as “throughput improved by 2×” is meaningful only if the benchmark itself measures streaming and concurrency correctly. Otherwise, a large apparent improvement may come from a timing boundary, token-counting assumption, warm-up policy, or throughput formula rather than from the system being tested.

A real model server is a poor place to validate a new benchmark client because too many variables are unknown at once. If the first token arrives after 100 ms, the measurement alone cannot tell us whether that time came from:

- prompt prefill;
- queueing and scheduler delay;
- model execution;
- network or HTTP buffering;
- client-side parsing;
- multiple tokens being combined into one stream event;
- or an error in the benchmark implementation.

Experiment 001 therefore uses a controlled server with known behavior. Prefill delay, token delay, output length, cold-start penalty, and tail-latency injection are configured in advance. Because the expected behavior is known, the client can be checked against a reference rather than trusted merely because it produced plausible numbers.

This experiment establishes five foundations for everything that follows:

1. **Separate latency phases.** TTFT, inter-token latency, and end-to-end latency describe different user-visible and system-level behavior.
2. **Correct concurrent throughput.** Throughput must use the wall-clock duration of the complete run, not the sum of overlapping request latencies.
3. **Visible tail behavior.** Means and medians can hide the slow requests that determine production SLOs.
4. **Explicit warm-up policy.** Cold-start and steady-state performance must be measured and reported separately.
5. **Useful capacity.** Goodput counts requests that satisfy the SLO, while raw throughput counts requests even when they are too slow to be useful.

The output of Experiment 001 is not a claim about model performance. It is evidence that the measurement method used by subsequent model-performance experiments has known semantics, auditable raw traces, and tested calculations.

## Question

How should latency, throughput, tail behavior, warm-up, and SLO-aware goodput be measured for a concurrent streaming inference service?

## Hypothesis

A request-level trace schema plus wall-clock benchmark timing will correctly distinguish latency from capacity. The experiment should also show that common shortcuts—summing concurrent request latency, reporting only a median, or mixing cold starts into steady state—produce misleading conclusions.

## Controlled method

The experiment starts a local HTTP server that implements `/v1/chat/completions` with Server-Sent Events. Unlike a real model server, its timing is known:

- configurable prefill delay before the first content event;
- configurable delay between subsequent token events;
- exactly one known token per content event;
- deterministic tail-latency injection;
- optional first-request cold-start delay;
- final server-reported token usage.

The benchmark client uses a monotonic clock, runs requests concurrently, retains every request trace, and calculates throughput from total elapsed wall-clock time.

This is a real HTTP and streaming experiment, but it is not a model or hardware benchmark.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/llms-lab experiment-001
```

Raw results are written to:

```text
experiments/exp001_measurement_basics/results/
```

The narrative report is regenerated at `experiments/exp001_measurement_basics/RESULTS.md`.

## Manual server and client workflow

Terminal one:

```bash
llms-lab mock-server \
  --prefill-ms 20 \
  --token-delay-ms 5 \
  --tail-every 10 \
  --tail-delay-ms 100
```

Terminal two:

```bash
llms-lab benchmark-online \
  --base-url http://127.0.0.1:8000/v1 \
  --model controlled-mock \
  --requests 30 \
  --concurrency 8 \
  --ttft-slo-ms 80 \
  --latency-slo-ms 150
```

The all-in-one `experiment-001` command uses an internal controlled-mode flag so the mock server can receive exact token counts. The generic `benchmark-online` command does not send those mock-only fields.

## Acceptance criteria

- TTFT is measured from request start to the first non-empty streamed content event.
- ITL excludes TTFT and uses gaps between subsequent controlled token events.
- Concurrent throughput uses complete-run wall-clock duration.
- Server-reported usage determines token counts when available.
- Tail injection changes P95 while leaving the median comparatively stable.
- Goodput excludes requests that violate either configured latency SLO.
- Cold-start latency is reported separately from the warmed measurement.
- Raw request traces and environment metadata are preserved.

## What this experiment does not claim

- It does not compare model quality or inference engines.
- It does not model GPU scheduling, KV-cache pressure, or memory bandwidth.
- It does not assume that one SSE event equals one token on real servers.
- It does not treat a 30-request sample as a reliable production P99 estimate.

The validated measurement foundation is used by Experiment 002 for the first real Transformers-versus-vLLM comparison.

## Role in the experiment series

```text
Experiment 001: validate the measurement method
        |
        +-- request-level trace schema
        +-- concurrent streaming client
        +-- TTFT / ITL / end-to-end latency
        +-- wall-clock throughput and SLO goodput
        +-- warm-up and tail-latency policy
        |
        v
Experiment 002: Transformers versus vLLM
        |
        v
Quantization, batching, KV cache, prefix caching, and speculative decoding
```

The ordering matters: first establish that the ruler is trustworthy, then use it to compare real systems.
