# Experiment 008: Production Readiness

## Why this experiment exists

A fast engine is not a production system. Under overload it needs bounded admission, deadlines, observable failure modes, capacity assumptions, and a reversible model rollout. This experiment turns the benchmarked backend into a small but testable serving control plane.

## Part A: overload protection and errors

The gateway in `llm_systems_lab.production_gateway` limits in-flight requests, applies a queue deadline and an upstream request deadline, and emits structured JSON errors. The load test increases offered traffic beyond measured capacity and checks that admitted-request latency remains bounded while excess work receives HTTP 429 rather than timing out silently.

## Part B: observability

`/metrics` exposes Prometheus text metrics for requests, errors, rejections, in-flight work, routing, and latency. Logs contain a request ID, selected model lane, status, and elapsed time but never prompt text. The dashboard/report must distinguish client cancellation, queue rejection, upstream timeout, upstream failure, and success.

## Part C: capacity and cost

Use measured sustainable goodput, not peak microbenchmark throughput. Required replicas are `ceil(peak demand / (per-replica goodput × target utilization))`. Report input/output tokens per second, requests per second, GPU memory headroom, replicas, GPU-hours, and dollars per million total and output tokens. State the price source and date in the result artifact.

## Part D: canary and rollback

Traffic is assigned deterministically from `X-Request-ID`, so retries remain on one lane. Start at 5%, compare error rate, p95 latency, and frozen-set quality, and roll back when any configured guardrail fails after the minimum sample count. The demo must show both a promotion and an automatic rollback.

## Run

```bash
python -m llm_systems_lab production-gateway \
  --config experiments/exp008_production_readiness/config.json \
  --host 127.0.0.1 --port 9000
```

Then point the shared online benchmark at `http://127.0.0.1:9000/v1` and preserve raw gateway metrics before and after the overload and canary tests.
