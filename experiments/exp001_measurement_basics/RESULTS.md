# Experiment 001 Results: Measuring LLM Inference Correctly

## Result status

Completed against a controlled OpenAI-compatible streaming server. These values validate the benchmark method; they are not model or hardware performance claims.

## Controlled configuration

- Prefill delay: 20 ms
- Decode delay: 5 ms per token after the first token
- Output length: 8 exact mock tokens
- Tail injection: 100 ms on every tenth server request
- Measured requests per concurrency: 30
- Warm-up requests before each measured run: 2
- TTFT SLO: 80 ms
- End-to-end latency SLO: 150 ms

## Concurrency results

| Concurrency | TTFT P50 | TTFT P95 | E2E P95 | Throughput | Naive throughput | Goodput |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.441 ms | 124.545 ms | 168.496 ms | 12.876 req/s | 12.914 req/s | 11.589 req/s |
| 4 | 24.615 ms | 123.593 ms | 164.971 ms | 47.294 req/s | 13.027 req/s | 42.565 req/s |
| 8 | 24.895 ms | 122.05 ms | 164.013 ms | 80.737 req/s | 13.034 req/s | 72.663 req/s |

“Naive throughput” is intentionally calculated as `requests / sum(request latency)`. It is incorrect for concurrent execution because overlapping requests are counted as if they ran serially.

At concurrency 8, wall-clock throughput was 80.737 req/s, while the naive calculation produced 13.034 req/s—a 6.19× difference. This confirms that concurrent throughput must use the elapsed time of the entire benchmark run.

## Tail behavior

At concurrency 8, mean TTFT was 34.019 ms, P50 was 24.895 ms, and P95 was 122.05 ms. The injected slow requests have limited effect on the median but are visible in the tail percentile. Reporting only an average or median would hide this behavior.

## Warm-up effect

The isolated cold request had a TTFT of 174.109 ms. After warm-up, steady-state TTFT P50 was 23.348 ms. Cold-start behavior should be reported separately rather than silently mixed into steady-state measurements.

## Throughput versus goodput

Throughput counts all successful requests. Goodput counts only requests satisfying both SLOs. Because the server injects tail delays, the two values diverge even with a 100% transport success rate. The configuration with the highest raw throughput is therefore not automatically the most useful configuration.

## Conclusions

1. TTFT, inter-token latency, and end-to-end latency measure different phases and must remain separate.
2. Concurrent throughput must be derived from benchmark wall-clock time.
3. Tail percentiles expose behavior hidden by averages and medians.
4. Warm-up policy materially changes reported latency and must be explicit.
5. SLO-aware goodput is a stronger capacity metric than unconstrained throughput.

## Limitations

- The server uses controlled sleeps and does not execute a model.
- One SSE content event equals exactly one token only because the mock protocol guarantees it. Real servers may emit zero, one, or multiple tokens per event.
- Localhost removes real network variability and does not model distributed serving.
- Thirty requests are enough to demonstrate the instrumentation, not to estimate a production P99 reliably.
- Python scheduling and timer resolution introduce small deviations from configured delays.

## Reproduce

```bash
llms-lab experiment-001
```
