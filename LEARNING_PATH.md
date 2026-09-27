# Learning Path

This project assumes a working understanding of Transformer inference: tokenization, prefill, autoregressive decoding, attention, and the purpose of a KV cache. The labs focus on measuring and optimizing how models execute in real systems.

## Systems track

1. `labs/00_measurement_basics`: learn to measure before optimizing.
2. Transformers baseline: establish transparent, single-process behavior.
3. Memory anatomy: separate weights, activations, temporary buffers, and KV cache.
4. Batching: study latency-throughput trade-offs under concurrency.
5. Quantization: measure memory, speed, and quality together.
6. Efficient serving: compare scheduling and memory-management strategies.
7. Fine-tuning and evaluation: connect adaptation to measurable quality.
8. Production system: operate under SLO, capacity, reliability, and cost constraints.

Each lab follows the same discipline: question, mental model, hypothesis, controlled setup, raw results, analysis, failure cases, production implications, and exercises.

## How to study each lab

1. Write down your expected result before running the experiment.
2. Inspect the raw request traces before reading aggregate metrics.
3. Change one independent variable at a time.
4. Explain any surprising result before adding another optimization.
5. Re-run quality evaluation whenever model weights, precision, or decoding behavior changes.
6. State exactly which workloads and environments support your conclusion.
