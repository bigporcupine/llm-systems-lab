# Roadmap

This roadmap is ordered by evidentiary value, not by the number of technologies covered.

## Phase 1: Trustworthy measurement

- [x] Define raw request traces and aggregate metrics.
- [x] Add deterministic pipeline validation.
- [x] Document TTFT, ITL, throughput, tail latency, and goodput.
- [x] Add an OpenAI-compatible streaming benchmark driver.
- [x] Capture host runtime, framework versions, and Git commit automatically.
- [x] Add an explicit warm-up policy.
- [x] Capture accelerator, driver, model revision, and serving-engine versions.
- [x] Add repeated-run confidence intervals.

## Phase 2: Serving backends

- [x] Implement a transparent Hugging Face Transformers baseline server.
- [x] Add a Colab GPU pilot runner with isolated backend environments.
- [ ] Execute and review the Experiment 002 Colab pilot.
- [ ] Compare Transformers and vLLM under identical workloads.
- [x] Add vLLM and SGLang runners plus a controlled shared-prefix protocol.
- [x] Add a separate local llama.cpp runner and reporting policy.
- [x] Add NVIDIA TensorRT-LLM as an advanced backend runner.

## Phase 3: Optimization experiments

- [x] Implement the concurrency and batching sweep.
- [x] Implement FP16/BF16, INT8, and INT4 performance and quality gates.
- [x] Implement the KV-cache growth protocol.
- [x] Implement the prefix-cache break-even protocol.
- [x] Implement speculative-decoding acceptance sweeps.
- [x] Add bounded prefill and decode profiling scripts.

## Phase 4: Training and quality

- [x] Build a task-specific evaluation schema and smoke-test set.
- [x] Add a LoRA and QLoRA training runner.
- [x] Define rank, learning-rate, and dataset-size ablations.
- [x] Add post-training and post-quantization quality gates.

## Phase 5: Production system

- [x] Add load shedding, timeouts, and structured errors.
- [x] Export Prometheus-compatible metrics.
- [x] Define capacity planning and cost-per-million-token models.
- [x] Implement deterministic canary routing and rollback guardrails.
- [ ] Publish a versioned benchmark report with measured results and limitations.

The versioned report structure is defined in [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md); the item remains open until reviewed measurements are inserted.

Checked implementation items mean the experiment and validation path exist. Hardware-dependent execution remains tracked separately and is never implied by an implementation checkmark.
