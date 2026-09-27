# Roadmap

This roadmap is ordered by evidentiary value, not by the number of technologies covered.

## Phase 1: Trustworthy measurement

- [x] Define raw request traces and aggregate metrics.
- [x] Add deterministic pipeline validation.
- [x] Document TTFT, ITL, throughput, tail latency, and goodput.
- [x] Add an OpenAI-compatible streaming benchmark driver.
- [x] Capture host runtime, framework versions, and Git commit automatically.
- [x] Add an explicit warm-up policy.
- [ ] Capture accelerator, driver, model revision, and serving-engine versions.
- [ ] Add repeated-run confidence intervals.

## Phase 2: Serving backends

- [x] Implement a transparent Hugging Face Transformers baseline server.
- [x] Add a Colab GPU pilot runner with isolated backend environments.
- [ ] Execute and review the Experiment 002 Colab pilot.
- [ ] Compare Transformers and vLLM under identical workloads.
- [ ] Compare vLLM and SGLang, including shared-prefix workloads.
- [ ] Compare local llama.cpp results without conflating them with CUDA results.
- [ ] Add NVIDIA TensorRT-LLM as an advanced backend.

## Phase 3: Optimization experiments

- [ ] Sweep concurrency and batching limits.
- [ ] Measure FP16/BF16, INT8, and INT4 performance and quality.
- [ ] Measure KV-cache growth with sequence length.
- [ ] Identify the prefix-cache break-even point.
- [ ] Evaluate speculative decoding under different acceptance rates.
- [ ] Profile prefill and decode bottlenecks.

## Phase 4: Training and quality

- [ ] Build a task-specific evaluation set and baseline.
- [ ] Fine-tune with LoRA and QLoRA.
- [ ] Run rank, learning-rate, and dataset-size ablations.
- [ ] Measure post-training and post-quantization quality regressions.

## Phase 5: Production system

- [ ] Add load shedding, timeouts, and structured errors.
- [ ] Export Prometheus-compatible metrics.
- [ ] Define capacity planning and cost-per-million-token models.
- [ ] Demonstrate a canary model rollout and rollback.
- [ ] Publish a versioned benchmark report with limitations.
