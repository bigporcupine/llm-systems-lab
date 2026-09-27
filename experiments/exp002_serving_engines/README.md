# Experiment 002: From Model Execution to Production Serving

## Status

Colab-ready pilot infrastructure is implemented. No model-performance result is published yet.

## Question

When does a dedicated inference engine begin to outperform a transparent, batch-one Transformers baseline, and what happens to latency, throughput, and SLO-compliant goodput as concurrency increases?

## Part A scope

The first phase compares:

1. a minimal Hugging Face Transformers server that performs greedy, token-by-token generation and serializes GPU access;
2. vLLM serving the exact same model revision and precision through an OpenAI-compatible endpoint.

SGLang is intentionally reserved for Part B. Adding two production engines before validating the first real-model pipeline would make dependency, scheduling, and cache differences harder to isolate. Part B will reuse the same client to compare vLLM and SGLang, including shared-prefix workloads where SGLang's RadixAttention is directly relevant.

## Why Google Colab

The development machine does not have an NVIDIA GPU. Colab supplies an interactive Linux GPU runtime suitable for smoke tests and an initial benchmark. Colab hardware is not guaranteed or fixed, so every result is scoped to the exact GPU and runtime recorded in the artifact. A future fixed-host run can reuse the same configuration and commands.

## Fairness rules

- Run one backend at a time on the same assigned GPU.
- Use the same model and immutable Hugging Face revision.
- Use the same precision, prompts, requested output length, and benchmark client.
- Include HTTP, JSON, and SSE overhead for both backends.
- Use server-reported token counts rather than character or stream-event counts.
- Warm each server before measurement.
- Retain request-level traces and report every failure.
- Treat the Transformers server as an explicit batch-one reference, not as an optimized production server.

## Hypotheses

1. At concurrency one, vLLM may provide limited benefit because there is little scheduling opportunity.
2. As concurrency rises, vLLM should scale output-token throughput more effectively through continuous batching.
3. Higher concurrency may improve raw throughput while increasing TTFT and reducing SLO goodput.
4. Longer prompts should increase prefill pressure and may change the concurrency at which either backend saturates.

## Pilot configuration

The committed `pilot.json` uses Qwen3-0.6B, one input-size hint, two concurrency levels, and twenty measured requests. It exists to validate the GPU pipeline, not to support a final performance conclusion.

Before running, the Colab notebook resolves `model_revision` from `main` to an immutable Hugging Face commit SHA and writes a runtime-specific configuration outside the Git checkout.

## Formal configuration

`formal.template.json` is a starting point, not an automatic promise that the assigned Colab GPU has sufficient memory. The final model and precision are selected only after recording `nvidia-smi` output. A formal run requires at least three repetitions per cell.

## Run in Colab

Open [`notebooks/exp002_colab_runner.ipynb`](../../notebooks/exp002_colab_runner.ipynb) in Google Colab, select a GPU runtime, and execute cells in order.

The notebook:

1. records the runtime and GPU;
2. clones this repository at a visible commit;
3. resolves the model revision;
4. creates isolated Transformers and vLLM environments;
5. starts one backend at a time;
6. runs the same benchmark matrix;
7. stops each backend and checks GPU state;
8. packages raw results and logs for immediate download.

## Local work without a GPU

The configuration parser, workload construction, aggregation, notebook, and lifecycle scripts can all be tested locally. The Transformers server intentionally refuses to label a CPU or Apple Silicon run as a formal Experiment 002 measurement.

## Result acceptance criteria

A publishable result must include:

- exact Git commit and model revision;
- exact GPU, memory, driver, CUDA, Python, and package versions;
- identical workload and decoding settings;
- at least three measured repetitions;
- raw request traces;
- TTFT, ITL, end-to-end latency, throughput, goodput, and failures;
- a statement that Colab allocation is non-dedicated and not guaranteed;
- an official-tool cross-check before making a headline performance claim.

## Current limitations

- The client currently uses closed-loop concurrency; open-loop request-rate testing comes later.
- The input setting is a deterministic word-count hint. Actual input tokens come from server-reported usage and must be inspected for equality across backends.
- Generic SSE content-event spacing is retained for diagnostics, but it is not assumed to be exact token-level ITL. TPOT is estimated from observed generation span and server-reported output-token count.
- GPU utilization and power sampling are not implemented yet.
- The Transformers baseline is intentionally batch-one and does not represent Text Generation Inference or another optimized Hugging Face server.
- SGLang is not included in Part A.
