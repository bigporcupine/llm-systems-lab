# Experiment 002: From Model Execution to Production Serving

## Status

Colab-ready pilot infrastructure is implemented. No model-performance result is published yet.

## Why this experiment exists

Loading a language model and calling `model.generate()` proves that the model can produce text. It does not prove that the system can serve many users efficiently.

A single interactive request follows a simple path:

```text
one prompt -> model.generate() -> one response
```

A production inference service has a different problem:

```text
many arriving requests
        |
        +-- different prompt lengths
        +-- different generation lengths
        +-- requests entering and finishing at different times
        +-- limited GPU memory for weights and KV cache
        +-- latency SLOs and failure handling
        |
        v
scheduling + batching + memory management + model execution
```

The model weights may be identical in both cases, but system performance can be radically different. A GPU processes large parallel operations efficiently; serving requests one at a time may leave that parallel capacity underused. Naively launching multiple independent generations can instead create memory pressure, unsafe execution, or unpredictable contention. A serving engine exists to schedule this work deliberately.

Experiment 002 studies the transition from **model execution** to **model serving**. Its purpose is not merely to show that one library has a larger throughput number. It asks:

> At what workload does a dedicated inference engine become useful, and what latency, throughput, and SLO trade-offs appear as concurrency increases?

This question matters because “vLLM is faster” is not a useful engineering conclusion by itself. At concurrency one, a specialized scheduler may provide little benefit or even add overhead. At higher concurrency, continuous batching may increase throughput while queueing raises time to first token. The best configuration therefore depends on the workload and the service objective.

Experiment 001 established that the measurement method is trustworthy. Experiment 002 is the first use of that method on a real model and GPU.

## Question

When does a dedicated inference engine begin to outperform a transparent, batch-one Transformers baseline, and what happens to latency, throughput, and SLO-compliant goodput as concurrency increases?

The experiment breaks this into four concrete questions:

1. How much overhead does a serving engine add or remove at concurrency one?
2. How does each backend scale as concurrency increases?
3. Does higher raw throughput also produce higher SLO-compliant goodput?
4. How does prompt length change the saturation point?

## Part A scope

The first phase compares:

1. a minimal Hugging Face Transformers server that performs greedy, token-by-token generation and serializes GPU access;
2. vLLM serving the exact same model revision and precision through an OpenAI-compatible endpoint.

SGLang is intentionally reserved for Part B. Adding two production engines before validating the first real-model pipeline would make dependency, scheduling, and cache differences harder to isolate. Part B will reuse the same client to compare vLLM and SGLang, including shared-prefix workloads where SGLang's RadixAttention is directly relevant.

## What is being compared

### Transformers reference backend

The reference server loads a Hugging Face causal language model and performs greedy generation one token at a time. A process lock allows only one request to execute on the GPU at once. Additional requests wait in a queue.

This is intentional. It represents the simplest understandable serving design built around direct model execution:

```text
HTTP request
    -> tokenize
    -> wait for GPU lock
    -> prefill
    -> decode one token at a time
    -> stream tokens
    -> release GPU lock
```

It is not presented as the fastest possible use of Transformers, nor as a replacement for Hugging Face Text Generation Inference. Its value is transparency: the behavior is small enough to inspect, and the absence of continuous batching is explicit.

### vLLM serving backend

vLLM loads the same model revision and exposes the same OpenAI-compatible interface. Unlike the reference server, it is designed to schedule multiple active requests and batch work dynamically as requests enter and leave the system.

Conceptually:

```text
request queue
    -> scheduler
    -> continuous batches
    -> managed KV-cache allocation
    -> streamed responses
```

The experiment therefore compares two serving designs, not two different models:

| Controlled item | Transformers | vLLM |
|---|---|---|
| Model weights | Same revision | Same revision |
| Precision | Same | Same |
| Prompt set | Same | Same |
| Requested output length | Same | Same |
| HTTP/SSE client | Same | Same |
| GPU | Same Colab runtime | Same Colab runtime |
| GPU scheduling | Serialized batch-one reference | vLLM scheduler and continuous batching |

The scheduling and memory-management differences are the main independent variable.

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

These hypotheses are written before collecting model results. A result that contradicts them is still useful; the report must explain the observation rather than change the hypothesis after the fact.

## How the experiment works

### 1. Record the execution environment

The Colab notebook records the assigned GPU before installing packages. The benchmark artifacts also capture the operating system, Python version, NVIDIA driver, package versions, repository commit, model revision, and precision.

This is essential because Colab does not guarantee a fixed GPU type. Results from different GPU assignments are not merged into one comparison.

### 2. Resolve immutable inputs

The notebook clones a visible Git commit of this repository. It also resolves the Hugging Face model name to an immutable model commit SHA. Both backends must load that exact revision.

The workload generator uses a fixed seed to construct the same prompts for every backend. The configured input value is a size hint; actual prompt-token counts come from server-reported usage and are retained in every trace.

### 3. Isolate backend dependencies

Transformers and vLLM are installed into separate virtual environments:

```text
.venv-transformers/
.venv-vllm/
```

This prevents installing one backend from silently replacing the other's PyTorch, Triton, Transformers, or kernel dependencies. The version from each backend environment is saved with its results.

### 4. Run one backend at a time

The experiment never keeps both models loaded simultaneously:

```text
start Transformers
    -> wait for health check
    -> warm up
    -> run every configured workload
    -> save traces and logs
    -> terminate server
    -> verify GPU state

start vLLM
    -> repeat the identical sequence
```

Running sequentially avoids GPU memory and compute interference between backends.

### 5. Warm up before measuring

Each workload cell sends unmeasured warm-up requests before recording results. Warm-up can include model initialization, CUDA context creation, kernel loading, memory allocation, and cache population. Mixing these requests into steady-state latency would make backend comparisons depend on startup behavior rather than serving behavior.

Startup performance is a valid topic, but it requires a separate experiment.

### 6. Apply controlled concurrency

The first version uses a closed-loop workload. For concurrency `N`, the client keeps at most `N` requests active; when one finishes, another begins until the configured request count is complete.

For example:

```text
concurrency 1  -> one active request
concurrency 4  -> up to four overlapping requests
concurrency 16 -> up to sixteen overlapping requests
```

This reveals how each backend scales under a fixed number of simultaneous clients. A later extension will use open-loop arrival rates to find the point where requests arrive faster than the server can complete them.

### 7. Stream and retain every request trace

Both backends expose `/v1/chat/completions` and stream Server-Sent Events. The shared client records:

- request start;
- first non-empty content event;
- subsequent content-event timestamps;
- request completion;
- server-reported prompt and completion tokens;
- HTTP errors, timeouts, and empty responses.

Raw request traces are saved before aggregation. This makes every percentile and throughput result auditable.

### 8. Repeat and aggregate

The pilot runs once to validate the pipeline. A formal configuration runs each combination at least three times and reports mean and standard deviation across repetitions.

The matrix varies:

```text
backend x input-size hint x concurrency x repetition
```

Only one independent workload variable changes within each comparison.

## Metrics and what they answer

| Metric | Engineering question |
|---|---|
| TTFT P50/P95/P99 | How long does a user wait before generation begins, including queueing and prefill? |
| TPOT estimate | How quickly does generation progress after the first observed content event? |
| End-to-end P50/P95/P99 | How long until the complete response arrives? |
| Request throughput | How many successful requests complete per wall-clock second? |
| Output-token throughput | How much decode work completes per wall-clock second? |
| Success rate | Does the backend remain reliable under load? |
| Goodput | How many requests complete successfully while satisfying the configured latency SLOs? |

Generic OpenAI-compatible SSE does not guarantee that one content event equals one token. For that reason, raw event spacing is retained as a diagnostic, while TPOT is estimated using the observed generation span and server-reported output-token count. The report must not relabel arbitrary chunks as exact tokens.

## How results should be interpreted

The expected output is not a single winner. The useful result is a set of operating regions.

For example, a valid conclusion might be:

```text
At concurrency 1, the two backends had similar latency.
At concurrency 8, vLLM increased output throughput while remaining inside the TTFT SLO.
At concurrency 16, raw throughput increased again, but queueing caused goodput to fall.
For the recorded GPU and workload, concurrency 8 was the best operating point.
```

This is stronger than saying “vLLM was X times faster” because it identifies:

- the hardware and model;
- the workload where the advantage appeared;
- the concurrency at which the system saturated;
- the latency cost of additional throughput;
- and the configuration that remained useful under an explicit SLO.

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
- TTFT, TPOT estimate, end-to-end latency, throughput, goodput, and failures;
- a statement that Colab allocation is non-dedicated and not guaranteed;
- an official-tool cross-check before making a headline performance claim.

## Current limitations

- The client currently uses closed-loop concurrency; open-loop request-rate testing comes later.
- The input setting is a deterministic word-count hint. Actual input tokens come from server-reported usage and must be inspected for equality across backends.
- Generic SSE content-event spacing is retained for diagnostics, but it is not assumed to be exact token-level ITL. TPOT is estimated from observed generation span and server-reported output-token count.
- GPU utilization and power sampling are not implemented yet.
- The Transformers baseline is intentionally batch-one and does not represent Text Generation Inference or another optimized Hugging Face server.
- SGLang is not included in Part A.

## What this experiment cannot prove

- It cannot prove that one backend is universally faster across models, GPUs, versions, or workloads.
- It cannot attribute every difference to one internal optimization; vLLM changes scheduling, kernels, memory management, and execution together.
- It cannot represent production arrival patterns until open-loop request-rate testing is added.
- It cannot establish a stable cross-machine ranking from one non-dedicated Colab runtime.
- It cannot compare model quality because both backends intentionally use the same weights and deterministic decoding.
- It cannot evaluate prefix reuse fairly until a dedicated shared-prefix workload and cache-control policy are introduced.

The result is deliberately scoped to the recorded model revision, backend versions, GPU, precision, workload, and experiment configuration.

## Relationship to the experiment series

```text
Experiment 001
Validate the measurement method with known timing
        |
        v
Experiment 002 Part A
Transformers reference vs vLLM
Understand why a serving engine matters
        |
        v
Experiment 002 Part B
vLLM vs SGLang
Compare serving-engine design trade-offs
        |
        v
Later experiments
Quantization, batching, KV cache, prefix caching, and speculative decoding
```
