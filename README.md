# LLM Systems Lab — Learn, Measure, Optimize, and Serve Language Models

**A reproducible, experiment-driven guide to understanding and optimizing LLM training and inference—from PyTorch fundamentals to production serving.**

LLM Systems Lab is an experiment-driven project for production inference engineering. Every optimization is expected to include raw measurements, workload and environment metadata, tail latency, throughput, quality impact, limitations, and a reproducible command.

> Project status: Experiments 001–008 have runnable measurement protocols and validation paths. Hardware-dependent matrices are published only after execution and review. No synthetic result is presented as model or hardware performance.

## Experiments

| Experiment | Purpose | Status | Run |
|---|---|---|---|
| [001 — Measuring LLM Inference Correctly](experiments/exp001_measurement_basics/README.md) | Validate streaming latency, concurrent throughput, tail behavior, warm-up policy, and goodput using known server timing. | Completed | `llms-lab experiment-001` |
| [002 — From Model Execution to Production Serving](experiments/exp002_serving_engines/README.md) | Compare a transparent Transformers batch-one reference with vLLM on the same model, GPU, precision, and workload. | Colab pilot ready | [Open the Colab runner](https://colab.research.google.com/github/bigporcupine/llm-systems-lab/blob/main/notebooks/exp002_colab_runner.ipynb) |
| [003 — Batching and Saturation](experiments/exp003_batching_saturation/README.md) | Find the throughput/latency frontier and the safe concurrency knee. | Implementation ready | Endpoint matrix |
| [004 — Precision and Quantization](experiments/exp004_precision_quantization/README.md) | Compare FP16, BF16, INT8, and INT4 with an explicit quality gate. | Implementation ready | Performance + quality |
| [005 — KV and Prefix Cache](experiments/exp005_kv_prefix_cache/README.md) | Measure KV-memory growth and prefix-cache break-even reuse. | Protocol ready | vLLM + SGLang |
| [006 — Speculative Decoding and Profiling](experiments/exp006_speculative_profiling/README.md) | Relate draft-token acceptance to speedup and kernel bottlenecks. | Protocol ready | Endpoint + Nsight |
| [007 — Fine-tuning and Quality](experiments/exp007_finetuning_quality/README.md) | Train LoRA/QLoRA adapters and run controlled ablations. | Training runner ready | GPU training |
| [008 — Production Readiness](experiments/exp008_production_readiness/README.md) | Test overload protection, metrics, capacity/cost, and canary rollback. | Gateway ready | Load + rollout |

See the [experiment index](experiments/README.md) for scope, outputs, and the distinction between completed results and planned work.

## Current capability

The first systems lab defines and implements:

- time to first token (TTFT);
- inter-token latency (ITL);
- end-to-end latency and tail percentiles;
- request and output-token throughput;
- success rate;
- SLO-aware goodput;
- versioned, request-level JSON traces;
- deterministic synthetic data for testing the measurement pipeline.
- a controlled OpenAI-compatible streaming server;
- a concurrent streaming benchmark client;
- explicit warm-up, tail-latency, and SLO experiments;
- automatic runtime-environment capture.

Synthetic traces validate the tooling only. They are deliberately labeled and never used to claim model, backend, or hardware performance.

## Quick start

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/llms-lab benchmark --requests 50
```

The command prints aggregate metrics and creates:

```text
results/synthetic.json   # raw, request-level evidence
reports/synthetic.md     # generated report scaffold
```

Run all tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Learning path

Start with [Lab 00: Measurement Before Optimization](labs/00_measurement_basics/README.md), then follow [the systems learning path](LEARNING_PATH.md) and [project roadmap](ROADMAP.md).

Experiment sequence:

1. Measurement correctness and benchmark hygiene
2. Hugging Face Transformers versus vLLM
3. Batching, concurrency, and SLO goodput
4. FP16/BF16 versus INT8 and INT4
5. KV-cache growth and prefix-cache break-even
6. Speculative decoding and kernel profiling
7. LoRA/QLoRA fine-tuning and quality evaluation
8. Production serving, observability, capacity, and cost

## Repository map

```text
labs/          Guided explanations, exercises, and exit criteria
experiments/   Hypotheses, controlled methods, results, and conclusions
src/           Reusable benchmark, metrics, and reporting code
tests/         Correctness checks for metrics and benchmark tooling
```

## Evidence standard

A real benchmark report must state:

- model name and immutable revision;
- accelerator, memory, driver, runtime, and framework versions;
- precision and quantization method;
- input/output-length distribution and request-arrival pattern;
- concurrency, batching, warm-up, and repetition policy;
- TTFT, ITL, end-to-end tail latency, throughput, goodput, and failures;
- quality evaluation under the same model configuration;
- raw result location, limitations, and known confounders.

Results from different hardware or workloads are not treated as directly comparable merely because they use the same model.

Before a measured run, create an immutable manifest beside its artifacts:

```bash
python -m llm_systems_lab create-manifest \
  --experiment-dir experiments/exp003_batching_saturation \
  --output artifacts/exp003/manifest.json
```

The manifest records SHA-256 hashes for the experiment definition, repository commit, runtime, and detected accelerator. Resolve moving model names to immutable revisions before creating it.

## Contributing

Contributions should add evidence, not just another integration. A useful experiment includes a falsifiable hypothesis, controlled variables, raw traces, repeatable commands, analysis of failed or surprising cases, and a precise statement of limitations.

## License

MIT
