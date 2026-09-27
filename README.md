# LLM Systems Lab — Learn, Measure, Optimize, and Serve Language Models

**A reproducible, experiment-driven guide to understanding and optimizing LLM training and inference—from PyTorch fundamentals to production serving.**

LLM Systems Lab is an experiment-driven project for production inference engineering. Every optimization is expected to include raw measurements, workload and environment metadata, tail latency, throughput, quality impact, limitations, and a reproducible command.

> Project status: the measurement foundation is runnable. Real serving-backend benchmarks are the next milestone. No synthetic result in this repository is presented as hardware performance.

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

The first completed study is [Experiment 001: Measuring LLM Inference Correctly](experiments/exp001_measurement_basics/README.md), including its [measured results](experiments/exp001_measurement_basics/RESULTS.md).

Experiment 002 is now in its Colab pilot stage: [From Model Execution to Production Serving](experiments/exp002_serving_engines/README.md) compares a transparent Transformers baseline with vLLM on the same recorded GPU runtime.

Planned experiment sequence:

1. Measurement correctness and benchmark hygiene
2. Hugging Face Transformers versus vLLM
3. FP16/BF16 versus INT8 and INT4
4. Batching, concurrency, and SLO goodput
5. KV-cache growth and prefix-cache break-even
6. LoRA/QLoRA fine-tuning and quality evaluation
7. Production serving, observability, capacity, and cost

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

## Contributing

Contributions should add evidence, not just another integration. A useful experiment includes a falsifiable hypothesis, controlled variables, raw traces, repeatable commands, analysis of failed or surprising cases, and a precise statement of limitations.

## License

MIT
