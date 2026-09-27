# Experiment Index

Experiments are numbered in dependency order. Each one introduces a measurement or systems capability that later experiments reuse.

## Experiment 001: Measuring LLM Inference Correctly

**Status:** Completed controlled experiment.

Experiment 001 validates the benchmark method against an OpenAI-compatible streaming server with known timing. It demonstrates why TTFT, decode cadence, end-to-end latency, wall-clock throughput, tail percentiles, warm-up behavior, and SLO goodput must be measured separately.

- [Design and method](exp001_measurement_basics/README.md)
- [Measured controlled results](exp001_measurement_basics/RESULTS.md)

No model is executed in Experiment 001. Its purpose is to calibrate the measurement system before testing real inference engines.

## Experiment 002: From Model Execution to Production Serving

**Status:** Colab GPU pilot ready; model-performance results not yet published.

Experiment 002 is the first real-model experiment. Part A compares a transparent, serialized Transformers reference server with vLLM using the same model revision, precision, prompts, output length, GPU runtime, HTTP/SSE path, and benchmark client.

- [Motivation, hypotheses, and complete method](exp002_serving_engines/README.md)
- [Pilot configuration](exp002_serving_engines/pilot.json)
- [Formal configuration template](exp002_serving_engines/formal.template.json)
- [Open the Colab GPU runner](https://colab.research.google.com/github/bigporcupine/llm-systems-lab/blob/main/notebooks/exp002_colab_runner.ipynb)

Part B will compare vLLM and SGLang after the Part A pipeline is validated, including a shared-prefix workload designed to study prefix reuse.

## Status language

- **Completed:** The experiment has been executed and its scoped results are published.
- **Pilot ready:** The implementation is available, but its first real run or review is still pending.
- **Planned:** The design exists on the roadmap, but runnable infrastructure is not complete.

An experiment is never labeled completed merely because its code exists.
