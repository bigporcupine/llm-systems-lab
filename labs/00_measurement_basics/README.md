# Lab 00: Measurement Before Optimization

## Why this lab exists

An optimization claim is only as credible as its measurement method. This lab builds the vocabulary and data contract used by every later experiment. It deliberately uses deterministic synthetic traces first, so the metric pipeline can be tested without pretending that generated data represents real hardware.

The accompanying [Experiment 001](../../experiments/exp001_measurement_basics/README.md) then moves beyond generated traces to a controlled HTTP streaming service. Its guiding principle is to calibrate the ruler before measuring a real model: known server timing makes benchmark errors distinguishable from model, scheduler, and network behavior.

## Learning objectives

By the end of this lab, you should be able to:

- distinguish time to first token (TTFT), inter-token latency (ITL), and end-to-end latency;
- explain why request throughput and output-token throughput answer different questions;
- calculate throughput from wall-clock benchmark duration rather than summed request latency;
- explain why averages hide tail latency;
- define goodput as throughput that satisfies an explicit service-level objective (SLO);
- identify when a benchmark result cannot be compared fairly.

## Mental model

A streamed generation request has two visibly different phases:

```text
request ───── prefill ───── first token ─ decode ─ token ─ decode ─ token
        <----- TTFT ------>             <- ITL ->
        <---------------- end-to-end latency ---------------------------->
```

Prefill processes the input prompt and is often compute-bound. Decode repeatedly produces new tokens and is often constrained by memory movement. A single aggregate latency number hides this distinction.

## Metric definitions

| Metric | Definition | Why it matters |
|---|---|---|
| TTFT | Request start to first streamed token | User-perceived responsiveness and prefill cost |
| ITL | Time between consecutive streamed tokens | Smoothness of generation and decode performance |
| End-to-end latency | Request start to completion | Total user wait time |
| Request throughput | Successful requests / wall-clock seconds | System capacity for a given workload |
| Output throughput | Generated tokens / wall-clock seconds | Decode-oriented capacity |
| Goodput | Requests meeting the SLO / wall-clock seconds | Capacity that remains useful to users |

## Run the pipeline validation

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/llms-lab benchmark \
  --requests 50 \
  --ttft-slo-ms 250 \
  --latency-slo-ms 2500
```

This writes raw traces to `results/synthetic.json` and a Markdown report to `reports/synthetic.md`. Both outputs are ignored by Git because they are generated artifacts.

## Experiment questions

1. Tighten the TTFT SLO from 250 ms to 150 ms. Why can throughput remain unchanged while goodput falls?
2. Increase the number of requests. Which percentiles are unstable for very small samples?
3. Inspect a request's token timestamps. Why is the first timestamp not part of ITL?
4. Imagine two systems with equal output tokens per second. What workload differences could make the comparison invalid?

## Measurement traps

- Do not compare results from different prompt or output-length distributions without saying so.
- Do not calculate concurrent throughput by adding individual request durations.
- Do not call network chunks “tokens” unless the server or tokenizer establishes that they are tokens.
- Do not report only the mean; include tail percentiles and sample count.
- Do not mix warm-up requests with steady-state measurements without labeling them.
- Do not claim a win until quality is measured under the optimized configuration.

## Exit criteria

You have completed the lab when you can derive every reported metric from the raw trace file and explain what it does—and does not—say about a real inference system.
