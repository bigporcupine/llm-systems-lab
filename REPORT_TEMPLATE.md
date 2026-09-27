# LLM Systems Lab Benchmark Report vX.Y

## Claim

State the narrow, falsifiable conclusion supported by this report. Do not generalize beyond the measured model, workload, engine, and hardware.

## Reproduction identity

| Field | Value |
|---|---|
| Repository commit | |
| Report version | |
| Experiment configuration hash | |
| Model and immutable revision | |
| Model artifact hash | |
| Backend and version | |
| Accelerator, memory, and driver | |
| Precision / quantization | |
| Dataset or prompt-corpus hash | |

## Hypothesis and controlled variables

Document the independent variable, controls, expected mechanism, success criterion, and stopping rule before presenting results.

## Workload

Describe input/output token distributions, arrival pattern, concurrency, sampling, cache policy, warm-up, repetitions, timeouts, and SLOs.

## Results

Include mean, standard deviation, bootstrap 95% confidence interval, failures, quality, and memory. Link every aggregate table to committed raw artifacts.

## Mechanism

Explain why the result occurred using engine metrics or a bounded profile. Distinguish observation from inference.

## Failed runs and exclusions

List failed cells, unsupported configurations, restarts, throttling, and all exclusions with reasons. Never silently remove an outlier.

## Limitations

State hardware, model, dataset-size, measurement, cache, and external-validity limitations.

## Decision

Record the configuration selected, the quality/SLO guardrails it satisfies, and the scenario where the decision should be revisited.
