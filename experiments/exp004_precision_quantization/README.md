# Experiment 004: Precision, Quantization, and Quality

## Why this experiment exists

Lower precision is useful only when memory savings or speedups survive a quality gate. “The model loads in INT4” is not optimization evidence; the comparison must isolate precision, measure memory and latency, and test the same prompts with deterministic decoding.

## Method

- Compare FP16, BF16, INT8 weight-only, and AWQ INT4 variants derived from one immutable model revision.
- Use identical prompts, token limits, engine settings, GPU, and concurrency.
- Record peak GPU memory after load, steady-state GPU memory, startup time, TTFT, E2E, tokens/s, and goodput.
- Generate deterministic predictions (`temperature=0`) for `data/evaluation/systems_qa.jsonl`.
- Score normalized exact match and token F1. A candidate passes only when its exact-match drop is no greater than `max_exact_match_drop`.
- Treat unsupported BF16 hardware as a skipped cell with a reason, not as zero performance.

## Run

Serve one variant at a time and use `experiment-matrix` for performance. Save predictions as `{id: prediction}` and score them with:

```bash
python -m llm_systems_lab score-quality \
  --dataset data/evaluation/systems_qa.jsonl \
  --predictions artifacts/exp004/awq-int4/predictions.json \
  --output artifacts/exp004/awq-int4/quality.json
```

The final table must include both performance and quality columns. No winner is declared when the quality set or memory measurement is missing.
