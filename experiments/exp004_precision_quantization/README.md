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

Serve one variant at a time and use `experiment-matrix` for performance. Generate raw predictions and scores directly from the same endpoint with:

`scripts/start_vllm_variant.sh` launches an explicitly pinned model artifact. For example, pass `none` for an FP16/BF16 checkpoint, the engine's supported INT8 method for an INT8 artifact, or `awq` for an AWQ checkpoint. Record the exact model artifact and backend version; changing only the flag does not quantize an FP checkpoint.

```bash
python -m llm_systems_lab evaluate-endpoint \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3-1.7B \
  --model-revision "$MODEL_REVISION" \
  --backend-version "$BACKEND_VERSION" \
  --dataset data/evaluation/systems_qa.jsonl \
  --output-dir artifacts/exp004/awq-int4/evaluation
```

Saved predictions can be rescored without contacting the model:

```bash
python -m llm_systems_lab score-quality \
  --dataset data/evaluation/systems_qa.jsonl \
  --predictions artifacts/exp004/awq-int4/predictions.json \
  --output artifacts/exp004/awq-int4/quality.json
```

The final table must include both performance and quality columns. No winner is declared when the quality set or memory measurement is missing.
