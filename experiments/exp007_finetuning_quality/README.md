# Experiment 007: LoRA, QLoRA, Ablations, and Quality Regression

## Why this experiment exists

Serving optimization does not show that a practitioner can adapt a model or reason about the interaction between training choices, memory, and quality. This experiment builds a leakage-resistant baseline, fine-tunes with LoRA and QLoRA, and runs controlled ablations rather than showcasing one favorable checkpoint.

## Dataset protocol

- The repository includes separate small training and evaluation smoke-test sets so the pipeline is runnable without training/evaluation leakage. They are not large enough to support a useful model-quality claim.
- For a formal run, expand the same schema, record provenance/license, freeze a versioned snapshot, and split by semantic template before training.
- The test split is never used for early stopping, prompt selection, or hyperparameter choice.

## Method

1. Evaluate the immutable base model with deterministic decoding.
2. Train LoRA and NF4 QLoRA adapters with the same examples, order, epochs, and seeds.
3. Record peak allocated/reserved GPU memory, wall time, tokens/s, loss curve, package versions, and adapter size.
4. Sweep rank while holding learning rate and data fixed; sweep learning rate while holding rank and data fixed; sweep data fraction while holding rank and learning rate fixed.
5. Evaluate every checkpoint on the frozen test set across three seeds and report mean plus 95% confidence interval.
6. Quantize the best accepted checkpoint and rerun the same quality and serving matrices.

## Run

```bash
python scripts/train_adapter.py \
  --config experiments/exp007_finetuning_quality/config.json \
  --method lora --rank 8 --learning-rate 0.0001 --dataset-fraction 1.0 \
  --seed 7 --output-dir artifacts/exp007/lora-r8-lr1e-4-seed7
```

Generate the complete one-factor-at-a-time rank, learning-rate, and data-size plan without using GPU time:

```bash
python scripts/run_ablation_matrix.py \
  --config experiments/exp007_finetuning_quality/config.json \
  --output-root artifacts/exp007/ablations
```

After reviewing `ablation-plan.json`, add `--execute` to run all LoRA/QLoRA cells and seeds sequentially.

Each completed cell writes `run.json` containing the immutable training and evaluation dataset hashes, selected training IDs, trainer metrics and loss history, allocated and reserved peak GPU memory, adapter size, every frozen-set prediction, and deterministic exact-match/token-F1 scores. The evaluation set is loaded only after training finishes.

## Serving-quality cross-check

Use the same `evaluate-endpoint` command and frozen dataset for the base model, accepted adapter, and quantized-base-plus-adapter endpoints. Start an adapter endpoint with:

```bash
bash scripts/start_vllm_lora.sh \
  Qwen/Qwen3-0.6B "$MODEL_REVISION" systems-lora \
  artifacts/exp007/lora-r8-lr1e-4-seed7 float16 8000
```

For a supported quantized base, pass the engine's explicit quantization method as the seventh argument. The endpoint model name is the adapter name:

```bash
python -m llm_systems_lab evaluate-endpoint \
  --base-url http://127.0.0.1:8000/v1 \
  --model systems-lora --model-revision "$MODEL_REVISION" \
  --backend-version "$BACKEND_VERSION" \
  --dataset data/evaluation/systems_qa.jsonl \
  --output-dir artifacts/exp007/serving-quality/lora
```

Preserve separate endpoint artifacts for the unadapted base, accepted adapter, and quantized-base-plus-adapter configurations. Never compare the training-time evaluation with endpoint results unless prompt formatting and token limits are identical and documented.

The experiment is complete only when the base, adapted, and post-quantization models share the same evaluation protocol and raw predictions are retained.
