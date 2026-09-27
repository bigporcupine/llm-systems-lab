# Experiment 007: LoRA, QLoRA, Ablations, and Quality Regression

## Why this experiment exists

Serving optimization does not show that a practitioner can adapt a model or reason about the interaction between training choices, memory, and quality. This experiment builds a leakage-resistant baseline, fine-tunes with LoRA and QLoRA, and runs controlled ablations rather than showcasing one favorable checkpoint.

## Dataset protocol

- The repository includes a small smoke-test set so the pipeline is runnable. It is not large enough to support a useful model-quality claim.
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

The experiment is complete only when the base, adapted, and post-quantization models share the same evaluation protocol and raw predictions are retained.
