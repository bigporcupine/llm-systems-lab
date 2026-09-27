#!/usr/bin/env python3
"""Train a reproducible LoRA or QLoRA adapter from an Experiment 007 config."""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("lora", "qlora-nf4"), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--dataset-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not 0 < args.dataset_fraction <= 1:
        raise SystemExit("--dataset-fraction must be in (0, 1]")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install the training extra before running Experiment 007") from exc

    rows = [json.loads(line) for line in Path(config["dataset"]).read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: max(1, round(len(rows) * args.dataset_fraction))]
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["model_revision"])
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if args.method == "qlora-nf4" else None
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["model_revision"], quantization_config=quant, torch_dtype=torch.float16, device_map="auto")
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    if args.method == "qlora-nf4":
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora)

    def tokenize(row):
        text = f"Question: {row['prompt']}\nAnswer: {row['references'][0]}"
        encoded = tokenizer(text, truncation=True, max_length=config["max_length"])
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    dataset = Dataset.from_list(rows).map(tokenize, remove_columns=list(rows[0]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = TrainingArguments(output_dir=str(args.output_dir), num_train_epochs=config["epochs"], learning_rate=args.learning_rate, per_device_train_batch_size=1, gradient_accumulation_steps=8, logging_steps=1, save_strategy="epoch", seed=args.seed, data_seed=args.seed, report_to=[])
    trainer = Trainer(model=model, args=training, train_dataset=dataset, processing_class=tokenizer)
    result = trainer.train()
    trainer.save_model()
    metadata = {"config": config, "method": args.method, "rank": args.rank, "learning_rate": args.learning_rate, "dataset_fraction": args.dataset_fraction, "seed": args.seed, "train_metrics": result.metrics, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
