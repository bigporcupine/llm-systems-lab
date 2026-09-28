#!/usr/bin/env python3
"""Train a reproducible LoRA or QLoRA adapter from an Experiment 007 config."""

import argparse
import json
import random
from pathlib import Path

from llm_systems_lab.environment import capture_environment
from llm_systems_lab.config_resolution import require_immutable_revision
from llm_systems_lab.manifest import sha256_file
from llm_systems_lab.quality import load_jsonl, score_predictions


def evaluate_model(model, tokenizer, rows, max_new_tokens):
    """Run deterministic generation and retain every frozen-set prediction."""
    predictions = {}
    model.eval()
    for row in rows:
        prompt = f"Question: {row['prompt']}\nAnswer:"
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
        encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        predictions[str(row["id"])] = tokenizer.decode(
            generated[0][input_length:], skip_special_tokens=True
        ).strip()
    return predictions


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
        require_immutable_revision(config["model_revision"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.rank not in config["ranks"] or args.learning_rate not in config["learning_rates"]:
        raise SystemExit("Rank and learning rate must be declared in the experiment config")
    if args.dataset_fraction not in config["dataset_fractions"] or args.seed not in config["seeds"]:
        raise SystemExit("Dataset fraction and seed must be declared in the experiment config")
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("Install the training extra before running Experiment 007") from exc

    rows = [json.loads(line) for line in Path(config["training_dataset"]).read_text().splitlines() if line.strip()]
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
    evaluation_path = Path(config["evaluation_dataset"])
    evaluation_rows = load_jsonl(evaluation_path)
    predictions = evaluate_model(
        model, tokenizer, evaluation_rows, int(config.get("max_eval_new_tokens", 128))
    )
    scores = score_predictions(evaluation_rows, predictions)
    adapter_size_bytes = sum(
        path.stat().st_size for path in args.output_dir.rglob("*") if path.is_file()
    )
    metadata = {
        "schema_version": "1.0", "source": "training",
        "base_model": config["base_model"], "model_revision": config["model_revision"],
        "config": config, "method": args.method, "rank": args.rank,
        "learning_rate": args.learning_rate, "dataset_fraction": args.dataset_fraction,
        "seed": args.seed,
        "training_dataset_sha256": sha256_file(Path(config["training_dataset"])),
        "selected_training_ids": [row["id"] for row in rows],
        "evaluation_dataset_sha256": sha256_file(evaluation_path),
        "evaluation_ids": [row["id"] for row in evaluation_rows],
        "predictions": predictions, "evaluation_scores": scores,
        "environment": capture_environment(), "train_metrics": result.metrics,
        "trainer_log_history": trainer.state.log_history,
        "peak_gpu_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
        "adapter_size_bytes": adapter_size_bytes,
    }
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
