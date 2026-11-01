from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .alignment import rationale_token_positions
from .attention import (
    STRATEGIES,
    deterministic_random_anchors,
    logical_pattern,
    pattern_stats,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one controlled RGSA experiment.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--tokenizer-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--window-radius", type=int, default=4)
    parser.add_argument("--anchor-budget", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument(
        "--attention-backend",
        choices=["dense_mask", "structured_sparse"],
        default="dense_mask",
        help=(
            "dense_mask preserves the original experiment; structured_sparse "
            "computes local/global edges without an n-by-n attention tensor"
        ),
    )
    parser.add_argument("--label-source", choices=["gold", "teacher"], default="gold")
    parser.add_argument("--invalid-rationale", choices=["drop", "keep", "error"], default="drop")
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def preprocessor(tokenizer, args, label_to_id):
    def apply(record):
        encoded = tokenizer(
            record["text"],
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        candidate_positions = [
            index for index, (start, end) in enumerate(offsets) if int(end) > int(start)
        ]
        if args.strategy == "rationale":
            positions = rationale_token_positions(
                record["text"], record.get("rationale_tokens", []), offsets
            )
        elif args.strategy == "shuffled":
            rationale_positions = rationale_token_positions(
                record["text"], record.get("rationale_tokens", []), offsets
            )
            positions = deterministic_random_anchors(
                candidate_positions,
                min(args.anchor_budget, len(rationale_positions)),
                f"{record['sample_id']}:shuffled",
                args.seed,
            )
        elif args.strategy == "distilled":
            positions = [
                int(position)
                for position in record.get("distilled_anchor_positions", [])
                if int(position) in candidate_positions
            ]
        elif args.strategy == "tfidf":
            positions = rationale_token_positions(
                record["text"], record.get("tfidf_tokens", []), offsets
            )
        elif args.strategy == "random":
            positions = deterministic_random_anchors(
                candidate_positions, args.anchor_budget, record["sample_id"], args.seed
            )
        else:
            positions = []
        anchor_mask = [0] * len(encoded["input_ids"])
        for position in positions[: args.anchor_budget]:
            anchor_mask[position] = 1
        label_field = "gold_label" if args.label_source == "gold" else "teacher_label"
        encoded["anchor_mask"] = anchor_mask
        encoded["labels"] = label_to_id[record[label_field]]
        return encoded

    return apply


class RGSACollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch

        copied = [dict(feature) for feature in features]
        anchors = [feature.pop("anchor_mask") for feature in copied]
        batch = self.tokenizer.pad(copied, padding=True, return_tensors="pt")
        max_length = int(batch["input_ids"].shape[1])
        batch["anchor_mask"] = torch.tensor(
            [row + [0] * (max_length - len(row)) for row in anchors],
            dtype=torch.long,
        )
        return batch


def connectivity_for_dataset(dataset, strategy: str, radius: int) -> dict[str, float]:
    active = full = 0.0
    for record in dataset:
        valid = [bool(x) for x in record["attention_mask"]]
        anchors = [bool(x) for x in record["anchor_mask"]]
        stats = pattern_stats(
            logical_pattern(valid, anchors, strategy=strategy, window_radius=radius), valid
        )
        active += stats["active_connections"]
        full += stats["full_connections"]
    ratio = active / full if full else 0.0
    return {
        "active_connections": active,
        "full_connections": full,
        "connectivity_ratio": ratio,
        "sparsity": 1.0 - ratio if full else 0.0,
    }


def main() -> None:
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    from .model import SmallTransformerClassifier, SmallTransformerConfig

    args = arguments()
    set_seed(args.seed)
    data_files = {
        "train": str(args.train_file),
        "validation": str(args.validation_file),
    }
    if args.test_file:
        data_files["test"] = str(args.test_file)
    dataset = load_dataset("json", data_files=data_files)
    if "annotation_valid" in dataset["train"].column_names:
        invalid = {
            split: sum(not bool(row.get("annotation_valid")) for row in dataset[split])
            for split in dataset
        }
        if args.invalid_rationale == "error" and any(invalid.values()):
            raise ValueError(f"Invalid rationale records found: {invalid}")
        if args.invalid_rationale == "drop":
            dataset = dataset.filter(lambda row: bool(row.get("annotation_valid")))
    label_field = "gold_label" if args.label_source == "gold" else "teacher_label"
    labels = sorted(set(dataset["train"][label_field]))
    label_to_id = {label: index for index, label in enumerate(labels)}
    for split in ("validation", "test"):
        if split not in dataset:
            continue
        unknown = set(dataset[split][label_field]) - set(labels)
        if unknown:
            raise ValueError(f"{split} contains unseen labels: {sorted(unknown)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for offset mapping")
    processed = dataset.map(
        preprocessor(tokenizer, args, label_to_id),
        remove_columns=dataset["train"].column_names,
    )
    student_config = SmallTransformerConfig(
        vocab_size=len(tokenizer),
        num_labels=len(labels),
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_size=args.ffn_size,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    model = SmallTransformerClassifier(
        student_config,
        strategy=args.strategy,
        window_radius=args.window_radius,
        attention_backend=args.attention_backend,
    )

    def metrics(prediction):
        predicted = np.argmax(prediction.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(prediction.label_ids, predicted),
            "macro_f1": f1_score(prediction.label_ids, predicted, average="macro"),
        }

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=args.disable_tqdm,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        data_collator=RGSACollator(tokenizer),
        compute_metrics=metrics,
    )
    trainer.train()
    results = trainer.evaluate()
    results.update(
        connectivity_for_dataset(processed["validation"], args.strategy, args.window_radius)
    )
    results["strategy"] = args.strategy
    results["anchor_budget"] = args.anchor_budget
    results["window_radius"] = args.window_radius
    results["attention_backend"] = args.attention_backend
    results["seed"] = args.seed
    results["train_samples"] = len(processed["train"])
    results["validation_samples"] = len(processed["validation"])
    if "test" in processed:
        results["test_samples"] = len(processed["test"])
    results["trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if "test" in processed:
        prediction_output = trainer.predict(
            processed["test"], metric_key_prefix="test"
        )
        results.update(prediction_output.metrics)
        test_connectivity = connectivity_for_dataset(
            processed["test"], args.strategy, args.window_radius
        )
        results.update({f"test_{key}": value for key, value in test_connectivity.items()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if "test" in processed:
        id_to_label = {value: key for key, value in label_to_id.items()}
        predicted_ids = np.argmax(prediction_output.predictions, axis=-1)
        with (args.output_dir / "test_predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for record, predicted_id in zip(
                dataset["test"], predicted_ids, strict=True
            ):
                predicted_label = id_to_label[int(predicted_id)]
                output_record = {
                    "sample_id": record.get("sample_id"),
                    "gold_label": record.get("gold_label"),
                    "teacher_label": record.get("teacher_label"),
                    "predicted_label": predicted_label,
                    "correct": predicted_label == record.get("gold_label"),
                    "teacher_correct": (
                        record.get("teacher_label") == record.get("gold_label")
                        if record.get("teacher_label") is not None
                        else None
                    ),
                    "rationale_tokens": record.get("rationale_tokens", []),
                    "distilled_anchor_positions": record.get(
                        "distilled_anchor_positions", []
                    ),
                }
                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    with (args.output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, default=str, indent=2, sort_keys=True)
    with (args.output_dir / "student_config.json").open("w", encoding="utf-8") as handle:
        json.dump(student_config.to_dict(), handle, indent=2, sort_keys=True)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
