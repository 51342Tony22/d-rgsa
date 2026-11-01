from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .alignment import rationale_token_positions
from .selector import AnchorSelector, AnchorSelectorConfig


@dataclass(frozen=True)
class EncodedExample:
    record: dict[str, Any]
    input_ids: list[int]
    attention_mask: list[int]
    selector_mask: list[int]
    anchor_targets: list[int]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill teacher rationale anchors into a lightweight token selector."
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predicted-data-dir", type=Path, required=True)
    parser.add_argument("--teacher-name", default="deepseek")
    parser.add_argument("--tokenizer-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--architecture", choices=["bigru", "mlp"], default="bigru")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--anchor-budget", type=int, default=4)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--selector-hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-positive-weight", type=float, default=20.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number} has no text")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def encode_records(
    records: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
    anchor_budget: int,
) -> list[EncodedExample]:
    encoded_rows: list[EncodedExample] = []
    for record in records:
        encoded = tokenizer(
            record["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        selector_mask = [
            int(int(end) > int(start)) for start, end in offsets
        ]
        target_positions = rationale_token_positions(
            record["text"],
            record.get("rationale_tokens", []),
            offsets,
        )[:anchor_budget]
        targets = [0] * len(encoded["input_ids"])
        for position in target_positions:
            if selector_mask[position]:
                targets[position] = 1
        encoded_rows.append(
            EncodedExample(
                record=record,
                input_ids=[int(value) for value in encoded["input_ids"]],
                attention_mask=[int(value) for value in encoded["attention_mask"]],
                selector_mask=selector_mask,
                anchor_targets=targets,
            )
        )
    return encoded_rows


class SelectorDataset(Dataset):
    def __init__(self, rows: list[EncodedExample]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.rows[index]


class SelectorCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    @staticmethod
    def _pad(values: list[int], length: int, fill: int) -> list[int]:
        return values + [fill] * (length - len(values))

    def __call__(self, rows: list[EncodedExample]) -> dict[str, Any]:
        max_length = max(len(row.input_ids) for row in rows)
        return {
            "input_ids": torch.tensor(
                [
                    self._pad(row.input_ids, max_length, self.pad_token_id)
                    for row in rows
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [self._pad(row.attention_mask, max_length, 0) for row in rows],
                dtype=torch.long,
            ),
            "selector_mask": torch.tensor(
                [self._pad(row.selector_mask, max_length, 0) for row in rows],
                dtype=torch.bool,
            ),
            "anchor_targets": torch.tensor(
                [self._pad(row.anchor_targets, max_length, 0) for row in rows],
                dtype=torch.float32,
            ),
            "rows": rows,
        }


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def positive_weight(rows: list[EncodedExample], maximum: float) -> float:
    positives = sum(sum(row.anchor_targets) for row in rows)
    candidates = sum(sum(row.selector_mask) for row in rows)
    negatives = max(0, candidates - positives)
    if positives <= 0:
        raise ValueError("Training data contain no aligned rationale anchors")
    return min(maximum, negatives / positives)


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    selector_mask: torch.Tensor,
    *,
    positive_class_weight: float,
) -> torch.Tensor:
    weights = torch.where(
        targets > 0.5,
        torch.full_like(targets, positive_class_weight),
        torch.ones_like(targets),
    )
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        weight=weights,
        reduction="none",
    )
    mask = selector_mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def topk_positions(
    logits: torch.Tensor,
    selector_mask: torch.Tensor,
    budget: int,
) -> list[list[int]]:
    selected: list[list[int]] = []
    for row_logits, row_mask in zip(logits, selector_mask, strict=True):
        candidates = row_mask.nonzero(as_tuple=False).flatten()
        count = min(max(0, budget), int(candidates.numel()))
        if count == 0:
            selected.append([])
            continue
        candidate_scores = row_logits[candidates]
        indices = torch.topk(candidate_scores, k=count).indices
        selected.append([int(value) for value in candidates[indices].tolist()])
    return selected


def selection_metrics(
    predictions: list[list[int]],
    targets: list[list[int]],
) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    jaccards: list[float] = []
    exact = 0
    for predicted_values, target_values in zip(predictions, targets, strict=True):
        predicted = set(predicted_values)
        target = set(target_values)
        true_positive += len(predicted & target)
        false_positive += len(predicted - target)
        false_negative += len(target - predicted)
        union = predicted | target
        jaccards.append(len(predicted & target) / len(union) if union else 1.0)
        exact += int(predicted == target)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1.0e-12, precision + recall)
    return {
        "anchor_precision": precision,
        "anchor_recall": recall,
        "anchor_f1": f1,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "exact_match": exact / max(1, len(predictions)),
        "samples": float(len(predictions)),
    }


@torch.no_grad()
def predict(
    model: AnchorSelector,
    loader: DataLoader,
    *,
    device: torch.device,
    anchor_budget: int,
) -> tuple[list[list[int]], list[list[int]], list[list[float]], float]:
    model.eval()
    predictions: list[list[int]] = []
    targets: list[list[int]] = []
    scores: list[list[float]] = []
    loss_total = 0.0
    batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        selector_mask = batch["selector_mask"].to(device)
        logits = model(input_ids, attention_mask)
        batch_predictions = topk_positions(logits, selector_mask, anchor_budget)
        for index, positions in enumerate(batch_predictions):
            predictions.append(positions)
            targets.append(
                [
                    int(position)
                    for position in (
                        batch["anchor_targets"][index] > 0.5
                    ).nonzero(as_tuple=False).flatten().tolist()
                ]
            )
            scores.append(
                [
                    float(torch.sigmoid(logits[index, position]).cpu())
                    for position in positions
                ]
            )
        batches += 1
    return predictions, targets, scores, loss_total / max(1, batches)


def make_loader(
    rows: list[EncodedExample],
    *,
    batch_size: int,
    pad_token_id: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        SelectorDataset(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=SelectorCollator(pad_token_id),
    )


def write_predictions(
    path: Path,
    rows: list[EncodedExample],
    predictions: list[list[int]],
    scores: list[list[float]],
    tokenizer,
    *,
    teacher_name: str,
    architecture: str,
    anchor_budget: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row, positions, probabilities in zip(
            rows,
            predictions,
            scores,
            strict=True,
        ):
            record = dict(row.record)
            record.update(
                {
                    "distilled_anchor_positions": positions,
                    "distilled_anchor_scores": probabilities,
                    "distilled_anchor_tokens": tokenizer.convert_ids_to_tokens(
                        [row.input_ids[position] for position in positions]
                    ),
                    "distilled_selector_teacher": teacher_name,
                    "distilled_selector_architecture": architecture,
                    "distilled_anchor_budget": anchor_budget,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    from transformers import AutoTokenizer

    args = arguments()
    if args.anchor_budget <= 0:
        raise ValueError("anchor-budget must be positive")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required")
    raw = {
        "train": read_jsonl(args.train_file),
        "validation": read_jsonl(args.validation_file),
        "test": read_jsonl(args.test_file),
    }
    encoded = {
        split: encode_records(
            records,
            tokenizer,
            max_length=args.max_length,
            anchor_budget=args.anchor_budget,
        )
        for split, records in raw.items()
    }
    usable = {
        split: [
            row
            for row in rows
            if bool(row.record.get("annotation_valid", True))
            and any(row.anchor_targets)
        ]
        for split, rows in encoded.items()
    }
    selector_config = AnchorSelectorConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_length,
        embedding_size=args.embedding_size,
        hidden_size=args.selector_hidden_size,
        architecture=args.architecture,
        dropout=args.dropout,
    )
    model = AnchorSelector(selector_config)
    device = choose_device()
    model.to(device)
    class_weight = positive_weight(usable["train"], args.max_positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(
        usable["train"],
        batch_size=args.batch_size,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=True,
        seed=args.seed,
    )
    evaluation_loaders = {
        split: make_loader(
            rows,
            batch_size=args.batch_size,
            pad_token_id=tokenizer.pad_token_id,
            shuffle=False,
            seed=args.seed,
        )
        for split, rows in usable.items()
    }
    output_loaders = {
        split: make_loader(
            rows,
            batch_size=args.batch_size,
            pad_token_id=tokenizer.pad_token_id,
            shuffle=False,
            seed=args.seed,
        )
        for split, rows in encoded.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "selector.pt"
    best_f1 = -math.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            loss = masked_bce_loss(
                logits,
                batch["anchor_targets"].to(device),
                batch["selector_mask"].to(device),
                positive_class_weight=class_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            batches += 1
        validation_predictions, validation_targets, _, _ = predict(
            model,
            evaluation_loaders["validation"],
            device=device,
            anchor_budget=args.anchor_budget,
        )
        validation_metrics = selection_metrics(
            validation_predictions,
            validation_targets,
        )
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(1, batches),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True))
        if validation_metrics["anchor_f1"] > best_f1:
            best_f1 = validation_metrics["anchor_f1"]
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    all_metrics: dict[str, Any] = {
        "teacher_name": args.teacher_name,
        "architecture": args.architecture,
        "seed": args.seed,
        "anchor_budget": args.anchor_budget,
        "positive_class_weight": class_weight,
        "device": str(device),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "usable_samples": {split: len(rows) for split, rows in usable.items()},
        "all_samples": {split: len(rows) for split, rows in encoded.items()},
        "history": history,
    }
    for split, loader in evaluation_loaders.items():
        predictions, targets, _, _ = predict(
            model,
            loader,
            device=device,
            anchor_budget=args.anchor_budget,
        )
        all_metrics[split] = selection_metrics(predictions, targets)
        output_predictions, _, output_scores, _ = predict(
            model,
            output_loaders[split],
            device=device,
            anchor_budget=args.anchor_budget,
        )
        write_predictions(
            args.predicted_data_dir / f"{split}.jsonl",
            encoded[split],
            output_predictions,
            output_scores,
            tokenizer,
            teacher_name=args.teacher_name,
            architecture=args.architecture,
            anchor_budget=args.anchor_budget,
        )
    with (args.output_dir / "selector_config.json").open("w", encoding="utf-8") as handle:
        json.dump(selector_config.to_dict(), handle, indent=2, sort_keys=True)
    with (args.output_dir / "distillation_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(vars(args), handle, default=str, indent=2, sort_keys=True)
    with (args.output_dir / "selector_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(all_metrics, handle, indent=2, sort_keys=True)
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps(all_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
