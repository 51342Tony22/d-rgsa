from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .alignment import rationale_token_positions


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two teacher label and rationale-anchor datasets."
    )
    parser.add_argument("--teacher-a", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-b", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher-a-name", default="DeepSeek")
    parser.add_argument("--teacher-b-name", default="Qwen")
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id", ""))
                if not sample_id:
                    raise ValueError(f"{path}:{line_number} has no sample_id")
                if sample_id in records:
                    raise ValueError(f"Duplicate sample_id: {sample_id}")
                records[sample_id] = row
    return records


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def jaccard(first: set[int], second: set[int]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def compare(
    teacher_a: dict[str, dict[str, Any]],
    teacher_b: dict[str, dict[str, Any]],
    tokenizer,
    *,
    teacher_a_name: str,
    teacher_b_name: str,
    max_length: int,
) -> dict[str, Any]:
    common_ids = sorted(set(teacher_a) & set(teacher_b))
    if not common_ids:
        raise ValueError("The teacher files have no common sample IDs")
    label_agreements = 0
    correct_counts = Counter()
    valid_counts = Counter()
    anchor_jaccards: list[float] = []
    exact_anchor_matches = 0
    anchor_counts_a: list[float] = []
    anchor_counts_b: list[float] = []
    correctness = Counter()
    per_split = Counter()
    for sample_id in common_ids:
        first = teacher_a[sample_id]
        second = teacher_b[sample_id]
        if first["text"] != second["text"] or first["gold_label"] != second["gold_label"]:
            raise ValueError(f"Mismatched source example for {sample_id}")
        label_agreements += first.get("teacher_label") == second.get("teacher_label")
        correct_a = first.get("teacher_label") == first.get("gold_label")
        correct_b = second.get("teacher_label") == second.get("gold_label")
        correct_counts[teacher_a_name] += correct_a
        correct_counts[teacher_b_name] += correct_b
        valid_counts[teacher_a_name] += bool(first.get("annotation_valid"))
        valid_counts[teacher_b_name] += bool(second.get("annotation_valid"))
        correctness[
            f"{teacher_a_name}_{'correct' if correct_a else 'wrong'}__"
            f"{teacher_b_name}_{'correct' if correct_b else 'wrong'}"
        ] += 1
        encoded = tokenizer(
            first["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        positions_a = set(
            rationale_token_positions(
                first["text"], first.get("rationale_tokens", []), offsets
            )
        )
        positions_b = set(
            rationale_token_positions(
                second["text"], second.get("rationale_tokens", []), offsets
            )
        )
        anchor_jaccards.append(jaccard(positions_a, positions_b))
        exact_anchor_matches += positions_a == positions_b
        anchor_counts_a.append(float(len(positions_a)))
        anchor_counts_b.append(float(len(positions_b)))
        per_split[str(first.get("split", "unknown"))] += 1
    size = len(common_ids)
    return {
        "common_samples": size,
        "per_split": dict(per_split),
        "label_agreement": label_agreements / size,
        "teacher_accuracy": {
            teacher_a_name: correct_counts[teacher_a_name] / size,
            teacher_b_name: correct_counts[teacher_b_name] / size,
        },
        "anchor_valid_rate": {
            teacher_a_name: valid_counts[teacher_a_name] / size,
            teacher_b_name: valid_counts[teacher_b_name] / size,
        },
        "mean_anchor_position_jaccard": safe_mean(anchor_jaccards),
        "exact_anchor_position_match": exact_anchor_matches / size,
        "mean_anchor_positions": {
            teacher_a_name: safe_mean(anchor_counts_a),
            teacher_b_name: safe_mean(anchor_counts_b),
        },
        "correctness_overlap": dict(correctness),
        "only_in_teacher_a": len(set(teacher_a) - set(teacher_b)),
        "only_in_teacher_b": len(set(teacher_b) - set(teacher_a)),
    }


def main() -> None:
    from transformers import AutoTokenizer

    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required")
    result = compare(
        read_jsonl(args.teacher_a),
        read_jsonl(args.teacher_b),
        tokenizer,
        teacher_a_name=args.teacher_a_name,
        teacher_b_name=args.teacher_b_name,
        max_length=args.max_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
