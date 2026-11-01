from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


LABEL_NAMES = ("negative", "positive")


def stratified_indices(
    labels: list[int],
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[int]]:
    from sklearn.model_selection import train_test_split

    total = train_size + validation_size + test_size
    if total > len(labels):
        raise ValueError(f"Requested {total} examples, but only {len(labels)} are available")
    all_indices = list(range(len(labels)))
    selected, _ = train_test_split(
        all_indices,
        train_size=total,
        random_state=seed,
        stratify=labels,
    )
    selected_labels = [labels[index] for index in selected]
    train, remainder = train_test_split(
        selected,
        train_size=train_size,
        random_state=seed + 1,
        stratify=selected_labels,
    )
    remainder_labels = [labels[index] for index in remainder]
    validation, test = train_test_split(
        remainder,
        train_size=validation_size,
        test_size=test_size,
        random_state=seed + 2,
        stratify=remainder_labels,
    )
    result = {
        "train": sorted(train),
        "validation": sorted(validation),
        "test": sorted(test),
    }
    flattened = [index for split in result.values() for index in split]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Data split overlap detected")
    return result


def write_jsonl(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a stratified labeled SST-2 subset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=4800)
    parser.add_argument("--validation-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = arguments()
    source = load_dataset("stanfordnlp/sst2", split="train")
    labels = [int(value) for value in source["label"]]
    splits = stratified_indices(
        labels,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    manifest = {
        "dataset": "stanfordnlp/sst2",
        "source_split": "train",
        "seed": args.seed,
        "total_size": sum(len(indices) for indices in splits.values()),
        "splits": {},
    }
    for split, indices in splits.items():
        records = [
            {
                "sample_id": f"sst2-6k:{split}:{position}",
                "source_idx": int(source[index]["idx"]),
                "text": str(source[index]["sentence"]).strip(),
                "gold_label": LABEL_NAMES[int(source[index]["label"])],
            }
            for position, index in enumerate(indices)
        ]
        path = args.output_dir / f"{split}.jsonl"
        digest = write_jsonl(path, records)
        distribution = dict(sorted(Counter(row["gold_label"] for row in records).items()))
        manifest["splits"][split] = {
            "file": path.name,
            "size": len(records),
            "label_distribution": distribution,
            "sha256": digest,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
