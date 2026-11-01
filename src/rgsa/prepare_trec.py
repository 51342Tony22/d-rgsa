from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .prepare import write_jsonl


LABEL_NAMES = (
    "abbreviation",
    "entity",
    "description",
    "human",
    "location",
    "numeric",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare labeled TREC train/validation/test.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    args = arguments()
    # Parquet conversion of CogComp/trec; compatible with datasets>=5 where
    # executable dataset scripts are no longer supported.
    source = load_dataset("omkar334/trec")
    train_source = source["train"]
    test_source = source["test"]
    all_train_indices = list(range(len(train_source)))
    train_indices, validation_indices = train_test_split(
        all_train_indices,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=[int(value) for value in train_source["coarse_label"]],
    )
    split_sources = {
        "train": (train_source, sorted(train_indices)),
        "validation": (train_source, sorted(validation_indices)),
        "test": (test_source, list(range(len(test_source)))),
    }
    manifest = {
        "dataset": "omkar334/trec (Parquet conversion of CogComp/trec)",
        "seed": args.seed,
        "label_names": list(LABEL_NAMES),
        "splits": {},
    }
    for split, (dataset, indices) in split_sources.items():
        records = [
            {
                "sample_id": f"trec:{split}:{position}",
                "source_idx": int(index),
                "text": str(dataset[index]["text"]).strip(),
                "gold_label": LABEL_NAMES[int(dataset[index]["coarse_label"])],
            }
            for position, index in enumerate(indices)
        ]
        path = args.output_dir / f"{split}.jsonl"
        digest = write_jsonl(path, records)
        manifest["splits"][split] = {
            "file": path.name,
            "size": len(records),
            "label_distribution": dict(
                sorted(Counter(row["gold_label"] for row in records).items())
            ),
            "sha256": digest,
        }
    manifest["total_size"] = sum(item["size"] for item in manifest["splits"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
