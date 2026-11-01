from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .alignment import validate_rationale_tokens
from .data import SPECS, load_benchmark, normalize
from .schema import make_annotation
from .teacher import PROMPT_VERSION, TeacherClient


def read_local_examples(
    path: Path, *, dataset_name: str, split: str
) -> tuple[list[dict[str, str]], list[str]]:
    examples: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("text", "gold_label"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{path} row {index + 1} has invalid {field!r}")
            examples.append(
                {
                    "sample_id": str(
                        row.get("sample_id", f"{dataset_name}:{split}:{index}")
                    ),
                    "text": row["text"].strip(),
                    "gold_label": row["gold_label"].strip(),
                }
            )
    if not examples:
        raise ValueError(f"No examples found in {path}")
    labels = sorted({row["gold_label"] for row in examples})
    return examples, labels


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result.add(str(json.loads(line)["sample_id"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return result


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher rationale annotations.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=sorted(SPECS))
    source.add_argument("--input-file", type=Path)
    parser.add_argument("--dataset-name", default="local")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek", "qwen", "siliconflow", "compatible", "local-hf"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-rationale-tokens", type=int, default=6)
    parser.add_argument(
        "--prompt-profile",
        choices=["generic", "trec", "agnews"],
        default="generic",
    )
    parser.add_argument("--provide-gold-label", action="store_true")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.input_file:
        examples, label_names = read_local_examples(
            args.input_file,
            dataset_name=args.dataset_name,
            split=args.split,
        )
        dataset_name = args.dataset_name
    else:
        dataset, spec = load_benchmark(args.dataset, args.split)
        examples = (
            normalize(raw, spec, args.dataset, args.split, index)
            for index, raw in enumerate(dataset)
        )
        label_names = list(spec.label_names)
        dataset_name = args.dataset
    teacher = TeacherClient(args.provider, args.model, base_url=args.base_url)
    done = completed_ids(args.output)
    written = failures = 0
    for index, example in enumerate(examples):
        if index < args.start_index:
            continue
        if example["sample_id"] in done:
            continue
        if args.limit is not None and written >= args.limit:
            break
        for attempt in range(args.max_retries):
            try:
                generated = teacher.generate(
                    text=example["text"],
                    labels=label_names,
                    gold_label=example["gold_label"] if args.provide_gold_label else None,
                    max_rationale_tokens=args.max_rationale_tokens,
                    prompt_profile=args.prompt_profile,
                )
                valid = validate_rationale_tokens(
                    example["text"], generated.rationale_tokens
                )[: args.max_rationale_tokens]
                record = make_annotation(
                    sample=example,
                    dataset=dataset_name,
                    split=args.split,
                    result=generated,
                    valid_tokens=valid,
                    provider=args.provider,
                    model=args.model,
                    prompt_version=PROMPT_VERSION,
                )
                append_jsonl(args.output, record)
                written += 1
                print(
                    f"[{written}] {example['sample_id']} "
                    f"anchors={len(valid)} valid={bool(valid)}"
                )
                break
            except Exception as exc:
                if attempt + 1 == args.max_retries:
                    failures += 1
                    print(f"FAILED {example['sample_id']}: {exc}")
                    break
                jitter = random.Random(args.seed + index + attempt).random()
                delay = min(60.0, 2**attempt + jitter)
                print(f"Retrying {example['sample_id']} in {delay:.1f}s: {exc}")
                time.sleep(delay)
    print(f"Done: wrote={written}, failures={failures}, output={args.output}")


if __name__ == "__main__":
    main()
