from __future__ import annotations

import argparse
import json
from pathlib import Path

from .alignment import validate_rationale_tokens


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add TF-IDF anchor tokens to JSONL data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-input", type=Path)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2], default=1)
    return parser.parse_args()


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    args = arguments()
    rows = read_jsonl(args.input)
    fit_rows = read_jsonl(args.fit_input) if args.fit_input else rows
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, args.ngram_max),
        stop_words="english",
        token_pattern=r"(?u)\b\w[\w'-]*\b",
    )
    vectorizer.fit([row["text"] for row in fit_rows])
    matrix = vectorizer.transform([row["text"] for row in rows])
    features = vectorizer.get_feature_names_out()
    for row_index, row in enumerate(rows):
        vector = matrix.getrow(row_index)
        ranked = sorted(
            zip(vector.indices, vector.data),
            key=lambda item: (-float(item[1]), str(features[item[0]])),
        )
        candidates = [str(features[index]) for index, _ in ranked[: args.top_k]]
        row["tfidf_tokens"] = validate_rationale_tokens(row["text"], candidates)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()

