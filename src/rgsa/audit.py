from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit teacher annotation quality.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    rows: list[dict] = []
    for path in args.input:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("No annotation records found")
    valid = [row for row in rows if row.get("annotation_valid")]
    agreements = [
        row for row in rows if row.get("teacher_label") == row.get("gold_label")
    ]
    token_counts = [len(row.get("rationale_tokens", [])) for row in rows]
    result = {
        "samples": len(rows),
        "teacher_gold_agreement": len(agreements) / len(rows),
        "teacher_errors": len(rows) - len(agreements),
        "anchor_valid_rate": len(valid) / len(rows),
        "anchor_invalid": len(rows) - len(valid),
        "mean_rationale_phrases": sum(token_counts) / len(token_counts),
        "gold_distribution": dict(Counter(row["gold_label"] for row in rows)),
        "teacher_distribution": dict(Counter(row["teacher_label"] for row in rows)),
        "providers": dict(Counter(row.get("teacher_provider", "unknown") for row in rows)),
        "models": dict(Counter(row.get("teacher_model", "unknown") for row in rows)),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
