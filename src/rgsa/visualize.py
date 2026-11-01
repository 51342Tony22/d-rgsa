from __future__ import annotations

import argparse
import json
from pathlib import Path

from .alignment import rationale_token_positions
from .attention import STRATEGIES, logical_pattern


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a logical attention mask.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--strategy", choices=STRATEGIES, default="rationale")
    parser.add_argument("--tokenizer-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--window-radius", type=int, default=4)
    parser.add_argument("--anchor-budget", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from transformers import AutoTokenizer

    args = arguments()
    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    row = rows[args.sample_index]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    encoded = tokenizer(
        row["text"],
        truncation=True,
        max_length=args.max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    source = row.get("rationale_tokens", []) if args.strategy == "rationale" else []
    if args.strategy == "tfidf":
        source = row.get("tfidf_tokens", [])
    positions = rationale_token_positions(row["text"], source, offsets)[: args.anchor_budget]
    anchors = [index in positions for index in range(len(encoded["input_ids"]))]
    valid = [bool(x) for x in encoded["attention_mask"]]
    graph = logical_pattern(
        valid, anchors, strategy=args.strategy, window_radius=args.window_radius
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    figure_size = max(6.0, min(14.0, len(tokens) * 0.35))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    ax.imshow(np.asarray(graph, dtype=int), cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=7)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    ax.set_title(f"{args.strategy} attention pattern")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
