from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .alignment import validate_rationale_tokens
from .annotate import append_jsonl, completed_ids, read_local_examples
from .schema import make_annotation
from .teacher import PROMPT_VERSION, TeacherClient


_THREAD_LOCAL = threading.local()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable concurrent teacher annotation via an OpenAI-compatible API."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--provider",
        choices=["deepseek", "qwen", "siliconflow", "compatible"],
        default="siliconflow",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--prompt-profile", choices=["generic", "trec", "agnews"])
    parser.add_argument("--max-rationale-tokens", type=int, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--provide-gold-label", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_prompt_profile(dataset_name: str) -> str:
    normalized = dataset_name.lower().replace("_", "-")
    if normalized == "trec":
        return "trec"
    if normalized in {"agnews", "ag-news", "agnews-6k", "ag-news-6k"}:
        return "agnews"
    return "generic"


def thread_client(args: argparse.Namespace) -> TeacherClient:
    client = getattr(_THREAD_LOCAL, "teacher_client", None)
    if client is None:
        client = TeacherClient(
            args.provider,
            args.model,
            base_url=args.base_url,
            max_new_tokens=args.max_output_tokens,
        )
        _THREAD_LOCAL.teacher_client = client
    return client


def annotate_one(
    example: dict[str, str],
    labels: list[str],
    args: argparse.Namespace,
    prompt_profile: str,
    ordinal: int,
) -> dict[str, Any]:
    for attempt in range(args.max_retries):
        try:
            generated = thread_client(args).generate(
                text=example["text"],
                labels=labels,
                gold_label=example["gold_label"] if args.provide_gold_label else None,
                max_rationale_tokens=args.max_rationale_tokens,
                prompt_profile=prompt_profile,
            )
            valid = validate_rationale_tokens(
                example["text"], generated.rationale_tokens
            )[: args.max_rationale_tokens]
            return make_annotation(
                sample=example,
                dataset=args.dataset_name,
                split=args.split,
                result=generated,
                valid_tokens=valid,
                provider=args.provider,
                model=args.model,
                prompt_version=PROMPT_VERSION,
            )
        except Exception:
            if attempt + 1 == args.max_retries:
                raise
            jitter = random.Random(args.seed + ordinal * 1009 + attempt).random()
            time.sleep(min(60.0, 2**attempt + jitter))
    raise AssertionError("retry loop ended unexpectedly")


def main() -> None:
    args = arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")
    examples, labels = read_local_examples(
        args.input,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    done = completed_ids(args.output)
    pending = [row for row in examples if row["sample_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    profile = args.prompt_profile or default_prompt_profile(args.dataset_name)
    failure_path = args.output.with_name(f"{args.output.stem}.failures.jsonl")
    written = failures = 0
    futures: dict[Future[dict[str, Any]], dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for ordinal, example in enumerate(pending):
            future = executor.submit(
                annotate_one,
                example,
                labels,
                args,
                profile,
                ordinal,
            )
            futures[future] = example
        for future in as_completed(futures):
            example = futures[future]
            try:
                record = future.result()
                append_jsonl(args.output, record)
                written += 1
                if (
                    written <= 10
                    or written % args.log_every == 0
                    or written == len(pending)
                ):
                    print(
                        f"[{written}/{len(pending)}] {example['sample_id']} "
                        f"anchors={len(record['rationale_tokens'])} "
                        f"valid={record['annotation_valid']}",
                        flush=True,
                    )
            except Exception as exc:
                failures += 1
                append_jsonl(
                    failure_path,
                    {"sample_id": example["sample_id"], "error": str(exc)},
                )
                print(f"FAILED {example['sample_id']}: {exc}", flush=True)
    print(
        json.dumps(
            {
                "already_completed": len(done),
                "submitted": len(pending),
                "written": written,
                "failures": failures,
                "output": str(args.output),
                "provider": args.provider,
                "model": args.model,
                "prompt_profile": profile,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
