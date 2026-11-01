from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .alignment import validate_rationale_tokens
from .annotate import read_local_examples
from .schema import TeacherResult, make_annotation
from .teacher import (
    PROMPT_VERSION,
    PROVIDERS,
    SYSTEM_PROMPT,
    parse_json,
    user_prompt,
)


TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=json_default,
        )


def json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def siliconflow_client():
    key = os.getenv("SILICONFLOW_API_KEY")
    if not key:
        raise RuntimeError("Set SILICONFLOW_API_KEY before using SiliconFlow batch API")
    from openai import OpenAI

    return OpenAI(api_key=key, base_url=PROVIDERS["siliconflow"].base_url)


def prepare_requests(
    input_path: Path,
    output_path: Path,
    *,
    dataset_name: str,
    split: str,
    model: str,
    max_rationale_tokens: int,
    max_output_tokens: int,
    provide_gold_label: bool,
    prompt_profile: str,
    limit: int | None,
) -> int:
    examples, labels = read_local_examples(
        input_path, dataset_name=dataset_name, split=split
    )
    if limit is not None:
        examples = examples[:limit]
    if len(examples) > 5000:
        raise ValueError("SiliconFlow allows at most 5000 requests per batch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            prompt = user_prompt(
                example["text"],
                labels,
                gold_label=example["gold_label"] if provide_gold_label else None,
                max_rationale_tokens=max_rationale_tokens,
                prompt_profile=prompt_profile,
            )
            request = {
                "custom_id": example["sample_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0,
                    "max_tokens": max_output_tokens,
                },
            }
            if model.startswith("Qwen/"):
                request["body"]["enable_thinking"] = False
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    return len(examples)


def object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def submit_batch(request_path: Path, job_path: Path, *, description: str) -> dict[str, Any]:
    client = siliconflow_client()
    with request_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    input_file_id = object_value(uploaded, "id")
    if not input_file_id:
        data = object_value(uploaded, "data", {})
        input_file_id = object_value(data, "id")
    if not input_file_id:
        raise RuntimeError(f"Could not read uploaded file ID from {uploaded!r}")
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    record = {
        "batch_id": object_value(batch, "id"),
        "input_file_id": input_file_id,
        "request_file": str(request_path),
        "status": object_value(batch, "status"),
        "output_file_id": object_value(batch, "output_file_id"),
        "error_file_id": object_value(batch, "error_file_id"),
    }
    write_json(job_path, record)
    return record


def refresh_job(job_path: Path) -> dict[str, Any]:
    client = siliconflow_client()
    record = read_json(job_path)
    batch = client.batches.retrieve(record["batch_id"])
    record.update(
        {
            "status": object_value(batch, "status"),
            "output_file_id": object_value(batch, "output_file_id"),
            "error_file_id": object_value(batch, "error_file_id"),
            "request_counts": object_value(batch, "request_counts"),
        }
    )
    write_json(job_path, record)
    return record


def recover_job(batch_id: str, job_path: Path) -> dict[str, Any]:
    client = siliconflow_client()
    batch = client.batches.retrieve(batch_id)
    record = {
        "batch_id": batch_id,
        "input_file_id": object_value(batch, "input_file_id"),
        "status": object_value(batch, "status"),
        "output_file_id": object_value(batch, "output_file_id"),
        "error_file_id": object_value(batch, "error_file_id"),
        "request_counts": object_value(batch, "request_counts"),
    }
    write_json(job_path, record)
    return record


def cancel_job(job_path: Path) -> dict[str, Any]:
    client = siliconflow_client()
    record = read_json(job_path)
    batch = client.batches.cancel(record["batch_id"])
    record["status"] = object_value(batch, "status")
    write_json(job_path, record)
    return record


def response_text(file_response: Any) -> str:
    text = getattr(file_response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(file_response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if hasattr(file_response, "read"):
        raw = file_response.read()
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    raise RuntimeError("Unsupported file-content response from SiliconFlow")


def collect_batch(
    job_path: Path,
    source_path: Path,
    output_path: Path,
    *,
    dataset_name: str,
    split: str,
    model: str,
    max_rationale_tokens: int,
    prompt_version: str,
) -> dict[str, int]:
    client = siliconflow_client()
    job = refresh_job(job_path)
    if job["status"] != "completed":
        raise RuntimeError(f"Batch {job['batch_id']} is {job['status']}, not completed")
    examples, labels = read_local_examples(
        source_path, dataset_name=dataset_name, split=split
    )
    by_id = {row["sample_id"]: row for row in examples}
    output_reference = str(job["output_file_id"])
    if output_reference.startswith(("http://", "https://")):
        import httpx

        response = httpx.get(output_reference, timeout=300.0, follow_redirects=True)
        response.raise_for_status()
        downloaded: Any = response
    else:
        downloaded = client.files.content(output_reference)
    raw_results = [
        json.loads(line)
        for line in response_text(downloaded).splitlines()
        if line.strip()
    ]
    annotations: dict[str, dict[str, Any]] = {}
    failure_records: list[dict[str, str]] = []
    for item in raw_results:
        custom_id = str(item.get("custom_id", ""))
        try:
            if item.get("error"):
                raise ValueError(str(item["error"]))
            body = item["response"]["body"]
            content = body["choices"][0]["message"]["content"]
            result = TeacherResult.from_dict(parse_json(content))
            if result.label not in labels:
                raise ValueError(f"Unexpected teacher label: {result.label}")
            example = by_id[custom_id]
            valid_tokens = validate_rationale_tokens(
                example["text"], result.rationale_tokens
            )[:max_rationale_tokens]
            annotations[custom_id] = make_annotation(
                sample=example,
                dataset=dataset_name,
                split=split,
                result=result,
                valid_tokens=valid_tokens,
                provider="siliconflow",
                model=model,
                prompt_version=prompt_version,
            )
        except Exception as exc:
            failure_records.append({"sample_id": custom_id, "error": str(exc)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            record = annotations.get(example["sample_id"])
            if record is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    missing_ids = sorted(set(by_id) - set(annotations))
    recorded_failure_ids = {row["sample_id"] for row in failure_records}
    for sample_id in missing_ids:
        if sample_id not in recorded_failure_ids:
            failure_records.append({"sample_id": sample_id, "error": "missing batch result"})
    failure_path = output_path.with_name(f"{output_path.stem}.failures.jsonl")
    with failure_path.open("w", encoding="utf-8") as handle:
        for record in failure_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    anchor_valid = sum(bool(record["annotation_valid"]) for record in annotations.values())
    return {
        "expected": len(examples),
        "received": len(raw_results),
        "parsed": len(annotations),
        "anchor_valid": anchor_valid,
        "failures": len(failure_records),
        "missing": len(examples) - len(annotations),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SiliconFlow batch annotation workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--dataset-name", default="sst2-6k")
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    prepare.add_argument("--max-rationale-tokens", type=int, default=6)
    prepare.add_argument("--max-output-tokens", type=int, default=256)
    prepare.add_argument("--provide-gold-label", action="store_true")
    prepare.add_argument(
        "--prompt-profile",
        choices=["generic", "trec", "agnews"],
        default="generic",
    )
    prepare.add_argument("--limit", type=int)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--requests", type=Path, required=True)
    submit.add_argument("--job", type=Path, required=True)
    submit.add_argument("--description", default="RGSA teacher annotation")

    status = subparsers.add_parser("status")
    status.add_argument("--job", type=Path, required=True)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--batch-id", required=True)
    recover.add_argument("--job", type=Path, required=True)

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--job", type=Path, required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--job", type=Path, required=True)
    collect.add_argument("--source", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--dataset-name", default="sst2-6k")
    collect.add_argument("--split", required=True)
    collect.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    collect.add_argument("--max-rationale-tokens", type=int, default=6)
    collect.add_argument("--prompt-version", default=PROMPT_VERSION)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.command == "prepare":
        count = prepare_requests(
            args.input,
            args.output,
            dataset_name=args.dataset_name,
            split=args.split,
            model=args.model,
            max_rationale_tokens=args.max_rationale_tokens,
            max_output_tokens=args.max_output_tokens,
            provide_gold_label=args.provide_gold_label,
            prompt_profile=args.prompt_profile,
            limit=args.limit,
        )
        print(json.dumps({"requests": count, "output": str(args.output)}, indent=2))
    elif args.command == "submit":
        print(
            json.dumps(
                submit_batch(args.requests, args.job, description=args.description),
                indent=2,
                default=str,
            )
        )
    elif args.command == "status":
        print(json.dumps(refresh_job(args.job), indent=2, default=str))
    elif args.command == "recover":
        print(
            json.dumps(
                recover_job(args.batch_id, args.job),
                indent=2,
                default=json_default,
            )
        )
    elif args.command == "cancel":
        print(json.dumps(cancel_job(args.job), indent=2, default=str))
    elif args.command == "collect":
        print(
            json.dumps(
                collect_batch(
                    args.job,
                    args.source,
                    args.output,
                    dataset_name=args.dataset_name,
                    split=args.split,
                    model=args.model,
                    max_rationale_tokens=args.max_rationale_tokens,
                    prompt_version=args.prompt_version,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
