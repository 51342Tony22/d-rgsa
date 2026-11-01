from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


TEACHER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string"},
        "short_rationale": {"type": "string"},
        "rationale_tokens": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["label", "short_rationale", "rationale_tokens"],
}


@dataclass(frozen=True)
class TeacherResult:
    label: str
    short_rationale: str
    rationale_tokens: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherResult":
        if not isinstance(value, dict):
            raise ValueError("Teacher output must be a JSON object")
        label = value.get("label")
        rationale = value.get("short_rationale")
        tokens = value.get("rationale_tokens")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("short_rationale must be a non-empty string")
        if not isinstance(tokens, list) or not all(isinstance(x, str) for x in tokens):
            raise ValueError("rationale_tokens must be a list of strings")
        cleaned = tuple(x.strip() for x in tokens if x.strip())
        if not cleaned:
            raise ValueError("rationale_tokens cannot be empty")
        return cls(label.strip(), rationale.strip(), cleaned)


def make_annotation(
    *,
    sample: dict[str, str],
    dataset: str,
    split: str,
    result: TeacherResult,
    valid_tokens: list[str],
    provider: str,
    model: str,
    prompt_version: str,
) -> dict[str, Any]:
    return {
        **sample,
        "dataset": dataset,
        "split": split,
        "teacher_label": result.label,
        "short_rationale": result.short_rationale,
        "rationale_tokens": valid_tokens,
        "annotation_valid": bool(valid_tokens),
        "teacher_provider": provider,
        "teacher_model": model,
        "prompt_version": prompt_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

