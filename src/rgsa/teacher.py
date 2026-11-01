from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .schema import TEACHER_JSON_SCHEMA, TeacherResult


PROMPT_VERSION = "rgsa-v1"
SYSTEM_PROMPT = """You annotate text-classification examples for a sparse-attention study.
Return only one JSON object with label, short_rationale, and rationale_tokens.
short_rationale must be one concise evidence-based sentence, not private chain-of-thought.
Every rationale_tokens item must be a short exact contiguous substring copied from the input.
Choose only evidence that supports the classification. Never return indices or paraphrases."""

TREC_LABEL_GUIDANCE = """TREC coarse labels are defined by the expected answer type:
- abbreviation: an abbreviation or its expansion.
- entity: an animal, object, product, food, language, substance, creative work, event, or other thing.
- description: a definition, explanation, manner, reason, or abstract description.
- human: a person, group of people, profession, or human title.
- location: a city, country, state, mountain, or other place.
- numeric: a date, count, amount, distance, duration, percentage, size, or other number.
Classify by the expected answer type, not merely by the first question word."""

AGNEWS_LABEL_GUIDANCE = """AG News labels are topic categories:
- world: international affairs, politics, diplomacy, war, disasters, or global public events.
- sports: athletic competitions, teams, players, scores, leagues, or sporting events.
- business: companies, markets, finance, trade, corporate actions, jobs, or the economy.
- sci/tech: science, technology, software, hardware, space, medicine, research, or gadgets.
Return exactly one of these lowercase labels: world, sports, business, sci/tech.
Use the article's main topic rather than incidental named entities."""

PROMPT_PROFILES = {
    "generic": "",
    "trec": TREC_LABEL_GUIDANCE,
    "agnews": AGNEWS_LABEL_GUIDANCE,
}


@dataclass(frozen=True)
class Provider:
    base_url: str | None
    key_environment_variable: str


PROVIDERS = {
    "openai": Provider(None, "OPENAI_API_KEY"),
    "deepseek": Provider("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "qwen": Provider(
        "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"
    ),
    "siliconflow": Provider("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
    "compatible": Provider(None, "TEACHER_API_KEY"),
}


def user_prompt(
    text: str,
    labels: list[str],
    *,
    gold_label: str | None,
    max_rationale_tokens: int,
    prompt_profile: str = "generic",
) -> str:
    if prompt_profile not in PROMPT_PROFILES:
        raise ValueError(f"Unknown prompt profile: {prompt_profile}")
    label_instruction = (
        f"The verified gold label is {json.dumps(gold_label)}; return exactly that label."
        if gold_label is not None
        else "Predict exactly one label from the label set."
    )
    task_prompt = (
        f"Label set: {json.dumps(labels, ensure_ascii=False)}\n"
        f"{label_instruction}\n"
        f"Select between 1 and {max_rationale_tokens} evidence substrings.\n"
        f"Input text:\n{text}"
    )
    guidance = PROMPT_PROFILES[prompt_profile]
    return f"{guidance}\n\n{task_prompt}" if guidance else task_prompt


def parse_json(text: str) -> dict[str, Any]:
    clean = text.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not match:
            raise ValueError("Teacher did not return a JSON object")
        return json.loads(match.group(0))


class TeacherClient:
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        base_url: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_new_tokens = max_new_tokens
        self._client: Any = None
        self._pipeline: Any = None
        if provider == "local-hf":
            self.base_url = None
            return
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        settings = PROVIDERS[provider]
        self.base_url = base_url or settings.base_url
        api_key = os.getenv(settings.key_environment_variable)
        if not api_key:
            raise RuntimeError(f"Set {settings.key_environment_variable} first")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=self.base_url)

    def generate(
        self,
        *,
        text: str,
        labels: list[str],
        gold_label: str | None,
        max_rationale_tokens: int,
        prompt_profile: str = "generic",
    ) -> TeacherResult:
        prompt = user_prompt(
            text,
            labels,
            gold_label=gold_label,
            max_rationale_tokens=max_rationale_tokens,
            prompt_profile=prompt_profile,
        )
        if self.provider == "local-hf":
            output = self._local(prompt)
        elif self.provider == "openai":
            output = self._openai(prompt)
        else:
            output = self._compatible(prompt)
        result = TeacherResult.from_dict(parse_json(output))
        if result.label not in labels:
            raise ValueError(f"Teacher label {result.label!r} is not in {labels}")
        if len(result.rationale_tokens) > max_rationale_tokens:
            result = TeacherResult(
                result.label,
                result.short_rationale,
                result.rationale_tokens[:max_rationale_tokens],
            )
        return result

    def _openai(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "teacher_annotation",
                    "strict": True,
                    "schema": TEACHER_JSON_SCHEMA,
                }
            },
        )
        return response.output_text

    def _compatible(self, prompt: str) -> str:
        extra_body = None
        if self.provider == "siliconflow" and self.model.startswith("Qwen/"):
            # Qwen reasoning can consume the complete output budget before
            # emitting `content`; annotation needs only the concise JSON answer.
            extra_body = {"enable_thinking": False}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=self.max_new_tokens,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Teacher returned empty content")
        return content

    def _local(self, prompt: str) -> str:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="text-generation",
                model=self.model,
                dtype="auto",
                device_map="auto",
            )
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        generated = self._pipeline(chat, max_new_tokens=self.max_new_tokens, do_sample=False)
        value = generated[0]["generated_text"]
        if isinstance(value, list):
            return str(value[-1]["content"])
        return str(value)
