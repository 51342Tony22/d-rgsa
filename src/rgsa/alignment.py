from __future__ import annotations

import re
from collections.abc import Iterable, Sequence


def surface_spans(text: str, phrases: Iterable[str]) -> list[tuple[int, int]]:
    """Find all case-insensitive literal occurrences of teacher-selected phrases."""
    spans: set[tuple[int, int]] = set()
    for phrase in phrases:
        clean = phrase.strip()
        if not clean:
            continue
        for match in re.finditer(re.escape(clean), text, flags=re.IGNORECASE):
            spans.add((match.start(), match.end()))
    return sorted(spans)


def validate_rationale_tokens(text: str, tokens: Iterable[str]) -> list[str]:
    """Keep unique strings that really occur in the original input."""
    valid: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        clean = token.strip()
        key = clean.casefold()
        if clean and key not in seen and surface_spans(text, [clean]):
            valid.append(clean)
            seen.add(key)
    return valid


def spans_to_token_positions(
    spans: Sequence[tuple[int, int]],
    offset_mapping: Sequence[Sequence[int]],
) -> list[int]:
    """Map character spans to all overlapping student-tokenizer positions."""
    positions: list[int] = []
    for position, pair in enumerate(offset_mapping):
        start, end = int(pair[0]), int(pair[1])
        if start == end:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            positions.append(position)
    return positions


def rationale_token_positions(
    text: str,
    phrases: Iterable[str],
    offset_mapping: Sequence[Sequence[int]],
) -> list[int]:
    # Preserve teacher/TF-IDF ranking when an anchor budget truncates the list.
    positions: list[int] = []
    seen: set[int] = set()
    for phrase in phrases:
        for position in spans_to_token_positions(surface_spans(text, [phrase]), offset_mapping):
            if position not in seen:
                positions.append(position)
                seen.add(position)
    return positions
