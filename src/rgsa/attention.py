from __future__ import annotations

import hashlib
from collections.abc import Sequence


STRATEGIES = (
    "full",
    "local",
    "random",
    "tfidf",
    "rationale",
    "distilled",
    "shuffled",
)


def deterministic_random_anchors(
    candidates: Sequence[int], budget: int, sample_id: str, seed: int
) -> list[int]:
    """Choose repeatable pseudo-random anchors without Python hash randomization."""
    ranked = sorted(
        candidates,
        key=lambda position: hashlib.sha256(
            f"{seed}:{sample_id}:{position}".encode("utf-8")
        ).digest(),
    )
    return ranked[: max(0, budget)]


def logical_pattern(
    valid: Sequence[bool],
    anchors: Sequence[bool],
    *,
    strategy: str,
    window_radius: int,
) -> list[list[bool]]:
    """Create one query-by-key logical attention graph."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if len(valid) != len(anchors):
        raise ValueError("valid and anchors must have equal length")
    if window_radius < 0:
        raise ValueError("window_radius must be non-negative")
    length = len(valid)
    result = [[False] * length for _ in range(length)]
    for query in range(length):
        for key in range(length):
            if valid[query] and valid[key]:
                result[query][key] = strategy == "full" or abs(query - key) <= window_radius
    if strategy in {"random", "tfidf", "rationale", "distilled", "shuffled"}:
        for anchor, selected in enumerate(anchors):
            if not (selected and valid[anchor]):
                continue
            for position, is_valid in enumerate(valid):
                if is_valid:
                    result[position][anchor] = True
                    result[anchor][position] = True
    return result


def pattern_stats(pattern: Sequence[Sequence[bool]], valid: Sequence[bool]) -> dict[str, float]:
    token_count = sum(bool(x) for x in valid)
    full = token_count * token_count
    active = sum(bool(x) for row in pattern for x in row)
    ratio = active / full if full else 0.0
    return {
        "active_connections": float(active),
        "full_connections": float(full),
        "connectivity_ratio": ratio,
        "sparsity": 1.0 - ratio if full else 0.0,
    }


def batched_pattern(valid_mask, anchor_mask, *, strategy: str, window_radius: int):
    """Vectorized PyTorch equivalent of logical_pattern."""
    import torch

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    valid = valid_mask.bool()
    valid_pairs = valid[:, :, None] & valid[:, None, :]
    if strategy == "full":
        return valid_pairs
    length = valid.shape[1]
    positions = torch.arange(length, device=valid.device)
    local = (positions[:, None] - positions[None, :]).abs() <= window_radius
    graph = valid_pairs & local.unsqueeze(0)
    if strategy in {"random", "tfidf", "rationale", "distilled", "shuffled"}:
        anchors = anchor_mask.bool() & valid
        global_edges = anchors[:, :, None] | anchors[:, None, :]
        graph |= valid_pairs & global_edges
    return graph
