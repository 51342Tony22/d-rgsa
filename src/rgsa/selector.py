from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class AnchorSelectorConfig:
    """Configuration for the teacher-free global-anchor selector."""

    vocab_size: int
    pad_token_id: int
    max_length: int = 128
    embedding_size: int = 128
    hidden_size: int = 64
    architecture: str = "bigru"
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.architecture not in {"bigru", "mlp"}:
            raise ValueError("architecture must be 'bigru' or 'mlp'")
        if min(
            self.vocab_size,
            self.max_length,
            self.embedding_size,
            self.hidden_size,
        ) <= 0:
            raise ValueError("selector dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnchorSelector(nn.Module):
    """Predict one rationale-anchor score per input token."""

    def __init__(self, config: AnchorSelectorConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(config.max_length, config.embedding_size)
        self.embedding_norm = nn.LayerNorm(config.embedding_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        if config.architecture == "bigru":
            self.contextualizer: nn.Module = nn.GRU(
                input_size=config.embedding_size,
                hidden_size=config.hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            score_input_size = config.hidden_size * 2
        else:
            self.contextualizer = nn.Sequential(
                nn.Linear(config.embedding_size, config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            score_input_size = config.hidden_size
        self.score_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(score_input_size, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = input_ids.shape
        if length > self.config.max_length:
            raise ValueError(
                f"Sequence length {length} exceeds selector maximum "
                f"{self.config.max_length}"
            )
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden = self.embedding_dropout(self.embedding_norm(hidden))
        if self.config.architecture == "bigru":
            hidden, _ = self.contextualizer(hidden)
        else:
            hidden = self.contextualizer(hidden)
        logits = self.score_head(hidden).squeeze(-1)
        if attention_mask is not None:
            logits = logits.masked_fill(~attention_mask.bool(), -1.0e4)
        return logits

