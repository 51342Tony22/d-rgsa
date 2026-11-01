from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .attention import batched_pattern


@dataclass(frozen=True)
class SmallTransformerConfig:
    """Architecture shared by every attention-pattern variant."""

    vocab_size: int
    num_labels: int
    pad_token_id: int
    max_length: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    ffn_size: int = 512
    dropout: float = 0.1
    pooling: str = "cls"

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.hidden_size <= 0 or self.ffn_size <= 0:
            raise ValueError("hidden_size, num_heads, and ffn_size must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        if min(self.vocab_size, self.num_labels, self.max_length, self.num_layers) <= 0:
            raise ValueError("vocab_size, num_labels, max_length, and num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ControlledSelfAttention(nn.Module):
    """Multi-head attention with a per-sample query-by-key connectivity graph."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        *,
        backend: str = "dense_mask",
        window_radius: int = 4,
    ) -> None:
        super().__init__()
        if backend not in {"dense_mask", "structured_sparse"}:
            raise ValueError("backend must be 'dense_mask' or 'structured_sparse'")
        self.num_heads = num_heads
        self.backend = backend
        self.window_radius = window_radius
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        allowed_graph: torch.Tensor,
        *,
        output_attentions: bool,
        valid_tokens: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        strategy: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            self.backend == "structured_sparse"
            and strategy == "full"
            and not output_attentions
        ):
            if valid_tokens is None:
                raise ValueError("optimized full attention requires valid_tokens")
            output, _ = self.attention(
                hidden_states,
                hidden_states,
                hidden_states,
                key_padding_mask=~valid_tokens.bool(),
                need_weights=False,
            )
            return output, None
        if (
            self.backend == "structured_sparse"
            and strategy != "full"
            and not output_attentions
        ):
            if valid_tokens is None or anchor_mask is None:
                raise ValueError(
                    "structured_sparse attention requires valid_tokens and anchor_mask"
                )
            return (
                self._structured_sparse_forward(
                    hidden_states,
                    valid_tokens.bool(),
                    anchor_mask.bool(),
                    use_global=strategy
                    in {"random", "tfidf", "rationale", "distilled", "shuffled"},
                ),
                None,
            )
        batch, length, _ = hidden_states.shape
        if allowed_graph.shape != (batch, length, length):
            raise ValueError(
                f"Expected graph shape {(batch, length, length)}, got {tuple(allowed_graph.shape)}"
            )
        # Fully masked padding queries would produce NaN after softmax. Give each such
        # throwaway query one safe key; its output is zeroed by the encoder layer.
        safe_graph = allowed_graph.bool().clone()
        empty_queries = ~safe_graph.any(dim=-1)
        batch_indices, query_indices = empty_queries.nonzero(as_tuple=True)
        safe_graph[batch_indices, query_indices, 0] = True
        # PyTorch bool attn_mask uses True for blocked edges, the inverse of our graph.
        blocked = ~safe_graph
        blocked = (
            blocked[:, None, :, :]
            .expand(batch, self.num_heads, length, length)
            .reshape(batch * self.num_heads, length, length)
        )
        output, weights = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=blocked,
            need_weights=output_attentions,
            average_attn_weights=False,
        )
        return output, weights if output_attentions else None

    def _structured_sparse_forward(
        self,
        hidden_states: torch.Tensor,
        valid_tokens: torch.Tensor,
        anchor_mask: torch.Tensor,
        *,
        use_global: bool,
    ) -> torch.Tensor:
        """Compute local-plus-global attention without materializing an n-by-n mask.

        Non-anchor queries attend a sliding local window and all selected global
        anchors. Anchor queries attend every valid key. This is exactly the graph
        used by ``batched_pattern`` but requires O(nw + nk) attention scores for
        window width ``w`` and ``k`` anchors.
        """

        batch, length, embed_dim = hidden_states.shape
        head_dim = embed_dim // self.num_heads
        qkv = F.linear(
            hidden_states,
            self.attention.in_proj_weight,
            self.attention.in_proj_bias,
        )
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.num_heads, head_dim).transpose(1, 2)

        query = split_heads(query) * (head_dim**-0.5)
        key = split_heads(key)
        value = split_heads(value)

        radius = self.window_radius
        width = 2 * radius + 1
        padded_key = F.pad(key, (0, 0, radius, radius))
        padded_value = F.pad(value, (0, 0, radius, radius))
        local_key = padded_key.unfold(2, width, 1).permute(0, 1, 2, 4, 3)
        local_value = padded_value.unfold(2, width, 1).permute(0, 1, 2, 4, 3)
        local_valid = F.pad(valid_tokens, (radius, radius), value=False).unfold(
            1, width, 1
        )
        local_anchor = F.pad(anchor_mask, (radius, radius), value=False).unfold(
            1, width, 1
        )
        local_allowed = local_valid
        if use_global:
            # Global keys are appended below, so remove local duplicates to keep
            # the softmax mathematically identical to the dense logical graph.
            local_allowed = local_allowed & ~local_anchor
        local_scores = torch.einsum("bhnd,bhnwd->bhnw", query, local_key)
        score_parts = [local_scores]
        mask_parts = [local_allowed[:, None, :, :]]
        value_parts = [local_value]

        global_indices: torch.Tensor | None = None
        global_valid: torch.Tensor | None = None
        if use_global:
            anchor_mask = anchor_mask & valid_tokens
            maximum_anchors = int(anchor_mask.sum(dim=1).max().item())
            if maximum_anchors > 0:
                global_indices = torch.topk(
                    anchor_mask.to(torch.int8),
                    k=maximum_anchors,
                    dim=1,
                ).indices
                global_valid = anchor_mask.gather(1, global_indices)
                gather_index = global_indices[:, None, :, None].expand(
                    batch,
                    self.num_heads,
                    maximum_anchors,
                    head_dim,
                )
                global_key = key.gather(2, gather_index)
                global_value = value.gather(2, gather_index)
                global_scores = torch.einsum(
                    "bhnd,bhkd->bhnk", query, global_key
                )
                score_parts.append(global_scores)
                mask_parts.append(
                    valid_tokens[:, None, :, None]
                    & global_valid[:, None, None, :]
                )
                value_parts.append(
                    global_value[:, :, None, :, :].expand(
                        batch,
                        self.num_heads,
                        length,
                        maximum_anchors,
                        head_dim,
                    )
                )

        combined_scores = torch.cat(score_parts, dim=-1)
        combined_allowed = torch.cat(
            [
                mask.expand(batch, self.num_heads, length, mask.shape[-1])
                for mask in mask_parts
            ],
            dim=-1,
        )
        # Padding queries have no valid edges. Give them one harmless score and
        # zero the result after softmax to prevent NaNs.
        empty_queries = ~combined_allowed.any(dim=-1)
        if empty_queries.any():
            combined_allowed = combined_allowed.clone()
            first_column = combined_allowed[..., 0]
            combined_allowed[..., 0] = first_column | empty_queries
        combined_scores = combined_scores.masked_fill(
            ~combined_allowed,
            torch.finfo(combined_scores.dtype).min,
        )
        probabilities = F.softmax(combined_scores, dim=-1)
        probabilities = F.dropout(
            probabilities,
            p=float(self.attention.dropout),
            training=self.training,
        )
        combined_values = torch.cat(value_parts, dim=3)
        attended = torch.einsum(
            "bhnc,bhncd->bhnd",
            probabilities,
            combined_values,
        )
        attended = attended * valid_tokens[:, None, :, None].to(attended.dtype)

        if (
            use_global
            and global_indices is not None
            and global_valid is not None
        ):
            maximum_anchors = global_indices.shape[1]
            gather_index = global_indices[:, None, :, None].expand(
                batch,
                self.num_heads,
                maximum_anchors,
                head_dim,
            )
            global_query = query.gather(2, gather_index)
            global_query_scores = torch.einsum(
                "bhkd,bhnd->bhkn", global_query, key
            )
            global_query_allowed = (
                global_valid[:, None, :, None]
                & valid_tokens[:, None, None, :]
            ).expand(
                batch,
                self.num_heads,
                maximum_anchors,
                length,
            )
            global_query_scores = global_query_scores.masked_fill(
                ~global_query_allowed,
                torch.finfo(global_query_scores.dtype).min,
            )
            global_probabilities = F.softmax(global_query_scores, dim=-1)
            global_probabilities = F.dropout(
                global_probabilities,
                p=float(self.attention.dropout),
                training=self.training,
            )
            global_attended = torch.einsum(
                "bhkn,bhnd->bhkd", global_probabilities, value
            )
            global_attended = global_attended * global_valid[
                :, None, :, None
            ].to(global_attended.dtype)
            replacements = torch.zeros_like(attended).scatter(
                2,
                gather_index,
                global_attended,
            )
            replacement_mask = torch.zeros(
                (batch, 1, length, 1),
                dtype=torch.bool,
                device=hidden_states.device,
            ).scatter(
                2,
                global_indices[:, None, :, None],
                global_valid[:, None, :, None],
            )
            attended = torch.where(replacement_mask, replacements, attended)

        attended = attended.transpose(1, 2).contiguous().view(
            batch, length, embed_dim
        )
        return self.attention.out_proj(attended)


class SmallTransformerLayer(nn.Module):
    """Pre-norm Transformer encoder block implemented independently of BERT."""

    def __init__(
        self,
        config: SmallTransformerConfig,
        *,
        attention_backend: str,
        window_radius: int,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = ControlledSelfAttention(
            config.hidden_size,
            config.num_heads,
            config.dropout,
            backend=attention_backend,
            window_radius=window_radius,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_size, config.hidden_size),
        )
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        allowed_graph: torch.Tensor,
        valid_tokens: torch.Tensor,
        *,
        output_attentions: bool,
        anchor_mask: torch.Tensor,
        strategy: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, weights = self.attention(
            self.attention_norm(hidden_states),
            allowed_graph,
            output_attentions=output_attentions,
            valid_tokens=valid_tokens,
            anchor_mask=anchor_mask,
            strategy=strategy,
        )
        hidden_states = hidden_states + self.attention_dropout(attended)
        hidden_states = hidden_states + self.ffn_dropout(self.ffn(self.ffn_norm(hidden_states)))
        # Padding queries are not used, but zeroing them avoids carrying meaningless states.
        hidden_states = hidden_states * valid_tokens.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states, weights


class SmallTransformerClassifier(nn.Module):
    """Lightweight student; variants differ only in their attention graph."""

    def __init__(
        self,
        config: SmallTransformerConfig,
        *,
        strategy: str = "full",
        window_radius: int = 4,
        attention_backend: str = "dense_mask",
    ) -> None:
        super().__init__()
        self.config = config
        self.strategy = strategy
        self.window_radius = window_radius
        self.attention_backend = attention_backend
        self.num_labels = config.num_labels
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(config.max_length, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            SmallTransformerLayer(
                config,
                attention_backend=attention_backend,
                window_radius=window_radius,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
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
        anchor_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...] | None]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if anchor_mask is None:
            anchor_mask = torch.zeros_like(attention_mask)
        batch, length = input_ids.shape
        if length > self.config.max_length:
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum {self.config.max_length}"
            )
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden_states = self.token_embeddings(input_ids) * math.sqrt(self.config.hidden_size)
        hidden_states = hidden_states + self.position_embeddings(positions)
        hidden_states = self.embedding_dropout(self.embedding_norm(hidden_states))
        valid = attention_mask.bool()
        if self.attention_backend == "structured_sparse" and not output_attentions:
            # Both the optimized full path and structured sparse path avoid the
            # dense query-by-key graph. Keep an empty placeholder because the
            # attention module ignores it in these two fast paths.
            graph = torch.empty(0, dtype=torch.bool, device=input_ids.device)
        else:
            graph = batched_pattern(
                valid,
                anchor_mask,
                strategy=self.strategy,
                window_radius=self.window_radius,
            )
        all_attentions: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, weights = layer(
                hidden_states,
                graph,
                valid,
                output_attentions=output_attentions,
                anchor_mask=anchor_mask,
                strategy=self.strategy,
            )
            if weights is not None:
                all_attentions.append(weights)
        hidden_states = self.final_norm(hidden_states)
        if self.config.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            weights = valid.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(pooled)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        result: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {"logits": logits}
        if loss is not None:
            result["loss"] = loss
        if output_attentions:
            result["attentions"] = tuple(all_attentions)
        return result
