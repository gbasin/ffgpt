from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Mask with True in future positions that must be hidden."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 6
    d_model: int = 64
    n_heads: int = 2
    n_blocks: int = 2
    mlp_hidden: int = 256
    dropout: float = 0.0


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.act(self.fc1(x))
        out = self.fc2(self.dropout(hidden))
        return out, hidden


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if causal:
            c_mask = causal_mask(seq_len, device=x.device)
            attn_scores = attn_scores.masked_fill(c_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if pad_mask is not None:
            key_mask = (~pad_mask).unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(context)

        if pad_mask is not None:
            out = out * pad_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, hidden_dim=mlp_hidden, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_in = self.ln1(x)
        attn_out = self.attn(attn_in, pad_mask=pad_mask, causal=causal)
        x = x + attn_out

        mlp_in = self.ln2(x)
        mlp_out, goodness_activations = self.mlp(mlp_in)
        x = x + mlp_out

        return x, goodness_activations


class FFTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_hidden=config.mlp_hidden,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        return self.token_embedding(tokens) + self.position_embedding(positions)

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        causal: bool = False,
        detach_between_blocks: bool = True,
        inter_block_norm: str = "none",
        inter_block_norm_eps: float = 1e-5,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if pad_mask is None:
            pad_mask = torch.ones_like(tokens, dtype=torch.bool)

        norm_mode = inter_block_norm.lower()
        if norm_mode not in {"none", "layernorm", "rmsnorm", "l2"}:
            raise ValueError(
                f"inter_block_norm must be one of ['none', 'layernorm', 'rmsnorm', 'l2'], got {inter_block_norm}"
            )
        if inter_block_norm_eps <= 0.0:
            raise ValueError(f"inter_block_norm_eps must be > 0, got {inter_block_norm_eps}")

        x = self.embed(tokens)
        block_outputs: list[torch.Tensor] = []
        goodness_activations: list[torch.Tensor] = []

        for block_idx, block in enumerate(self.blocks):
            if detach_between_blocks and block_idx > 0:
                x = x.detach()
            if block_idx > 0 and norm_mode != "none":
                if norm_mode == "layernorm":
                    x = F.layer_norm(x, (x.shape[-1],), eps=inter_block_norm_eps)
                elif norm_mode == "rmsnorm":
                    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + inter_block_norm_eps)
                else:
                    x = x / x.norm(dim=-1, keepdim=True).clamp(min=inter_block_norm_eps)
            x, acts = block(x, pad_mask=pad_mask, causal=causal)
            block_outputs.append(x)
            goodness_activations.append(acts)

        return block_outputs, goodness_activations

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.token_embedding.weight


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_hidden=config.mlp_hidden,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        return self.token_embedding(tokens) + self.position_embedding(positions)

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if pad_mask is None:
            pad_mask = torch.ones_like(tokens, dtype=torch.bool)

        x = self.embed(tokens)
        goodness_activations: list[torch.Tensor] = []

        for block in self.blocks:
            x, acts = block(x, pad_mask=pad_mask, causal=causal)
            goodness_activations.append(acts)

        logits = self.lm_head(x)
        return logits, x, goodness_activations
