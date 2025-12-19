"""Lightweight Transformer encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # (B, N, D)


@dataclass
class EncoderTransformerConfig:
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5


class EncoderTransformerBlock(nn.Module):
    def __init__(self, cfg: EncoderTransformerConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.attention_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(
            cfg.hidden_dim, elementwise_affine=True, eps=cfg.layer_norm_eps
        )
        self.attn_dropout = nn.Dropout(cfg.dropout)

        if cfg.activation.lower() == "gelu":
            activation = nn.GELU()
        elif cfg.activation.lower() in {"silu", "swish"}:
            activation = nn.SiLU()
        elif cfg.activation.lower() == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.GELU()

        self.mlp_norm = nn.LayerNorm(
            cfg.hidden_dim, elementwise_affine=True, eps=cfg.layer_norm_eps
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.mlp_dim),
            activation,
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_dim, cfg.hidden_dim),
        )
        self.mlp_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:

        attn_in = self.attn_norm(tokens)
        attn_out, _ = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask if key_padding_mask is not None else None,
            attn_mask=attn_mask,
            need_weights=False,
        )

        tokens = tokens + self.attn_dropout(attn_out)

        mlp_in = self.mlp_norm(tokens)
        mlp_out = self.mlp_dropout(self.mlp(mlp_in))
        tokens = tokens + mlp_out
        return tokens


class EncoderTransformer(nn.Module):
    def __init__(self, cfg: EncoderTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            EncoderTransformerBlock(cfg) for _ in range(cfg.num_layers)
        )
        self.final_norm = nn.LayerNorm(
            cfg.hidden_dim, elementwise_affine=True, eps=cfg.layer_norm_eps
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _prepare_attention_mask(
        mask: Optional[torch.Tensor], *, device: torch.device, dtype: torch.dtype
    ):
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            float_mask = mask.to(dtype=dtype)
            float_mask.masked_fill_(float_mask > 0, float("-inf"))
            return float_mask
        return mask.to(device=device, dtype=dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = self._prepare_attention_mask(
            attn_mask, device=tokens.device, dtype=tokens.dtype
        )
        x = tokens
        for block in self.blocks:
            x = block(
                x,
                attn_mask=mask,
                key_padding_mask=key_padding_mask,
            )
        return self.final_norm(x)


__all__ = ["EncoderTransformer", "EncoderTransformerConfig"]
