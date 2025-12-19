"""Lightweight Transformer decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # (B, N, D)


@dataclass
class DecoderTransformerConfig:
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5


class _AdaLayerNormZero(nn.Module):
    def __init__(self, hidden_dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mlp(cond).chunk(2, dim=-1)
        return _modulate(self.norm(x), shift, scale)


class DecoderTransformerBlock(nn.Module):
    def __init__(self, cfg: DecoderTransformerConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.attention_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(
            cfg.hidden_dim, elementwise_affine=False, eps=cfg.layer_norm_eps
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

        self.cross_norm = nn.LayerNorm(
            cfg.hidden_dim, elementwise_affine=False, eps=cfg.layer_norm_eps
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.attention_dropout,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(cfg.dropout)

        self.mlp_norm = nn.LayerNorm(
            cfg.hidden_dim, elementwise_affine=False, eps=cfg.layer_norm_eps
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.mlp_dim),
            activation,
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_dim, cfg.hidden_dim),
        )
        self.mlp_dropout = nn.Dropout(cfg.dropout)

        self.ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(cfg.hidden_dim, 9 * cfg.hidden_dim, bias=True)
        )
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_kpm: torch.Tensor,
        *,
        memory: torch.Tensor,
        encoder_kpm: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada_ln(cond).chunk(9, dim=-1)

        attn_in = _modulate(self.attn_norm(tokens), shift_msa, scale_msa)
        attn_out, _ = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=tokens_kpm,
            need_weights=False,
        )

        tokens = tokens + gate_msa.unsqueeze(1) * self.attn_dropout(attn_out)

        cross_in = _modulate(self.cross_norm(tokens), shift_cross, scale_cross)
        cross_out, _ = self.cross_attn(
            cross_in,
            memory,
            memory,
            key_padding_mask=encoder_kpm,
            need_weights=False,
        )

        tokens = tokens + gate_cross.unsqueeze(1) * self.cross_dropout(cross_out)

        mlp_in = _modulate(self.mlp_norm(tokens), shift_mlp, scale_mlp)
        mlp_out = self.mlp_dropout(self.mlp(mlp_in))
        tokens = tokens + gate_mlp.unsqueeze(1) * mlp_out
        return tokens


class DecoderTransformer(nn.Module):
    def __init__(self, cfg: DecoderTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            DecoderTransformerBlock(cfg) for _ in range(cfg.num_layers)
        )
        self.final_norm = _AdaLayerNormZero(cfg.hidden_dim, cfg.layer_norm_eps)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_kpm: torch.Tensor,
        *,
        memory: torch.Tensor,
        encoder_kpm: torch.Tensor,
        diffusion_time_cond: torch.Tensor,
    ) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(
                x,
                tokens_kpm=tokens_kpm,
                memory=memory,
                encoder_kpm=encoder_kpm,
                cond=diffusion_time_cond,
            )
        return self.final_norm(x, diffusion_time_cond)


__all__ = ["DecoderTransformer", "DecoderTransformerConfig"]
