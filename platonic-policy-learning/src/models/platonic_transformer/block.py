import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable

from .attention import DensePlatonicAttention
from .linear import PlatonicLinear
from .groups import PLATONIC_GROUPS


def drop_path(
    x: Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
) -> Tensor:
    """Implements sample-wise stochastic depth."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor  # [...same as x...]


class DropPath(nn.Module):
    """Wraps stochastic depth in an nn.Module so it can be slotted into nn.Sequential."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class DensePlatonicBlock(nn.Module):
    """Transformer-style residual block for Platonic-symmetric token sets."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        solid_name: str,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        layer_norm_eps: float = 1e-5,
        spatial_dims: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
        freq_sigma: float = 1.0,
        freq_init: str = "random",
        learned_freqs: bool = True,
        mean_aggregation: bool = False,
        attention: bool = False,
        use_key: bool = False,
        conditioning: bool = False,
    ) -> None:
        super().__init__()

        # Store the Platonic group so we can reshape around it later on.
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        self.conditioning = conditioning

        # d_model and dim_feedforward must be divisible by the group order so that we can
        # reshape the channel dimension into ``[G, C]`` without remainder.
        if d_model % self.num_G != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by group size ({self.num_G})."
            )
        if dim_feedforward % self.num_G != 0:
            raise ValueError(
                f"dim_feedforward ({dim_feedforward}) must be divisible by group size ({self.num_G})."
            )
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        # Remember the per-group channel width for the group-aware LayerNorm.
        self.dim_per_g = d_model // self.num_G

        # Create the dense self-attention module that never falls back to sparse logic.
        self.interaction = DensePlatonicAttention(
            in_channels=d_model,
            out_channels=d_model,
            embed_dim=d_model,
            num_heads=nhead,
            solid_name=solid_name,
            spatial_dims=spatial_dims,
            freq_sigma=freq_sigma,
            freq_init=freq_init,
            learned_freqs=learned_freqs,
            mean_aggregation=mean_aggregation,
            attention=attention,
            use_key=use_key,
        )

        # Optional diffusion-style conditioning via adaptive LayerNorm.
        if conditioning:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True)
            )
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)

        # Feed-forward sub-layer built from equivariant linear layers.
        self.linear1 = PlatonicLinear(d_model, dim_feedforward, solid=solid_name)
        self.linear2 = PlatonicLinear(dim_feedforward, d_model, solid=solid_name)

        # Group-aware LayerNorms operate on the per-group channel dimension.
        self.norm1 = nn.LayerNorm(self.dim_per_g, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.dim_per_g, eps=layer_norm_eps)

        # Residual helpers.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.activation = activation

        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.gamma_1 = (
            nn.Parameter(layer_scale_init_value * torch.ones((d_model)))
            if layer_scale_init_value is not None
            else None
        )
        self.gamma_2 = (
            nn.Parameter(layer_scale_init_value * torch.ones((d_model)))
            if layer_scale_init_value is not None
            else None
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        conditioning: Optional[Tensor] = None,
        token_normaliser: Optional[float] = None,
    ) -> Tensor:
        """
        Args:
            x: Token features with flattened group channels, shape ``[B, N, G*C]``.
            pos: Token coordinates, shape ``[B, N, D]`` where ``D`` is typically 3.
            conditioning: Optional conditioning embedding ``[B, d_model]``.
            token_normaliser: Optional scalar used to normalise the linear attention kernel.
        """
        # Prepare conditioning modulation/gating parameters if requested.
        if self.conditioning and conditioning is not None:
            modulation_chunks = self.adaLN_modulation(conditioning).chunk(6, dim=-1)
            shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = [
                self._broadcast_condition(chunk, x) for chunk in modulation_chunks
            ]
        else:
            shift_msa = scale_msa = gate_msa = shift_ffn = scale_ffn = gate_ffn = None

        # --- Attention branch with pre-normalisation ---
        normed_x = self._normalize(x, self.norm1)  # [B, N, G*C]
        normed_x = self._apply_condition(normed_x, shift_msa, scale_msa)  # [B, N, G*C]
        attn_out = self._interaction_block(normed_x, pos, token_normaliser)  # [B, N, G*C]
        if self.gamma_1 is not None:
            attn_out = self.gamma_1 * attn_out  # [B, N, G*C]
        attn_out = self.drop_path1(attn_out)  # [B, N, G*C]
        x = x + (gate_msa * attn_out if gate_msa is not None else attn_out)  # [B, N, G*C]

        # --- Feed-forward branch with pre-normalisation ---
        normed_ff = self._normalize(x, self.norm2)  # [B, N, G*C]
        normed_ff = self._apply_condition(normed_ff, shift_ffn, scale_ffn)  # [B, N, G*C]
        ff_out = self._ff_block(normed_ff)  # [B, N, G*C]
        if self.gamma_2 is not None:
            ff_out = self.gamma_2 * ff_out  # [B, N, G*C]
        ff_out = self.drop_path2(ff_out)  # [B, N, G*C]
        x = x + (gate_ffn * ff_out if gate_ffn is not None else ff_out)  # [B, N, G*C]

        return x

    @staticmethod
    def _apply_condition(
        x: Tensor, shift: Optional[Tensor], scale: Optional[Tensor]
    ) -> Tensor:
        """Applies AdaLN-style affine modulation."""
        if shift is None or scale is None:
            return x
        return x * (1 + scale) + shift  # [B, N, G*C]

    @staticmethod
    def _broadcast_condition(params: Tensor, x: Tensor) -> Tensor:
        """
        Broadcasts conditioning parameters over the token dimension so they
        can be fused with ``x``. Parameters are always shaped ``[B, C]``.
        """
        params = params.to(dtype=x.dtype)
        return params[:, None, :].expand(-1, x.shape[1], -1)  # [B, N, G*C]

    def _normalize(self, x: Tensor, norm_layer: nn.LayerNorm) -> Tensor:
        """Applies LayerNorm on the per-group channel dimension."""
        leading_dims = x.shape[:-1]
        reshaped = x.view(*leading_dims, self.num_G, self.dim_per_g)  # [..., G, C]
        normed = norm_layer(reshaped)  # [..., G, C]
        return normed.view(*leading_dims, -1)  # [..., G*C]

    def _interaction_block(
        self,
        x: Tensor,
        pos: Tensor,
        token_normaliser: Optional[float],
    ) -> Tensor:
        """Runs the dense attention layer followed by dropout."""
        attn = self.interaction(x, pos, token_normaliser)  # [B, N, G*C]
        return self.dropout1(attn)  # [B, N, G*C]

    def _ff_block(self, x: Tensor) -> Tensor:
        """Standard two-layer feed-forward network with GELU activation."""
        hidden = self.linear1(x)  # [B, N, dim_feedforward]
        hidden = self.activation(hidden)  # [B, N, dim_feedforward]
        hidden = self.ffn_dropout(hidden)  # [B, N, dim_feedforward]
        output = self.linear2(hidden)  # [B, N, G*C]
        return self.dropout2(output)  # [B, N, G*C]
