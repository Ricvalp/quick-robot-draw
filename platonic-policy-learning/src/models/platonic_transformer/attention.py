import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# PyTorch 2.2 exposes scaled dot product attention without the new API.
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore

    _SDP_AVAILABLE = True
except ImportError:  # pragma: no cover - compatibility path
    SDPBackend = None  # type: ignore
    sdpa_kernel = None  # type: ignore
    _SDP_AVAILABLE = False

from .rope import PlatonicRoPE
from .linear import PlatonicLinear
from .groups import PLATONIC_GROUPS


class DensePlatonicAttention(nn.Module):
    """
    Multi-head attention layer for uniform token batches with Platonic symmetry.

    Queries, keys, and values are projected in a group-aware manner and optionally
    enriched with rotary positional embeddings. Setting ``attention=True`` uses the
    softmax attention rule; otherwise a linear kernelised path is taken.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_heads: int,
        solid_name: str,
        spatial_dims: int = 3,
        freq_sigma: float = 1.0,
        freq_init: str = "random",
        learned_freqs: bool = True,
        bias: bool = True,
        mean_aggregation: bool = False,
        attention: bool = False,
        use_key: bool = False,
    ) -> None:
        super().__init__()

        # Cache group metadata so we can reshape the last dimension to ``[G, C]``.
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G

        if in_channels % self.num_G != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by group size ({self.num_G})."
            )
        if out_channels % self.num_G != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by group size ({self.num_G})."
            )
        if num_heads % self.num_G != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by group size ({self.num_G})."
            )

        self.in_channels_g = in_channels // self.num_G
        self.out_channels_g = out_channels // self.num_G

        # Each group element owns ``effective_num_heads`` attention heads.
        self.effective_num_heads = num_heads // self.num_G

        # embed_dim is also grouped, so it must divide accordingly.
        if embed_dim % self.num_G != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by group size ({self.num_G})."
            )
        self.embed_dim = embed_dim
        self.embed_dim_g = embed_dim // self.num_G

        if self.embed_dim_g % self.effective_num_heads != 0:
            raise ValueError(
                f"embed_dim per group ({self.embed_dim_g}) must be divisible by heads per group "
                f"({self.effective_num_heads})."
            )

        self.head_dim = self.embed_dim_g // self.effective_num_heads
        self.mean_aggregation = mean_aggregation
        self.attention = attention
        self.use_key = use_key

        # Group-aware linear projections for queries, keys, and values.
        self.q_proj = PlatonicLinear(in_channels, embed_dim, solid_name, bias=bias)
        self.v_proj = PlatonicLinear(in_channels, embed_dim, solid_name, bias=bias)
        if freq_sigma is None or use_key:
            self.k_proj = PlatonicLinear(in_channels, embed_dim, solid_name, bias=bias)
        else:
            self.register_buffer("k_proj", None)

        # Rotary positional embeddings remain unchanged from the dense path.
        if freq_sigma is not None:
            self.rope_emb = PlatonicRoPE(
                embed_dim=embed_dim,
                num_heads=self.effective_num_heads,
                head_dim=self.head_dim,
                solid_name=solid_name,
                spatial_dims=spatial_dims,
                freq_sigma=freq_sigma,
                learned_freqs=learned_freqs,
                freq_init=freq_init,
            )
        else:
            self.register_buffer("rope_emb", None)

        # Final projection mixes heads back into the flattened ``[G * C]`` layout.
        self.out_proj = PlatonicLinear(embed_dim, out_channels, solid_name, bias=bias)

    def forward(
        self, x: Tensor, pos: Tensor, token_normaliser: Optional[float] = None
    ) -> Tensor:
        """Apply attention to tokens ``[B, N, G*C_in]`` at positions ``[B, N, D]``.

        Args:
            x: Token features.
            pos: Token coordinates.
            token_normaliser: Optional scalar used to normalise the linear attention kernel.
        """
        B, N, _ = x.shape

        # --- 1. Compute equivariant Q/K/V projections ---
        q_raw = self.q_proj(x)  # [B, N, embed_dim]
        v_raw = self.v_proj(x)  # [B, N, embed_dim]
        if self.k_proj is not None:
            k_raw = self.k_proj(x)  # [B, N, embed_dim]
        else:
            k_raw = torch.ones_like(q_raw)  # [B, N, embed_dim]

        # --- 2. Reshape to explicit group/head axes ---
        q = q_raw.view(B, N, self.num_G, self.effective_num_heads, self.head_dim)  # [B, N, G, H, Dh]
        k = k_raw.view(B, N, self.num_G, self.effective_num_heads, self.head_dim)  # [B, N, G, H, Dh]
        v = v_raw.view(B, N, self.num_G, self.effective_num_heads, self.head_dim)  # [B, N, G, H, Dh]

        # --- 3. Inject positional structure with RoPE ---
        if self.rope_emb is not None:
            q = self.rope_emb(q, pos)  # [B, N, G, H, Dh]
            k = self.rope_emb(k, pos)  # [B, N, G, H, Dh]

        if self.attention:
            # --- 4a. Standard softmax attention via SDPA ---
            q_sdpa = q.permute(0, 2, 3, 1, 4).reshape(
                B, self.num_G * self.effective_num_heads, N, self.head_dim
            )  # [B, G*H, N, Dh]
            k_sdpa = k.permute(0, 2, 3, 1, 4).reshape(
                B, self.num_G * self.effective_num_heads, N, self.head_dim
            )  # [B, G*H, N, Dh]
            v_sdpa = v.permute(0, 2, 3, 1, 4).reshape(
                B, self.num_G * self.effective_num_heads, N, self.head_dim
            )  # [B, G*H, N, Dh]

            key_padding = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            combined_mask = key_padding[:, None, None, :].expand(-1, q_sdpa.shape[1], -1, -1)

            if _SDP_AVAILABLE:
                with sdpa_kernel(  # type: ignore[misc]
                    [
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.MATH,
                        SDPBackend.EFFICIENT_ATTENTION,
                    ]
                ):
                    attn_output = F.scaled_dot_product_attention(
                        q_sdpa, k_sdpa, v_sdpa, attn_mask=combined_mask
                    )  # [B, G*H, N, Dh]
            else:
                attn_output = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, attn_mask=combined_mask
                )

            output = attn_output.reshape(
                B, self.num_G, self.effective_num_heads, N, self.head_dim
            ).permute(0, 3, 1, 2, 4)  # [B, N, G, H, Dh]
        else:
            kv_kernel = torch.einsum("bsghd,bsghe->bghde", k, v)  # [B, G, H, Dh, Dh]

            if token_normaliser is not None:
                normaliser = max(token_normaliser, 1.0)
            else:
                normaliser = max(float(N), 1.0)
            kv_kernel = kv_kernel / normaliser

            output = torch.einsum("bsghd,bghde->bsghe", q, kv_kernel)  # [B, N, G, H, Dh]

        output = output.reshape(B, N, self.embed_dim)  # [B, N, embed_dim]
        return self.out_proj(output)  # [B, N, G*C_out]
