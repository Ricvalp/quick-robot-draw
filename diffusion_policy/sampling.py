"""Utilities for qualitative sampling and visualisation of QuickDraw policies."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

__all__ = [
    "make_start_token",
    "sample_quickdraw_tokens",
    "tokens_to_figure",
]


def make_start_token(
    batch_size: int,
    feature_dim: int,
    device: torch.device,
    *,
    pen_down: bool = False,
) -> torch.Tensor:
    """Return a batch of `[0, 0, pen]` start tokens centered at the origin."""

    token = torch.zeros(batch_size, 1, feature_dim, device=device)
    if pen_down:
        token[..., 2] = 1.0
    return token


@torch.no_grad()
def sample_quickdraw_tokens(
    policy: torch.nn.Module,
    total_tokens: int,
    *,
    start_token: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Autoregressively sample `total_tokens` conditioned on the growing context."""

    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive.")

    device = next(policy.parameters()).device
    feature_dim = policy.cfg.point_feature_dim

    if start_token is None:
        context = make_start_token(1, feature_dim, device)
    else:
        if start_token.shape[-1] != feature_dim:
            raise ValueError(
                f"start_token feature dim {start_token.shape[-1]} != {feature_dim}."
            )
        context = start_token.to(device=device)

    horizon = policy.cfg.horizon
    num_chunks = math.ceil(total_tokens / horizon)
    samples: list[torch.Tensor] = []

    for _ in range(num_chunks):
        actions = policy.sample_actions(context, generator=generator)
        samples.append(actions)
        context = torch.cat([context, actions], dim=1)

    generated = torch.cat(samples, dim=1)
    return generated[:, :total_tokens, :]


def tokens_to_figure(
    tokens: torch.Tensor | np.ndarray,
    *,
    coordinate_mode: str = "absolute",
    color: str = "black",
    linewidth: float = 1.5,
):
    """Convert `(N, 3)` tokens into a Matplotlib figure for logging or saving."""

    import matplotlib.pyplot as plt  # defer import to keep CLI light

    array = tokens.detach().cpu().numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("tokens must have shape (N, 3).")

    coords = array[:, :2].copy()
    if coordinate_mode == "delta":
        coords = np.cumsum(coords, axis=0)
    elif coordinate_mode != "absolute":
        raise ValueError(f"Unsupported coordinate_mode '{coordinate_mode}'.")

    pen_state = array[:, 2]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    for idx in range(1, coords.shape[0]):
        start = coords[idx - 1]
        end = coords[idx]
        active = pen_state[idx] >= 0.5
        line_color = color if active else "tab:red"
        linestyle = "-" if active else "--"
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=line_color,
            linewidth=linewidth,
            linestyle=linestyle,
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    fig.tight_layout()
    return fig
