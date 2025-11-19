"""Utilities for qualitative sampling and visualisation of QuickDraw policies."""

from __future__ import annotations

import math
from io import BytesIO
from typing import Optional, List, Dict

import numpy as np
import torch

__all__ = [
    "InContextDiffusionCollatorEval",
    "make_start_token",
    "sample_quickdraw_tokens",
    "sample_quickdraw_tokens_unconditinoal",
    "tokens_to_figure",
    "tokens_to_gif",
]



class InContextDiffusionCollatorEval:

    def __init__(
        self,
    ):
        pass

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        points_batch: List[torch.Tensor] = []
        points_lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]
            reset_idx = (tokens[:, 5] == 1.0).nonzero(as_tuple=True)[0]
            context = tokens[:reset_idx].clone()
            points_batch.append(context)
            points_lengths.append(context.shape[0])

        max_len = max(points_lengths)
        batch_size = len(points_batch)
        feature_dim = context[0].shape[-1]

        points = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_len, dtype=torch.float32)

        for idx, (pts, points_len) in enumerate(zip(points_batch, points_lengths)):
            points[idx, -points_len:] = pts
            mask[idx, -points_len:] = True

        return {"points": points, "mask": mask}



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
def sample_quickdraw_tokens_unconditional(
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


@torch.no_grad()
def sample_quickdraw_tokens(
    policy: torch.nn.Module,
    max_tokens: int,
    context: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Autoregressively sample `total_tokens` conditioned on the growing context."""

    device = next(policy.parameters()).device
    feature_dim = policy.cfg.point_feature_dim

    if context['points'].shape[-1] != feature_dim:
        raise ValueError(
            f"start_token feature dim {context['points'].shape[-1]} != {feature_dim}."
        )

    context = {key: v.to(device) for key, v in context.items()}

    horizon = policy.cfg.horizon
    max_chunks = math.ceil(max_tokens / horizon)
    samples: list[torch.Tensor] = []

    for _ in range(max_chunks):
        actions = policy.sample_actions(observation=context, generator=generator)
        samples.append(actions)
        context = {
            'points': torch.cat([context['points'], actions], dim=1),
            'mask': torch.cat([context['mask'], torch.ones(actions.shape[:2]).to(device)], dim=1)
        }

    generated = torch.cat(samples, dim=1)
    sketches = clean_sketches(generated)
    return sketches


def clean_sketches(generated):

    sketches = []
    for sketch in generated:
        end_idx = (sketch[:, 6] >= 0.5).nonzero(as_tuple=True)[0]
        if end_idx.numel() > 0:
            end_idx = end_idx[0]
        else:
            end_idx = sketch.shape[0]
        sketches.append(sketch[:end_idx, :3])

    return sketches


def tokens_to_figure(
    tokens: torch.Tensor | np.ndarray,
    *,
    coordinate_mode: str = "absolute",
    color: str = "black",
    linewidth: float = 1.5,
):
    """Convert `(N, 3)` tokens into a Matplotlib figure for logging or saving."""

    import matplotlib.pyplot as plt  # defer import to keep CLI light

    array = (
        tokens.detach().cpu().numpy()
        if isinstance(tokens, torch.Tensor)
        else np.asarray(tokens)
    )
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


def tokens_to_gif(
    tokens: torch.Tensor | np.ndarray,
    *,
    coordinate_mode: str = "absolute",
    color: str = "black",
    pen_up_color: str = "tab:red",
    linewidth: float = 1.5,
    interval_ms: int = 80,
    loop: int = 0,
    output_path: Optional[str] = None,
) -> bytes:
    """Render an animated GIF showing the drawing unfolding token by token."""

    import matplotlib.pyplot as plt  # defer import to keep CLI light
    from PIL import Image

    array = (
        tokens.detach().cpu().numpy()
        if isinstance(tokens, torch.Tensor)
        else np.asarray(tokens)
    )
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("tokens must have shape (N, 3).")

    coords = array[:, :2].copy()
    if coordinate_mode == "delta":
        coords = np.cumsum(coords, axis=0)
    elif coordinate_mode != "absolute":
        raise ValueError(f"Unsupported coordinate_mode '{coordinate_mode}'.")

    pen_state = array[:, 2]

    def _figure_to_rgb(current_fig):
        canvas = current_fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        return data.reshape(height, width, 4)[..., :3].copy()

    frames: list[Image.Image] = []
    num_points = coords.shape[0]
    steps = max(2, num_points)

    for step in range(1, steps):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        for idx in range(1, min(step + 1, num_points)):
            start = coords[idx - 1]
            end = coords[idx]
            active = pen_state[idx] >= 0.5
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color if active else pen_up_color,
                linewidth=linewidth,
                linestyle="-" if active else "--",
            )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        fig.tight_layout()
        frame = _figure_to_rgb(fig)
        frames.append(Image.fromarray(frame))
        plt.close(fig)

    if not frames:
        fig = tokens_to_figure(
            tokens, coordinate_mode=coordinate_mode, color=color, linewidth=linewidth
        )
        frame = _figure_to_rgb(fig)
        frames.append(Image.fromarray(frame))
        plt.close(fig)

    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=max(1, interval_ms),
        loop=max(0, loop),
        disposal=2,
    )
    data = buffer.getvalue()
    if output_path is not None:
        with open(output_path, "wb") as handle:
            handle.write(data)
    buffer.close()
    for frame in frames:
        frame.close()
    return data
