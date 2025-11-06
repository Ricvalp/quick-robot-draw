"""
Visualisation utilities for inspecting QuickDraw sketches and episodes.
"""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

from .preprocess import ProcessedSketch

__all__ = ["plot_sketch", "plot_episode_tokens"]


def _reconstruct_absolute_from_tokens(tokens: np.ndarray) -> np.ndarray:
    """Reconstruct absolute coordinates from (dx, dy) tokens."""
    absolute = np.cumsum(tokens[:, :2], axis=0)
    return absolute


def plot_sketch(
    sketch: ProcessedSketch,
    *,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    color: str = "black",
    linewidth: float = 1.5,
) -> plt.Axes:
    """
    Plot a processed sketch using matplotlib.

    Parameters
    ----------
    sketch : ProcessedSketch
        Sketch to visualise.
    ax : Optional[plt.Axes]
        Optional axes to draw onto. Creates a new figure when omitted.
    show : bool
        Call `plt.show()` after plotting.
    color : str
        Line colour.
    linewidth : float
        Line width for drawing strokes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    points = sketch.absolute
    pen = sketch.pen

    segments = []
    current = [points[0]]
    for idx in range(1, points.shape[0]):
        if pen[idx] < 0.5:
            segments.append(np.stack(current))
            current = [points[idx]]
        else:
            current.append(points[idx])
    if current:
        segments.append(np.stack(current))

    for segment in segments:
        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=linewidth)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(f"{sketch.family_id} ({sketch.sample_id})")
    ax.axis("off")
    if show:
        plt.show()
    return ax


def plot_episode_tokens(
    tokens: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    show: bool = False,
    color: str = "tab:blue",
    separator_color: str = "tab:red",
) -> plt.Axes:
    """
    Plot the trajectory encoded by an episode token matrix.

    Special tokens (start/sep/reset/stop) are visualised as coloured markers.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    absolute = _reconstruct_absolute_from_tokens(tokens)
    pen = tokens[:, 2]

    segments = []
    current = [absolute[0]]
    for idx in range(1, absolute.shape[0]):
        if pen[idx] < 0.5:
            if current:
                segments.append(np.stack(current))
            current = [absolute[idx]]
        else:
            current.append(absolute[idx])
    if current:
        segments.append(np.stack(current))

    for segment in segments:
        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1.2)

    sep_indices = np.where(tokens[:, 4] > 0.5)[0]
    reset_indices = np.where(tokens[:, 5] > 0.5)[0]
    start_indices = np.where(tokens[:, 3] > 0.5)[0]
    stop_indices = np.where(tokens[:, 6] > 0.5)[0]

    ax.scatter(
        absolute[sep_indices, 0],
        absolute[sep_indices, 1],
        color=separator_color,
        marker="x",
        label="SEP",
    )
    ax.scatter(
        absolute[reset_indices, 0],
        absolute[reset_indices, 1],
        color="tab:orange",
        marker="s",
        label="RESET",
    )
    ax.scatter(
        absolute[start_indices, 0],
        absolute[start_indices, 1],
        color="tab:green",
        marker="^",
        label="START",
    )
    ax.scatter(
        absolute[stop_indices, 0],
        absolute[stop_indices, 1],
        color="tab:red",
        marker="o",
        label="STOP",
    )

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(loc="upper right")
    if show:
        plt.show()
    return ax
