"""
Visualisation utilities for inspecting QuickDraw sketches and episodes.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .preprocess import ProcessedSketch

__all__ = ["plot_sketch", "plot_episode_tokens"]


def _reconstruct_absolute_from_tokens(
    tokens: np.ndarray, mode: str = "delta"
) -> np.ndarray:
    """Reconstruct absolute coordinates from tokens."""
    if mode == "absolute":
        return tokens[:, :2]
    return np.cumsum(tokens[:, :2], axis=0)


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
    coordinate_mode: str = "delta",
) -> plt.Axes:
    """
    Plot the trajectory encoded by an episode token matrix.

    Special tokens (start/sep/reset/stop) are visualised as coloured markers.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    absolute = _reconstruct_absolute_from_tokens(tokens, coordinate_mode)
    pen = tokens[:, 2]

    for idx in range(1, absolute.shape[0]):
        start = absolute[idx - 1]
        end = absolute[idx]
        if pen[idx] >= 0.5:
            line_color = color
            linestyle = "-"
        else:
            line_color = "tab:red"
            linestyle = "--"
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=line_color,
            linewidth=1.2,
            linestyle=linestyle,
        )

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
    # Distinguish between global START control tokens and per-sketch START flags.
    global_start = []
    sketch_start = []
    for idx in start_indices:
        if idx == 0:
            global_start.append(idx)
        elif tokens[idx - 1, 6] > 0.5 or tokens[idx - 1, 5] > 0.5:
            global_start.append(idx)
        else:
            sketch_start.append(idx)

    ax.scatter(
        absolute[global_start, 0],
        absolute[global_start, 1],
        color="tab:green",
        marker="^",
        label="GLOBAL START",
    )
    ax.scatter(
        absolute[sketch_start, 0],
        absolute[sketch_start, 1],
        color="tab:olive",
        marker="s",
        label="SKETCH START",
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
