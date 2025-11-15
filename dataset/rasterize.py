"""Utilities for rasterising QuickDraw sketches into pixel images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import numpy as np

from .preprocess import ProcessedSketch

try:  # Pillow is optional until rasterisation is used.
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


__all__ = [
    "RasterizerConfig",
    "rasterize_absolute_points",
    "rasterize_processed_sketch",
]


@dataclass(frozen=True)
class RasterizerConfig:
    """Configuration describing how sketches should be rasterised."""

    img_size: int = 64
    antialias: int = 2
    line_width: float = 1.5
    background_value: float = 0.0
    stroke_value: float = 1.0
    normalize_inputs: bool = False


def _ensure_array(data: np.ndarray | Iterable[float]) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return array


def _normalize_points(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = float(np.max(maxs - mins))
    if scale <= 0:
        scale = 1.0
    normalized = (points - center) / (scale / 2.0)
    return np.clip(normalized, -1.0, 1.0)


def _map_to_canvas(points: np.ndarray, size: int) -> np.ndarray:
    scale = (points + 1.0) * 0.5 * (size - 1)
    return np.clip(scale, 0.0, size - 1)


if TYPE_CHECKING:
    from PIL import ImageDraw as _PILImageDraw


def _draw_points(draw: "_PILImageDraw.ImageDraw", points: np.ndarray, radius: int, value: int) -> None:
    if radius <= 0:
        radius = 1
    for x, y in points:
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, fill=value)


def rasterize_absolute_points(
    absolute: np.ndarray,
    pen: np.ndarray,
    *,
    config: RasterizerConfig,
) -> np.ndarray:
    """Render a sequence of absolute points + pen states to a `[H, W]` image."""

    if Image is None or ImageDraw is None:  # pragma: no cover - dependency optional
        raise RuntimeError(
            "Pillow is required for rasterisation. Install it via 'pip install pillow'."
        )

    absolute = _ensure_array(absolute)
    pen = np.asarray(pen, dtype=np.float32).reshape(-1)
    if absolute.shape[0] == 0:
        return np.full((config.img_size, config.img_size), config.background_value, dtype=np.float32)

    if absolute.shape[0] != pen.shape[0]:
        raise ValueError("absolute and pen arrays must have the same length")

    points = absolute.astype(np.float32, copy=False)
    if config.normalize_inputs:
        points = _normalize_points(points)

    canvas_size = int(config.img_size * max(1, config.antialias))
    points = _map_to_canvas(points, canvas_size)

    background = int(round(config.background_value * 255.0))
    stroke = int(round(config.stroke_value * 255.0))
    img = Image.new("L", (canvas_size, canvas_size), color=background)
    draw = ImageDraw.Draw(img)
    width = max(1, int(round(config.line_width * config.antialias)))

    for idx in range(1, points.shape[0]):
        if pen[idx] < 0.5:
            continue
        start = tuple(points[idx - 1])
        end = tuple(points[idx])
        draw.line([start, end], fill=stroke, width=width, joint="round")

    stroke_radius = max(1, width // 2)
    start_indices = [0]
    start_indices.extend(idx for idx, flag in enumerate(pen) if flag < 0.5 and idx > 0)
    start_points = points[start_indices]
    _draw_points(draw, start_points, stroke_radius, stroke)

    if config.antialias > 1:
        img = img.resize((config.img_size, config.img_size), resample=Image.BOX)

    array = np.asarray(img, dtype=np.float32) / 255.0
    return array.astype(np.float32, copy=False)


def rasterize_processed_sketch(sketch: ProcessedSketch, config: RasterizerConfig) -> np.ndarray:
    """Render a preprocessed sketch into a grayscale image."""

    return rasterize_absolute_points(
        sketch.absolute,
        sketch.pen,
        config=config,
    )
