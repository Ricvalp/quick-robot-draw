from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from dataset.rasterize import RasterizerConfig, rasterize_absolute_points


def _trim_stroke_sequence(sequence: np.ndarray) -> np.ndarray:
    eos = np.where(sequence[:, 4] > 0.5)[0]
    if eos.size > 0:
        return sequence[: eos[0] + 1]
    return sequence


def _stroke5_to_absolute(sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seq = _trim_stroke_sequence(sequence)
    if seq.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    absolute = np.cumsum(seq[:, :2], axis=0, dtype=np.float32)
    pen = seq[:, 2].astype(np.float32).copy()
    pen[0] = 0.0
    pen_up = np.where(seq[:, 3] > 0.5)[0] + 1
    pen_up = pen_up[pen_up < pen.shape[0]]
    pen[pen_up] = 0.0
    return absolute, pen


def sketches_to_image_batch(
    sketches: torch.Tensor,
    img_size: int = 64,
) -> torch.Tensor:
    """Convert a batch of stroke-5 sketches to images."""

    if sketches.ndim != 3 or sketches.shape[-1] != 5:
        raise ValueError("sketches must have shape (B, S, 5)")

    device = sketches.device
    batch_size = sketches.shape[0]
    if batch_size == 0:
        return torch.zeros(0, 1, img_size, img_size, device=device)

    raster_config = RasterizerConfig(img_size=img_size, normalize_inputs=True)
    images = []
    for sequence in sketches.detach().cpu().numpy():
        absolute, pen = _stroke5_to_absolute(sequence)
        if absolute.shape[0] == 0:
            rendered = np.zeros((img_size, img_size), dtype=np.float32)
        else:
            rendered = rasterize_absolute_points(absolute, pen, config=raster_config)
        images.append(rendered)

    batch = np.stack(images, axis=0)
    tensor = torch.from_numpy(batch).unsqueeze(1)
    return tensor.to(device=device)
