"""
Utility helpers for working with SketchRNN stroke tensors.
"""

from __future__ import annotations

import math
from typing import List

import torch

__all__ = ["trim_strokes_to_eos", "strokes_to_tokens", "WarmupCosineScheduler"]


def trim_strokes_to_eos(strokes: torch.Tensor) -> List[torch.Tensor]:
    """
    Split a batch of stroke tensors into individual sequences trimmed at EOS.

    Parameters
    ----------
    strokes : torch.Tensor
        Tensor shaped ``(B, T, 5)`` representing `(Δx, Δy, p1, p2, p3)` tokens.

    Returns
    -------
    List[torch.Tensor]
        List containing per-sample tensors whose length stops immediately after
        the first EOS marker (or uses the full sequence when EOS is absent).
    """
    trimmed: List[torch.Tensor] = []
    for seq in strokes:
        eos = torch.nonzero(seq[:, -1] > 0.5, as_tuple=False)
        if eos.numel() > 0:
            end = int(eos[0].item()) + 1
        else:
            end = seq.shape[0]
        trimmed.append(seq[:end].detach().clone())
    return trimmed


def strokes_to_tokens(strokes: torch.Tensor) -> torch.Tensor:
    """
    Convert `(Δx, Δy, p1, p2, p3)` strokes into `(Δx, Δy, pen)` tokens.

    The returned tensor can be rendered with `diffusion_policy.sampling.tokens_to_figure`
    using `coordinate_mode="delta"`, since the deltas accumulate to absolute
    coordinates and the pen channel corresponds to "pen down" activations.
    """
    tokens = torch.zeros(
        strokes.shape[0], 3, dtype=strokes.dtype, device=strokes.device
    )
    tokens[:, :2] = strokes[:, :2]
    tokens[:, 2] = strokes[:, 2]
    return tokens


class WarmupCosineScheduler:
    """
    Linear warmup to max_lr, then cosine decay to min_lr.
    Step this once per optimizer update.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1

        if self.step_num < self.warmup_steps:
            # ------- Linear warmup -------
            lr = self.max_lr * (self.step_num / self.warmup_steps)

        else:
            # ------- Cosine decay -------
            progress = (self.step_num - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        # apply LR to optimizer
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        return lr  # so you can log it

    def get_last_lr(self):
        return self.optimizer.param_groups[0]["lr"]
