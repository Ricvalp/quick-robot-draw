"""
Utility helpers for working with SketchRNN stroke tensors.
"""

from __future__ import annotations

from typing import List

import torch

__all__ = ["trim_strokes_to_eos", "strokes_to_tokens"]


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
        eos = torch.nonzero(seq[:, 4] > 0.5, as_tuple=False)
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
