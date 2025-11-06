"""
Helpers for preparing QuickDraw episodes for diffusion-policy training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .loader import quickdraw_collate_fn

__all__ = ["DiffusionCollatorConfig", "DiffusionCollator"]


@dataclass
class DiffusionCollatorConfig:
    """Configuration describing the context/target windows for diffusion training."""

    horizon: int


class DiffusionCollator:
    """
    Collate callable that pads batches and builds masks for diffusion policies.

    The collator always reveals a random number of query tokens per episode,
    uniformly sampled from the valid range `[0, max(query_len - horizon, 0)]`
    so that at least `horizon` tokens remain for denoising.
    """

    def __init__(
        self,
        horizon: int,
        seed: int = 0,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.horizon = horizon
        self.rng = np.random.RandomState(seed)

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        """Pad the batch and append context/target masks."""
        base = quickdraw_collate_fn(batch)
        tokens = base["tokens"]
        mask = base["mask"]
        B, T, _ = tokens.shape
        context_mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        target_mask = torch.zeros(B, T, dtype=torch.bool, device=tokens.device)
        context_lengths = torch.zeros(B, dtype=torch.long, device=tokens.device)
        target_lengths = torch.zeros(B, dtype=torch.long, device=tokens.device)
        observed_query_tokens = torch.zeros(B, dtype=torch.long, device=tokens.device)

        for idx in range(B):
            meta = base["metadata"][idx]
            query_start = self._resolve_query_start(meta, tokens[idx])
            if query_start is None:
                raise RuntimeError("Unable to resolve query start index for sample.")
            query_length = int(
                meta.get("query_length", mask[idx, query_start:].sum().item())
            )
            query_length = max(query_length, 0)
            observe = self._sample_observe_tokens(query_length)
            target_len = min(self.horizon, max(query_length - observe, 0))

            context_end = min(T, query_start + observe)
            target_start = context_end
            target_end = min(T, target_start + target_len)

            context_mask[idx, :context_end] = True
            context_mask[idx] &= mask[idx]
            target_mask[idx, target_start:target_end] = True
            target_mask[idx] &= mask[idx]

            context_lengths[idx] = int(context_mask[idx].sum().item())
            target_lengths[idx] = int(target_mask[idx].sum().item())
            observed_query_tokens[idx] = int(observe)

        base["context_mask"] = context_mask
        base["target_mask"] = target_mask
        base["context_lengths"] = context_lengths
        base["target_lengths"] = target_lengths
        base["observed_query_tokens"] = observed_query_tokens
        base["horizon"] = self.horizon
        return base

    @staticmethod
    def _resolve_query_start(
        meta: Dict[str, object], tokens: torch.Tensor
    ) -> Optional[int]:
        """Return the index of the first query token based on metadata or tokens."""
        if "query_start_index" in meta:
            return int(meta["query_start_index"])

        reset = (tokens[:, 5] > 0.5).nonzero(as_tuple=False)
        start = (tokens[:, 3] > 0.5).nonzero(as_tuple=False)
        if reset.numel() == 0 or start.numel() == 0:
            return None
        reset_idx = int(reset[0, 0].item())
        starts_after = start[start[:, 0] > reset_idx]
        if starts_after.numel() == 0:
            return None
        return int(starts_after[0, 0].item() + 1)

    def _sample_observe_tokens(self, query_length: int) -> int:
        """Return the number of observable query tokens for a single sample."""
        if query_length <= 0:
            return 0
        max_before_horizon = max(query_length - self.horizon, 0)
        if max_before_horizon == 0:
            return 0
        return int(self.rng.randint(0, max_before_horizon + 1))
