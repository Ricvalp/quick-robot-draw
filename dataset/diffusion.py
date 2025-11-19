"""
Helpers for preparing QuickDraw episodes for diffusion-policy training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

__all__ = ["DiffusionCollator", "InContextDiffusionCollator"]


class DiffusionCollator:
    """
    Unconditional diffusion collator: treats the entire sketch history as context and
    learns to denoise a future horizon chunk.
    """

    def __init__(self, horizon: int, seed: int = 0) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        points_batch: List[torch.Tensor] = []
        actions_batch: List[int] = []
        trim_lengths: List[int] = []
        points_lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]
            filtered = self._filter_tokens(tokens)
            seq_len = filtered.shape[0]
            if seq_len < self.horizon + 1:
                continue
            start_idx = int(self.rng.integers(1, seq_len - self.horizon + 1))
            trim_len = start_idx + self.horizon
            points = filtered[:start_idx].clone()
            actions = filtered[start_idx:trim_len].clone()

            points_batch.append(points)
            actions_batch.append(actions)
            trim_lengths.append(trim_len)
            points_lengths.append(start_idx)

        if not points_batch:
            raise ValueError("No valid samples for diffusion collator.")

        max_len = max(trim_lengths)
        batch_size = len(points_batch)
        feature_dim = points_batch[0].shape[-1]

        points = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
        actions = torch.zeros(
            batch_size, self.horizon, feature_dim, dtype=torch.float32
        )
        mask = torch.zeros(batch_size, max_len + self.horizon, dtype=torch.bool)

        for idx, (pts, acts, points_len) in enumerate(
            zip(points_batch, actions_batch, points_lengths)
        ):
            points[idx, -points_len:] = pts
            mask[idx, -(points_len + self.horizon) :] = True
            actions[idx, : self.horizon] = acts

        return {"points": points, "actions": actions, "mask": mask}

    @staticmethod
    def _filter_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """Drop special tokens, keeping only actual sketch deltas."""
        reset_idx = (tokens[:, 5] == 1.0).nonzero(as_tuple=True)[0]
        if reset_idx.numel() > 0:
            filtered = tokens[reset_idx[0] + 1 : -1]
        else:
            raise ValueError("No reset token found in sketch tokens.")
        return filtered[:, :3]  # Keep only x, y, pen_state


class InContextDiffusionCollator:
    """
    In-context diffusion collator.
    """

    def __init__(self, horizon: int, seed: int = 0) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        points_batch: List[torch.Tensor] = []
        actions_batch: List[int] = []
        trim_lengths: List[int] = []
        points_lengths: List[int] = []

        for sample in batch:
            tokens = sample["tokens"]
            reset_idx = (tokens[:, 5] == 1.0).nonzero(as_tuple=True)[0]
            query_len = tokens.shape[0] - reset_idx
            start_idx = int(self.rng.integers(reset_idx, reset_idx + query_len))
            points = tokens[:start_idx].clone()
            actions = tokens[start_idx : start_idx + self.horizon].clone()

            if actions.shape[0] < self.horizon:
                actions = self._pad_actions(actions)

            points_batch.append(points)
            actions_batch.append(actions)
            trim_lengths.append(points.shape[0] + actions.shape[0])
            points_lengths.append(points.shape[0])

        if not points_batch:
            raise ValueError("No valid samples for diffusion collator.")

        max_len = max(trim_lengths)
        batch_size = len(points_batch)
        feature_dim = points_batch[0].shape[-1]

        points = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
        actions = torch.zeros(
            batch_size, self.horizon, feature_dim, dtype=torch.float32
        )
        mask = torch.zeros(batch_size, max_len + self.horizon, dtype=torch.bool)

        for idx, (pts, acts, points_len) in enumerate(
            zip(points_batch, actions_batch, points_lengths)
        ):
            points[idx, -points_len:] = pts
            mask[idx, -(points_len + self.horizon) :] = True
            actions[idx, : self.horizon] = acts

        return {"points": points, "actions": actions, "mask": mask}

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Pads actions that are shorter than the horizon with end-tokens."""
        pad_len = self.horizon - actions.shape[0]
        padding = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).tile((pad_len, 1))

        return torch.cat([actions, padding])
