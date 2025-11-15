from typing import List, Dict
import io
import numpy as np
import torch
import webdataset as wds
from glob import glob
from torch.utils.data import DataLoader

__all__ = [
    "EpisodeToImage",
    "get_cached_loader",
]

class EpisodeToImage:
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
        actions = torch.zeros(batch_size, self.horizon, feature_dim, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_len + self.horizon, dtype=torch.bool)

        for idx, (pts, acts, points_len) in enumerate(zip(points_batch, actions_batch, points_lengths)):
            points[idx, -points_len:] = pts
            mask[idx, -(points_len + self.horizon):] = True
            actions[idx, : self.horizon] = acts

        return {
            "points": points,
            "actions": actions,
            "mask": mask
        }

    @staticmethod
    def _filter_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """Drop special tokens, keeping only actual sketch deltas."""
        reset_idx = (tokens[:, 5] == 1.0).nonzero(as_tuple=True)[0]
        if reset_idx.numel() > 0:
            filtered = tokens[reset_idx[0] + 1:-1]
        else:
            raise ValueError("No reset token found in sketch tokens.")
        return filtered[:, :3]  # Keep only x, y, pen_state


def decode_pt(sample):
    """
    sample: a tuple (strokes_bytes, lengths_bytes, mask_bytes, ...)
    This function converts bytes to tensors.
    """

    decoded = {}
    for key, value in sample.items():
        if key.endswith(".pt"):
            buf = io.BytesIO(value)
            decoded[key[:-3]] = torch.load(buf)   # remove ".pt"
    return decoded

def cached_collate(samples):
    # Simply concatenate along batch dimension
    batch = {}
    for key in samples[0].keys():
        batch[key] = torch.cat([s[key] for s in samples], dim=0)
    return batch

def get_cached_loader(shard_glob, batch_size, num_workers=4):
    shards = sorted(glob(shard_glob))

    dataset = (
        wds.WebDataset(shards)
           .decode()                               # identity, we decode manually
           .to_tuple("strokes.pt", "lengths.pt", "mask.pt")
           .map(lambda tup: {"strokes.pt": tup[0], "lengths.pt": tup[1], "mask.pt": tup[2]})
           .map(decode_pt)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=cached_collate,
        pin_memory=True,
        persistent_workers=True
    )


