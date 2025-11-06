"""
PyTorch dataset utilities for loading QuickDraw K-shot episodes.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .episode_builder import EpisodeBuilder
from .storage import DatasetManifest, SketchStorage, StorageConfig

__all__ = ["QuickDrawEpisodes", "quickdraw_collate_fn"]


class QuickDrawEpisodes(Dataset):
    """
    PyTorch dataset that lazily assembles K-shot prompt/query episodes.

    Parameters
    ----------
    root : str
        Root directory containing the processed dataset and manifest.
    split : str
        Dataset split (train/val/test).
    K : int
        Number of prompt examples per episode.
    max_seq_len : int
        Maximum allowed number of tokens per episode (guard).
    backend : str
        Storage backend to use. Should match preprocessing stage.
    augment : bool
        Whether to apply online augmentations during sampling.
    storage_config : Optional[StorageConfig]
        Optional explicit storage configuration. When omitted, a default is
        derived from arguments.
    seed : Optional[int]
        Base seed for deterministic episode sampling across workers.
    augment_config : Optional[dict]
        Augmentation overrides passed to `EpisodeBuilder`.
    """

    def __init__(
        self,
        root: str,
        *,
        split: str = "train",
        K: int = 5,
        max_seq_len: int = 512,
        backend: str = "lmdb",
        augment: bool = True,
        storage_config: Optional[StorageConfig] = None,
        seed: int = 0,
        augment_config: Optional[Dict[str, object]] = None,
    ) -> None:
        self.root = root
        self.split = split
        self.k_shot = K
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.seed = seed

        manifest_path = os.path.join(root, "DatasetManifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. Build the dataset first."
            )
        self.manifest = DatasetManifest.load(manifest_path)

        if storage_config is None:
            storage_config = StorageConfig(root=root, backend=backend)
        self.storage_config = storage_config

        self.sketch_storage = SketchStorage(storage_config, mode="r")
        all_families = self.sketch_storage.families()
        self.family_to_samples: Dict[str, List[str]] = {}

        assigned_families = self._resolve_split_families(all_families)
        for family in assigned_families:
            samples = self.sketch_storage.samples_for_family(family)
            if len(samples) > 0:
                self.family_to_samples[family] = samples

        self.family_ids = sorted(self.family_to_samples.keys())
        if not self.family_ids:
            raise RuntimeError(f"No sketches found for split '{split}'.")

        self.base_rng = np.random.RandomState(seed)
        self.builder = EpisodeBuilder(
            fetch_family=lambda fam: self.family_to_samples[fam],
            fetch_sketch=self.sketch_storage.get,
            family_ids=self.family_ids,
            k_shot=self.k_shot,
            max_seq_len=self.max_seq_len,
            seed=seed,
            augment_config=augment_config,
        )

        self._episode_space = sum(
            max(0, len(samples) - self.k_shot) for samples in self.family_to_samples.values()
        )
        if self._episode_space == 0:
            self._episode_space = len(self.family_ids)

    def __len__(self) -> int:
        """Return the nominal number of unique episode samples.

        This enables PyTorch to size epoch loops even though sampling is stochastic.
        """
        return self._episode_space

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Assemble and return a single K-shot episode as PyTorch tensors.

        Deterministic seeds derived from `index` keep behaviour stable per epoch.
        """
        worker = get_worker_info()
        if worker is None:
            seed = (self.seed + index) % (2**32)
        else:
            seed = (self.seed + worker.id * 10_000 + index) % (2**32)
        rng = np.random.RandomState(seed)
        episode = self.builder.build_episode(augment=self.augment, rng=rng)
        tokens = torch.from_numpy(episode.tokens.astype(np.float32, copy=False))
        return {
            "tokens": tokens,
            "length": tokens.shape[0],
            "family_id": episode.family_id,
            "prompt_ids": episode.metadata.get("prompt_ids", []),
            "query_id": episode.metadata.get("query_id"),
            "metadata": episode.metadata,
        }

    def set_epoch(self, epoch: int) -> None:
        """Update the base random seed for deterministic epoch-level shuffles."""
        self.seed = (self.seed + epoch * 131) % (2**32)

    def _resolve_split_families(self, family_ids: List[str]) -> List[str]:
        """Return the subset of families assigned to the requested split.

        Defaults to all families if no explicit split map is recorded.
        """
        split_map = self.manifest.config.get("family_split_map")
        if split_map:
            return [fam for fam in family_ids if split_map.get(fam, "train") == self.split]
        return family_ids

    def close(self) -> None:
        """Release underlying storage resources.

        Explicit closure is useful when loaders are short-lived CLI tools.
        """
        self.sketch_storage.close()


def quickdraw_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Collate function for DataLoader batching.

    Pads token sequences to the maximum length within the batch and returns
    masks describing valid timesteps.
    """
    if not batch:
        raise ValueError("Empty batch encountered in collate function.")

    max_len = max(item["tokens"].shape[0] for item in batch)
    token_dim = batch[0]["tokens"].shape[1]
    batch_size = len(batch)

    tokens = torch.zeros(batch_size, max_len, token_dim, dtype=batch[0]["tokens"].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    family_ids: List[str] = []
    prompt_ids: List[List[str]] = []
    query_ids: List[str] = []
    metadata: List[Dict[str, object]] = []

    for idx, item in enumerate(batch):
        length = item["tokens"].shape[0]
        tokens[idx, :length] = item["tokens"]
        mask[idx, :length] = True
        lengths[idx] = length
        family_ids.append(item["family_id"])
        prompt_ids.append(list(item["prompt_ids"]))
        query_ids.append(item["query_id"])
        metadata.append(item["metadata"])

    return {
        "tokens": tokens,
        "mask": mask,
        "lengths": lengths,
        "family_id": family_ids,
        "prompt_ids": prompt_ids,
        "query_id": query_ids,
        "metadata": metadata,
    }
