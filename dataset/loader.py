"""
PyTorch dataset utilities for loading QuickDraw K-shot episodes.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .episode_builder import EpisodeBuilder
from .storage import DatasetManifest, SketchStorage, StorageConfig

__all__ = ["QuickDrawEpisodes"]


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
        max_seq_len : Optional[int]
            Maximum allowed number of tokens per episode. When provided,
            the loader will resample episodes until the limit is met (with
            a warning) or fail after several attempts.
    backend : str
        Storage backend to use. Should match preprocessing stage.
    augment : bool
        Whether to apply online augmentations during sampling.
    coordinate_mode : str
        `"delta"` (default) for motion deltas or `"absolute"` for absolute positions.
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
        max_seq_len: Optional[int] = 512,
        backend: str = "lmdb",
        augment: bool = True,
        storage_config: Optional[StorageConfig] = None,
        seed: int = 0,
        augment_config: Optional[Dict[str, object]] = None,
        coordinate_mode: str = "delta",
    ) -> None:
        self.root = root
        self.split = split
        self.k_shot = K
        self.max_seq_len = max_seq_len
        self.augment = augment
        self.seed = seed
        self.augment_config = augment_config
        self.coordinate_mode = coordinate_mode

        manifest_path = os.path.join(root, "DatasetManifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. Build the dataset first."
            )
        self.manifest = DatasetManifest.load(manifest_path)

        if storage_config is None:
            storage_config = StorageConfig(root=root, backend=backend)
        self.storage_config = storage_config

        tmp_sketch_storage = SketchStorage(storage_config, mode="r")
        all_families = tmp_sketch_storage.families()
        self.family_to_samples: Dict[str, List[str]] = {}

        assigned_families = self._resolve_split_families(all_families)
        for family in assigned_families:
            samples = tmp_sketch_storage.samples_for_family(family)
            if len(samples) > 0:
                self.family_to_samples[family] = samples

        tmp_sketch_storage.close()

        self.family_ids = sorted(self.family_to_samples.keys())
        if not self.family_ids:
            raise RuntimeError(f"No sketches found for split '{split}'.")

        self.base_rng = np.random.RandomState(seed)

        self.sketch_storage = None
        self.builder = None
        self._worker_pid = None

        self.retry_on_overflow = self.max_seq_len is not None
        self.max_retry_attempts = 8
        if self.retry_on_overflow:
            warnings.warn(
                f"QuickDrawEpisodes will resample episodes until they fit "
                f"max_seq_len={self.max_seq_len} (up to {self.max_retry_attempts} attempts).",
                RuntimeWarning,
                stacklevel=2,
            )

        self._episode_space = sum(
            max(0, len(samples) - self.k_shot)
            for samples in self.family_to_samples.values()
        )
        if self._episode_space == 0:
            self._episode_space = len(self.family_ids)

    def __getstate__(self):
        """Customize pickling to avoid non-fork-safe resources."""
        state = self.__dict__.copy()
        state["sketch_storage"] = None
        state["builder"] = None
        state["_worker_pid"] = None
        return state

    def __len__(self) -> int:
        """Return the nominal number of unique episode samples.

        This enables PyTorch to size epoch loops even though sampling is stochastic.
        """
        return self._episode_space

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Assemble and return a single K-shot episode as PyTorch tensors.

        Deterministic seeds derived from `index` keep behaviour stable per epoch.
        """

        self._ensure_worker_state()  # Initialize SketchStorage and EpisodeBuilder per worker

        worker = get_worker_info()
        if worker is None:
            base_seed = (self.seed + index) % (2**32)
        else:
            base_seed = (self.seed + worker.id * 10_000 + index) % (2**32)

        attempts = self.max_retry_attempts if self.retry_on_overflow else 1
        episode = None
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            rng_seed = (base_seed + attempt * 9773) % (2**32)
            rng = np.random.RandomState(rng_seed)
            try:
                episode = self.builder.build_episode(augment=self.augment, rng=rng)
                break
            except ValueError as exc:
                if not self.retry_on_overflow or "Episode length" not in str(exc):
                    raise
                last_exc = exc
                continue
        if episode is None:
            raise RuntimeError(
                f"Unable to sample an episode ≤ {self.max_seq_len} tokens after "
                f"{attempts} attempts."
            ) from last_exc
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
            return [
                fam for fam in family_ids if split_map.get(fam, "train") == self.split
            ]
        return family_ids

    def close(self) -> None:
        """Release underlying storage resources.

        Explicit closure is useful when loaders are short-lived CLI tools.
        """
        if self.sketch_storage is not None:
            self.sketch_storage.close()

    def _ensure_worker_state(self) -> None:
        """Ensure per-worker state is initialized for multi-process loading.

        Necessary because SketchStorage and EpisodeBuilder are not fork-safe.
        """
        pid = os.getpid()
        if self.builder is not None and self._worker_pid == pid:
            return
        if self.sketch_storage is not None:
            self.sketch_storage.close()
        self.sketch_storage = SketchStorage(self.storage_config, mode="r")
        self.builder = EpisodeBuilder(
            fetch_family=self._fetch_family,
            fetch_sketch=self.sketch_storage.get,
            family_ids=self.family_ids,
            k_shot=self.k_shot,
            max_seq_len=self.max_seq_len,
            seed=self.seed,
            augment_config=self.augment_config,
            coordinate_mode=self.coordinate_mode,
        )
        self._worker_pid = pid

    def _fetch_family(self, family_id: str) -> List[str]:
        return self.family_to_samples[family_id]
