#!/usr/bin/env python3
"""
Verify QuickDraw dataset integrity.
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch

from dataset.loader import QuickDrawEpisodes, quickdraw_collate_fn
from dataset.storage import DatasetManifest, SketchStorage, StorageConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI options for dataset verification checks.

    The parsed namespace controls which split/backends get inspected.
    """
    parser = argparse.ArgumentParser(description="Verify processed QuickDraw dataset.")
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument("--backend", default="lmdb")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-episodes", type=int, default=16)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def verify_sketch_storage(
    storage: SketchStorage, manifest: DatasetManifest
) -> Dict[str, int]:
    """Ensure cached sketches match manifest counts and basic sanity checks.

    Returns a dictionary of families whose counts deviate from the manifest.
    """
    discrepancies: Dict[str, int] = {}
    for family, expected in manifest.sketch_counts.items():
        samples = storage.samples_for_family(family)
        actual = len(samples)
        if actual != expected:
            discrepancies[family] = actual
        # Load a single sketch for sanity.
        if samples:
            sketch = storage.get(family, samples[0])
            if sketch.length <= 0:
                raise ValueError(f"Sketch {family}/{samples[0]} has zero length.")
            if np.isnan(sketch.absolute).any():
                raise ValueError(f"Sketch {family}/{samples[0]} contains NaNs.")
    return discrepancies


def verify_episode_loader(
    root: str,
    backend: str,
    split: str,
    K: int,
    max_seq_len: int,
    seed: int,
    num_samples: int,
) -> None:
    """Sample a handful of episodes and verify their tensor integrity.

    Raises ValueError if NaNs, length mismatches, or overflow are detected.
    """
    dataset = QuickDrawEpisodes(
        root,
        split=split,
        K=K,
        backend=backend,
        max_seq_len=max_seq_len,
        augment=False,
        seed=seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(4, num_samples),
        num_workers=0,
        collate_fn=quickdraw_collate_fn,
    )
    checked = 0
    for batch in loader:
        tokens = batch["tokens"]
        if torch.isnan(tokens).any():
            raise ValueError("Detected NaNs in episode tokens.")
        if (tokens.abs() > 10).any():
            print("Warning: large token magnitude detected.")
        lengths = batch["lengths"]
        if (lengths <= 0).any():
            raise ValueError("Episode with non-positive length encountered.")
        if (lengths > max_seq_len).any():
            raise ValueError("Episode exceeds configured max sequence length.")
        checked += tokens.shape[0]
        if checked >= num_samples:
            break
    dataset.close()


def main() -> None:
    """Entry point for the verification script.

    Loads the manifest, performs storage checks, and optionally samples episodes.
    """
    args = parse_args()
    manifest_path = f"{args.root}/DatasetManifest.json"
    manifest = DatasetManifest.load(manifest_path)
    print("Loaded manifest:")
    print(manifest.to_dict())

    storage = SketchStorage(
        StorageConfig(root=args.root, backend=args.backend), mode="r"
    )
    discrepancies = verify_sketch_storage(storage, manifest)
    storage.close()

    if discrepancies:
        print("Warning: Sketch count discrepancies detected:")
        for family, actual in discrepancies.items():
            print(
                f"  {family}: manifest={manifest.sketch_counts[family]}, actual={actual}"
            )
    else:
        print("Sketch storage verified.")

    if args.sample_episodes > 0:
        verify_episode_loader(
            args.root,
            args.backend,
            args.split,
            args.K,
            args.max_seq_len,
            args.seed,
            args.sample_episodes,
        )
        print(f"Verified {args.sample_episodes} sampled episodes successfully.")


if __name__ == "__main__":
    main()
