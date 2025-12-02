#!/usr/bin/env python3
"""
Inspect a built QuickDraw dataset by plotting sample sketches.

The script reads configuration from configs/dataset/inspect.py via
ml_collections config_flags. Figures are saved under the configured output
directory (defaults to figures/inspect).
"""

from __future__ import annotations

import os
import random
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")  # use non-interactive backend for headless environments

import matplotlib.pyplot as plt
from absl import app
from ml_collections import config_flags

from dataset.storage import DatasetManifest, SketchStorage, StorageConfig
from dataset.visualize import plot_sketch

_CONFIG = config_flags.DEFINE_config_file(
    "config", default="configs/dataset/inspect.py"
)


def _select(items: Sequence[str], limit: int, rng: random.Random) -> List[str]:
    """Return up to `limit` items sampled (deterministically) from the list."""
    pool = sorted(items)
    if limit is None or limit <= 0 or limit >= len(pool):
        return pool
    return rng.sample(pool, k=limit)


def _resolve_families(
    available: Iterable[str], requested: Sequence[str] | None
) -> List[str]:
    """Filter available families based on the requested list."""
    available_set = set(available)
    if not requested:
        return sorted(available_set)
    resolved = [fam for fam in requested if fam in available_set]
    if not resolved:
        raise RuntimeError("Requested families not found in storage.")
    return sorted(resolved)


def main(_) -> None:
    cfg = _CONFIG.value
    rng = random.Random(int(cfg.get("seed", 0)))

    storage_cfg = StorageConfig(
        root=cfg.root,
        backend=cfg.get("backend", "lmdb"),
    )

    output_dir = cfg.get("output_dir", "figures/inspect")
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(cfg.root, "DatasetManifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Build the dataset first."
        )
    DatasetManifest.load(manifest_path)  # Used as a sanity check.

    sketch_storage = SketchStorage(storage_cfg, mode="r")
    try:
        families = _resolve_families(sketch_storage.families(), cfg.get("families"))
        if not families:
            raise RuntimeError("No families available for inspection.")

        sketches_per_family = int(cfg.get("num_sketches_per_family", 4))
        for fam in families:
            sample_ids = sketch_storage.samples_for_family(fam)
            chosen = _select(sample_ids, sketches_per_family, rng)
            for idx, sample_id in enumerate(chosen, start=1):
                sketch = sketch_storage.get(fam, sample_id)
                ax = plot_sketch(sketch, show=False)
                fig = ax.get_figure()
                fig.savefig(
                    os.path.join(output_dir, f"{fam}_sketch_{idx}.png"),
                    bbox_inches="tight",
                )
                plt.close(fig)
    finally:
        sketch_storage.close()


if __name__ == "__main__":
    app.run(main)
