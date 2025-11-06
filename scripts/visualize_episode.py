#!/usr/bin/env python3
"""
Visualise a sampled QuickDraw K-shot episode.

Example:
    python scripts/visualize_episode.py --root data/ --split train --backend lmdb
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt

from dataset.loader import QuickDrawEpisodes
from dataset.storage import SketchStorage, StorageConfig
from dataset.visualize import plot_episode_tokens, plot_sketch


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the visualization script.

    Returns a namespace describing dataset location and output preferences.
    """
    parser = argparse.ArgumentParser(description="Visualise QuickDraw episodes.")
    parser.add_argument("--root", required=True, help="Processed dataset root.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--backend", default="lmdb", help="Storage backend to use.")
    parser.add_argument("--K", type=int, default=5, help="Number of prompts.")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Sequence limit.")
    parser.add_argument("--index", type=int, default=0, help="Sample index to draw.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--save-dir",
        default="figures",
        help="Directory to store generated figures (set empty to disable saving).",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=["delta", "absolute"],
        default="delta",
        help="Which coordinate representation to visualize.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display; only save figures.",
    )
    return parser.parse_args()


def main() -> None:
    """Sample an episode, plot its tokens/sketches, and optionally save figures.

    The routine also writes PNGs into `--save-dir` for later inspection.
    """
    args = parse_args()

    dataset = QuickDrawEpisodes(
        args.root,
        split=args.split,
        K=args.K,
        backend=args.backend,
        max_seq_len=args.max_seq_len,
        augment=False,
        seed=args.seed,
        coordinate_mode=args.coordinate_mode,
    )
    episode = dataset[args.index]

    storage = SketchStorage(
        StorageConfig(root=args.root, backend=args.backend), mode="r"
    )
    family_id: str = episode["family_id"]
    prompt_ids: List[str] = episode["prompt_ids"]
    query_id: str = episode["query_id"]

    tokens = episode["tokens"].numpy()
    token_ax = plot_episode_tokens(
        tokens, show=False, coordinate_mode=args.coordinate_mode
    )
    token_fig = token_ax.figure

    num_prompts = len(prompt_ids)
    cols = min(num_prompts + 1, 3)
    rows = (num_prompts + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, sample_id in enumerate(prompt_ids):
        sketch = storage.get(family_id, sample_id)
        plot_sketch(sketch, ax=axes[idx], show=False)
        axes[idx].set_title(f"Prompt {idx+1}")

    query_sketch = storage.get(family_id, query_id)
    plot_sketch(query_sketch, ax=axes[num_prompts], show=False, color="tab:orange")
    axes[num_prompts].set_title("Query")
    for idx in range(num_prompts + 1, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        safe_family = family_id.replace(" ", "_")
        base_name = f"{safe_family}_idx{args.index}"
        token_path = os.path.join(args.save_dir, f"{base_name}_tokens.png")
        grid_path = os.path.join(args.save_dir, f"{base_name}_sketches.png")
        token_fig.savefig(token_path, dpi=200, bbox_inches="tight")
        fig.savefig(grid_path, dpi=200, bbox_inches="tight")
        print(f"Saved figures to {token_path} and {grid_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(token_fig)
        plt.close(fig)

    storage.close()
    dataset.close()


if __name__ == "__main__":
    main()
