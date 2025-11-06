#!/usr/bin/env python3
"""
Visualise diffusion conditioning for a sampled QuickDraw episode.

Example:
    python scripts/visualize_diffusion_episode.py --root data/ --horizon 64
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from dataset.loader import QuickDrawEpisodes
from dataset.diffusion import DiffusionCollator
from dataset.visualize import plot_episode_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise diffusion episode masks.")
    parser.add_argument("--root", required=True, help="Processed dataset root.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--backend", default="lmdb", help="Storage backend.")
    parser.add_argument("--K", type=int, default=5, help="Number of prompts.")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Sequence limit.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--horizon", type=int, default=64, help="Diffusion horizon H.")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to visualise (after collate).",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=["delta", "absolute"],
        default="delta",
        help="Coordinate representation for plotting.",
    )
    parser.add_argument(
        "--save-dir",
        default="figures",
        help="Directory to store generated figures (set empty to disable saving).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display; only save figures.",
    )
    return parser.parse_args()


def main() -> None:
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

    collator = DiffusionCollator(horizon=args.horizon, seed=args.seed)
    batch = collator([dataset[args.index]])

    tokens = batch["tokens"][0].numpy()
    context_mask = batch["context_mask"][0].numpy().astype(bool)
    target_mask = batch["target_mask"][0].numpy().astype(bool)

    ax = plot_episode_tokens(
        tokens,
        show=False,
        coordinate_mode=args.coordinate_mode,
        color="lightgray",
    )
    absolute = tokens[:, :2] if args.coordinate_mode == "absolute" else np.cumsum(
        tokens[:, :2], axis=0
    )

    ax.scatter(
        absolute[context_mask, 0],
        absolute[context_mask, 1],
        s=6,
        color="tab:blue",
        label="Context",
        alpha=0.8,
    )
    ax.scatter(
        absolute[target_mask, 0],
        absolute[target_mask, 1],
        s=40,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.5,
        label="Target",
    )
    ax.legend(loc="best")
    ax.set_title(
        f"Diffusion episode — observed={int(batch['context_lengths'][0])}, "
        f"target={int(batch['target_lengths'][0])}"
    )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, f"diffusion_episode_idx{args.index}.png")
        ax.figure.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(ax.figure)

    dataset.close()


if __name__ == "__main__":
    main()
