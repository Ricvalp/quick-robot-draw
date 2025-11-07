#!/usr/bin/env python3
"""
Visualise a sampled QuickDraw K-shot episode.

Example:
    python scripts/visualize_episode.py --root data/ --split train --backend lmdb
"""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from dataset.loader import QuickDrawEpisodes, quickdraw_collate_fn
from dataset.diffusion import DiffusionCollator


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
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader.")
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
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=DiffusionCollator(horizon=5),
        # collate_fn=quickdraw_collate_fn,
        shuffle=True
    )

    import tqdm

    for batch in tqdm.tqdm(loader):
        # Process the batch (e.g., visualize or analyze)
        pass


if __name__ == "__main__":
    main()
