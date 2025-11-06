#!/usr/bin/env python3
"""
Profile QuickDraw episode loading throughput.
"""

from __future__ import annotations

import argparse
import time

from torch.utils.data import DataLoader

from dataset.loader import QuickDrawEpisodes, quickdraw_collate_fn


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for profiling the data pipeline.

    The resulting namespace feeds directly into the profiling routine.
    """
    parser = argparse.ArgumentParser(description="Profile QuickDraw data loading.")
    parser.add_argument("--root", required=True, help="Processed dataset root directory.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--backend", default="lmdb")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true", help="Enable online augmentations.")
    return parser.parse_args()


def main() -> None:
    """Run a short profiling loop and report throughput statistics.

    Prints aggregate episodes/sec and tokens/sec to STDOUT for quick tuning.
    """
    args = parse_args()
    dataset = QuickDrawEpisodes(
        args.root,
        split=args.split,
        K=args.K,
        backend=args.backend,
        max_seq_len=args.max_seq_len,
        augment=args.augment,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=quickdraw_collate_fn,
        pin_memory=True,
    )

    start = time.time()
    episodes = 0
    tokens = 0
    for step, batch in enumerate(loader):
        episodes += batch["tokens"].shape[0]
        tokens += batch["tokens"].shape[0] * batch["tokens"].shape[1]
        if step + 1 >= args.steps:
            break
    elapsed = time.time() - start
    eps_per_sec = episodes / elapsed if elapsed > 0 else 0.0
    tokens_per_sec = tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"Steps: {min(args.steps, step + 1)} | Episodes/sec: {eps_per_sec:.2f} | "
        f"Tokens/sec: {tokens_per_sec:.2f}"
    )

    dataset.close()


if __name__ == "__main__":
    main()
