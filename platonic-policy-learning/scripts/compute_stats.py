from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

# Ensure the project root is on the path so `src` imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rlbench.datasets.imitation_learning.cached_dataset import get_temporal_cached_dataloader

from src.utils.stats import RunningStat, NormalizationStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dataset normalization statistics.")
    parser.add_argument("--cache-path", type=str, required=True, help="RLBench imitation cache root directory.")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Subset of RLBench tasks to include.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size for the dataloader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pinned memory for the dataloader.")
    parser.add_argument("--drop-last", action="store_true", help="Drop the last incomplete batch.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples while computing stats.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches processed.")
    parser.add_argument("--output_dir", type=str, default="artifacts/stats", help="Directory to write the computed stats JSON.")
    parser.add_argument("--output", type=str, default="normalization_stats.json", help="Path to write the computed stats JSON.")
    return parser.parse_args()


def process_batch(
    batch: dict,
    position_stats: RunningStat,
    color_stats: RunningStat,
) -> None:
    point_cloud_sequence = batch["observation"]["point_cloud_sequence"]
    for views in point_cloud_sequence:
        for view in views:
            points = view["points"].to(dtype=torch.float32)
            colors = view["colors"].to(dtype=torch.float32)
            position_stats.update(points)
            color_stats.update(colors)

    proprio = batch["observation"]["proprio_sequence"][..., :3].to(dtype=torch.float32)
    position_stats.update(proprio)

    actions = batch["action"][..., :3].to(dtype=torch.float32)
    position_stats.update(actions)


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_path).expanduser().resolve()

    dataloader = get_temporal_cached_dataloader(
        cache_root,
        tasks=args.tasks,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )

    position_stats = RunningStat(dim=3)
    color_stats = RunningStat(dim=3)

    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None

    progress = tqdm(dataloader, total=total_batches, desc="Computing stats", leave=False)

    for step, batch in enumerate(progress, start=1):
        process_batch(batch, position_stats, color_stats)
        if total_batches:
            progress.set_postfix(processed=step)
        if args.max_batches is not None and step >= args.max_batches:
            progress.close()
            break
    else:
        progress.close()

    # Make sure output directory exists.
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output

    stats = NormalizationStats(
        position_mean=position_stats.mean,
        position_std=position_stats.std,
        color_mean=color_stats.mean,
        color_std=color_stats.std,
    )
    stats.to_json(output_path)

    print(f"Wrote normalization stats to {output_path}")
    print(f"Position mean: {stats.position_mean.tolist()}")
    print(f"Position std: {stats.position_std.tolist()}")
    print(f"Color mean: {stats.color_mean.tolist()}")
    print(f"Color std: {stats.color_std.tolist()}")


if __name__ == "__main__":
    main()
