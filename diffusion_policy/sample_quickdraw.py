#!/usr/bin/env python3
"""Sample sketches from a trained DiT diffusion policy checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from diffusion_policy import DiTDiffusionPolicy, DiTDiffusionPolicyConfig
from diffusion_policy.sampling import make_start_token, sample_quickdraw_tokens, tokens_to_figure

try:  # PyTorch 2.6+ defaults to weights_only=True and blocks custom globals.
    from torch.serialization import add_safe_globals

    add_safe_globals([DiTDiffusionPolicyConfig])
except Exception:  # pragma: no cover - older PyTorch versions
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample QuickDraw sketches with a diffusion policy checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the saved policy checkpoint (.pt).")
    parser.add_argument("--device", default="cuda", help="Computation device (e.g. cuda or cpu).")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of sketches to sample in a batch.")
    parser.add_argument("--tokens-per-sample", type=int, default=256, help="How many tokens to generate per sketch.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic sampling.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Override the diffusion sampler steps.")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory to store generated figures (set empty to skip saving).")
    parser.add_argument("--prefix", type=str, default="sample", help="Filename prefix for saved figures.")
    parser.add_argument("--coordinate-mode", choices=["absolute", "delta"], default="absolute", help="Whether the tokens represent absolute coords or deltas.")
    parser.add_argument("--show", action="store_true", help="Display the figures interactively after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available, falling back to CPU.")
            device = torch.device("cpu")
    except RuntimeError:
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    policy_cfg = checkpoint["config"]
    policy = DiTDiffusionPolicy(policy_cfg).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    if args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

    start = make_start_token(args.num_samples, policy.cfg.point_feature_dim, device)
    samples = sample_quickdraw_tokens(
        policy,
        args.tokens_per_sample,
        start_token=start,
        generator=generator,
    )

    output_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    for idx, sample in enumerate(samples):
        fig = tokens_to_figure(sample, coordinate_mode=args.coordinate_mode)
        if output_dir is not None:
            path = output_dir / f"{args.prefix}_{idx:03d}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Saved {path}")
        figures.append(fig)

    if args.show:
        for fig in figures:
            fig.show()
        plt.show()

    for fig in figures:
        plt.close(fig)


if __name__ == "__main__":
    main()
