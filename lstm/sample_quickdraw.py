#!/usr/bin/env python3
"""
Sample sketches from a trained SketchRNN checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple
from ml_collections import ConfigDict, config_flags

import torch

from diffusion_policy.sampling import tokens_to_figure
from lstm import SketchRNN, SketchRNNConfig, strokes_to_tokens, trim_strokes_to_eos


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Sample sketches from a SketchRNN checkpoint.")
#     parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint.")
#     parser.add_argument("--num-samples", type=int, default=8, help="Number of sketches to draw.")
#     parser.add_argument("--steps", type=int, default=256, help="Maximum decoding steps per sketch.")
#     parser.add_argument("--temperature", type=float, default=0.65, help="Sampling temperature.")
#     parser.add_argument("--greedy", action="store_true", help="Use greedy decoding for mixture + pen states.")
#     parser.add_argument("--device", type=str, default="cpu", help="Device for sampling.")
#     parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling noise.")
#     parser.add_argument("--output-dir", type=str, default="samples", help="Directory to store PNGs.")
#     parser.add_argument("--prefix", type=str, default="sketch", help="Filename prefix for saved samples.")
#     return parser.parse_args()


def load_config(_CONFIG_FILE: str) -> ConfigDict:

    cfg = _CONFIG_FILE.value

    return cfg


def load_model(
    checkpoint_path: str, device: torch.device
) -> Tuple[SketchRNN, Optional[int]]:
    state = torch.load(checkpoint_path, map_location=device)
    cfg_data = state.get("config")
    if isinstance(cfg_data, SketchRNNConfig):
        cfg = cfg_data
    else:
        cfg = SketchRNNConfig(**cfg_data)
    model = SketchRNN(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state.get("epoch")


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="lstm/configs/sample.py"
)


def main() -> None:

    cfg = load_config(_CONFIG_FILE)

    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    model, epoch = load_model(cfg.checkpoint, device)
    print(f"Loaded checkpoint '{cfg.checkpoint}' (epoch={epoch}).")

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    samples = model.sample(
        cfg.steps,
        num_samples=cfg.num_samples,
        temperature=cfg.temperature,
        greedy=cfg.greedy,
        generator=generator,
    )
    trimmed = trim_strokes_to_eos(samples)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    for idx, seq in enumerate(trimmed):
        tokens = strokes_to_tokens(seq)
        fig = tokens_to_figure(tokens, coordinate_mode="delta")
        out_path = out_dir / f"{cfg.prefix}_{idx:03d}.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    from absl import app

    app.run(main)
