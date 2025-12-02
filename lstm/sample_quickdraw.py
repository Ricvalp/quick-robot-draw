#!/usr/bin/env python3
"""
Sample sketches from a trained SketchRNN checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from ml_collections import ConfigDict, config_flags

from diffusion.sampling import tokens_to_figure
from lstm import SketchRNN, SketchRNNConfig, strokes_to_tokens, trim_strokes_to_eos

_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="configs/lstm/sample.py"
)


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
