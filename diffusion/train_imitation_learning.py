#!/usr/bin/env python3
"""
Train a DiT-based diffusion policy on QuickDraw episodes using the existing dataset pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch
from ml_collections import ConfigDict, config_flags
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset.diffusion import DiffusionCollator
from dataset.loader import QuickDrawEpisodes
from diffusion import DiTDiffusionPolicy, DiTDiffusionPolicyConfig
from diffusion.sampling import (
    make_start_token,
    sample_quickdraw_tokens_unconditional,
    tokens_to_figure,
)


def load_config(_CONFIG_FILE: str) -> ConfigDict:

    cfg = _CONFIG_FILE.value

    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_qualitative_samples(
    policy: DiTDiffusionPolicy,
    cfg: argparse.Namespace,
    global_step: int,
    device: torch.device,
) -> None:
    """Sample sketches and push them to WandB for quick visual inspection."""

    if cfg.wandb_project is None or cfg.eval_samples <= 0:
        return

    prev_mode = policy.training
    policy.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval_sample_seed + global_step)

    start = make_start_token(cfg.eval_samples, policy.cfg.point_feature_dim, device)
    samples = sample_quickdraw_tokens_unconditional(
        policy,
        cfg.eval_max_tokens,
        start_token=start,
        generator=generator,
    )

    images = []
    batch = len(samples)
    for idx in range(batch):
        fig = tokens_to_figure(samples[idx], coordinate_mode="absolute")
        images.append(wandb.Image(fig, caption=f"step {global_step + 1} sample {idx}"))
        plt.close(fig)

    if images:
        wandb.log({"samples/sketches": images}, step=global_step + 1)

    if prev_mode:
        policy.train()


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="diffusion_policy/configs/imitation_learning.py"
)


def main(_) -> None:
    cfg = load_config(_CONFIG_FILE)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    dataset = QuickDrawEpisodes(
        root=cfg.data_root,
        split=cfg.split,
        K=cfg.K,
        backend=cfg.backend,
        max_seq_len=cfg.max_seq_len,
        augment=False,
        seed=cfg.seed,
        coordinate_mode="absolute",
    )
    collator = DiffusionCollator(horizon=cfg.horizon, seed=cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    noise_scheduler_kwargs = {
        "num_train_timesteps": cfg.num_train_timesteps,
        "beta_start": cfg.beta_start,
        "beta_end": cfg.beta_end,
        "beta_schedule": cfg.beta_schedule,
    }

    policy_cfg = DiTDiffusionPolicyConfig(
        context_length=0,
        horizon=cfg.horizon,
        point_feature_dim=7,  # 2 positions + 1 pen state + 4 special tokens
        action_dim=7,  # x, y, pen
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        mlp_dim=cfg.mlp_dim,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        num_inference_steps=cfg.num_inference_steps,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
    )
    policy = DiTDiffusionPolicy(policy_cfg).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    save_dir = Path(cfg.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb_use:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config={
                **vars(cfg),
                "model": policy_cfg,
            },
        )
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"Model parameter count: {total_params:,}")
        wandb.log({"model/parameters": total_params}, step=0)
    else:
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"Model parameter count: {total_params:,}")

    global_step = 0

    for epoch in range(cfg.epochs):
        policy.train()
        running_loss = 0.0
        step = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for batch in progress:

            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = policy.compute_loss(batch)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            step += 1

            running_loss += float(loss.detach().cpu())
            progress.set_postfix({"mse": metrics["mse"]})

            if (
                cfg.wandb_use
                and cfg.loss_log_every > 0
                and global_step % cfg.loss_log_every == 0
            ):
                wandb.log({"train/batch_loss": metrics["mse"]}, step=global_step)

            if global_step % cfg.save_checkpoint_every == 0:
                checkpoint_path = save_dir / f"policy_epoch_{global_step:06d}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": policy_cfg,
                    },
                    checkpoint_path,
                )

        avg_loss = running_loss / step
        print(f"Epoch {epoch+1}: avg loss {avg_loss:.6f}")
        if cfg.wandb_use:
            wandb.log({"train/mse": avg_loss, "epoch": epoch + 1}, step=global_step)

        _log_qualitative_samples(policy, cfg, global_step, device)

    if cfg.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    from absl import app

    app.run(main)
