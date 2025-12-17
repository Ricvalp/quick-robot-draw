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
from diffusion.sampling import sample_quickdraw_tokens_unconditional, tokens_to_figure


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

    # if cfg.wandb.use is False or cfg.eval.samples <= 0:
    #     return

    prev_mode = policy.training
    policy.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed + global_step)

    start = (
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        .tile((cfg.eval.samples, 1, 1))
        .to(device=device)
    )
    samples = sample_quickdraw_tokens_unconditional(
        policy,
        cfg.eval.steps,
        context=start,
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
    "config", default="configs/diffusion/imitation_learning.py"
)


def main(_) -> None:
    cfg = load_config(_CONFIG_FILE)
    set_seed(cfg.run.seed)
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    dataset = QuickDrawEpisodes(
        root=cfg.data.root,
        split=cfg.data.split,
        K=cfg.data.K,
        backend=cfg.data.backend,
        max_seq_len=cfg.data.max_query_len,
        retry_on_overflow=True,
        augment=False,
        seed=cfg.run.seed,
        coordinate_mode="absolute",
    )

    collator = DiffusionCollator(horizon=cfg.model.horizon, seed=cfg.run.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    noise_scheduler_kwargs = {
        "num_train_timesteps": cfg.model.num_train_timesteps,
        "beta_start": cfg.model.beta_start,
        "beta_end": cfg.model.beta_end,
        "beta_schedule": cfg.model.beta_schedule,
    }

    policy_cfg = DiTDiffusionPolicyConfig(
        context_length=0,
        horizon=cfg.model.horizon,
        point_feature_dim=7,  # 2 positions + 1 pen state + 4 special tokens
        action_dim=7,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_dim=cfg.model.mlp_dim,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        num_inference_steps=cfg.eval.num_inference_steps,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
    )
    policy = DiTDiffusionPolicy(policy_cfg).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    save_dir = Path(cfg.checkpoint.dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.use:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
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

    for epoch in range(cfg.training.epochs):
        policy.train()
        running_loss = 0.0
        step = 0
        progress = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}", leave=False
        )
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
                cfg.wandb.use
                and cfg.logging.loss_log_every > 0
                and global_step % cfg.logging.loss_log_every == 0
            ):
                wandb.log({"train/batch_loss": metrics["mse"]}, step=global_step)

            # if global_step % cfg.checkpoint.save_interval == 0:

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
        if cfg.wandb.use:
            wandb.log({"train/mse": avg_loss, "epoch": epoch + 1}, step=global_step)

        _log_qualitative_samples(policy, cfg, global_step, device)

    if cfg.wandb.project:
        wandb.finish()


if __name__ == "__main__":
    from absl import app

    app.run(main)
