#!/usr/bin/env python3
"""
Train a DiT-based diffusion policy on QuickDraw episodes using the existing dataset pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict
from ml_collections import ConfigDict, config_flags

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from diffusion_policy import DiTDiffusionPolicy, DiTDiffusionPolicyConfig
from diffusion_policy.sampling import InContextDiffusionCollatorEval, sample_quickdraw_tokens, tokens_to_figure
from dataset.loader import QuickDrawEpisodes
from dataset.diffusion import InContextDiffusionCollator


def load_config(_CONFIG_FILE: str) -> ConfigDict:

    cfg = _CONFIG_FILE.value

    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_qualitative_samples(
    policy: DiTDiffusionPolicy,
    context: torch.Tensor,
    cfg: dict,
    step: int,
    device: torch.device,
) -> None:
    """Sample sketches and push them to WandB for quick visual inspection."""

    if cfg.wandb_project is None or cfg.eval_samples <= 0:
        return

    prev_mode = policy.training
    policy.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval_sample_seed + step)

    samples = sample_quickdraw_tokens(
        policy=policy,
        max_tokens=cfg.eval_max_tokens,
        context=context,
        generator=generator,
    )

    import matplotlib.pyplot as plt

    images = []
    batch_size = len(samples)
    for idx in range(batch_size):
        fig = tokens_to_figure(samples[idx], coordinate_mode="absolute")
        images.append(wandb.Image(fig, caption=f"step {step + 1} sample {idx}"))
        plt.close(fig)

    if images:
        wandb.log({"samples/sketches": images}, step=step + 1)

    if prev_mode:
        policy.train()


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="diffusion_policy/configs/in_context_imitation_learning.py"
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
    collator = InContextDiffusionCollator(horizon=cfg.horizon, seed=cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )
    
    eval_collator = InContextDiffusionCollatorEval()
    eval_dataloader = DataLoader(
        dataset,
        batch_size=cfg.eval_samples,
        shuffle=True,
        collate_fn=eval_collator,
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

    if cfg.wandb_project:
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
        eval_iterator = iter(eval_dataloader)

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

            running_loss += float(loss.detach().cpu())
            global_step += 1
            step += 1
            progress.set_postfix({"mse": metrics["mse"]})

            if (
                cfg.wandb_project
                and cfg.loss_log_every > 0
                and global_step % cfg.loss_log_every == 0
            ):
                wandb.log({"train/batch_loss": metrics["mse"]}, step=global_step)

            if global_step % cfg.eval_interval == 0:
                try:
                    context = next(eval_iterator)
                    _log_qualitative_samples(
                        policy=policy,
                        context=context,
                        cfg=cfg,
                        step=global_step,
                        device=device,
                        )
                except StopIteration as err:
                    print("WARNING:" , err)


        avg_loss = running_loss / step
        print(f"Epoch {epoch+1}: avg loss {avg_loss:.6f}")
        if cfg.wandb_project:
            wandb.log({"train/mse": avg_loss, "epoch": epoch + 1}, step=global_step)

        checkpoint_path = save_dir / f"policy_epoch_{epoch+1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": policy_cfg,
            },
            checkpoint_path,
        )

    if cfg.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    from absl import app

    app.run(main)
