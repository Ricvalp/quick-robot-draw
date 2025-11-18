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
from diffusion_policy.sampling import make_start_token, sample_quickdraw_tokens, tokens_to_figure
from dataset.loader import QuickDrawEpisodes
from dataset.diffusion import DiffusionCollator


def load_config(
    _CONFIG_FILE: str
    ) -> ConfigDict:
    
    cfg = _CONFIG_FILE.value
    
    return cfg

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy_inputs(
    batch: Dict[str, torch.Tensor],
    horizon: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    tokens = batch["tokens"]  # (B, T, 7)
    context_mask = batch["context_mask"]  # (B, T)
    target_mask = batch["target_mask"]  # (B, T)

    proprio_list = []
    positions_list = []
    colours_list = []
    actions_list = []

    for i in range(tokens.shape[0]):
        ctx_tokens = tokens[i][context_mask[i]]
        tgt_tokens = tokens[i][target_mask[i]]

        if ctx_tokens.numel() == 0 or tgt_tokens.shape[0] != horizon:
            continue

        ctx_tokens = ctx_tokens.to(device=device, dtype=torch.float32)
        tgt_tokens = tgt_tokens.to(device=device, dtype=torch.float32)

        ctx_len = ctx_tokens.shape[0]
        absolute = ctx_tokens[:, :2]
        deltas = torch.zeros_like(absolute)
        deltas[0] = absolute[0]
        if ctx_len > 1:
            deltas[1:] = absolute[1:] - absolute[:-1]

        proprio = deltas
        positions = torch.cat([absolute, torch.zeros(ctx_len, 1, device=device)], dim=1).view(ctx_len, 1, 3)
        colours = torch.stack(
            [
                ctx_tokens[:, 2],
                ctx_tokens[:, 4],
                ctx_tokens[:, 6],
            ],
            dim=-1,
        ).view(ctx_len, 1, 3)
        target_absolute = tgt_tokens[:, :2]
        target_deltas = torch.zeros_like(target_absolute)
        if ctx_len > 0:
            target_deltas[0] = target_absolute[0] - absolute[-1]
        else:
            target_deltas[0] = target_absolute[0]
        if horizon > 1:
            target_deltas[1:] = target_absolute[1:] - target_absolute[:-1]

        proprio_list.append(proprio)
        positions_list.append(positions)
        colours_list.append(colours)
        actions_list.append(torch.stack([target_deltas[:, 0], target_deltas[:, 1], tgt_tokens[:, 2]], dim=-1))

    if not proprio_list:
        raise ValueError("No valid samples in batch after preprocessing.")

    return {
        "proprio": torch.stack(proprio_list, dim=0),
        "observation": {
            "positions": torch.stack(positions_list, dim=0),
            "colors": torch.stack(colours_list, dim=0),
        },
        "actions": torch.stack(actions_list, dim=0),
    }


def _log_qualitative_samples(policy: DiTDiffusionPolicy, args: argparse.Namespace, epoch: int, device: torch.device) -> None:
    """Sample sketches and push them to WandB for quick visual inspection."""

    if args.wandb_project is None or args.eval_samples <= 0:
        return
    if (epoch + 1) % max(1, args.eval_interval) != 0:
        return

    prev_mode = policy.training
    policy.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(args.eval_sample_seed + epoch)

    start = make_start_token(args.eval_samples, policy.cfg.point_feature_dim, device)
    samples = sample_quickdraw_tokens(
        policy,
        args.eval_tokens,
        start_token=start,
        generator=generator,
    )

    import matplotlib.pyplot as plt

    images = []
    batch = samples.shape[0]
    for idx in range(batch):
        fig = tokens_to_figure(samples[idx], coordinate_mode="absolute")
        images.append(wandb.Image(fig, caption=f"epoch {epoch + 1} sample {idx}"))
        plt.close(fig)

    if images:
        wandb.log({"samples/sketches": images}, step=epoch + 1)

    if prev_mode:
        policy.train()


_CONFIG_FILE = config_flags.DEFINE_config_file("config", default="diffusion_policy/configs/in_context_imitation_learning.py")


def main() -> None:
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
        point_feature_dim=3,  # 2 positions + 1 pen state
        action_dim=3,  # x, y, pen
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
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    save_dir = Path(cfg.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb_project:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run,
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
        total_batches = 0
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

            running_loss += float(loss.detach().cpu())
            total_batches += 1
            progress.set_postfix({"mse": metrics["mse"]})

            if (
                cfg.wandb_project
                and cfg.loss_log_every > 0
                and global_step % cfg.loss_log_every == 0
            ):
                wandb.log({"train/batch_loss": metrics["mse"]}, step=global_step)

        if total_batches == 0:
            raise RuntimeError("No valid batches processed; consider reducing the horizon or batch size.")
        avg_loss = running_loss / total_batches
        print(f"Epoch {epoch+1}: avg loss {avg_loss:.6f}")
        if cfg.wandb_project:
            wandb.log({"train/mse": avg_loss, "epoch": epoch + 1}, step=epoch + 1)

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

        _log_qualitative_samples(policy, cfg, epoch, device)

    if cfg.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    from absl import app
    app.run(main)
