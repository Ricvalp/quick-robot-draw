#!/usr/bin/env python3
"""
Train a DiT-based diffusion policy on QuickDraw episodes using the existing dataset pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from diffusion_policy import DiTDiffusionPolicy, DiTDiffusionPolicyConfig
from diffusion_policy.sampling import make_start_token, sample_quickdraw_tokens, tokens_to_figure
from dataset.loader import QuickDrawEpisodes
from dataset.diffusion import DiffusionCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DiTDiffusionPolicy on QuickDraw.")
    parser.add_argument("--data-root", type=str, default="data/", help="Processed dataset root.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--backend", type=str, default="lmdb", help="Storage backend.")
    parser.add_argument("--K", type=int, default=5, help="Number of prompts per episode.")
    parser.add_argument("--horizon", type=int, default=64, help="Prediction horizon.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Transformer hidden size.")
    parser.add_argument("--num-layers", type=int, default=8, help="Transformer depth.")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads per layer.")
    parser.add_argument("--mlp-dim", type=int, default=2048, help="Transformer MLP width.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Residual dropout.")
    parser.add_argument("--attention-dropout", type=float, default=0.0, help="Attention dropout.")
    parser.add_argument("--beta-start", type=float, default=0.0001, help="DDPM beta start.")
    parser.add_argument("--beta-end", type=float, default=0.02, help="DDPM beta end.")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="DDPM schedule.")
    parser.add_argument("--num-train-timesteps", type=int, default=1000, help="DDPM training timesteps.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Sampling steps (for inference).")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to store checkpoints.")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Episode max sequence length.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional Weights & Biases project name.")
    parser.add_argument("--wandb-run", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity/team.")
    parser.add_argument("--loss-log-every", type=int, default=200, help="Steps between logging batch loss to W&B.")
    parser.add_argument("--eval-samples", type=int, default=4, help="Number of sketches to sample per epoch for qualitative monitoring.")
    parser.add_argument("--eval-tokens", type=int, default=256, help="Tokens to generate per qualitative sample.")
    parser.add_argument("--eval-interval", type=int, default=1, help="Epoch interval for qualitative sampling (set >1 to reduce frequency).")
    parser.add_argument("--eval-sample-seed", type=int, default=42, help="Base seed for qualitative sampling noise.")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = QuickDrawEpisodes(
        root=args.data_root,
        split=args.split,
        K=args.K,
        backend=args.backend,
        max_seq_len=args.max_seq_len,
        augment=False,
        seed=args.seed,
        coordinate_mode="absolute",
    )
    collator = DiffusionCollator(horizon=args.horizon, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    noise_scheduler_kwargs = {
        "num_train_timesteps": args.num_train_timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "beta_schedule": args.beta_schedule,
    }

    policy_cfg = DiTDiffusionPolicyConfig(
        context_length=0,
        horizon=args.horizon,
        point_feature_dim=3,  # 2 positions + 1 pen state
        action_dim=3,  # x, y, pen
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        num_inference_steps=args.num_inference_steps,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
    )
    policy = DiTDiffusionPolicy(policy_cfg).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            entity=args.wandb_entity,
            config={
                **vars(args),
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

    for epoch in range(args.epochs):
        policy.train()
        running_loss = 0.0
        total_batches = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
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
                args.wandb_project
                and args.loss_log_every > 0
                and global_step % args.loss_log_every == 0
            ):
                wandb.log({"train/batch_loss": metrics["mse"]}, step=global_step)

        if total_batches == 0:
            raise RuntimeError("No valid batches processed; consider reducing the horizon or batch size.")
        avg_loss = running_loss / total_batches
        print(f"Epoch {epoch+1}: avg loss {avg_loss:.6f}")
        if args.wandb_project:
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

        _log_qualitative_samples(policy, args, epoch, device)

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
