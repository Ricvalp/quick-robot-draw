#!/usr/bin/env python3
"""
Train a DiT-based diffusion policy on QuickDraw episodes using the existing dataset pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import torch
from ml_collections import ConfigDict, config_flags
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.episode_builder import EpisodeBuilderSimilar

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb
from dataset.diffusion import ContextQueryInContextDiffusionCollator
from dataset.loader import QuickDrawEpisodes
from diffusion import DiTEncDecDiffusionPolicy, DiTEncDecDiffusionPolicyConfig
from diffusion.sampling import sample_quickdraw_tokens


def load_config(_CONFIG_FILE: str) -> ConfigDict:

    cfg = _CONFIG_FILE.value

    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_qualitative_samples(
    policy: DiTEncDecDiffusionPolicy,
    context: dict,
    split: str,
    cfg: dict,
    step: int,
    device: torch.device,
) -> None:
    """Sample sketches and push them to WandB for visual inspection."""

    if (not cfg.wandb.use) or cfg.wandb.project is None or cfg.eval.samples <= 0:
        return

    prev_mode = policy.training
    policy.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed + step)

    samples = sample_quickdraw_tokens(
        policy=policy,
        max_tokens=cfg.data.max_query_len,
        demos=context,
        generator=generator,
    )

    def _plot_tokens(
        ax, tokens: torch.Tensor, title: str, coordinate_mode: str
    ) -> None:
        """Render `(N, 3)` tokens on the provided axis."""
        array = tokens.detach().cpu().numpy()
        coords = (
            array[:, :2].cumsum(axis=0) if coordinate_mode == "delta" else array[:, :2]
        )
        pen_state = array[:, 2]
        for token_idx in range(1, coords.shape[0]):
            start = coords[token_idx - 1]
            end = coords[token_idx]
            active = pen_state[token_idx] >= 0.5
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="black" if active else "tab:red",
                linewidth=1.5,
                linestyle="-" if active else "--",
            )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    def _split_context_prompts(ctx_tokens: torch.Tensor) -> list[torch.Tensor]:
        """Split the concatenated context tokens into individual prompt sketches."""
        sketches = []
        current = []
        for token in ctx_tokens:
            if token[5] > 0.5:  # stop token marks the end of the episode
                break
            if token[4] > 0.5:  # separator between sketches
                if current:
                    sketches.append(torch.stack(current))
                    current = []
                continue
            current.append(token[[0, 1, 2]])
        if current:
            sketches.append(torch.stack(current))
        return sketches[: cfg.data.K]

    images = []
    batch_size = len(samples)
    for idx in range(batch_size):
        ctx_tokens = context["context"][idx]
        ctx_mask = context["context_mask"][idx]
        valid_ctx = ctx_tokens[ctx_mask].detach().cpu()
        prompts = _split_context_prompts(valid_ctx)
        sample_tokens = samples[idx]

        total_plots = len(prompts) + 1
        cols = min(total_plots, 3)
        rows = (total_plots + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for prompt_idx, prompt_tokens in enumerate(prompts):
            _plot_tokens(
                axes[prompt_idx],
                prompt_tokens,
                f"Context {prompt_idx + 1}",
                cfg.data.coordinate_mode,
            )

        _plot_tokens(
            axes[len(prompts)],
            sample_tokens,
            "Sample",
            cfg.data.coordinate_mode,
        )

        for ax in axes[total_plots:]:
            ax.axis("off")

        fig.tight_layout()
        images.append(wandb.Image(fig, caption=f"step {step + 1} sample {idx}"))
        plt.close(fig)

    if images:
        wandb.log({f"samples/sketches/{split}": images}, step=step + 1)

    if prev_mode:
        policy.train()


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config",
    default="diffusion_policy/configs/encoder_decoder_in_context_imitation_learning.py",
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
        max_seq_len=cfg.data.max_seq_len,
        seed=cfg.run.seed,
        coordinate_mode=cfg.data.coordinate_mode,
        builder_cls=EpisodeBuilderSimilar,  # EpidodeBuilder
        index_dir=cfg.data.index_dir,
        ids_dir=cfg.data.ids_dir,
    )
    collator = ContextQueryInContextDiffusionCollator(
        horizon=cfg.model.horizon, seed=cfg.run.seed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
    )

    eval_collator = ContextQueryInContextDiffusionCollator(
        horizon=cfg.model.horizon, seed=cfg.run.seed
    )
    eval_dataloader = DataLoader(
        dataset,
        batch_size=cfg.eval.samples,
        shuffle=True,
        collate_fn=eval_collator,
    )

    noise_scheduler_kwargs = {
        "num_train_timesteps": cfg.model.num_train_timesteps,
        "beta_start": cfg.model.beta_start,
        "beta_end": cfg.model.beta_end,
        "beta_schedule": cfg.model.beta_schedule,
    }

    policy_cfg = DiTEncDecDiffusionPolicyConfig(
        horizon=cfg.model.horizon,
        point_feature_dim=cfg.model.input_dim,  # 2 positions + 1 pen state + 4 special tokens
        action_dim=cfg.model.output_dim,  # x, y, pen
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_dim=cfg.model.mlp_dim,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        num_inference_steps=cfg.eval.num_inference_steps,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
    )
    policy = DiTEncDecDiffusionPolicy(policy_cfg).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    save_dir = Path(cfg.checkpoint.dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.use and cfg.wandb.project:
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

    eval_iterator = iter(eval_dataloader)

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

            running_loss += float(loss.detach().cpu())
            global_step += 1
            step += 1
            progress.set_postfix({"mse": metrics["mse"]})

            if (
                cfg.wandb.use
                and cfg.logging.loss_log_every > 0
                and global_step % cfg.logging.loss_log_every == 0
            ):
                wandb.log({"train/batch_loss": metrics["mse"]}, step=global_step)

            if cfg.wandb.use and global_step % cfg.eval.eval_every == 0:
                try:
                    eval_batch = next(eval_iterator)
                except StopIteration:
                    eval_iterator = iter(eval_dataloader)
                    eval_batch = next(eval_iterator)

                _log_qualitative_samples(
                    policy=policy,
                    context=eval_batch,
                    cfg=cfg,
                    step=global_step,
                    device=device,
                    split="eval",
                )

                _log_qualitative_samples(
                    policy=policy,
                    context={
                        key: batch[key][: cfg.eval.samples] for key in batch.keys()
                    },
                    cfg=cfg,
                    step=global_step,
                    device=device,
                    split="train",
                )

        avg_loss = running_loss / step
        print(f"Epoch {epoch+1}: avg loss {avg_loss:.6f}")
        if cfg.wandb.use:
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

    if cfg.wandb.use:
        wandb.finish()


if __name__ == "__main__":
    from absl import app

    app.run(main)
