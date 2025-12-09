#!/usr/bin/env python3
"""
Train a SketchRNN imitation policy on QuickDraw episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch
from ml_collections import config_flags
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb
from dataset.loader import QuickDrawEpisodes
from dataset.lstm import ILRNNCollator
from lstm import SketchRNN, SketchRNNConfig
from lstm.utils import strokes_to_tokens, trim_strokes_to_eos

_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="lstm/configs/imitation_learning.py"
)


def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value
    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_kl_weight(step: int, cfg: argparse.Namespace) -> float:
    if cfg.kl.anneal_steps <= 0:
        return cfg.kl.end
    progress = min(step / max(1, cfg.kl.anneal_steps), 1.0)
    return cfg.kl.start + (cfg.kl.end - cfg.kl.start) * progress


def _log_qualitative_samples(
    policy: SketchRNN,
    context: torch.Tensor,
    context_lengths: torch.Tensor,
    cfg: dict,
    step: int,
    device: torch.device,
) -> None:
    """Sample sketches and push them to WandB for quick visual inspection."""

    if not cfg.wandb.use or cfg.eval.samples <= 0:
        return

    prev_mode = policy.training
    policy.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval.seed + step)

    samples = policy.sample(
        num_steps=cfg.eval.steps,
        contexts=context.to(device=device),
        context_lengths=context_lengths.to(device=device),
        num_samples=cfg.eval.samples,
        unconditional=False,
        deterministic=True,
        generator=generator,
    )

    trimmed = trim_strokes_to_eos(samples)

    def _plot_tokens(
        ax, tokens: torch.Tensor, title: str, coordinate_mode: str
    ) -> None:
        """Render `(N, 3)` tokens on the provided axis."""
        array = tokens.detach().cpu().numpy()
        coords = (
            array[:, :2].cumsum(axis=0) if coordinate_mode == "delta" else array[:, :2]
        )
        pen_state = array[:, 2]
        for idx in range(1, coords.shape[0]):
            start = coords[idx - 1]
            end = coords[idx]
            active = pen_state[idx] >= 0.5
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

    images = []
    for idx, seq in enumerate(trimmed):
        ctx_len = int(context_lengths[idx].item())
        ctx_tokens = context[idx, :ctx_len, :3]
        sample_tokens = strokes_to_tokens(seq)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
        _plot_tokens(axes[0], ctx_tokens, "Context", cfg.data.coordinate_mode)
        _plot_tokens(axes[1], sample_tokens, "Sample", "delta")

        images.append(wandb.Image(fig, caption=f"step {step + 1} sample {idx}"))
        plt.close(fig)
    if images:
        wandb.log({"samples/sketches": images}, step=step + 1)

    if prev_mode:
        policy.train()


def main(_) -> None:

    cfg = load_cfgs(_CONFIG_FILE)

    set_seed(cfg.run.seed)
    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    dataset = QuickDrawEpisodes(
        root=cfg.data.root,
        split=cfg.data.split,
        K=cfg.data.K,
        backend=cfg.data.backend,
        max_query_len=cfg.data.max_query_len,
        augment=False,
        seed=cfg.run.seed,
        coordinate_mode=cfg.data.coordinate_mode,
    )

    collator = ILRNNCollator(
        max_query_len=cfg.data.max_query_len,
        teacher_forcing_with_eos=cfg.model.teacher_forcing_with_eos,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
        prefetch_factor=4,
        persistent_workers=True,
    )

    eval_dataset = QuickDrawEpisodes(
        root=cfg.data.root,
        split="val",  # cfg.data.split if cfg.eval.eval_on_train else "val",
        K=cfg.data.K,
        backend=cfg.data.backend,
        max_query_len=cfg.data.max_query_len,
        augment=False,
        seed=cfg.run.seed + 1,
        coordinate_mode=cfg.data.coordinate_mode,
    )
    eval_collator = ILRNNCollator(
        max_query_len=cfg.data.max_query_len,
        teacher_forcing_with_eos=cfg.model.teacher_forcing_with_eos,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.eval.samples,
        shuffle=True,
        collate_fn=eval_collator,
    )
    eval_iterator = iter(eval_dataloader)

    sketch_config = SketchRNNConfig(
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        latent_dim=cfg.model.latent_dim,
        encoder_hidden=cfg.model.encoder_hidden,
        encoder_num_layers=cfg.model.encoder_num_layers,
        decoder_hidden=cfg.model.decoder_hidden,
        decoder_num_layers=cfg.model.decoder_num_layers,
        num_mixtures=cfg.model.num_mixtures,
        dropout=cfg.model.dropout,
    )
    model = SketchRNN(sketch_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    # scheduler = CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=1e-6)

    save_dir = Path(cfg.checkpoint.dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total_params:,}")

    if cfg.profiling.use:
        import os

        from torch.profiler import ProfilerActivity, profile

        os.makedirs(cfg.profiling.trace_dir, exist_ok=True)

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        with profile(activities=activities) as prof:

            for step, batch in enumerate(dataloader):
                strokes = batch["strokes"].to(device, non_blocking=True)
                lengths = batch["lengths"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                kl_weight = compute_kl_weight(step, cfg)
                loss, metrics = model.compute_loss(
                    strokes, lengths, kl_weight=kl_weight
                )
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.training.grad_clip
                )
                optimizer.step()

                if step > 3:
                    break
        prof.export_chrome_trace(cfg.profiling.trace_dir + "lstm_trace.json")
        print(f"Saved profiling trace to {cfg.profiling.trace_dir}lstm_trace.json")
        return

    if cfg.wandb.use:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config={**cfg.to_dict(), "model": sketch_config.__dict__},
        )
        wandb.log({"model/parameters": total_params}, step=0)

    global_step = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        running = 0.0
        batches = 0
        progress = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs}", leave=False
        )

        for batch in progress:

            input_queries = batch["input_queries"].to(device)
            output_queries = batch["output_queries"].to(device)
            input_queries_lengths = batch["input_queries_lengths"].to(device)
            output_queries_lengths = batch["output_queries_lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            kl_weight = compute_kl_weight(global_step, cfg)
            loss, metrics = model.compute_loss(
                queries=output_queries,
                queries_lengths=output_queries_lengths,
                contexts=input_queries,
                contexts_lengths=input_queries_lengths,
                kl_weight=kl_weight,
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=cfg.training.grad_clip
            )
            optimizer.step()

            running += float(loss.detach().cpu())
            batches += 1
            progress.set_postfix({"loss": metrics["loss"], "kl": metrics["kl"]})

            global_step += 1
            if (
                cfg.wandb.use
                and cfg.logging.loss_log_every > 0
                and global_step % cfg.logging.loss_log_every == 0
            ):
                wandb.log(
                    {
                        "train/batch_loss": metrics["loss"],
                        "train/recon": metrics["recon"],
                        "train/kl": metrics["kl"],
                        "train/kl_weight": metrics["kl_weight"],
                    },
                    step=global_step,
                )

            break

        if batches == 0:
            raise RuntimeError(
                "No valid batches processed; consider reducing max_seq_len or batch size."
            )

        avg_loss = running / batches
        print(f"Epoch {epoch + 1}: avg loss {avg_loss:.6f}")

        if cfg.wandb.use:
            wandb.log({"train/loss": avg_loss, "epoch": epoch + 1})

        if (
            cfg.checkpoint.save_interval is not None
            and (epoch + 1) % max(1, cfg.checkpoint.save_interval) == 0
        ):
            checkpoint_path = save_dir / f"sketchrnn_epoch_{epoch + 1:03d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": sketch_config,
                },
                checkpoint_path,
            )

        try:
            eval_batch = next(eval_iterator)
            eval_query = eval_batch["input_queries"].to(device)
            eval_queries_lengths = eval_batch["input_queries_lengths"].to(device)
        except StopIteration:
            eval_iterator = iter(eval_dataloader)
            eval_batch = next(eval_iterator)
            eval_query = eval_batch["input_queries"].to(device)
            eval_queries_lengths = eval_batch["input_queries_lengths"].to(device)

        print("evaluating qualitative samples...")
        _log_qualitative_samples(
            policy=model,
            context=eval_query,
            context_lengths=eval_queries_lengths,
            cfg=cfg,
            step=global_step,
            device=device,
        )

    if cfg.wandb.project:
        wandb.finish()


if __name__ == "__main__":
    import absl.app as app

    app.run(main)
