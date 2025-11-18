#!/usr/bin/env python3
"""
Train a SketchRNN imitation policy on QuickDraw episodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from ml_collections import config_flags

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb


from dataset.loader import QuickDrawEpisodes
from dataset.lstm import LSTMCollator
from diffusion_policy.sampling import tokens_to_figure
from lstm import SketchRNN, SketchRNNConfig, strokes_to_tokens, trim_strokes_to_eos


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Train SketchRNN on QuickDraw.")
#     parser.add_argument("--data-root", type=str, default="data/", help="Processed dataset root.")
#     parser.add_argument("--split", type=str, default="train", help="Dataset split.")
#     parser.add_argument("--backend", type=str, default="lmdb", help="Storage backend.")
#     parser.add_argument("--K", type=int, default=5, help="Number of prompt sketches per episode.")
#     parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
#     parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
#     parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
#     parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
#     parser.add_argument("--num-workers", type=int, default=16, help="DataLoader worker count.")
#     parser.add_argument("--device", type=str, default="cuda", help="Training device.")
#     parser.add_argument("--seed", type=int, default=0, help="Random seed.")
#     parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length (incl. EOS).")
#     parser.add_argument("--coordinate-mode", type=str, default="delta", help="Token coordinate mode.")
#     parser.add_argument("--encoder-hidden", type=int, default=256, help="BiLSTM hidden size per direction.")
#     parser.add_argument("--encoder-layers", type=int, default=1, help="Number of encoder layers.")
#     parser.add_argument("--decoder-hidden", type=int, default=512, help="Decoder LSTM hidden size.")
#     parser.add_argument("--decoder-layers", type=int, default=1, help="Number of decoder layers.")
#     parser.add_argument("--latent-dim", type=int, default=128, help="Latent vector dimension.")
#     parser.add_argument("--num-mixtures", type=int, default=20, help="Number of GMM components.")
#     parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for encoder/decoder stacks.")
#     parser.add_argument("--kl-start", type=float, default=0.1, help="Initial KL weight.")
#     parser.add_argument("--kl-end", type=float, default=1.0, help="Final KL weight.")
#     parser.add_argument("--kl-anneal-steps", type=int, default=20000, help="Linear KL anneal steps.")
#     parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
#     parser.add_argument("--checkpoint-dir", type=str, default="lstm/checkpoints", help="Directory for checkpoints.")
#     parser.add_argument("--save-interval", type=int, default=10, help="Epoch interval between saving checkpoints.")
#     parser.add_argument("--wandb-project", type=str, default=None, help="Optional Weights & Biases project.")
#     parser.add_argument("--wandb-run", type=str, default=None, help="Weights & Biases run name.")
#     parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity/team.")
#     parser.add_argument("--loss-log-every", type=int, default=200, help="Steps between batch loss logs.")
#     parser.add_argument("--eval-samples", type=int, default=4, help="Sketches sampled per eval.")
#     parser.add_argument("--eval-steps", type=int, default=200, help="Decoding steps per qualitative sample.")
#     parser.add_argument("--eval-interval", type=int, default=1, help="Epoch interval between eval sampling.")
#     parser.add_argument("--eval-temperature", type=float, default=0.65, help="Sampling temperature.")
#     parser.add_argument("--eval-seed", type=int, default=42, help="Seed used for qualitative sampling.")
#     parser.add_argument("--greedy-eval", action="store_true", help="Use greedy sampling for eval sketches.")
#     parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiling and save trace.")
#     parser.add_argument("--trace-dir", type=str, default="profiling/lstm/", help="Directory to save profiling trace.")
#     return parser.parse_args()


_CONFIG_FILE = config_flags.DEFINE_config_file("config", default="lstm/configs/imitation_learning.py")

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
    if cfg.kl_anneal_steps <= 0:
        return cfg.kl_end
    progress = min(step / max(1, cfg.kl_anneal_steps), 1.0)
    return cfg.kl_start + (cfg.kl_end - cfg.kl_start) * progress


def _log_eval_samples(model: SketchRNN, cfg: argparse.Namespace, step: int, device: torch.device) -> None:
    if cfg.wandb_project is None or cfg.eval_samples <= 0:
        return

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.eval_seed + step)
    samples = model.sample(
        cfg.eval_steps,
        num_samples=cfg.eval_samples,
        temperature=cfg.eval_temperature,
        greedy=cfg.greedy_eval,
        generator=generator,
    )
    trimmed = trim_strokes_to_eos(samples)

    import matplotlib.pyplot as plt

    images = []
    for idx, seq in enumerate(trimmed):
        tokens = strokes_to_tokens(seq)
        fig = tokens_to_figure(tokens, coordinate_mode="delta")
        images.append(wandb.Image(fig, caption=f"step {step + 1} sample {idx}"))
        plt.close(fig)
    if images:
        wandb.log({"samples/sketches": images}, step=step + 1)


def main() -> None:
    
    cfg = load_cfgs(_CONFIG_FILE)
    
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
        coordinate_mode=cfg.coordinate_mode,
    )
    collator = LSTMCollator(max_seq_len=cfg.max_seq_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
        prefetch_factor=4,
        persistent_workers=True,
    )

    sketch_config = SketchRNNConfig(
        input_dim=5,
        latent_dim=cfg.latent_dim,
        encoder_hidden_size=cfg.encoder_hidden,
        encoder_num_layers=cfg.encoder_layers,
        decoder_hidden_size=cfg.decoder_hidden,
        decoder_num_layers=cfg.decoder_layers,
        num_mixtures=cfg.num_mixtures,
        dropout=cfg.dropout,
    )
    model = SketchRNN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    save_dir = Path(cfg.checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total_params:,}")
    
    if cfg.profile:
        from torch.profiler import profile, ProfilerActivity
        import os
        
        os.makedirs(cfg.trace_dir, exist_ok=True)
        
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        with profile(activities=activities) as prof:

            for step, batch in enumerate(dataloader):
                strokes = batch["strokes"].to(device, non_blocking=True)
                lengths = batch["lengths"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                kl_weight = compute_kl_weight(step, cfg)
                loss, metrics = model.compute_loss(strokes, lengths, kl_weight=kl_weight)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()
                
                if step > 3:
                    break
        prof.export_chrome_trace(cfg.trace_dir + f"lstm_trace.json")
        print(f"Saved profiling trace to {cfg.trace_dir}lstm_trace.json")
        return

    if cfg.wandb_project:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run,
            entity=cfg.wandb_entity,
            config={**vars(cfg), "model": sketch_config.__dict__},
        )
        wandb.log({"model/parameters": total_params}, step=0)

    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        batches = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)

        for batch in progress:
            strokes = batch["strokes"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)
            kl_weight = compute_kl_weight(global_step, cfg)
            loss, metrics = model.compute_loss(strokes, lengths, kl_weight=kl_weight)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            running += float(loss.detach().cpu())
            batches += 1
            progress.set_postfix({"loss": metrics["loss"], "kl": metrics["kl"]})

            global_step += 1
            if cfg.wandb_project and cfg.loss_log_every > 0 and global_step % cfg.loss_log_every == 0:
                wandb.log(
                    {
                        "train/batch_loss": metrics["loss"],
                        "train/recon": metrics["recon"],
                        "train/kl": metrics["kl"],
                        "train/kl_weight": metrics["kl_weight"],
                    },
                    step=global_step,
                )

        if batches == 0:
            raise RuntimeError("No valid batches processed; consider reducing max_seq_len or batch size.")

        avg_loss = running / batches
        print(f"Epoch {epoch + 1}: avg loss {avg_loss:.6f}")

        if cfg.wandb_project:
            wandb.log({"train/loss": avg_loss, "epoch": epoch + 1})

        if cfg.save_interval is not None and (epoch + 1) % max(1, cfg.save_interval) == 0:
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

        if cfg.eval_interval is not None and global_step % max(1, cfg.eval_interval) == 0:
            _log_eval_samples(model, cfg, global_step, device)

    if cfg.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    import absl.app as app
    
    app.run(main)
