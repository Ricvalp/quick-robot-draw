from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import torch

from tqdm.auto import tqdm
import wandb

from rlbench.datasets.imitation_learning.cached_dataset import get_temporal_cached_dataloader
from rlbench.datasets.in_context_imitation_learning.cached_dataset import get_in_context_cached_dataloader

from src.policies import (
    DiTDiffusionPolicy,
    DiTDiffusionPolicyConfig,
    PlatonicDiffusionPolicy,
    PlatonicDiffusionPolicyConfig,
)
from src.models.platonic_transformer.groups import PLATONIC_GROUPS
from src.utils import (
    point_cloud_with_actions,
    NormalizationStats,
    NormalizationTransform,
    WarmupCosineLRSchedulerConfig,
    build_warmup_cosine_scheduler,
)


def parse_args() -> argparse.Namespace:
    # Configure the CLI so experiments can adjust hyperparameters without touching code.
    parser = argparse.ArgumentParser(description="Train a minimal DiT diffusion policy.")
    parser.add_argument("--cache-path", type=str, default="/home/dknigge/project_dir/data/robotics/rlbench/imitation_learning/", help="RLBench imitation learning cache root dir.")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="List of RLBench tasks to train on. Uses all if unspecified.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory to store checkpoints.")
    parser.add_argument("--policy", type=str, default="dit", choices=["dit", "platonic"], help="Selects which policy backbone to train.")

    # Training loop hyperparameters that control optimisation.
    parser.add_argument("--epochs", type=int, default=48, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=48, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for AdamW.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker processes.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle RLBench samples each epoch.")
    parser.add_argument("--drop-last", action="store_true", help="Drop the final incomplete batch.")
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable DataLoader pinned-memory optimisation.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Override batches per epoch when the dataloader length is undefined.",
    )
    parser.add_argument(
        "--pointcloud-max-points",
        type=int,
        default=512,
        help="Optional cap on points per frame; keeps the first N points if provided.",
    )
    parser.add_argument(
        "--normalization-stats",
        type=str,
        default="artifacts/stats/normalization_stats.json",
        help="Path to dataset normalization stats JSON.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["none", "warmup_cosine"],
        default="none",
        help="Optional learning-rate schedule applied per optimisation step.",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="Linear warmup steps before cosine decay engages.",
    )
    parser.add_argument(
        "--lr-min-factor",
        type=float,
        default=0.1,
        help="Final cosine learning rate expressed as a fraction of the base LR.",
    )
    
    # Model architecture knobs exposed for quick sweeps.
    parser.add_argument("--hidden-dim", type=int, default=512, help="Transformer hidden size.")
    parser.add_argument("--num-layers", type=int, default=8, help="Transformer depth.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--mlp-dim", type=int, default=2048, help="Transformer MLP hidden size.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Token dropout.")
    parser.add_argument("--attention-dropout", type=float, default=0.0, help="Attention dropout.")

    # Diffusion noise schedule settings passed straight into the DDPM scheduler.
    parser.add_argument("--num-train-timesteps", type=int, default=1000, help="Noise scheduler train steps.")
    parser.add_argument("--beta-start", type=float, default=0.0001, help="Noise schedule beta start.")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Noise schedule beta end.")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="Noise schedule type.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Sampling steps (for future use).")

    # Platonic transformer specific knobs.
    parser.add_argument("--platonic-solid", type=str, default="icosahedron", help="Platonic group governing rotational equivariance.")
    parser.add_argument("--platonic-hidden-dim", type=int, default=480, help="Hidden size for the platonic transformer (must be divisible by group order).")
    parser.add_argument("--platonic-num-layers", type=int, default=8, help="Number of transformer blocks in the platonic backbone.")
    parser.add_argument("--platonic-num-heads", type=int, default=60, help="Attention heads for the platonic transformer (must be divisible by group order).")
    parser.add_argument("--platonic-ffn-dim-factor", type=int, default=4, help="Expansion factor for platonic transformer feed-forward layers.")
    parser.add_argument("--platonic-dropout", type=float, default=0.0, help="Dropout probability inside the platonic transformer.")
    parser.add_argument("--platonic-drop-path", type=float, default=0.0, help="Stochastic depth rate for the platonic transformer.")
    parser.add_argument("--platonic-mean-aggregation", action="store_true", help="Average tokens instead of sum inside attention kernels.")
    parser.add_argument("--platonic-softmax-attention", action="store_true", help="Enable softmax attention instead of linear kernel attention.")
    parser.add_argument("--platonic-rope-sigma", type=float, default=1.0, help="Frequency scale for platonic rotary embeddings.")
    parser.add_argument("--platonic-fixed-freqs", action="store_true", help="Freeze platonic rotary frequencies instead of learning them.")

    # Miscellaneous training conveniences.
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs.")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping threshold (0 disables).")
    parser.add_argument("--validate-every", type=int, default=1, help="Run denoising validation every N epochs (0 disables).")
    parser.add_argument("--wandb-project", type=str, default='ppl', help="Weights & Biases project for logging.")
    parser.add_argument("--wandb-entity", type=str, default='davidmknigge', help="Weights & Biases entity (optional).")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Custom name for the Weights & Biases run.")
    return parser.parse_args()


def run_validation(
    policy,
    dataloader,
    device,
    run,
    epoch: int,
    normalizer: NormalizationTransform | None,
    max_points: int | None,
) -> None:
    if run is None:
        return
    policy.eval()
    with torch.no_grad():
        raw_batch = next(iter(dataloader))
        batch = normalizer.normalize_batch(raw_batch) if normalizer else raw_batch
        observation = batch["observation"]["point_cloud_sequence"]
        proprio = batch["observation"]["proprio_sequence"].to(device)
        pc_points = observation["points"].to(device)
        pc_colors = observation["colors"].to(device)
        if max_points is not None:
            pc_points = pc_points[..., :max_points, :]
            pc_colors = pc_colors[..., :max_points, :]
        gt_actions = batch["action"].to(device)
        point_cloud_tokens = {
            "positions": pc_points,
            "colors": pc_colors,
        }
        pred_actions = policy.sample_actions(proprio, point_cloud_tokens)
    if normalizer:
        cloud_normalized = batch["observation"]["point_cloud_sequence"]
        cloud_denorm = normalizer.denormalize_point_cloud_sequence(cloud_normalized)
        cloud_source = {}
        for key, value in cloud_denorm.items():
            if value is None:
                continue
            cloud_source[key] = value[0].detach().cpu()
        gt_vis = normalizer.denormalize_action_positions(gt_actions.detach().cpu())[0]
        pred_vis = normalizer.denormalize_action_positions(pred_actions.detach().cpu())[0]
    else:
        raw_cloud = raw_batch["observation"]["point_cloud_sequence"]
        cloud_source = {}
        for key, value in raw_cloud.items():
            if value is None:
                continue
            cloud_source[key] = value[0].detach().cpu()
        gt_vis = gt_actions.cpu()[0]
        pred_vis = pred_actions.cpu()[0]
    cloud = point_cloud_with_actions(
        cloud_source,
        pred_vis,
        gt_vis,
    )
    run.log({"validation/point_cloud": cloud, "validation/epoch": epoch})


def main() -> None:
    args = parse_args()
    # Decide whether to leverage GPU acceleration; fall back to CPU when unavailable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve rlbench dataloader from the cache root supplied on the CLI.
    cache_root = Path(args.cache_path).expanduser().resolve()
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache path not found: {cache_root}")
    
    # List all dirs under cache root as possible tasks, and list which are selected.
    available_tasks = [d.name for d in cache_root.iterdir() if d.is_dir()]
    if args.tasks is None:
        args.tasks = available_tasks
    else:
        for task in args.tasks:
            if task not in available_tasks:
                raise ValueError(f"Requested task '{task}' not found in cache at {cache_root}")
    print(f"Using tasks: {args.tasks} from available: {available_tasks}")

    dataloader = get_temporal_cached_dataloader(
        cache_root,
        tasks=args.tasks,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )

    # Peek at a reference batch so model dimensions align with cached dataset structure.
    sample = next(iter(dataloader))
    observations = sample["observation"]["proprio_sequence"]  # (B, N_time, N_proprio)
    actions = sample["action"]  # (B, N_action, N_action_dims)
    context_length = int(observations.shape[1])
    horizon = int(actions.shape[1])
    obs_dim = int(observations.shape[2])
    action_dim = int(actions.shape[2])
    pc_dict = sample["observation"]["point_cloud_sequence"]
    point_feature_dim = pc_dict["points"].shape[-1] + pc_dict["colors"].shape[-1]

    scheduler_kwargs = {
        "num_train_timesteps": args.num_train_timesteps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "beta_schedule": args.beta_schedule,
        "prediction_type": "epsilon",
    }

    if args.policy == "dit":
        # Construct the DiT-backed policy using inferred shapes plus CLI overrides.
        policy_cfg = DiTDiffusionPolicyConfig(
            context_length=context_length,
            horizon=horizon,
            proprio_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            noise_scheduler_kwargs=scheduler_kwargs,
            num_inference_steps=args.num_inference_steps,
            point_feature_dim=point_feature_dim,
        )
        policy = DiTDiffusionPolicy(policy_cfg).to(device)
    else:
        solid = args.platonic_solid.lower()
        if solid not in PLATONIC_GROUPS:
            raise ValueError(f"Unknown platonic solid '{solid}'. Available groups: {list(PLATONIC_GROUPS.keys())}")
        group = PLATONIC_GROUPS[solid]
        if args.platonic_hidden_dim % group.G != 0:
            raise ValueError(f"Platonic hidden_dim ({args.platonic_hidden_dim}) must be divisible by group order ({group.G}).")
        if args.platonic_num_heads % group.G != 0:
            raise ValueError(f"Platonic num_heads ({args.platonic_num_heads}) must be divisible by group order ({group.G}).")

        # Platonic policy acts on pose tensors: obs/actions are 8D (pos + quat + gripper).
        policy_cfg = PlatonicDiffusionPolicyConfig(
            context_length=context_length,
            horizon=horizon,
            hidden_dim=args.platonic_hidden_dim,
            num_layers=args.platonic_num_layers,
            num_heads=args.platonic_num_heads,
            solid_name=solid,
            ffn_dim_factor=args.platonic_ffn_dim_factor,
            dropout=args.platonic_dropout,
            drop_path_rate=args.platonic_drop_path,
            mean_aggregation=args.platonic_mean_aggregation,
            use_softmax_attention=args.platonic_softmax_attention,
            rope_sigma=args.platonic_rope_sigma,
            learned_freqs=not args.platonic_fixed_freqs,
            noise_scheduler_kwargs=scheduler_kwargs,
            num_inference_steps=args.num_inference_steps,
            scalar_channels=4,
        )
        policy = PlatonicDiffusionPolicy(policy_cfg).to(device)

    # AdamW optimiser keeps weight decay decoupled from gradient magnitude.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Ensure a place exists for checkpoints, even on the first run.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load normalization stats and prepare normalizer if a path is provided.
    if not Path(args.normalization_stats).exists():
        raise FileNotFoundError(f"Normalization stats file not found: {args.normalization_stats}, run compute_stats.py first.")
    stats = NormalizationStats.from_json(args.normalization_stats)
    normalizer = NormalizationTransform(stats)

    # Create a wandb run to log training metrics and model config.
    wandb_run = None
    if args.wandb_project:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={**asdict(policy_cfg), **vars(args)},
        )

    # Print architecture, number of parameters, and training device.
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("Policy architecture:\n")
    print(policy)
    print(
        f"\nParameters | total: {total_params:,} | trainable: {trainable_params:,} | device: {device.type}"
    )

    try:
        batches_per_epoch = len(dataloader)
    except TypeError:
        batches_per_epoch = None

    steps_per_epoch = batches_per_epoch
    if steps_per_epoch is None and args.steps_per_epoch is not None:
        steps_per_epoch = args.steps_per_epoch
    if args.lr_scheduler != "none" and steps_per_epoch is None:
        raise ValueError(
            "LR scheduler requested but steps per epoch is unknown. "
            "Pass --steps-per-epoch when using iterable-style dataloaders."
        )

    scheduler = None
    if args.lr_scheduler == "warmup_cosine":
        total_steps = int(steps_per_epoch) * args.epochs
        scheduler_cfg = WarmupCosineLRSchedulerConfig(
            total_steps=total_steps,
            warmup_steps=args.lr_warmup_steps,
            min_lr_ratio=args.lr_min_factor,
        )
        scheduler = build_warmup_cosine_scheduler(optimizer, scheduler_cfg)

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        policy.train()
        running_loss = 0.0
        batches_processed = 0

        progress = tqdm(
            dataloader,
            total=batches_per_epoch,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
        )

        for batch in progress:
            if normalizer is not None:
                batch = normalizer.normalize_batch(batch)

            # Unpack and transfer mini-batch to the training device.
            proprio = batch["observation"]["proprio_sequence"].to(device)  # (B, N_time, 8)
            pc_points = batch["observation"]["point_cloud_sequence"]["points"].to(device)  # (B, N_time, N_points, 3)
            pc_colors = batch["observation"]["point_cloud_sequence"]["colors"].to(device)  # (B, N_time, N_points, 3)
            max_points = args.pointcloud_max_points
            if max_points is not None:
                pc_points = pc_points[..., :max_points, :]  # (B, N_time, max_points, 3)
                pc_colors = pc_colors[..., :max_points, :]  # (B, N_time, max_points, 3)
            actions = batch["action"].to(device)  # (B, N_action, 8)

            # Reconstruct the batch.
            policy_batch = {
                "proprio": proprio,
                "point_cloud": {
                    "positions": pc_points,
                    "colors": pc_colors,
                },
                "actions": actions,
            }  # Matches `PlatonicDiffusionPolicy.compute_loss` contract.

            # Standard diffusion training: zero gradients, compute loss, backprop, and step.
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy.compute_loss(policy_batch)  # loss: scalar
            loss.backward()

            if args.grad_clip > 0:
                # Optional gradient clipping for stability with large models or learning rates.
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                wandb_run and wandb_run.log({"train/lr": current_lr}, commit=False)
            batches_processed += 1
            running_loss += loss.item()
            avg_loss = running_loss / batches_processed
            progress.set_postfix(loss=f"{avg_loss:.6f}")
            
            # Log batch loss for more granular training curves.
            wandb_run and wandb_run.log({"train/batch_loss": loss.item()})

        progress.close()
        avg_loss = running_loss / max(1, batches_processed)
        # Report scalar loss so training progress is visible without extra tooling.
        tqdm.write(f"epoch {epoch:03d} | loss {avg_loss:.6f}")
        if wandb_run is not None:
            wandb_run.log({"train/epoch_loss": avg_loss, "epoch": epoch})

        if args.validate_every > 0 and epoch % args.validate_every == 0:
            run_validation(policy, dataloader, device, wandb_run, epoch, normalizer, args.pointcloud_max_points)

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = output_dir / f"checkpoints/policy_epoch{epoch:03d}.pt"
            # Persist model weights, optimiser state, and config for reproducibility.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": policy.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(policy_cfg),
                },
                checkpoint_path,
            )
            print(f"saved checkpoint to {checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
