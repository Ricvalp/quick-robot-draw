from pathlib import Path

import torch

from rlbench.datasets.imitation_learning.cached_dataset import get_temporal_cached_dataloader

from src.policies.dit_policy import DiTDiffusionPolicy, DiTDiffusionPolicyConfig


DATA_ROOT = Path("/home/dknigge/project_dir/data/robotics/rlbench/imitation_learning")


def _fetch_batch(batch_size=2):
    dataloader = get_temporal_cached_dataloader(
        DATA_ROOT,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return next(iter(dataloader))


def _build_policy_from_batch(batch):
    proprio = batch["observation"]["proprio_sequence"]
    point_cloud = batch["observation"]["point_cloud_sequence"]
    actions = batch["action"]

    context_length = proprio.shape[1]
    horizon = actions.shape[1]
    proprio_dim = proprio.shape[2]
    action_dim = actions.shape[2]
    point_feature_dim = point_cloud["points"].shape[-1] + point_cloud["colors"].shape[-1]

    cfg = DiTDiffusionPolicyConfig(
        context_length=context_length,
        horizon=horizon,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        hidden_dim=128,
        num_layers=1,
        num_heads=8,
        mlp_dim=256,
        dropout=0.0,
        attention_dropout=0.0,
        num_inference_steps=5,
        point_feature_dim=point_feature_dim,
        noise_scheduler_kwargs={
            "num_train_timesteps": 200,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
    )
    policy = DiTDiffusionPolicy(cfg)
    policy.eval()
    return policy, cfg


def test_dit_policy_compute_loss_shapes():
    batch = _fetch_batch(batch_size=2)
    policy, cfg = _build_policy_from_batch(batch)

    proprio = batch["observation"]["proprio_sequence"]
    point_cloud = batch["observation"]["point_cloud_sequence"]
    actions = batch["action"]

    policy_batch = {
        "proprio": proprio,
        "observation": {
            "positions": point_cloud["points"],
            "colors": point_cloud["colors"],
        },
        "actions": actions,
    }

    loss, metrics = policy.compute_loss(policy_batch)

    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    assert "mse" in metrics


def test_dit_policy_sample_actions_shapes():
    batch = _fetch_batch(batch_size=1)
    policy, cfg = _build_policy_from_batch(batch)

    proprio = batch["observation"]["proprio_sequence"]
    point_cloud = batch["observation"]["point_cloud_sequence"]

    with torch.no_grad():
        sample = policy.sample_actions(
            proprio,
            {
                "positions": point_cloud["points"],
                "colors": point_cloud["colors"],
            },
            generator=torch.Generator().manual_seed(0),
        )

    assert sample.shape == (proprio.shape[0], cfg.horizon, cfg.action_dim)
