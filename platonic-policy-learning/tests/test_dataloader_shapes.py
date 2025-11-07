from pathlib import Path

import torch

from rlbench.datasets.imitation_learning.cached_dataset import (
    TemporalPointCloudCachedDataset,
    get_temporal_cached_dataloader,
)


DATA_ROOT = Path("/home/dknigge/project_dir/data/robotics/rlbench/imitation_learning")


def test_temporal_point_cloud_dataset_shapes():
    dataset = TemporalPointCloudCachedDataset(DATA_ROOT)
    sample = dataset[0]

    observation = sample["observation"]
    point_cloud = observation["point_cloud_sequence"]
    proprio = observation["proprio_sequence"]
    action = sample["action"]

    assert isinstance(point_cloud, dict)
    assert set(point_cloud.keys()) == {"points", "colors", "masks"}

    # Single sample tensors: (T, P, 3) for points/colors, (T, P) for masks.
    points = point_cloud["points"]
    colors = point_cloud["colors"]
    masks = point_cloud["masks"]

    assert points.ndim == 3 and points.shape[-1] == 3
    assert colors.shape == points.shape
    assert masks.shape == points.shape[:-1]

    # Proprio has shape (T, D) with D=8 for RLBench pose + gripper, action matches horizon.
    assert proprio.ndim == 2 and proprio.shape[-1] == 8
    assert action.ndim == 2 and action.shape[-1] == 8


def test_temporal_dataloader_collate_shapes():
    dataloader = get_temporal_cached_dataloader(
        DATA_ROOT,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(dataloader))

    observation = batch["observation"]
    point_cloud = observation["point_cloud_sequence"]
    proprio = observation["proprio_sequence"]
    action = batch["action"]

    assert isinstance(point_cloud, dict)

    points = point_cloud["points"]
    colors = point_cloud["colors"]
    masks = point_cloud["masks"]

    # Batched tensors should be (B, T, P, 3) and (B, T, P).
    assert points.ndim == 4 and points.shape[-1] == 3
    assert colors.shape == points.shape
    assert masks.shape == points.shape[:-1]

    batch_size = points.shape[0]
    context_length = points.shape[1]
    num_points = points.shape[2]

    assert proprio.shape == (batch_size, context_length, 8)
    assert action.ndim == 3 and action.shape[0] == batch_size and action.shape[-1] == 8
    assert torch.all(masks.logical_or(~masks))
