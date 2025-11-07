from pathlib import Path

import torch

from rlbench.datasets.imitation_learning.cached_dataset import get_temporal_cached_dataloader

from src.models.platonic_transformer.groups import PLATONIC_GROUPS
from src.policies.platonic_policy import (
    PlatonicDiffusionPolicy,
    PlatonicDiffusionPolicyConfig,
    _components_to_pose,
    _pack_components,
    _pose_to_components,
    _unpack_components,
)
from src.utils.geometry import quaternion_to_matrix


CACHE_ROOT = Path("/home/dknigge/project_dir/data/robotics/rlbench/imitation_learning")


def _rotate_pose_sequence(pose: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate pose tensors by first converting to the scalar/vector format."""
    grasp, orientation, position = _pose_to_components(pose)
    orientation = torch.einsum("ij,...kj->...ki", rotation, orientation)
    position = torch.matmul(position, rotation.t())
    return _components_to_pose(grasp, orientation, position)


def _translate_pose_sequence(pose: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Translate pose tensors by shifting position channels."""
    grasp, orientation, position = _pose_to_components(pose)
    position = position + translation.to(dtype=position.dtype, device=position.device)
    return _components_to_pose(grasp, orientation, position)


def _rototranslate_pose_sequence(
    pose: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    """Apply rotation followed by translation to pose tensors."""
    return _translate_pose_sequence(_rotate_pose_sequence(pose, rotation), translation)


def _rotate_point_cloud(
    point_cloud: dict[str, torch.Tensor], rotation: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Rotate point cloud positions [B, N_time, N_pts, 3]; colours unaffected."""
    positions = torch.matmul(
        point_cloud["positions"], rotation.t()
    )
    return {
        "positions": positions,
        "colors": point_cloud["colors"],
    }


def _translate_point_cloud(
    point_cloud: dict[str, torch.Tensor], translation: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Translate point cloud positions by a fixed vector."""
    translation = translation.view(1, 1, 1, 3).to(
        dtype=point_cloud["positions"].dtype,
        device=point_cloud["positions"].device,
    )
    positions = point_cloud["positions"] + translation
    return {
        "positions": positions,
        "colors": point_cloud["colors"],
    }


def _rototranslate_point_cloud(
    point_cloud: dict[str, torch.Tensor],
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply rotation followed by translation to point cloud positions."""
    rotated = _rotate_point_cloud(point_cloud, rotation)
    return _translate_point_cloud(rotated, translation)


def _rotate_packed(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate packed (flattened) action tensors used by the diffusion scheduler."""
    grasp, orientation, position = _unpack_components(tensor)
    orientation = torch.einsum("ij,...kj->...ki", rotation, orientation)
    position = torch.matmul(position, rotation.t())
    return _pack_components(grasp, orientation, position)


def _translate_packed(tensor: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Translate the position channel within packed action tensors."""
    # Relative representation is translation invariant; no update required.
    return tensor


def _rototranslate_packed(
    tensor: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    """Apply rotation followed by translation within packed action tensors."""
    rotated = _rotate_packed(tensor, rotation)
    return _translate_packed(rotated, translation)


def _fetch_batch():
    """Load a single sample so we can test equivariance with real data."""
    dataloader = get_temporal_cached_dataloader(
        CACHE_ROOT,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    batch = next(iter(dataloader))
    proprio = batch["observation"]["proprio_sequence"]
    point_cloud_raw = batch["observation"]["point_cloud_sequence"]
    point_cloud = {
        "positions": point_cloud_raw["points"],
        "colors": point_cloud_raw["colors"],
    }
    actions = batch["action"]
    return proprio, point_cloud, actions


def _random_quaternion(batch_shape: tuple[int, ...]) -> torch.Tensor:
    q = torch.randn(*batch_shape, 4)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def test_platonic_policy_equivariance():
    """Noise predictions should rotate exactly with inputs."""
    observations, point_cloud, actions = _fetch_batch()
    # observations shape: (B=1, T_obs, 8); actions shape: (B=1, T_action, 8)
    group = PLATONIC_GROUPS["icosahedron"]

    cfg = PlatonicDiffusionPolicyConfig(
        context_length=observations.shape[1],
        horizon=actions.shape[1],
        hidden_dim=group.G * 2,
        num_layers=1,
        num_heads=group.G,
        solid_name="icosahedron",
        ffn_dim_factor=2,
        noise_scheduler_kwargs={
            "num_train_timesteps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
        num_inference_steps=10,
    )
    policy = PlatonicDiffusionPolicy(cfg).cpu()
    policy.eval()

    obs_scalars, obs_vectors, obs_positions, anchor = policy._build_context_tokens(
        observations, point_cloud
    )
    action_gripper, action_orientation, action_positions = policy._split_actions(actions, anchor)
    action_model = _pack_components(
        action_gripper,
        action_orientation,
        action_positions,
    )  # (B, T_action, 10)

    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(
        action_model.shape,
        generator=generator,
        device=action_model.device,
        dtype=action_model.dtype,
    )
    timesteps = torch.randint(
        0,
        policy.scheduler.config.num_train_timesteps,
        (actions.shape[0],),
        generator=generator,
        dtype=torch.long,
    )
    noisy = policy.scheduler.add_noise(action_model, noise, timesteps)  # forward diffusion
    noisy_gripper, noisy_orientation, noisy_positions = _unpack_components(noisy)
    noisy_action_scalars, noisy_action_vectors, noisy_action_positions = policy._build_action_tokens(
        noisy_gripper,
        noisy_orientation,
        noisy_positions,
    )
    pred_noise = policy._tokenise(
        obs_scalars,
        obs_vectors,
        obs_positions,
        noisy_action_scalars,
        noisy_action_vectors,
        noisy_action_positions,
        timesteps,
    )

    rotation = group.elements[5].to(dtype=observations.dtype)
    observations_rot = _rotate_pose_sequence(observations, rotation)
    point_cloud_rot = _rotate_point_cloud(point_cloud, rotation)
    actions_rot = _rotate_pose_sequence(actions, rotation)

    obs_rot_scalars, obs_rot_vectors, obs_rot_positions, anchor_rot = policy._build_context_tokens(
        observations_rot, point_cloud_rot
    )
    action_rot_gripper, action_rot_orientation, action_rot_positions = policy._split_actions(
        actions_rot, anchor_rot
    )
    action_rot_model = _pack_components(
        action_rot_gripper,
        action_rot_orientation,
        action_rot_positions,
    )

    noise_rot = _rotate_packed(noise, rotation)
    noisy_rot = policy.scheduler.add_noise(action_rot_model, noise_rot, timesteps)
    assert torch.allclose(noisy_rot, _rotate_packed(noisy, rotation), atol=1e-5)

    noisy_rot_gripper, noisy_rot_orientation, noisy_rot_positions = _unpack_components(noisy_rot)
    noisy_rot_action_scalars, noisy_rot_action_vectors, noisy_rot_action_positions = policy._build_action_tokens(
        noisy_rot_gripper,
        noisy_rot_orientation,
        noisy_rot_positions,
    )
    pred_noise_rot = policy._tokenise(
        obs_rot_scalars,
        obs_rot_vectors,
        obs_rot_positions,
        noisy_rot_action_scalars,
        noisy_rot_action_vectors,
        noisy_rot_action_positions,
        timesteps,
    )
    assert torch.allclose(pred_noise_rot, _rotate_packed(pred_noise, rotation), atol=2e-4)


def test_rotate_packed_consistency():
    group = PLATONIC_GROUPS["icosahedron"]
    tensor = torch.randn(2, 5, 10)
    rotation = group.elements[3].to(dtype=tensor.dtype)

    rotated = _rotate_packed(tensor, rotation)
    grasp, orientation, position = _unpack_components(rotated)
    grasp_ref, orientation_ref, position_ref = _unpack_components(tensor)
    manual_orientation = torch.einsum("ij,...kj->...ki", rotation, orientation_ref)
    manual_position = torch.matmul(position_ref, rotation.t())

    assert torch.allclose(grasp, grasp_ref)
    assert torch.allclose(orientation, manual_orientation)
    assert torch.allclose(position, manual_position)


def test_tokenise_equivariance_random_actions():
    observations, point_cloud, _ = _fetch_batch()
    group = PLATONIC_GROUPS["icosahedron"]

    cfg = PlatonicDiffusionPolicyConfig(
        context_length=observations.shape[1],
        horizon=4,
        hidden_dim=group.G * 2,
        num_layers=1,
        num_heads=group.G,
        solid_name="icosahedron",
        ffn_dim_factor=2,
        noise_scheduler_kwargs={
            "num_train_timesteps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
        num_inference_steps=10,
    )
    policy = PlatonicDiffusionPolicy(cfg).cpu()
    policy.eval()

    rotation = group.elements[11].to(dtype=observations.dtype)
    observations_rot = _rotate_pose_sequence(observations, rotation)
    point_cloud_rot = _rotate_point_cloud(point_cloud, rotation)

    rand_pos = torch.randn(observations.shape[0], cfg.horizon, 3)
    rand_quat = _random_quaternion((observations.shape[0], cfg.horizon))
    rand_grip = torch.randn(observations.shape[0], cfg.horizon, 1)
    random_actions = torch.cat([rand_pos, rand_quat, rand_grip], dim=-1)
    random_actions_rot = _rotate_pose_sequence(random_actions, rotation)

    obs_scalars, obs_vectors, obs_positions, anchor = policy._build_context_tokens(
        observations, point_cloud
    )
    action_scalars, action_orientation, action_positions = policy._split_actions(
        random_actions, anchor
    )
    action_model = _pack_components(action_scalars, action_orientation, action_positions)

    obs_rot_scalars, obs_rot_vectors, obs_rot_positions, anchor_rot = policy._build_context_tokens(
        observations_rot, point_cloud_rot
    )
    action_rot_scalars, action_rot_orientation, action_rot_positions = policy._split_actions(
        random_actions_rot, anchor_rot
    )
    action_rot_model = _pack_components(
        action_rot_scalars,
        action_rot_orientation,
        action_rot_positions,
    )

    timesteps = torch.randint(0, policy.scheduler.config.num_train_timesteps, (observations.shape[0],))
    noise = torch.randn_like(action_model)
    noisy = policy.scheduler.add_noise(action_model, noise, timesteps)
    noisy_gripper, noisy_orientation, noisy_positions = _unpack_components(noisy)
    noisy_action_scalars, noisy_action_vectors, noisy_action_positions = policy._build_action_tokens(
        noisy_gripper,
        noisy_orientation,
        noisy_positions,
    )
    pred_noise = policy._tokenise(
        obs_scalars,
        obs_vectors,
        obs_positions,
        noisy_action_scalars,
        noisy_action_vectors,
        noisy_action_positions,
        timesteps,
    )

    noisy_rot = policy.scheduler.add_noise(
        action_rot_model,
        _rotate_packed(noise, rotation),
        timesteps,
    )
    noisy_rot_scalars, noisy_rot_orientation, noisy_rot_positions = _unpack_components(noisy_rot)
    noisy_rot_action_scalars, noisy_rot_action_vectors, noisy_rot_action_positions = policy._build_action_tokens(
        noisy_rot_scalars,
        noisy_rot_orientation,
        noisy_rot_positions,
    )
    pred_noise_rot = policy._tokenise(
        obs_rot_scalars,
        obs_rot_vectors,
        obs_rot_positions,
        noisy_rot_action_scalars,
        noisy_rot_action_vectors,
        noisy_rot_action_positions,
        timesteps,
    )

    assert torch.allclose(pred_noise_rot, _rotate_packed(pred_noise, rotation), atol=2e-4)

def test_sampling_equivariance_deterministic():
    observations, point_cloud, _ = _fetch_batch()
    group = PLATONIC_GROUPS["icosahedron"]

    cfg = PlatonicDiffusionPolicyConfig(
        context_length=observations.shape[1],
        horizon=4,
        hidden_dim=group.G * 2,
        num_layers=1,
        num_heads=group.G,
        solid_name="icosahedron",
        ffn_dim_factor=2,
        noise_scheduler_kwargs={
            "num_train_timesteps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
        num_inference_steps=10,
    )
    policy = PlatonicDiffusionPolicy(cfg).cpu()
    policy.eval()

    noise = torch.randn(
        (
            observations.shape[0],
            cfg.horizon,
            10,
        )
    )
    grip, orientation, position = policy.sample_actions(
        observations,
        point_cloud,
        initial_noise=noise,
        deterministic=True,
        return_features=True,
    )
    actions = _components_to_pose(grip, orientation, position).cpu()

    rotation = group.elements[7].to(dtype=observations.dtype)
    observations_rot = _rotate_pose_sequence(observations, rotation)
    point_cloud_rot = _rotate_point_cloud(point_cloud, rotation)

    noise_rot = _rotate_packed(noise, rotation)
    grip_rot, orientation_rot, position_rot = policy.sample_actions(
        observations_rot,
        point_cloud_rot,
        initial_noise=noise_rot,
        deterministic=True,
        return_features=True,
    )
    rot_orientation = torch.einsum("ij,...kj->...ki", rotation, orientation)
    rot_position = torch.matmul(position, rotation.t())
    assert torch.allclose(grip_rot.cpu(), grip.cpu(), atol=1e-3)
    assert torch.allclose(orientation_rot.cpu(), rot_orientation.cpu(), atol=5e-4)
    assert torch.allclose(position_rot.cpu(), rot_position.cpu(), atol=5e-4)

    actions_rot = _components_to_pose(grip_rot, orientation_rot, position_rot).cpu()

    rotated_actions = _rotate_pose_sequence(actions, rotation)

    def canonicalise(pose: torch.Tensor) -> torch.Tensor:
        quat = pose[..., 3:7]
        sign = torch.where(quat[..., -1:] < 0, -1.0, 1.0)
        pose = pose.clone()
        pose[..., 3:7] = quat * sign
        return pose

    actions_rot = canonicalise(actions_rot)
    rotated_actions = canonicalise(rotated_actions)

    assert torch.allclose(actions_rot[..., :3], rotated_actions[..., :3], atol=5e-4)
    assert torch.allclose(actions_rot[..., 7:], rotated_actions[..., 7:], atol=5e-4)

    rot_m = quaternion_to_matrix(actions_rot[..., 3:7])
    ref_m = quaternion_to_matrix(rotated_actions[..., 3:7])
    assert torch.allclose(rot_m, ref_m, atol=5e-4)


def test_sampling_equivariance_rototranslation():
    observations, point_cloud, _ = _fetch_batch()
    group = PLATONIC_GROUPS["icosahedron"]

    cfg = PlatonicDiffusionPolicyConfig(
        context_length=observations.shape[1],
        horizon=4,
        hidden_dim=group.G * 2,
        num_layers=1,
        num_heads=group.G,
        solid_name="icosahedron",
        ffn_dim_factor=2,
        noise_scheduler_kwargs={
            "num_train_timesteps": 100,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "prediction_type": "epsilon",
        },
        num_inference_steps=10,
    )
    policy = PlatonicDiffusionPolicy(cfg).cpu()
    policy.eval()

    noise = torch.randn(
        (
            observations.shape[0],
            cfg.horizon,
            10,
        )
    )
    grip, orientation, position = policy.sample_actions(
        observations,
        point_cloud,
        initial_noise=noise,
        deterministic=True,
        return_features=True,
    )
    actions = _components_to_pose(grip, orientation, position).cpu()

    rotation = group.elements[9].to(dtype=observations.dtype)
    translation = torch.tensor([0.15, -0.08, 0.05], dtype=observations.dtype)

    observations_rt = _rototranslate_pose_sequence(observations, rotation, translation)
    point_cloud_rt = _rototranslate_point_cloud(point_cloud, rotation, translation)

    noise_rt = _rototranslate_packed(noise, rotation, translation)
    grip_rt, orientation_rt, position_rt = policy.sample_actions(
        observations_rt,
        point_cloud_rt,
        initial_noise=noise_rt,
        deterministic=True,
        return_features=True,
    )
    actions_rt = _components_to_pose(grip_rt, orientation_rt, position_rt).cpu()

    reference = _rototranslate_pose_sequence(actions, rotation, translation)

    def canonicalise(pose: torch.Tensor) -> torch.Tensor:
        quat = pose[..., 3:7]
        sign = torch.where(quat[..., -1:] < 0, -1.0, 1.0)
        pose = pose.clone()
        pose[..., 3:7] = quat * sign
        return pose

    actions_rt = canonicalise(actions_rt)
    reference = canonicalise(reference)

    assert torch.allclose(actions_rt[..., :3], reference[..., :3], atol=5e-4)
    assert torch.allclose(actions_rt[..., 7:], reference[..., 7:], atol=5e-4)
    rot_m = quaternion_to_matrix(actions_rt[..., 3:7])
    ref_m = quaternion_to_matrix(reference[..., 3:7])
    assert torch.allclose(rot_m, ref_m, atol=5e-4)
