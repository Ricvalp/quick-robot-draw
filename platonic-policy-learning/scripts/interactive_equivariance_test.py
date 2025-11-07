#!/usr/bin/env python3
"""Interactive viewer for inspecting Platonic policy equivariance.

The script boots a small `viser` server that visualises:
  * The most recent point cloud frame from an RLBench cache sample.
  * The ground-truth action trajectory as a chain of frames and line segments.
  * A randomly initialised Platonic diffusion policy's sampled trajectory.

Clicking the provided GUI button applies a random rotation sampled from the
chosen Platonic symmetry group to the entire sample (observations, actions,
and point cloud). The viewer then recomputes the policy rollout to help verify
that predictions rotate consistently with the inputs.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
import viser
from rlbench.datasets.imitation_learning.cached_dataset import (
    get_temporal_cached_dataloader,
)

from src.models.platonic_transformer.groups import PLATONIC_GROUPS
from src.policies import PlatonicDiffusionPolicy, PlatonicDiffusionPolicyConfig
from src.policies.platonic_policy import _pack_components, _unpack_components
from src.utils import NormalizationStats, NormalizationTransform
from src.utils.geometry import (
    apply_rotation,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


# ---------------------------------------------------------------------------
# Utility dataclasses used to keep the viewer state tidy.


@dataclass
class SampleBatch:
    """Container for a single RLBench sample pulled from the cached dataset."""

    proprio: torch.Tensor  # (T_obs, 8) sequence of proprio poses
    point_cloud_sequence: List[List[dict]]  # nested view dictionaries
    actions: torch.Tensor  # (T_action, 8) action rollout


@dataclass
class TrajectoryVisuals:
    """Handles to all geometry we add to the viser scene."""

    point_cloud: viser.PointCloudHandle | None = None
    gt_segments: viser.LineSegmentsHandle | None = None
    pred_segments: viser.LineSegmentsHandle | None = None
    gt_axes: viser.BatchedAxesHandle | None = None
    pred_axes: viser.BatchedAxesHandle | None = None


# ---------------------------------------------------------------------------
# Quaternion / pose helpers


def rotate_pose_sequence(pose: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate a pose sequence by left-multiplying with the supplied rotation matrix.

    Args:
        pose: Pose tensor of shape ``(..., 8)`` holding ``(x, y, z, qx, qy, qz, qw, g)``.
        rotation: Rotation matrix in shape ``(3, 3)``.
    """
    positions = apply_rotation(rotation, pose[..., :3])  # (..., 3)
    basis = quaternion_to_matrix(pose[..., 3:7])  # (..., 3, 3)
    rotated_basis = torch.einsum("ij,...jk->...ik", rotation, basis)  # (..., 3, 3)
    quaternions = matrix_to_quaternion(rotated_basis)  # (..., 4)
    return torch.cat([positions, quaternions, pose[..., 7:]], dim=-1)


def rotate_point_cloud_sequence(
    sequence: Iterable[Iterable[dict]],
    rotation: torch.Tensor,
) -> List[List[dict]]:
    """Rotate every point cloud in the cached sample (in-place clone)."""
    rotated_sequence: List[List[dict]] = []
    for views in sequence:
        rotated_views: List[dict] = []
        for view in views:
            points = view["points"]  # (N, 3)
            colors = view["colors"]
            rotated_views.append(
                {
                    **view,
                    "points": apply_rotation(rotation, points),
                    "colors": colors,
                }
            )
        rotated_sequence.append(rotated_views)
    return rotated_sequence


# ---------------------------------------------------------------------------
# Data loading and visualisation helpers.


def fetch_single_sample(
    cache_path: Path,
    tasks: list[str] | None,
) -> Optional[SampleBatch]:
    """Pull a single sample from the cache; return ``None`` when unavailable."""
    print(f"[viewer] Loading sample from {cache_path}")
    if not cache_path.exists():
        print(f"[viewer] Cache path not found: {cache_path}")
        return None
    dataloader = get_temporal_cached_dataloader(
        cache_path,
        tasks=tasks,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("[viewer] Dataloader yielded no samples. Check cache contents.")
        return None
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"[viewer] Failed to load sample: {exc}")
        return None

    observation = batch["observation"]
    proprio = observation["proprio_sequence"][0].to(dtype=torch.float32)  # (T_obs, 8)
    # Deep-clone the point cloud entries so we can rotate them freely.
    point_cloud_sequence: List[List[dict]] = []
    for views in observation["point_cloud_sequence"]:
        view_list: List[dict] = []
        for view in views:
            view_list.append(
                {
                    "points": view["points"].clone().to(dtype=torch.float32),
                    "colors": view["colors"].clone().to(dtype=torch.float32),
                }
            )
        if not view_list:
            print("[viewer] Encountered an empty point cloud view; skipping sample.")
            return None
        point_cloud_sequence.append(view_list)

    actions = batch["action"][0].to(dtype=torch.float32)  # (T_action, 8)
    print(
        "[viewer] Loaded sample with "
        f"{proprio.shape[0]} obs steps and {actions.shape[0]} action steps."
    )
    return SampleBatch(
        proprio=proprio,
        point_cloud_sequence=point_cloud_sequence,
        actions=actions,
    )


def point_cloud_to_numpy(views: Iterable[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Stack multiple camera views into one array of points and colours."""
    points = torch.cat([view["points"] for view in views], dim=0)  # (N, 3)
    colors = torch.cat([view["colors"] for view in views], dim=0)  # (N, 3)
    # Ensure colors live in the expected 0..255 uint8 range.
    if colors.max() <= 1.0:
        colors = colors * 255.0
    colors_uint8 = colors.clamp(0.0, 255.0).to(torch.uint8)
    return points.cpu().numpy(), colors_uint8.cpu().numpy()


def trajectory_segments(positions: np.ndarray) -> np.ndarray:
    """Convert a sequence of points (T, 3) into the (T-1, 2, 3) segment format."""
    if positions.shape[0] < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([positions[:-1], positions[1:]], axis=1).astype(np.float32)


def quat_xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """viser expects quaternions in (w, x, y, z) order."""
    return np.concatenate([quaternion[..., 3:], quaternion[..., :3]], axis=-1)


# ---------------------------------------------------------------------------
# Main viewer implementation.


class EquivarianceViewer:
    """Interactive viser-based visualiser for Platonic diffusion policies."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cache_root = Path(args.cache_path).expanduser().resolve()
        self.checkpoint_path: Path | None = None
        self.checkpoint_state: dict[str, torch.Tensor] | None = None
        self.checkpoint_config: dict[str, Any] | None = None
        self.horizon = args.horizon
        self.normalizer = self._load_normalizer(args.normalization_stats)

        if args.checkpoint is not None:
            self._prepare_checkpoint(args.checkpoint)

        solid_name = (
            self.checkpoint_config.get("solid_name", args.platonic_solid)
            if self.checkpoint_config is not None
            else args.platonic_solid
        ).lower()
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(
                f"Unknown platonic solid '{solid_name}'. "
                f"Available groups: {list(PLATONIC_GROUPS.keys())}"
            )
        self.group = PLATONIC_GROUPS[solid_name]
        self.solid_name = solid_name
        self.sample: Optional[SampleBatch] = fetch_single_sample(
            self.cache_root,
            args.tasks,
        )
        self.initial_noise: Optional[torch.Tensor] = None

        # Build a randomly initialised Platonic policy matching the sample geometry.
        self.policy: Optional[PlatonicDiffusionPolicy]
        if self.sample is not None:
            print("[viewer] Initialising Platonic policy.")
            self.policy = self._build_policy(self.sample)
            self.policy.eval()
            if self.checkpoint_state is not None:
                self._load_checkpoint_weights()
            print("[viewer] Policy initialised.")
            self.generator = torch.Generator(device=self.policy.device).manual_seed(0)
        else:
            self.policy = None
            print("[viewer] Policy skipped because no sample was loaded.")

        # Cached handles for scene elements so we can update them in-place.
        self.handles = TrajectoryVisuals()

        self.server = viser.ViserServer(
            host=args.host,
            port=args.port,
            label="Platonic Equivariance Explorer",
        )
        print(f"[viewer] Viser server running on {args.host}:{args.port}")
        self._build_gui()
        # Render the unrotated sample on startup.
        if self.sample is not None:
            print("[viewer] Rendering initial sample.")
            self._update_scene(self.sample, initialise_noise=True)
            print("[viewer] Initial render complete.")
        else:
            print(
                "[viewer] Viewer running without data. "
                "Verify cache path and tasks, then restart once data is available."
            )

    # ------------------------------------------------------------------ GUI / policy setup

    def _load_normalizer(self, stats_path: str | None) -> NormalizationTransform | None:
        """Load dataset normalization stats for consistent policy inputs."""
        if stats_path is None:
            return None
        path = Path(stats_path).expanduser().resolve()
        if not path.exists():
            print(f"[viewer] Normalization stats not found at {path}; proceeding without normalization.")
            return None
        try:
            stats = NormalizationStats.from_json(path)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[viewer] Failed to load normalization stats '{path}': {exc}")
            return None
        print(f"[viewer] Loaded normalization stats from {path}")
        return NormalizationTransform(stats)

    def _prepare_checkpoint(self, checkpoint: str) -> None:
        """Load checkpoint metadata so we can restore weights after building the policy."""
        path = Path(checkpoint).expanduser().resolve()
        self.checkpoint_path = path
        if not path.exists():
            print(f"[viewer] Checkpoint not found: {path}")
            return
        try:
            blob = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[viewer] Failed to load checkpoint '{path}': {exc}")
            return

        self.checkpoint_state = None

        if isinstance(blob, dict):
            if "model_state" in blob:
                self.checkpoint_state = blob["model_state"]
            elif all(isinstance(v, torch.Tensor) for v in blob.values()):
                # Pure state dict case.
                self.checkpoint_state = blob  # type: ignore[assignment]
            config = blob.get("config")
            if isinstance(config, dict):
                self.checkpoint_config = config
                horizon_override = config.get("horizon")
                if horizon_override is not None and int(horizon_override) != self.horizon:
                    print(
                        f"[viewer] Overriding horizon to {int(horizon_override)} "
                        "from checkpoint config."
                    )
                    self.horizon = int(horizon_override)
        if self.checkpoint_state is None:
            print(
                "[viewer] Checkpoint loaded, but no model_state found. "
                "Continuing with randomly initialised weights."
            )

    def _load_checkpoint_weights(self) -> None:
        """Restore model weights from the loaded checkpoint blob."""
        if self.policy is None or self.checkpoint_state is None:
            return
        try:
            load_status = self.policy.load_state_dict(self.checkpoint_state, strict=False)
        except RuntimeError as exc:
            print(f"[viewer] Failed to load checkpoint weights: {exc}")
            return
        missing, unexpected = load_status
        if missing:
            print(f"[viewer] Missing keys while loading checkpoint: {sorted(missing)}")
        if unexpected:
            print(f"[viewer] Unexpected keys while loading checkpoint: {sorted(unexpected)}")
        if self.checkpoint_path is not None:
            print(f"[viewer] Loaded checkpoint weights from {self.checkpoint_path}")

    def _build_policy(self, sample: SampleBatch) -> PlatonicDiffusionPolicy:
        """Initialise a Platonic policy whose tensor shapes match the sample."""
        context_length = int(sample.proprio.shape[0])
        if self.checkpoint_config is not None:
            cfg_dict = dict(self.checkpoint_config)
            cfg = PlatonicDiffusionPolicyConfig(**cfg_dict)
            if cfg.horizon <= 0:
                raise ValueError("Checkpoint horizon must be positive.")
            if cfg.context_length != context_length:
                print(
                    f"[viewer] Warning: checkpoint context_length={cfg.context_length} "
                    f"differs from sample context_length={context_length}."
                )
            if cfg.scalar_channels < 4:
                cfg.scalar_channels = 4
            self.horizon = cfg.horizon
            return PlatonicDiffusionPolicy(cfg)

        horizon = int(self.horizon)
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")

        if self.args.platonic_hidden_dim % self.group.G != 0:
            raise ValueError(
                f"platonic-hidden-dim ({self.args.platonic_hidden_dim}) must be divisible by group order {self.group.G}"
            )
        if self.args.platonic_num_heads % self.group.G != 0:
            raise ValueError(
                f"platonic-num-heads ({self.args.platonic_num_heads}) must be divisible by group order {self.group.G}"
            )

        cfg = PlatonicDiffusionPolicyConfig(
            context_length=context_length,
            horizon=horizon,
            hidden_dim=self.args.platonic_hidden_dim,
            num_layers=self.args.platonic_num_layers,
            num_heads=self.args.platonic_num_heads,
            solid_name=self.solid_name,
            ffn_dim_factor=self.args.platonic_ffn_dim_factor,
            dropout=self.args.platonic_dropout,
            drop_path_rate=self.args.platonic_drop_path,
            mean_aggregation=self.args.platonic_mean_aggregation,
            use_softmax_attention=self.args.platonic_softmax_attention,
            rope_sigma=self.args.platonic_rope_sigma,
            learned_freqs=not self.args.platonic_fixed_freqs,
            noise_scheduler_kwargs={
                "num_train_timesteps": self.args.num_train_timesteps,
                "beta_start": self.args.beta_start,
                "beta_end": self.args.beta_end,
                "beta_schedule": self.args.beta_schedule,
                "prediction_type": "epsilon",
            },
            num_inference_steps=self.args.num_inference_steps,
            scalar_channels=4,
        )
        return PlatonicDiffusionPolicy(cfg)

    def _build_gui(self) -> None:
        """Create GUI controls for the viewer."""
        self.server.gui.add_markdown(
            "## Platonic Policy Equivariance\n"
            "Use the button below to draw a random group element and rotate the cached sample.\n"
            "Green: ground truth action frames; Red: policy prediction."
        )

        button = self.server.gui.add_button("Apply random rotation")
        if self.sample is None:
            button.disabled = True
            self.server.gui.add_markdown(
                "⚠️ No cached sample could be loaded. "
                "Check `--cache-path` and `--tasks`, then restart once data is available.",
                order=1,
            )
            print("[viewer] Rotate button disabled; no sample available.")
        else:
            print("[viewer] GUI button initialised.")

        @button.on_click
        def _(_event: viser.GuiEvent[viser.GuiButtonHandle]) -> None:
            if self.sample is None:
                print("[viewer] Cannot rotate without a cached sample.")
                return
            self._apply_random_rotation()

    # ------------------------------------------------------------------ Scene update logic

    def _run_policy(
        self,
        proprio: torch.Tensor,
        noise: Optional[torch.Tensor],
    ) -> np.ndarray:
        """Sample a trajectory from the Platonic policy given an observation sequence."""
        if self.policy is None:
            raise RuntimeError("Policy is not initialised.")
        point_cloud = self._stack_point_cloud(sample.point_cloud_sequence)
        if self.normalizer is not None:
            proprio = self.normalizer.normalize_proprio(proprio)
            normalized_pc = self.normalizer.normalize_point_cloud_sequence(point_cloud)
            point_cloud = {
                "points": normalized_pc["points"],
                "colors": normalized_pc["colors"],
            }
        else:
            point_cloud = {
                "points": point_cloud["points"],
                "colors": point_cloud["colors"],
            }

        max_points = getattr(self.args, "pointcloud_max_points", None)
        if max_points is not None:
            point_cloud = {
                "points": point_cloud["points"][..., : max_points, :],
                "colors": point_cloud["colors"][..., : max_points, :],
            }

        point_cloud = {
            "positions": point_cloud["points"].unsqueeze(0).to(self.policy.device),
            "colors": point_cloud["colors"].unsqueeze(0).to(self.policy.device),
        }

        with torch.no_grad():
            # Policy expects shape (B, T_obs, obs_dim); we only use batch size 1.
            batch = proprio.unsqueeze(0).to(self.policy.device)
            if noise is not None:
                prediction = self.policy.sample_actions(
                    batch,
                    point_cloud,
                    initial_noise=noise,
                    deterministic=True,
                )
            else:
                prediction = self.policy.sample_actions(
                    batch,
                    point_cloud,
                    generator=self.generator,
                    deterministic=True,
                )
        return prediction[0].cpu().numpy()  # (T_action, 8)

    def _update_scene(self, sample: SampleBatch, initialise_noise: bool = False) -> None:
        """Render the provided sample and policy outputs inside viser."""
        if self.policy is None:
            print("[viewer] Policy is unavailable; skipping scene update.")
            return
        print("[viewer] Updating scene with new sample.")
        # Latest point cloud frame (index -1) provides the richest context.
        last_views = sample.point_cloud_sequence[-1]
        if not last_views:
            print("[viewer] Skipping update; last point cloud view is empty.")
            return
        points_np, colors_np = point_cloud_to_numpy(last_views)

        gt_actions = sample.actions.cpu().numpy()  # (T_action, 8)
        if self.policy.cfg.horizon < gt_actions.shape[0]:
            gt_actions = gt_actions[: self.policy.cfg.horizon]
        if initialise_noise or getattr(self, "initial_noise", None) is None:
            self.initial_noise = torch.randn(
                (
                    1,
                    self.policy.cfg.horizon,
                    self.policy.packed_action_dim,
                ),
                generator=self.generator,
                device=self.policy.device,
            )
        pred_actions = self._run_policy(
            sample.proprio,
            noise=self.initial_noise.clone() if self.initial_noise is not None else None,
        )
        if self.normalizer is not None:
            pred_actions_tensor = torch.from_numpy(pred_actions).to(dtype=torch.float32)
            pred_actions_tensor = self.normalizer.denormalize_action_positions(pred_actions_tensor)
            pred_actions = pred_actions_tensor.cpu().numpy()

        self._update_point_cloud(points_np, colors_np)
        self._update_trajectories(gt_actions, pred_actions)
        print("[viewer] Scene update complete.")

    def _update_point_cloud(self, points: np.ndarray, colors: np.ndarray) -> None:
        """Add or refresh the point cloud handle."""
        if self.handles.point_cloud is None:
            self.handles.point_cloud = self.server.scene.add_point_cloud(
                name="/scene/point_cloud",
                points=points,
                colors=colors,
                point_size=0.005,
                precision="float32",
            )
            print("[viewer] Created point cloud handle.")
        else:
            self.handles.point_cloud.points = points
            self.handles.point_cloud.colors = colors
            print("[viewer] Updated existing point cloud handle.")

    def _update_trajectories(self, gt_actions: np.ndarray, pred_actions: np.ndarray) -> None:
        """Draw both ground-truth and predicted trajectories with frames."""
        gt_positions = gt_actions[:, :3].astype(np.float32)  # (T, 3)
        pred_positions = pred_actions[:, :3].astype(np.float32)  # (T, 3)

        gt_segments = trajectory_segments(gt_positions)
        pred_segments = trajectory_segments(pred_positions)

        if self.handles.gt_segments is None:
            self.handles.gt_segments = self.server.scene.add_line_segments(
                "/scene/gt_traj",
                points=gt_segments,
                colors=(0, 255, 0),
                line_width=4,
            )
            print("[viewer] Created ground-truth trajectory segments.")
        else:
            self.handles.gt_segments.points = gt_segments
            print("[viewer] Updated ground-truth trajectory segments.")

        if self.handles.pred_segments is None:
            self.handles.pred_segments = self.server.scene.add_line_segments(
                "/scene/pred_traj",
                points=pred_segments,
                colors=(255, 0, 0),
                line_width=3,
            )
            print("[viewer] Created predicted trajectory segments.")
        else:
            self.handles.pred_segments.points = pred_segments
            print("[viewer] Updated predicted trajectory segments.")

        gt_quat = quat_xyzw_to_wxyz(gt_actions[:, 3:7])
        pred_quat = quat_xyzw_to_wxyz(pred_actions[:, 3:7])

        if self.handles.gt_axes is None:
            self.handles.gt_axes = self.server.scene.add_batched_axes(
                "/scene/gt_frames",
                batched_wxyzs=gt_quat,
                batched_positions=gt_positions,
                axes_length=0.05,
                axes_radius=0.003,
            )
            print("[viewer] Created ground-truth frame axes.")
        else:
            self.handles.gt_axes.batched_wxyzs = gt_quat
            self.handles.gt_axes.batched_positions = gt_positions
            print("[viewer] Updated ground-truth frame axes.")

        if self.handles.pred_axes is None:
            self.handles.pred_axes = self.server.scene.add_batched_axes(
                "/scene/pred_frames",
                batched_wxyzs=pred_quat,
                batched_positions=pred_positions,
                axes_length=0.06,
                axes_radius=0.004,
            )
            print("[viewer] Created predicted frame axes.")
        else:
            self.handles.pred_axes.batched_wxyzs = pred_quat
            self.handles.pred_axes.batched_positions = pred_positions
            print("[viewer] Updated predicted frame axes.")

    # ------------------------------------------------------------------ Random rotation interaction

    def _apply_random_rotation(self) -> None:
        """Sample a random group element, rotate the cached sample, and refresh the view."""
        if self.sample is None:
            print("[viewer] Cannot apply rotation; sample unavailable.")
            return
        index = random.randrange(self.group.G)
        rotation = self.group.elements[index].to(dtype=torch.float32)
        if self.policy is not None:
            rotation = rotation.to(device=self.policy.device)
        print(f"[viewer] Applying rotation index {index}.")
        rotated = SampleBatch(
            proprio=rotate_pose_sequence(self.sample.proprio, rotation),
            point_cloud_sequence=rotate_point_cloud_sequence(
                self.sample.point_cloud_sequence,
                rotation,
            ),
            actions=rotate_pose_sequence(self.sample.actions, rotation),
        )
        if self.policy is not None:
            if self.initial_noise is not None:
                grip, orientation, position = _unpack_components(self.initial_noise)
                rotated_orientation = torch.einsum("ij,...kj->...ki", rotation, orientation)
                rotated_position = torch.matmul(position, rotation.t())
                self.initial_noise = _pack_components(
                    grip,
                    rotated_orientation,
                    rotated_position,
                ).to(
                    device=self.policy.device
                )
            self.generator = self.generator.manual_seed(0)
        self._update_scene(rotated)

    # ------------------------------------------------------------------
    # Main loop

    def run(self) -> None:
        """Keep the viewer process alive until interrupted."""
        print("Viewer running. Open the URL above and press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[viewer] Shutting down.")


# ---------------------------------------------------------------------------
# CLI entry point.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Platonic policy equivariance viewer.")
    parser.add_argument("--cache-path", type=str, default="/home/dknigge/project_dir/data/robotics/rlbench/imitation_learning/", help="RLBench imitation learning cache root dir.")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Optional task subset to sample from.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address for the viser server.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viser server.")
    parser.add_argument("--horizon", type=int, default=1, help="Override the policy action horizon.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional policy checkpoint to restore before sampling.")
    parser.add_argument("--normalization-stats", type=str, default="artifacts/stats/normalization_stats.json", help="Path to dataset normalization stats JSON.")
    parser.add_argument("--platonic-solid", type=str, default="icosahedron", help="Platonic group used for rotations.")
    parser.add_argument("--platonic-hidden-dim", type=int, default=480, help="Transformer hidden width (must divide group order).")
    parser.add_argument("--platonic-num-heads", type=int, default=60, help="Number of attention heads (must divide group order).")
    parser.add_argument("--platonic-num-layers", type=int, default=8, help="Transformer depth.")
    parser.add_argument("--platonic-ffn-dim-factor", type=int, default=4, help="Expansion factor for the MLP tower.")
    parser.add_argument("--platonic-dropout", type=float, default=0.0, help="Dropout probability.")
    parser.add_argument("--platonic-drop-path", type=float, default=0.0, help="Stochastic depth rate.")
    parser.add_argument("--platonic-mean-aggregation", action="store_true", help="Enable mean aggregation inside attention kernels.")
    parser.add_argument("--platonic-softmax-attention", action="store_true", help="Use softmax attention instead of linear kernels.")
    parser.add_argument("--platonic-rope-sigma", type=float, default=1.0, help="Frequency scale for rotary embeddings.")
    parser.add_argument("--platonic-fixed-freqs", action="store_true", help="Freeze rotary frequencies (defaults to learnable).")
    parser.add_argument("--num-train-timesteps", type=int, default=1000, help="Diffusion training steps for the scheduler.")
    parser.add_argument("--beta-start", type=float, default=0.0001, help="Beta schedule start value.")
    parser.add_argument("--beta-end", type=float, default=0.02, help="Beta schedule end value.")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="Beta schedule name.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of reverse-diffusion sampling steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = EquivarianceViewer(args)
    viewer.run()


if __name__ == "__main__":
    main()
    def _stack_point_cloud(self, sequence: List[List[dict]]) -> dict[str, torch.Tensor]:
        points_list: list[torch.Tensor] = []
        colors_list: list[torch.Tensor] = []
        for frame_views in sequence:
            merged_points = torch.cat([view["points"] for view in frame_views], dim=0)
            merged_colors = torch.cat([view["colors"] for view in frame_views], dim=0)
            points_list.append(merged_points.to(dtype=torch.float32))
            colors_list.append(merged_colors.to(dtype=torch.float32))
        points = torch.stack(points_list, dim=0)
        colors = torch.stack(colors_list, dim=0)
        return {"points": points, "colors": colors}
