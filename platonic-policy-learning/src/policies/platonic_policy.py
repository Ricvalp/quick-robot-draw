"""Diffusion policy built on top of the Platonic Transformer backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from src.models.platonic_transformer import DensePlatonicTransformer
from src.utils.geometry import (
    quaternion_to_frame,
    frame_to_quaternion,
)


class SinusoidalTimeEmbedding(nn.Module):
    """Encode discrete time indices into fixed sinusoidal embeddings.

    Args:
        dim: Target embedding dimensionality.
        base: Geometric base controlling sinusoid frequencies.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Expected positive embedding dim, got {dim}.")

        # Pre-compute inverse frequencies so we only perform broadcasting at runtime.
        half_dim = max(1, dim // 2)
        freq_range = torch.arange(half_dim, dtype=torch.float32)  # (half_dim,)
        inv_freq = base ** (-freq_range / float(max(1, half_dim)))  # (half_dim,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert integer time indices to embeddings.

        Args:
            indices: (B, N) or (B,) tensor of scalar time indices.

        Returns:
            embeddings: (B, N, dim) sinusoidal encodings.
        """
        if indices.ndim == 1:
            indices = indices.unsqueeze(1)  # (B, 1)

        values = indices.to(self.inv_freq.dtype)  # (B, N)
        angles = values.unsqueeze(-1) * self.inv_freq  # (B, N, half_dim)
        sin_cos = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, N, 2 * half_dim)

        if sin_cos.shape[-1] < self.dim:
            pad = torch.zeros(
                *sin_cos.shape[:-1],
                self.dim - sin_cos.shape[-1],
                device=sin_cos.device,
                dtype=sin_cos.dtype,
            )  # (B, N, dim - 2*half_dim)
            sin_cos = torch.cat([sin_cos, pad], dim=-1)  # (B, N, dim)
        return sin_cos


# ---------------------------------------------------------------------------
def _pose_to_components(
    pose: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose robot poses into scalar, orientation, and position slots.

    Args:
        pose: (..., 8) tensor in `[x, y, z, qx, qy, qz, qw, grasp]` layout.

    Returns:
        grasp: (..., 1) gripper openness scalars.
        orientation: (..., 2, 3) forward/up axes representing the end-effector frame.
        position: (..., 3) Cartesian end-effector positions.
    """
    position = pose[..., :3]  # (..., 3)
    quaternion = pose[..., 3:7]  # (..., 4)
    grasp = pose[..., 7:8]  # (..., 1)
    forward, _, up = quaternion_to_frame(quaternion)  # each (..., 3)
    orientation = torch.stack([forward, up], dim=-2)  # (..., 2, 3)
    return grasp, orientation, position


def _components_to_pose(
    grasp: torch.Tensor,
    orientation: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the standard pose tensor from component slots.

    Args:
        grasp: (..., 1) gripper openness scalars.
        orientation: (..., 2, 3) forward/up axes spanning the local frame.
        position: (..., 3) Cartesian positions.

    Returns:
        pose: (..., 8) tensor `[x, y, z, qx, qy, qz, qw, grasp]`.
    """
    forward = orientation[..., 0, :]  # (..., 3)
    up = orientation[..., 1, :]  # (..., 3)
    quaternion = frame_to_quaternion(forward, up)  # (..., 4)
    return torch.cat([position, quaternion, grasp], dim=-1)  # (..., 8)


def _pack_components(
    grasp: torch.Tensor,
    orientation: torch.Tensor,
    position: torch.Tensor,
) -> torch.Tensor:
    """Flatten component features into the diffusion scheduler layout.

    Args:
        grasp: (B, N, 1) gripper openness scalars.
        orientation: (B, N, 2, 3) end-effector frame axes.
        position: (B, N, 3) Cartesian positions.

    Returns:
        packed: (B, N, 10) tensor `[orientation(6), position(3), grasp(1)]`.
    """
    orientation_flat = orientation.reshape(*orientation.shape[:-2], -1)  # (..., 6)
    return torch.cat([orientation_flat, position, grasp], dim=-1)  # (..., 10)


def _unpack_components(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`_pack_components`.

    Args:
        tensor: (B, N, 10) packed action tensor.

    Returns:
        grasp: (B, N, 1) scalars.
        orientation: (B, N, 2, 3) orientation axes.
        position: (B, N, 3) positions.
    """
    orientation_flat = tensor[..., :6]
    position = tensor[..., 6:9]
    grasp = tensor[..., 9:]
    orientation = orientation_flat.view(*tensor.shape[:-1], 2, 3)
    return grasp, orientation, position


# ---------------------------------------------------------------------------
# Policy configuration

@dataclass
class PlatonicDiffusionPolicyConfig:
    # Number of observation frames provided as context tokens.
    context_length: int
    # Action horizon (trajectory length) predicted by the policy.
    horizon: int
    # Transformer hidden size shared across scalar/vector streams.
    hidden_dim: int
    # Depth of the Platonic transformer (count of residual layers).
    num_layers: int
    # Number of attention heads per layer; often equals |G| for symmetry.
    num_heads: int
    # Platonic solid name selecting the equivariant symmetry group.
    solid_name: str
    # Expansion factor inside each feed-forward network branch.
    ffn_dim_factor: int = 4
    # Dropout applied to residual activations.
    dropout: float = 0.0
    # Drop-path rate for stochastic depth regularisation.
    drop_path_rate: float = 0.0
    # Average token outputs instead of taking the final action slice.
    mean_aggregation: bool = False
    # Switch between softmax and linear attention kernels.
    use_softmax_attention: bool = False
    # Scaling for rotary frequencies (RoPE) used by the backbone.
    rope_sigma: float = 1.0
    # Learnable frequency spectrum instead of fixed sinusoidal bands.
    learned_freqs: bool = True
    # Task-level routing for scalar features (e.g. dense vs sparse).
    scalar_task_level: str = "dense"
    # Task-level routing for vector features.
    vector_task_level: str = "dense"
    # Overrides forwarded to the diffusers DDIM scheduler.
    noise_scheduler_kwargs: Dict[str, object] | None = None
    # Number of reverse steps used at inference.
    num_inference_steps: int = 50

    # Count of scalar channels in packed action tokens.
    scalar_channels: int = 4
    # Number of vector-valued feature channels supplied on input.
    input_vector_channels: int = 2
    # Number of vector-valued channels predicted by the transformer head.
    output_vector_channels: int = 3
    # Amount of DDIM stochasticity (0 = fully deterministic sampler).
    ddim_eta: float = 0.0
    # Base frequency for sinusoidal world-time encodings.
    time_embedding_base: float = 10000.0


# ---------------------------------------------------------------------------
# Main policy

class PlatonicDiffusionPolicy(nn.Module):
    """Diffusion policy with Platonic rotational equivariance."""

    def __init__(self, cfg: PlatonicDiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        scheduler_kwargs = dict(cfg.noise_scheduler_kwargs or {})
        scheduler_kwargs.setdefault("prediction_type", "epsilon")
        scheduler_kwargs.setdefault("clip_sample", False)
        # Shared diffusion scheduler for training and sampling.
        self.scheduler = DDIMScheduler(**scheduler_kwargs)
        self.num_inference_steps = cfg.num_inference_steps

        self.transformer = DensePlatonicTransformer(
            input_dim=cfg.scalar_channels,
            input_dim_vec=cfg.input_vector_channels,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.scalar_channels,
            output_dim_vec=cfg.output_vector_channels,
            nhead=cfg.num_heads,
            num_layers=cfg.num_layers,
            solid_name=cfg.solid_name,
            dropout=cfg.dropout,
            drop_path_rate=cfg.drop_path_rate,
            mean_aggregation=cfg.mean_aggregation,
            attention=cfg.use_softmax_attention,
            ffn_dim_factor=cfg.ffn_dim_factor,
            rope_sigma=cfg.rope_sigma,
            learned_freqs=cfg.learned_freqs,
            scalar_task_level=cfg.scalar_task_level,
            vector_task_level=cfg.vector_task_level,
            time_conditioning=True,
        )

        # Separate scalar embedders for proprio, actions, and per-point colours.
        self.proprio_scalar_encoder = nn.Sequential(
            nn.Linear(1, cfg.scalar_channels),
            nn.SiLU(),
            nn.Linear(cfg.scalar_channels, cfg.scalar_channels),
        )
        self.action_scalar_encoder = nn.Sequential(
            nn.Linear(1, cfg.scalar_channels),
            nn.SiLU(),
            nn.Linear(cfg.scalar_channels, cfg.scalar_channels),
        )
        self.action_scalar_decoder = nn.Linear(cfg.scalar_channels, 1)
        self.point_rgb_encoder = nn.Sequential(
            nn.Linear(3, cfg.scalar_channels),
            nn.SiLU(),
            nn.Linear(cfg.scalar_channels, cfg.scalar_channels),
        )

        # Pre-compute deterministic time bases so we can tag each token with its world index.
        self.world_time_embedder = SinusoidalTimeEmbedding(
            dim=cfg.scalar_channels,
            base=cfg.time_embedding_base,
        )
        context_time = torch.arange(
            -cfg.context_length + 1,
            1,
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, N_context)
        self.register_buffer("context_time_indices", context_time, persistent=False)
        action_time = torch.arange(
            1,
            cfg.horizon + 1,
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, N_action)
        self.register_buffer("action_time_indices", action_time, persistent=False)

        # Orientation occupies two vector channels; channel 0 remains reserved for positions.
        self.orientation_channels = self.cfg.input_vector_channels
        if self.cfg.output_vector_channels < self.orientation_channels + 1:
            raise ValueError("output_vector_channels must be at least orientation_channels + 1.")
        self.packed_action_dim = self.orientation_channels * 3 + 3 + 1  # (orientation, position, gripper)

    # ------------------------------------------------------------------
    # Utility helpers

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_context_tokens(
        self,
        proprio: torch.Tensor,
        point_cloud: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine proprio and point cloud observations into transformer tokens.

        Args:
            proprio: (B, N_time, 8) proprioceptive pose history.
            point_cloud: optional dict with
                - positions: (B, N_time, N_points, 3)
                - colors: (B, N_time, N_points, 3)

        Returns:
            obs_scalars: (B, N_context, scalar_channels) scalar tokens.
            obs_vectors: (B, N_context, input_vector_channels, 3) vector tokens.
            obs_positions: (B, N_context, 3) relative coordinates.
            anchor: (B, 3) translation offset removed from all positions.
        """
        proprio_gripper, proprio_orientation, proprio_position = _pose_to_components(
            proprio
        )  # (B, N_time, 1), (B, N_time, 2, 3), (B, N_time, 3)

        # Use the first frame as the reference origin so every token becomes translation invariant.
        anchor = proprio_position[:, 0, :].clone()  # (B, 3)
        proprio_position = proprio_position - anchor[:, None, :]  # (B, N_time, 3)

        B, N_time = proprio.shape[:2]
        device = proprio.device
        dtype = proprio.dtype

        # Build absolute time embeddings so the transformer can disambiguate history order.
        context_times = self.context_time_indices.to(device=device, dtype=torch.float32)  # [-H+1, 0] indices → (1, N_time)
        context_times = context_times.expand(B, -1)  # (B, N_time)
        context_time_emb = self.world_time_embedder(context_times).to(dtype=dtype)  # (B, N_time, scalar_channels)

        # Embed gripper signals; orientation axes fill the vector stream.
        proprio_scalars = self.proprio_scalar_encoder(proprio_gripper)  # (B, N_time, scalar_channels)
        proprio_scalars = proprio_scalars + context_time_emb  # tag tokens with absolute time indices
        proprio_vectors = torch.zeros(
            B,
            N_time,
            self.cfg.input_vector_channels,
            3,
            device=device,
            dtype=dtype,
        )
        proprio_vectors[..., : self.orientation_channels, :] = proprio_orientation  # (B, N_time, 2, 3)

        # Bring point cloud payloads onto the same device/dtype as the model.
        points = point_cloud["positions"].to(device=device, dtype=dtype)
        colors = point_cloud["colors"].to(device=device, dtype=dtype)
        B, N_time, N_points, _ = points.shape

        # Centre each point around the shared anchor before flattening across time.
        points = points - anchor[:, None, None, :]  # (B, N_time, N_points, 3)
        point_time_emb = context_time_emb.unsqueeze(-2).expand(
            B,
            N_time,
            N_points,
            -1,
        ).reshape(B, N_time * N_points, -1)  # broadcast frame metadata across each point → (B, N_time * N_points, scalar_channels)

        point_scalars = self.point_rgb_encoder(
            colors.reshape(B, N_time * N_points, 3)
        )  # (B, N_time * N_points, scalar_channels)
        point_scalars = point_scalars + point_time_emb  # share time indices across each point set
        point_vectors = torch.zeros(
            B,
            N_time * N_points,
            self.cfg.input_vector_channels,
            3,
            device=device,
            dtype=dtype,
        )
        point_positions = points.reshape(B, N_time * N_points, 3)  # (B, N_time * N_points, 3)

        # Concatenate proprio and point cloud tokens along the sequence axis.
        combined_scalars = torch.cat([proprio_scalars, point_scalars], dim=1)
        combined_vectors = torch.cat([proprio_vectors, point_vectors], dim=1)
        combined_positions = torch.cat([proprio_position, point_positions], dim=1)

        return combined_scalars, combined_vectors, combined_positions, anchor

    def _build_action_tokens(
        self,
        gripper: torch.Tensor,
        orientation: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble action token features with time information.

        Args:
            gripper: (B, N_action, 1) noisy gripper scalars.
            orientation: (B, N_action, 2, 3) end-effector frame axes.
            positions: (B, N_action, 3) positions relative to the anchor.

        Returns:
            action_scalars: (B, N_action, scalar_channels) scalar token features.
            action_vectors: (B, N_action, input_vector_channels, 3) vector token features.
            positions: (B, N_action, 3) passthrough positions for RoPE.
        """
        assert gripper.ndim == 3 and gripper.shape[2] == 1, "Expected gripper shape (B, N_action, 1)."
        assert orientation.ndim == 4 and orientation.shape[2] == 2 and orientation.shape[3] == 3, "Expected orientation shape (B, N_action, 2, 3)."
        assert positions.ndim == 3 and positions.shape[2] == 3, "Expected positions shape (B, N_action, 3)."

        batch_size, num_tokens = gripper.shape[:2]

        # Encode scalar signals and attach the future-looking time embedding.
        action_scalars = self.action_scalar_encoder(gripper)  # (B, N_action, scalar_channels)
        action_times = self.action_time_indices.to(
            device=action_scalars.device, dtype=torch.float32)  # (1, N_action)
        action_times = action_times.expand(batch_size, -1)  # (B, N_action)
        action_time_emb = self.world_time_embedder(action_times).to(dtype=action_scalars.dtype)  # (B, N_action, scalar_channels)
        action_scalars = action_scalars + action_time_emb  # (B, N_action, scalar_channels)

        # Populate the vector stream with orientation axes; reserve channel 0 for positions.
        action_vectors = torch.zeros(
            batch_size,
            num_tokens,
            self.cfg.input_vector_channels,
            3,
            device=orientation.device,
            dtype=orientation.dtype,
        )  # (B, N_action, C_vec_in, 3)
        action_vectors[..., : self.orientation_channels, :] = orientation  # insert orientation channels

        return action_scalars, action_vectors, positions

    def _split_actions(
        self,
        actions: torch.Tensor,
        anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert absolute action poses into anchor-centred components.

        Args:
            actions: (B, N_action, 8) pose trajectory.
            anchor: (B, 3) translation offset captured from the context window.

        Returns:
            gripper: (B, N_action, 1) gripper openness scalars.
            orientation: (B, N_action, 2, 3) orientation axes.
            positions: (B, N_action, 3) positions expressed relative to the anchor.
        """
        gripper, orientation, positions = _pose_to_components(actions)  # (B, N_action, 1 / 2 / 3)
        positions = positions - anchor[:, None, :]  # (B, N_action, 3)
        return gripper, orientation, positions

    # ------------------------------------------------------------------
    # Diffusion interface

    def _predict_diffusion_residual(
        self,
        obs_scalars: torch.Tensor,
        obs_vectors: torch.Tensor,
        obs_positions: torch.Tensor,
        action_scalars: torch.Tensor,
        action_vectors: torch.Tensor,
        action_positions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the diffusion residual for the action tokens.

        Args:
            obs_scalars: (B, N_context, scalar_channels) context scalar tokens.
            obs_vectors: (B, N_context, input_vector_channels, 3) context vector tokens.
            obs_positions: (B, N_context, 3) relative positions for RoPE.
            action_scalars: (B, N_action, scalar_channels) noisy action scalar tokens.
            action_vectors: (B, N_action, input_vector_channels, 3) noisy action vector tokens.
            action_positions: (B, N_action, 3) noisy relative positions.
            timesteps: (B,) diffusion step indices.

        Returns:
            packed_noise: (B, N_action, 10) residual packed via `_pack_components`.
        """
        # Concatenate context tokens with the (noisy) action tokens.
        tokens_scalars = torch.cat(
            [obs_scalars, action_scalars],
            dim=1,
        )  # (B, N_context + N_action, scalar_channels)
        tokens_vectors = torch.cat(
            [obs_vectors, action_vectors],
            dim=1,
        )  # (B, N_context + N_action, C_vec_in, 3)
        token_positions = torch.cat([obs_positions, action_positions], dim=1)  # (B, N_context + N_action, 3)

        scalar_pred, vector_pred = self.transformer(
            scalars=tokens_scalars,
            pos=token_positions,
            vec=tokens_vectors,
            time_conditioning=timesteps,
        )

        # Transformer returns predictions for both context + horizon tokens.
        pred_scalar_features = scalar_pred[:, -self.cfg.horizon :, :]  # (B, N_action, scalar_channels)
        pred_vectors = vector_pred[
            :,
            -self.cfg.horizon :,
            :,
            :,
        ]  # (B, N_action, output_vector_channels, 3)
        pred_gripper = self.action_scalar_decoder(pred_scalar_features)  # (B, N_action, 1)
        pred_orientation = pred_vectors[..., 1:, :]  # (B, N_action, 2, 3)
        pred_positions = pred_vectors[..., 0, :]  # (B, N_action, 3)
        return _pack_components(pred_gripper, pred_orientation, pred_positions)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the diffusion training objective for a mini-batch.

        Args:
            batch: Dict containing
                - proprio: (B, N_time, 8)
                - actions: (B, N_action, 8)
                - point_cloud: optional dict as in `_build_context_tokens`.

        Returns:
            loss: Scalar MSE over the predicted diffusion residual.
            metrics: Dictionary of logging scalars.
        """
        proprio = batch["proprio"]
        point_cloud = batch.get("point_cloud")
        actions = batch["actions"]

        obs_scalars, obs_vectors, obs_positions, anchor = self._build_context_tokens(
            proprio,
            point_cloud,
        )
        action_gripper, action_orientation, action_positions = self._split_actions(actions, anchor)

        # Pack orientation, position, and scalar slots so the scheduler operates in a flat space.
        action_model = _pack_components(action_gripper, action_orientation, action_positions)
        noise = torch.randn_like(action_model)

        # Sample diffusion timesteps and run forward diffusion.
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
            dtype=torch.long,
        )
        noisy_actions = self.scheduler.add_noise(
            action_model,
            noise,
            timesteps,
        )  # (B, N_action, 10)
        noisy_gripper, noisy_orientation, noisy_positions = _unpack_components(noisy_actions)
        noisy_action_scalars, noisy_action_vectors, noisy_action_positions = self._build_action_tokens(
            gripper=noisy_gripper,
            orientation=noisy_orientation,
            positions=noisy_positions,
        )

        # Predict the Gaussian noise residual under the Platonic transformer.
        pred_noise = self._predict_diffusion_residual(
            obs_scalars,
            obs_vectors,
            obs_positions,
            noisy_action_scalars,
            noisy_action_vectors,
            noisy_action_positions,
            timesteps,
        )

        # Standard diffusion objective: match the sampled noise.
        loss = F.mse_loss(pred_noise, noise)
        metrics = {"mse": float(loss.detach().cpu())}
        return loss, metrics

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.compute_loss(batch)

    def sample_actions(
        self,
        proprio: torch.Tensor,
        point_cloud: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = False,
        return_features: bool = False,
    ) -> torch.Tensor:
        """Sample an action rollout via reverse diffusion.

        Args:
            proprio: (B, N_time, 8) proprio history.
            point_cloud: optional dict as in `_build_context_tokens`.
            generator: optional PRNG for reproducible sampling.
            initial_noise: optional latent `(B, N_action, 10)` to start from.
            deterministic: if True, set `eta=0` for DDIM updates.
            return_features: if True, return `(grip, orientation, position)`.

        Returns:
            actions: (B, N_action, 8) denoised action poses, or component
                tensors when `return_features` is requested.
        """
        # Tokenise observations once; they stay fixed during ancestral sampling.
        obs_scalars, obs_vectors, obs_positions, anchor = self._build_context_tokens(
            proprio,
            point_cloud,
        )

        # Configure the reverse diffusion schedule on the current device.
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            device=self.device,
        )

        # Start from isotropic Gaussian noise in the packed representation.
        if initial_noise is not None:
            sample = initial_noise.to(device=self.device)
        else:
            sample_shape = (
                proprio.shape[0],
                self.cfg.horizon,
                self.packed_action_dim,
            )
            sample = torch.randn(
                sample_shape,
                generator=generator,
                device=self.device,
            )

        # Sweep the configured inference timesteps in reverse order.
        for timestep in self.scheduler.timesteps:
            # Convert packed action state into scalar/vector view for the model.
            noisy_gripper, noisy_orientation, noisy_positions = _unpack_components(
                sample,
            )
            noisy_action_scalars, noisy_action_vectors, noisy_action_positions = self._build_action_tokens(
                gripper=noisy_gripper,
                orientation=noisy_orientation,
                positions=noisy_positions,
            )
            timestep_tensor = torch.full(
                (proprio.shape[0],),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            # Platonic transformer predicts the denoising residual for horizon tokens.
            noise_pred = self._predict_diffusion_residual(
                obs_scalars,
                obs_vectors,
                obs_positions,
                noisy_action_scalars,
                noisy_action_vectors,
                noisy_action_positions,
                timestep_tensor,
            )

            # DDIM update: eta governs stochasticity, while eta=0 recovers the
            # purely deterministic ancestral trajectory.
            eta = 0.0 if deterministic else self.cfg.ddim_eta
            scheduler_step = self.scheduler.step(
                noise_pred,
                int(timestep),
                sample,
                eta=eta,
                generator=generator,
            )
            # Feed the next iterate back into the loop in packed form.
            sample = scheduler_step.prev_sample.to(dtype=proprio.dtype)

        # After finishing the reverse process, convert back into pose tensors.
        final_gripper, final_orientation, final_positions = _unpack_components(
            sample,
        )
        final_positions = final_positions + anchor[:, None, :]
        if return_features:
            return final_gripper, final_orientation, final_positions
        # Convert the final scalar/vector representation back into pose layout.
        actions = _components_to_pose(final_gripper, final_orientation, final_positions)
        return actions


__all__ = [
    "PlatonicDiffusionPolicy",
    "PlatonicDiffusionPolicyConfig",
    "_pose_to_components",
    "_components_to_pose",
    "_pack_components",
    "_unpack_components",
]
