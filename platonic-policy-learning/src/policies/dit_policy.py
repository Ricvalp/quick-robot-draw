"""Minimal diffusion policy built around the DiT encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from src.models.dit import DiffusionTransformer, DiffusionTransformerConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        half_dim = max(1, dim // 2)
        inv_freq = base ** (-torch.arange(half_dim, dtype=torch.float32) / max(1, half_dim))  # (half_dim,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            indices = indices.unsqueeze(1)  # (B, 1)
        values = indices.to(self.inv_freq.dtype)  # (B, N)
        angles = values.unsqueeze(-1) * self.inv_freq  # (B, N, half_dim)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, N, min(dim, 2*half_dim))
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(*emb.shape[:-1], self.dim - emb.shape[-1], device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=-1)  # (B, N, dim)
        return emb


@dataclass
class DiTDiffusionPolicyConfig:
    # Number of observation tokens the policy attends over before action horizon.
    context_length: int
    # Length of the action rollout the policy predicts per sample.
    horizon: int
    # Number of features provided per point (position + colour).
    point_feature_dim: int
    # Dimensionality of each proprioceptive vector.
    proprio_dim: int
    # Dimensionality of each action vector.
    action_dim: int
    # Hidden size shared across transformer, embeddings, and output head.
    hidden_dim: int
    # Number of transformer blocks stacked in the DiT backbone.
    num_layers: int
    # Count of attention heads per block.
    num_heads: int
    # Width of the MLP sub-layer inside each transformer block.
    mlp_dim: int
    # Dropout controlling residual and feedforward stochasticity.
    dropout: float = 0.0
    # Dropout specific to attention probability matrices.
    attention_dropout: float = 0.0
    # Activation function used by the MLP branch.
    activation: str = "gelu"
    # Whether LayerNorm precedes the attention/MLP (AdaLN-Zero prefers True).
    norm_first: bool = True
    # Numerical epsilon fed into LayerNorm for stability.
    layer_norm_eps: float = 1e-5
    # Hidden dimension for simple linear embeddings of scalar signals.
    scalar_embedding_hidden_dim: int = 128
    # Base frequency for the absolute time positional encoding.
    time_embedding_base: float = 10000.0
    # Base frequency for the diffusion timestep encoding.
    diffusion_embedding_base: float = 10000.0
    # Number of sampling steps retained for later inference usage.
    num_inference_steps: int = 50
    # Parameters forwarded into the diffusers DDPM scheduler.
    noise_scheduler_kwargs: Dict[str, object] | None = None


class DiTDiffusionPolicy(nn.Module):
    def __init__(self, cfg: DiTDiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Translate policy hyper-parameters into a DiT backbone configuration.
        transformer_cfg = DiffusionTransformerConfig(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation=cfg.activation,
            norm_first=cfg.norm_first,
            layer_norm_eps=cfg.layer_norm_eps,
        )
        self.transformer = DiffusionTransformer(transformer_cfg)

        # Linear projections turn observations and actions into token embeddings.
        if cfg.proprio_dim <= 0:
            raise ValueError("proprio_dim must be positive.")
        self.proprio_encoder = nn.Linear(cfg.proprio_dim, cfg.hidden_dim)
        if cfg.point_feature_dim <= 0:
            raise ValueError("point_feature_dim must be positive.")
        self.point_feature_proj = nn.Sequential(
            nn.Linear(cfg.point_feature_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_encoder = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

        # Positional encoders tag each token with absolute time information.
        self.world_time_embedder = SinusoidalTimeEmbedding(cfg.hidden_dim, base=cfg.time_embedding_base)
        self.diffusion_time_embedder = SinusoidalTimeEmbedding(
            cfg.hidden_dim, base=cfg.diffusion_embedding_base
        )
        self.diffusion_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        # Instantiate a DDPM scheduler with user-provided hyperparameters.
        scheduler_kwargs = dict(cfg.noise_scheduler_kwargs or {})
        self.scheduler = DDPMScheduler(**scheduler_kwargs)
        self.num_inference_steps = cfg.num_inference_steps

        # Pre-compute absolute time indices for observations and actions.
        # Time indices anchor observations in [-H+1, 0] and actions in [1, horizon].
        context_idx = torch.arange(
            -cfg.context_length + 1,
            1,
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, N_time)
        self.register_buffer("context_time_indices", context_idx, persistent=False)
        action_idx = torch.arange(
            1,
            cfg.horizon + 1,
            dtype=torch.float32,
        ).unsqueeze(0)  # (1, N_action)
        self.register_buffer("action_time_indices", action_idx, persistent=False)

    # ------------------------------------------------------------------
    def _encode_context(
        self,
        proprio: torch.Tensor,
        points: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """Build the observation token block (proprio + point cloud).

        Args:
            proprio: (B, N_time, proprio_dim) proprio sequence.
            points: (B, N_time, N_points, 3) point cloud positions.
            colors: (B, N_time, N_points, 3) point colours.

        Returns:
            context: (B, N_observation_tokens, hidden_dim) observation tokens.
        """
        batch_size, num_frames = proprio.shape[:2]

        # Absolute indices span [-H+1, 0] so the model knows where each frame falls in history.
        frame_indices = self.context_time_indices.to(
            device=proprio.device,
            dtype=torch.float32,
        )  # (1, N_time)
        frame_indices = frame_indices.expand(batch_size, -1)  # (B, N_time)
        frame_time_emb = self.world_time_embedder(frame_indices)  # (B, N_time, hidden_dim)

        # Encode proprio signals and inject the frame-wise temporal metadata.
        proprio_tokens = self.proprio_encoder(proprio)  # (B, N_time, hidden_dim)
        proprio_tokens = proprio_tokens + frame_time_emb  # (B, N_time, hidden_dim)

        # Combine per-point geometry + RGB, project into hidden width, and tag with time.
        num_points = points.shape[2]  # number of points per frame
        point_features = torch.cat([points, colors], dim=-1)  # (B, N_time, N_points, point_feature_dim)
        point_tokens = self.point_feature_proj(point_features)  # (B, N_time, N_points, hidden_dim)
        point_tokens = point_tokens + frame_time_emb.unsqueeze(-2)  # broadcast frame metadata → (B, N_time, N_points, hidden_dim)
        point_tokens = point_tokens.reshape(batch_size, num_frames * num_points, -1)  # (B, N_time*N_points, hidden_dim)

        # Pack proprio followed by flattened point cloud tokens for downstream attention.
        return torch.cat([proprio_tokens, point_tokens], dim=1)  # (B, N_observation_tokens, hidden_dim)

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed noisy actions into the shared transformer width.

        Args:
            actions: (B, N_action, action_dim) noisy action samples.

        Returns:
            tokens: (B, N_action, hidden_dim) time-embedded action tokens.
        """
        tokens = self.action_encoder(actions)  # (B, N_action, hidden_dim)
        batch = tokens.shape[0]
        times = self.action_time_indices.to(device=actions.device, dtype=torch.float32)
        times = times.expand(batch, -1)  # (B, N_action)
        time_emb = self.world_time_embedder(times)  # (B, N_action, hidden_dim)
        return tokens + time_emb  # (B, N_action, hidden_dim)

    def _diffusion_condition(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Encode diffusion step indices for AdaLN conditioning.

        Args:
            timesteps: (B,) integer diffusion step indices.

        Returns:
            cond: (B, hidden_dim) conditioning embeddings.
        """
        emb = self.diffusion_time_embedder(timesteps.float().unsqueeze(1))[:, 0, :]  # (B, hidden_dim)
        return self.diffusion_proj(emb)  # (B, hidden_dim)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the diffusion training loss for a mini-batch.

        Args:
            batch: Dict with keys `proprio`, `actions`, and either `observation`
                or legacy `point_cloud`.

        Returns:
            loss: Scalar MSE between predicted and sampled noise.
            metrics: Dictionary containing logging scalars.
        """
        proprio = batch["proprio"]
        observation = batch.get("observation")
        if observation is None:
            observation = batch["point_cloud"]
        points = observation["positions"]
        colors = observation["colors"]
        actions = batch["actions"]

        if proprio.shape[1] != self.cfg.context_length:
            raise ValueError(
                f"Expected context length {self.cfg.context_length}, "
                f"got {proprio.shape[1]}."
            )
        if points.shape[1] != self.cfg.context_length:
            raise ValueError(
                f"Point cloud context mismatch: expected {self.cfg.context_length}, "
                f"got {points.shape[1]}."
            )

        # Sample standard Gaussian noise for the forward diffusion process.
        noise = torch.randn_like(actions)  # (B, N_action, action_dim)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
            dtype=torch.long,
        )  # (B,)
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)  # (B, N_action, action_dim)

        # Convert clean observations and perturbed actions into joint token sequence.
        context_tokens = self._encode_context(proprio, points, colors)  # (B, N_observation_tokens, hidden_dim)
        action_tokens = self._encode_actions(noisy_actions)  # (B, N_action, hidden_dim)
        tokens = torch.cat([context_tokens, action_tokens], dim=1)  # (B, N_observation_tokens + N_action, hidden_dim)

        # Condition the DiT backbone on the diffusion timestep embedding.
        diffusion_cond = self._diffusion_condition(timesteps)  # (B, hidden_dim)
        encoded = self.transformer(tokens, diffusion_time_cond=diffusion_cond)  # (B, N_observation_tokens + N_action, hidden_dim)

        # Take the tail slice corresponding to action tokens and predict the de-noised residual.
        pred = self.output_head(encoded[:, -self.cfg.horizon :, :])  # (B, N_action, action_dim)

        # Compare the network prediction to the original Gaussian noise; return scalars for logging.
        loss = F.mse_loss(pred, noise)  # scalar
        metrics = {"mse": float(loss.detach().cpu())}
        return loss, metrics

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.compute_loss(batch)

    def sample_actions(
        self,
        proprio: torch.Tensor,
        observation: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Run reverse diffusion to sample an action sequence.

        Args:
            proprio: (B, N_time, proprio_dim) proprio history.
            observation: dict containing
                - positions: (B, N_time, N_points, 3)
                - colors: (B, N_time, N_points, 3)
            generator: optional PRNG for deterministic sampling.

        Returns:
            actions: (B, N_action, action_dim) denoised actions.
        """
        # Infer batch sizing so the sampler emits a matching action rollout.
        batch_size = proprio.shape[0]  # ()
        device = proprio.device  # torch.device

        if proprio.shape[1] != self.cfg.context_length:
            raise ValueError(
                f"Expected context length {self.cfg.context_length}, "
                f"got {proprio.shape[1]}."
            )
        points = observation["positions"]
        colors = observation["colors"]
        if points.shape[1] != self.cfg.context_length:
            raise ValueError(
                f"Point cloud context mismatch: expected {self.cfg.context_length}, "
                f"got {points.shape[1]}."
            )

        # Prepare a Gaussian starting point for the reverse diffusion trajectory.
        sample = torch.randn(
            (batch_size, self.cfg.horizon, self.cfg.action_dim),
            generator=generator,
            device=device,
        )  # (B, N_action, action_dim)

        # Encode observations once since they stay fixed throughout denoising.
        context_tokens = self._encode_context(
            proprio, points, colors
        )  # (B, N_observation_tokens, hidden_dim)

        # Configure the scheduler timesteps for inference.
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)  # (N_diffusion_steps,)

        # Replay the reverse diffusion process until we reach the clean action manifold.
        for timestep in self.scheduler.timesteps:  # (num_steps,)
            # Broadcast the scalar timestep across the batch for conditioning.
            timesteps = torch.full(
                (batch_size,),
                timestep,
                device=device,
                dtype=torch.long,
            )  # (B,)

            # Encode the current noisy action tokens under the DiT backbone space.
            action_tokens = self._encode_actions(sample)  # (B, N_action, hidden_dim)

            # Concatenate observation and action tokens for joint attention.
            tokens = torch.cat([context_tokens, action_tokens], dim=1)  # (B, N_observation_tokens + N_action, hidden_dim)

            # Generate the adaptive LayerNorm modulation from the diffusion timestep.
            diffusion_cond = self._diffusion_condition(timesteps)  # (B, hidden_dim)

            # Run the DiT transformer to predict the score (noise) at this timestep.
            encoded = self.transformer(
                tokens,
                diffusion_time_cond=diffusion_cond,
            )  # (B, N_observation_tokens + N_action, hidden_dim)

            # Project the action slice back into the action space to recover noise estimates.
            noise_pred = self.output_head(encoded[:, -self.cfg.horizon :, :])  # (B, N_action, action_dim)

            # Apply the scheduler update to obtain the next, slightly cleaner action sample.
            scheduler_output = self.scheduler.step(
                noise_pred,
                timestep,
                sample,
                generator=generator,
            )  # prev_sample: (B, N_action, action_dim)

            sample = scheduler_output.prev_sample  # (B, N_action, action_dim)

        # Return the denoised actions matching the dataset's action tensor layout.
        return sample  # (B, N_action, action_dim)


__all__ = ["DiTDiffusionPolicy", "DiTDiffusionPolicyConfig"]
