"""Minimal diffusion policy built around the DiT encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.models.dit import DiffusionTransformer, DiffusionTransformerConfig


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        half_dim = max(1, dim // 2)
        inv_freq = base ** (
            -torch.arange(half_dim, dtype=torch.float32) / max(1, half_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim == 1:
            indices = indices.unsqueeze(1)
        values = indices.to(self.inv_freq.dtype)
        angles = values.unsqueeze(-1) * self.inv_freq
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(
                *emb.shape[:-1],
                self.dim - emb.shape[-1],
                device=emb.device,
                dtype=emb.dtype,
            )
            emb = torch.cat([emb, pad], dim=-1)
        return emb


@dataclass
class DiTDiffusionPolicyConfig:
    context_length: int
    horizon: int
    point_feature_dim: int
    action_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation: str = "gelu"
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    scalar_embedding_hidden_dim: int = 128
    time_embedding_base: float = 10000.0
    diffusion_embedding_base: float = 10000.0
    num_inference_steps: int = 50
    noise_scheduler_kwargs: Dict[str, object] | None = None


class DiTDiffusionPolicy(nn.Module):
    def __init__(self, cfg: DiTDiffusionPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

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

        self.world_time_embedder = SinusoidalTimeEmbedding(
            cfg.hidden_dim, base=cfg.time_embedding_base
        )
        self.diffusion_time_embedder = SinusoidalTimeEmbedding(
            cfg.hidden_dim, base=cfg.diffusion_embedding_base
        )
        self.diffusion_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        scheduler_kwargs = dict(cfg.noise_scheduler_kwargs or {})
        self.scheduler = DDPMScheduler(**scheduler_kwargs)
        self.num_inference_steps = cfg.num_inference_steps

        self.context_time_indices = None
        action_idx = torch.arange(1, cfg.horizon + 1, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("action_time_indices", action_idx, persistent=False)

    def _encode_context(self, points: torch.Tensor) -> torch.Tensor:
        batch_size, num_points = points.shape[:2]
        indices = torch.arange(
            -num_points + 1, 1, device=points.device, dtype=torch.float32
        )
        frame_time_emb = self.world_time_embedder(
            indices.unsqueeze(0).expand(batch_size, -1)
        )
        point_tokens = self.point_feature_proj(points)
        point_tokens = point_tokens + frame_time_emb

        return point_tokens

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        tokens = self.action_encoder(actions)
        batch = tokens.shape[0]
        times = self.action_time_indices.to(
            device=actions.device, dtype=torch.float32
        ).expand(batch, -1)
        time_emb = self.world_time_embedder(times)
        return tokens + time_emb

    def _diffusion_condition(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.diffusion_time_embedder(timesteps.float().unsqueeze(1))[:, 0, :]
        return self.diffusion_proj(emb)

    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        points = batch["points"]
        actions = batch["actions"]
        mask = batch["mask"]

        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
            dtype=torch.long,
        )
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)

        context_tokens = self._encode_context(points)
        action_tokens = self._encode_actions(noisy_actions)

        tokens = torch.cat([context_tokens, action_tokens], dim=1)

        diffusion_cond = self._diffusion_condition(timesteps)
        encoded = self.transformer(
            tokens, key_padding_mask=~mask, diffusion_time_cond=diffusion_cond
        )
        pred = self.output_head(encoded[:, -self.cfg.horizon :, :])

        loss = F.mse_loss(pred, noise)
        metrics = {"mse": float(loss.detach().cpu())}
        return loss, metrics

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.compute_loss(batch)

    def sample_actions(
        self,
        points: Optional[torch.Tensor] = None,
        observation: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Run the reverse diffusion process to synthesize a horizon chunk.

        Args:
            points: Optional context tensor shaped ``(B, T_ctx, point_feature_dim)``.
                When omitted, the method looks for ``"points"`` or ``"context"``
                inside ``observation``. The context provides the already-observed
                tokens that condition the diffusion rollout.
            observation: Optional dictionary that can include pre-padded context
                tensors (``"points"`` or ``"context"``) and an attention mask
                (``"mask"`` or ``"context_mask"``). Masks are expected to be
                boolean tensors where ``True`` marks valid tokens.
            generator: Optional ``torch.Generator`` for deterministic sampling.

        Returns:
            Tensor with shape ``(B, horizon, action_dim)`` containing the
            denoised action tokens for the next horizon window.
        """

        if observation is None:
            observation = {}

        context = observation.get("points")
        if context is None:
            context = observation.get("context")
            
        if context is None:
            context = points
        if context is None:
            raise ValueError("sample_actions requires context points to condition on.")

        batch_size, context_len, _ = context.shape
        device = context.device

        sample = torch.randn(
            (batch_size, self.cfg.horizon, self.cfg.action_dim),
            generator=generator,
            device=device,
        )

        context_tokens = self._encode_context(context)

        mask = observation.get("mask")
        if mask is None:
            mask = observation.get("context_mask")
            if mask is not None and mask.shape[1] != context_len:
                raise ValueError("context_mask must match the context length.")
            if mask is not None:
                mask = mask.to(device=device, dtype=torch.bool)
                mask = torch.cat(
                    [
                        mask,
                        torch.ones(
                            batch_size,
                            self.cfg.horizon,
                            device=device,
                            dtype=torch.bool,
                        ),
                    ],
                    dim=1,
                )
        else:
            mask = mask.to(device=device, dtype=torch.bool)
            if mask.shape[1] == context_len:
                mask = torch.cat(
                    [
                        mask,
                        torch.ones(
                            batch_size,
                            self.cfg.horizon,
                            device=device,
                            dtype=torch.bool,
                        ),
                    ],
                    dim=1,
                )
            elif mask.shape[1] != context_len + self.cfg.horizon:
                raise ValueError(
                    "mask must have length equal to context length or context length + horizon."
                )

        if mask is None:
            mask = torch.ones(
                batch_size,
                context_len + self.cfg.horizon,
                device=device,
                dtype=torch.bool,
            )

        self.scheduler.set_timesteps(self.num_inference_steps, device=device)

        for timestep in self.scheduler.timesteps:
            timesteps = torch.full(
                (batch_size,),
                timestep,
                device=device,
                dtype=torch.long,
            )

            action_tokens = self._encode_actions(sample)
            tokens = torch.cat([context_tokens, action_tokens], dim=1)
            diffusion_cond = self._diffusion_condition(timesteps)
            encoded = self.transformer(
                tokens,
                key_padding_mask=~mask,
                diffusion_time_cond=diffusion_cond,
            )
            noise_pred = self.output_head(encoded[:, -self.cfg.horizon :, :])
            scheduler_step = self.scheduler.step(
                noise_pred,
                timestep,
                sample,
                generator=generator,
            )
            sample = scheduler_step.prev_sample

        return sample


__all__ = ["DiTDiffusionPolicy", "DiTDiffusionPolicyConfig"]
