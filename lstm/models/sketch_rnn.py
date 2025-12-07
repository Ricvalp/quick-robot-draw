"""
SketchRNN implementation for QuickDraw imitation learning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ["SketchRNNConfig", "SketchRNN", "SketchRNNUnconditional"]


@dataclass
class SketchRNNConfig:
    """Configuration container for the SketchRNN architecture."""

    input_dim: int = 7
    output_dim: int = 6
    latent_dim: int = 128
    encoder_hidden: int = 256
    encoder_num_layers: int = 1
    decoder_hidden: int = 512
    decoder_num_layers: int = 1
    num_mixtures: int = 20
    dropout: float = 0.0


class MixtureDensityHead(nn.Module):
    """Project decoder states into 2D Gaussian mixture parameters + pen logits."""

    def __init__(
        self, hidden_size: int, num_mixtures: int, pen_classes: int = 3
    ) -> None:
        super().__init__()
        self.num_mixtures = num_mixtures
        self.pen_classes = pen_classes
        self.proj = nn.Linear(hidden_size, num_mixtures * 6 + pen_classes)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.proj(hidden)
        gauss_raw = raw[..., : self.num_mixtures * 6]
        pen_logits = raw[..., self.num_mixtures * 6 :]

        shape = hidden.shape[:-1] + (self.num_mixtures, 6)
        gauss_raw = gauss_raw.view(shape)

        pi_logits = gauss_raw[..., 0]
        mu_x = gauss_raw[..., 1]
        mu_y = gauss_raw[..., 2]
        log_sigma_x = gauss_raw[..., 3]
        log_sigma_y = gauss_raw[..., 4]
        rho = torch.tanh(gauss_raw[..., 5])

        return {
            "pi_logits": pi_logits,
            "mu_x": mu_x,
            "mu_y": mu_y,
            "log_sigma_x": log_sigma_x,
            "log_sigma_y": log_sigma_y,
            "rho": rho,
            "pen_logits": pen_logits,
        }


class SketchRNN(nn.Module):
    """Bidirectional LSTM encoder + autoregressive decoder for sketch modeling."""

    def __init__(self, cfg: SketchRNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        enc_dropout = cfg.dropout if cfg.encoder_num_layers > 1 else 0.0
        dec_dropout = cfg.dropout if cfg.decoder_num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.encoder_hidden,
            num_layers=cfg.encoder_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=enc_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=cfg.output_dim + cfg.latent_dim,
            hidden_size=cfg.decoder_hidden,
            num_layers=cfg.decoder_num_layers,
            batch_first=True,
            dropout=dec_dropout,
        )

        self.fc_mu = nn.Linear(2 * cfg.encoder_hidden, cfg.latent_dim)
        self.fc_logvar = nn.Linear(2 * cfg.encoder_hidden, cfg.latent_dim)
        self.latent_to_hidden = nn.Linear(
            cfg.latent_dim, cfg.decoder_hidden * cfg.decoder_num_layers
        )
        self.latent_to_cell = nn.Linear(
            cfg.latent_dim, cfg.decoder_hidden * cfg.decoder_num_layers
        )
        self.mdn_head = MixtureDensityHead(cfg.decoder_hidden, cfg.num_mixtures)

    def encode(
        self, strokes: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(
            strokes, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.encoder(packed)
        # Grab the last layer for both directions.
        h_forward = hidden[-2]
        h_backward = hidden[-1]
        embedding = torch.cat([h_forward, h_backward], dim=-1)
        mu = self.fc_mu(embedding)
        logvar = self.fc_logvar(embedding)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _init_decoder_state(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = z.shape[0]
        hidden = self.latent_to_hidden(z).view(
            batch, self.cfg.decoder_num_layers, self.cfg.decoder_hidden
        )
        cell = self.latent_to_cell(z).view(
            batch, self.cfg.decoder_num_layers, self.cfg.decoder_hidden
        )
        hidden = hidden.permute(1, 0, 2).contiguous()
        cell = cell.permute(1, 0, 2).contiguous()
        return hidden, cell

    def _decode_teacher_forcing(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        latent_context = z.unsqueeze(1).expand(-1, inputs.shape[1], -1)
        decoder_inputs = torch.cat([inputs, latent_context], dim=-1)
        packed = pack_padded_sequence(
            decoder_inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.decoder(packed, self._init_decoder_state(z))
        padded, _ = pad_packed_sequence(
            outputs, batch_first=True, total_length=inputs.shape[1]
        )
        return padded

    def _lengths_to_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        indices = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return indices < lengths.unsqueeze(1)

    def compute_loss(
        self,
        queries: torch.Tensor,
        queries_lengths: torch.Tensor,
        contexts: torch.Tensor,
        contexts_lengths: torch.Tensor,
        *,
        kl_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mu, logvar = self.encode(contexts, contexts_lengths)
        z = self.reparameterize(mu, logvar)

        inputs = queries[:, :-1, :]
        targets = queries[:, 1:, :]
        dec_lengths = queries_lengths - 1

        decoder_outputs = self._decode_teacher_forcing(inputs, dec_lengths, z)
        params = self.mdn_head(decoder_outputs)

        mask = self._lengths_to_mask(dec_lengths, inputs.shape[1])
        # mask = self._lengths_to_mask(torch.tile(torch.tensor([inputs.shape[1]]), (inputs.shape[0],)), inputs.shape[1]).to(inputs.device)

        xy = targets[..., :2]
        pen_targets = torch.cat([targets[..., 2:4], targets[..., -1:]], dim=-1)
        pen_targets = torch.argmax(pen_targets, dim=-1)

        mdn_loss = self._mdn_nll(xy, params, mask)
        pen_loss = self._pen_loss(params["pen_logits"], pen_targets, mask)
        recon = mdn_loss + pen_loss
        kl = self._kl_loss(mu, logvar)
        total = recon + kl_weight * kl

        metrics = {
            "loss": float(total.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "nll_xy": float(mdn_loss.detach().cpu()),
            "pen_ce": float(pen_loss.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "kl_weight": float(kl_weight),
        }
        return total, metrics

    def forward(
        self,
        strokes: torch.Tensor,
        lengths: torch.Tensor,
        *,
        kl_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.compute_loss(strokes, lengths, kl_weight=kl_weight)

    def _mdn_nll(
        self,
        targets: torch.Tensor,
        params: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        log_pi = F.log_softmax(params["pi_logits"], dim=-1)
        mu_x = params["mu_x"]
        mu_y = params["mu_y"]
        log_sigma_x = params["log_sigma_x"]
        log_sigma_y = params["log_sigma_y"]
        rho = params["rho"]

        sigma_x = torch.exp(log_sigma_x)
        sigma_y = torch.exp(log_sigma_y)

        x = targets[..., 0].unsqueeze(-1)
        y = targets[..., 1].unsqueeze(-1)

        norm_x = (x - mu_x) / sigma_x
        norm_y = (y - mu_y) / sigma_y
        z = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y

        eps = 1e-6
        denom = 1 - rho**2 + eps
        log_component = (
            log_pi
            - math.log(2 * math.pi)
            - log_sigma_x
            - log_sigma_y
            - 0.5 * torch.log(denom)
            - z / (2 * denom)
        )

        log_prob = torch.logsumexp(log_component, dim=-1)
        nll = -log_prob
        masked = nll * mask
        return masked.sum() / mask.sum().clamp_min(1.0)

    def _pen_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        )
        loss = loss.view_as(mask)
        masked = loss * mask
        return masked.sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()

    @torch.no_grad()
    def sample(
        self,
        num_steps: int,
        contexts: torch.Tensor,
        context_lengths: torch.Tensor,
        unconditional: bool = False,
        deterministic: bool = True,
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        greedy: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        device = next(self.parameters()).device
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(
                torch.randint(0, 2**31 - 1, (1,), device=device).item()
            )

        if unconditional:
            z = torch.randn(
                num_samples, self.cfg.latent_dim, device=device, generator=generator
            )
        elif contexts is not None and context_lengths is not None:
            mu, logvar = self.encode(contexts, context_lengths)
            if deterministic:
                z = mu
            else:
                z = self.reparameterize(mu, logvar)
        else:
            raise ValueError(
                "Either unconditional must be True or contexts and context_lengths must be provided."
            )
        hidden_state = self._init_decoder_state(z)
        prev = torch.zeros(z.shape[0], self.cfg.output_dim, device=device)
        prev[:, 4] = 1.0  # start with pen-up

        # eos_token = prev.clone()

        eos_token = torch.zeros(z.shape[0], 5, device=device)
        # eos_token[:, :] = 0.0
        eos_token[:, -1] = 1.0

        sequences: List[torch.Tensor] = []
        finished = torch.zeros(z.shape[0], dtype=torch.bool, device=device)

        for _ in range(num_steps):
            decoder_in = torch.cat([prev, z], dim=-1).unsqueeze(1)
            output, hidden_state = self.decoder(decoder_in, hidden_state)
            params = self.mdn_head(output.squeeze(1))

            stroke = self._sample_step(params, temperature, greedy, generator)
            stroke[finished] = eos_token[finished]
            sequences.append(stroke.unsqueeze(1))

            just_finished = stroke[:, -1] > 0.5
            finished = finished | just_finished
            prev[:, :4] = stroke[:, :4]
            prev[:, 4] = 0.0  # set sep token to zero
            prev[:, -1] = stroke[:, -1]  # reset EOS to zero for next step

            if bool(finished.all()):
                break

        if not sequences:
            return eos_token.unsqueeze(1)

        return torch.cat(sequences, dim=1)

    def _sample_step(
        self,
        params: Dict[str, torch.Tensor],
        temperature: float,
        greedy: bool,
        generator: torch.Generator,
    ) -> torch.Tensor:
        device = params["mu_x"].device
        pi_logits = params["pi_logits"] / max(temperature, 1e-3)
        if greedy:
            component = pi_logits.argmax(dim=-1)
        else:
            probs = F.softmax(pi_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            component = dist.sample()

        gather_index = component.unsqueeze(-1)
        mu_x = torch.gather(params["mu_x"], dim=-1, index=gather_index).squeeze(-1)
        mu_y = torch.gather(params["mu_y"], dim=-1, index=gather_index).squeeze(-1)
        log_sigma_x = torch.gather(
            params["log_sigma_x"], dim=-1, index=gather_index
        ).squeeze(-1)
        log_sigma_y = torch.gather(
            params["log_sigma_y"], dim=-1, index=gather_index
        ).squeeze(-1)
        rho = torch.gather(params["rho"], dim=-1, index=gather_index).squeeze(-1)

        sigma_x = torch.exp(log_sigma_x) * temperature
        sigma_y = torch.exp(log_sigma_y) * temperature
        eps = 1e-6
        z1 = torch.randn(mu_x.shape, device=device, generator=generator)
        z2 = torch.randn(mu_x.shape, device=device, generator=generator)
        dx = mu_x + sigma_x * z1
        sqrt_term = torch.sqrt(torch.clamp(1 - rho**2, min=eps))
        dy = mu_y + sigma_y * (rho * z1 + sqrt_term * z2)

        pen_logits = params["pen_logits"] / max(temperature, 1e-3)
        if greedy:
            pen_idx = pen_logits.argmax(dim=-1)
        else:
            pen_probs = F.softmax(pen_logits, dim=-1)
            pen_dist = torch.distributions.Categorical(pen_probs)
            pen_idx = pen_dist.sample()
        pen_one_hot = F.one_hot(pen_idx, num_classes=3).float()
        # Ensure EOS rows do not move the cursor.
        eos_mask = pen_one_hot[:, 2] > 0.5
        if eos_mask.any():
            dx = dx.masked_fill(eos_mask, 0.0)
            dy = dy.masked_fill(eos_mask, 0.0)

        stroke = torch.cat(
            [dx.unsqueeze(-1), dy.unsqueeze(-1), pen_one_hot],
            dim=-1,
        )
        return stroke


class SketchRNNUnconditional(SketchRNN):
    """SketchRNN model without encoder for unconditional generation."""

    def __init__(self, cfg: SketchRNNConfig) -> None:
        super().__init__(cfg)

    def forward(
        self,
        strokes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass

    def compute_loss(
        self,
        strokes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pass

    def sample(
        self,
        num_steps: int,
        *,
        num_samples: int = 1,
        temperature: float = 1.0,
        greedy: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        pass

    def _sample_step(self, params, temperature, greedy, generator):
        pass
