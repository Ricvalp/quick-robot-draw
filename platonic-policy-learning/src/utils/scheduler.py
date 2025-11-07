"""Learning-rate scheduling utilities shared across training scripts."""

from __future__ import annotations

import math
from dataclasses import dataclass
import torch


@dataclass
class WarmupCosineLRSchedulerConfig:
    """Configuration for a warmup followed by cosine decay scheduler."""

    total_steps: int  # Number of optimisation steps across the full run.
    warmup_steps: int = 0  # Linear warmup steps starting from 0 -> base LR.
    min_lr_ratio: float = 0.1  # Floor for cosine decay expressed as fraction of base LR.
    last_epoch: int = -1  # PyTorch-compatible last_epoch for resuming.

    def validate(self) -> None:
        """Ensure scheduler parameters are well-formed before use."""

        if self.total_steps <= 0:
            raise ValueError(f"total_steps must be positive, received {self.total_steps}.")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, received {self.warmup_steps}.")
        if self.warmup_steps > self.total_steps:
            raise ValueError(
                "warmup_steps must not exceed total_steps; "
                f"received warmup {self.warmup_steps} > total {self.total_steps}."
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(
                "min_lr_ratio must fall within [0, 1]. "
                f"Received {self.min_lr_ratio}."
            )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: WarmupCosineLRSchedulerConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a LambdaLR that performs linear warmup plus cosine decay.

    Args:
        optimizer: Optimiser whose learning rates will be scheduled.
        config: Hyperparameters controlling warmup length and cosine floor.

    Returns:
        LambdaLR schedule suitable for per-step ``scheduler.step()`` calls.
    """

    config.validate()

    warmup_steps = config.warmup_steps
    total_steps = config.total_steps
    min_lr_ratio = config.min_lr_ratio

    def schedule_lambda(step_index: int) -> float:
        """Piecewise LR multiplier computed for the provided step index."""

        if step_index < warmup_steps:
            warmup_progress = (step_index + 1) / max(1, warmup_steps)
            return warmup_progress

        if warmup_steps == total_steps:
            # Edge case: pure warmup, stay at base LR afterwards.
            return 1.0

        # Normalised progress across the cosine decay phase in [0, 1].
        decay_progress = (step_index - warmup_steps) / max(1, total_steps - warmup_steps)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine_term = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_term

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=schedule_lambda,
        last_epoch=config.last_epoch,
    )
