from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import json
import torch


class RunningStat:
    """Tracks running mean and variance for vector-valued samples."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self._sum = torch.zeros(dim, dtype=torch.float64)
        self._sum_sq = torch.zeros(dim, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        data = values.reshape(-1, self.dim).to(dtype=torch.float64)
        self.count += data.shape[0]
        self._sum += data.sum(dim=0)
        self._sum_sq += (data * data).sum(dim=0)

    @property
    def mean(self) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros(self.dim, dtype=torch.float32)
        return (self._sum / self.count).to(dtype=torch.float32)

    @property
    def std(self) -> torch.Tensor:
        if self.count == 0:
            return torch.ones(self.dim, dtype=torch.float32)
        mean = self._sum / self.count
        var = torch.clamp(self._sum_sq / self.count - mean * mean, min=1e-12)
        return var.sqrt().to(dtype=torch.float32)


@dataclass
class NormalizationStats:
    position_mean: torch.Tensor
    position_std: torch.Tensor
    color_mean: torch.Tensor
    color_std: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": {
                "mean": self.position_mean.tolist(),
                "std": self.position_std.tolist(),
            },
            "color": {
                "mean": self.color_mean.tolist(),
                "std": self.color_std.tolist(),
            },
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationStats":
        pos = data["position"]
        col = data["color"]
        return cls(
            position_mean=torch.tensor(pos["mean"], dtype=torch.float32),
            position_std=torch.tensor(pos["std"], dtype=torch.float32),
            color_mean=torch.tensor(col["mean"], dtype=torch.float32),
            color_std=torch.tensor(col["std"], dtype=torch.float32),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "NormalizationStats":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
