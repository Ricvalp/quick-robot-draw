from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision.models import resnet18

__all__ = [
    "ResNet18FeatureExtractor",
    "compute_fid",
]


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, pretrained_checkpoint_path: Union[str, Path]) -> None:
        super().__init__()
        self.model = resnet18(pretrained=False, num_classes=345)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        state_dict = torch.load(pretrained_checkpoint_path)
        self.model.load_state_dict(state_dict)
        self.model.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, x):
        return self.model(x)


def compute_fid(
    generated_features: torch.Tensor,
    statistics_path: str = None,
    gt_features: dict = None,
) -> float:
    """
    generated_images: Tensor of shape (N, 1, H, W)
    statistics_path: path to .npz file containing 'mu' and 'sigma' of real images
    returns: FID score
    """

    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)

    if gt_features is not None:
        mu_real = np.mean(gt_features, axis=0)
        sigma_real = np.cov(gt_features, rowvar=False)
    elif statistics_path is not None:
        stats = np.load(statistics_path)
        mu_real = stats["mu"]
        sigma_real = stats["sigma"]
    else:
        raise ValueError("Either statistics_path or statistics must be provided.")

    diff = mu_gen - mu_real
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_real, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = diff @ diff + np.trace(sigma_gen + sigma_real - 2 * covmean)
    return fid_score
