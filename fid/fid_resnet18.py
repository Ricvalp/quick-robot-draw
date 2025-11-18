from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torchvision.models import resnet18
from scipy import linalg
import numpy as np

__all__ = [
    "ResNet18FeatureExtractor",
]


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, prertained_checkpoint_path: Union[str, Path]) -> None:
        super().__init__()
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        state_dict = torch.load(prertained_checkpoint_path)
        self.model.load_state_dict(state_dict)
        self.model.fc = nn.Identity()  # Remove the final classification layer
    
    def forward(self, x):
        return self.model(x)
    

def compute_fid(generated_features: torch.Tensor, statistics_path: str = None, statistics: dict = None) -> float:
    """
    generated_images: Tensor of shape (N, 1, H, W)
    statistics_path: path to .npz file containing 'mu' and 'sigma' of real images
    returns: FID score
    """


    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    stats = np.load(statistics_path) if statistics_path is not None else statistics
    if stats is None:
        raise ValueError("Either statistics_path or statistics must be provided.")
    mu_real = stats['mu']
    sigma_real = stats['sigma']

    diff = mu_gen - mu_real
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_real, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = diff @ diff + np.trace(sigma_gen + sigma_real - 2 * covmean)
    return fid_score