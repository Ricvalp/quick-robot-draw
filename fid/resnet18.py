import torch
import torch.nn as nn
from torchvision.models import resnet18
from scipy import linalg
import numpy as np

__all__ = [
    "ResNet18FeatureExtractor",
]


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Identity()  # Remove the final classification layer
    
    def forward(self, x):
        return self.model(x)
    
def fid(images: torch.Tensor, resnet_checkpoint: str) -> torch.Tensor:
    """
    images: Tensor of shape (B, 1, H, W), grayscale images
    returns: Tensor of shape (B, 512), feature vectors
    """
    device = images.device
    model = ResNet18FeatureExtractor()
    state_dict = torch.load(resnet_checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()

def compute_fid(generated_images: torch.Tensor, statistics_path: str) -> float:
    """
    generated_images: Tensor of shape (N, 1, H, W)
    statistics_path: path to .npz file containing 'mu' and 'sigma' of real images
    returns: FID score
    """

    gen_features = fid(generated_images).cpu().numpy()

    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    stats = np.load(statistics_path)
    mu_real = stats['mu']
    sigma_real = stats['sigma']

    diff = mu_gen - mu_real
    covmean, _ = linalg.sqrtm(sigma_gen @ sigma_real, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = diff @ diff + np.trace(sigma_gen + sigma_real - 2 * covmean)
    return fid_score