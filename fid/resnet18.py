import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Identity()  # Remove the final classification layer
    
    def forward(self, x):
        return self.model(x)
    
