import torch
from torch import nn
from ml_collections import config_flags

from torch.utils.data import DataLoader
from fid.dataset import QuickDrawImages, ImageCollatror

from fid.resnet18 import ResNet18FeatureExtractor

_CONFIG_FILE = config_flags.DEFINE_config_file("task", default="fid/config.py")


def main(_):
    cfg = load_cfgs(_CONFIG_FILE)
    
    dataset = QuickDrawImages(
        root="data/imgs/",
        seed=cfg.seed,
    )
    
    collator = ImageCollatror()
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
    )

def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg