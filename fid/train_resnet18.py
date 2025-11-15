import torch
from torch import nn
from ml_collections import config_flags

from torch.utils.data import DataLoader
from fid import get_cached_loader

from fid.resnet18 import ResNet18FeatureExtractor

_CONFIG_FILE = config_flags.DEFINE_config_file("task", default="fid/config.py")

def main(_):
    cfg = load_cfgs(_CONFIG_FILE)

    dataloader = get_cached_loader(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg


if __name__ == "__main__":
    import absl.app as app

    app.run(main)