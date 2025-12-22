from ml_collections import ConfigDict


def get_config() -> ConfigDict:

    cfg = ConfigDict()
    cfg.dataset_path = "data/all-classes/train-val-split/"
    cfg.out_dir = "metrics/index/"
    cfg.checkpoint_path = "metrics/checkpoints/resnet18_step40000.pt"
    cfg.family_to_label_path = "metrics/family_to_label.yaml"
    cfg.num_workers = 16
    cfg.backend = "lmdb"

    return cfg
