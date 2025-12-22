from ml_collections import ConfigDict


def get_config() -> ConfigDict:

    cfg = ConfigDict()
    cfg.run = ConfigDict()
    cfg.run.seed = 2026

    cfg.data = ConfigDict()
    cfg.data.root = "data/all-classes/train-val-split/"
    cfg.data.backend = "lmdb"

    cfg.loader = ConfigDict()
    cfg.loader.batch_size = 128
    cfg.loader.num_workers = 0

    return cfg
