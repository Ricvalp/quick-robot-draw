from ml_collections import ConfigDict


def get_config() -> ConfigDict:

    cfg = ConfigDict()
    cfg.run = ConfigDict()
    cfg.run.seed = 2026

    cfg.data = ConfigDict()
    cfg.data.root = "data/all-classes/train-val-split/"
    cfg.data.split = "train"
    cfg.data.backend = "lmdb"
    cfg.data.K = 4
    cfg.data.max_query_len = 60
    cfg.data.max_context_len = 300
    cfg.data.max_seq_len = 360

    cfg.diffusion_model = ConfigDict()
    cfg.diffusion_model.horizon = 8

    cfg.loader = ConfigDict()
    cfg.loader.batch_size = 128
    cfg.loader.num_workers = 0

    return cfg
