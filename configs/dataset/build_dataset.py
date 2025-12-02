from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.root = "data/single_class/eyeglasses-simplified"
    cfg.raw_root = "raw/"
    cfg.backend = "lmdb"  # one of: lmdb, webdataset, hdf5
    cfg.num_prompts = 5
    cfg.max_seq_len = 512
    cfg.normalize = True
    cfg.num_workers = 0
    cfg.max_files = None

    cfg.resample = ConfigDict()
    cfg.resample.points = None
    cfg.resample.spacing = None

    cfg.simplify = ConfigDict()
    cfg.simplify.enabled = False
    cfg.simplify.epsilon = 1.0

    cfg.augmentations = ConfigDict()
    cfg.augmentations.rotation = True
    cfg.augmentations.scale = True
    cfg.augmentations.translation = True
    cfg.augmentations.jitter = False

    cfg.storage = ConfigDict()
    cfg.storage.compression = "zstd"
    cfg.storage.shards = 64

    cfg.families = ["eyeglasses"]
    cfg.max_sketches_per_file = 100000

    cfg.cache = ConfigDict()
    cfg.cache.sketches_subdir = "processed_sketches"
    cfg.cache.episodes_subdir = "episodes"

    cfg.seed = 1234

    cfg.split = ConfigDict()
    cfg.split.train_ratio = 1.0
    cfg.split.val_ratio = 0.0
    cfg.split.test_ratio = 0.0

    cfg.num_prebuilt_episodes = 0
    cfg.rebuild = False

    return cfg
