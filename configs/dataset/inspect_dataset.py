from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    cfg = ConfigDict()

    cfg.root = "data/single_class/eyeglasses-simplified"
    cfg.raw_root = "raw/"
    cfg.backend = "lmdb"  # one of: lmdb, webdataset, hdf5
    cfg.families = config_dict.FieldReference(
        (), field_type=tuple
    )  # override as a tuple: ("eyeglasses",)
    cfg.num_sketches_per_family = 4
    cfg.coordinate_mode = "delta"
    cfg.output_dir = "figures/inspect"
    cfg.seed = 123

    return cfg
