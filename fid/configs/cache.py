from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:

    cfg = ConfigDict()

    cfg.seed = 37
    cfg.dataset_path = config_dict.placeholder(str)
    cfg.output_dir = config_dict.placeholder(str)
    cfg.family_to_label_path = config_dict.placeholder(str)
    cfg.checkpoint_dir = "fid/checkpoints/"
    cfg.shard_size = 100000
    cfg.split_fracs = {'train': 0.8, 'val': 0.1}
    
    cfg.rasterizer_config = ConfigDict()
    cfg.rasterizer_config.img_size = 64
    cfg.rasterizer_config.antialias = 2
    cfg.rasterizer_config.line_width = 2.0
    cfg.rasterizer_config.background_value = 0.0
    cfg.rasterizer_config.stroke_value = 1.0
    cfg.rasterizer_config.normalize_inputs = False
    
    return cfg