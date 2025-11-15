from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:

    cfg = ConfigDict()

    cfg.seeds = tuple(list(range(20)))
    
    cfg.train = ConfigDict()
    cfg.train.train_to_target_psnr = config_dict.placeholder(float)
    
    return cfg