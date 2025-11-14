from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:

    cfg = ConfigDict()

    cfg.seeds = tuple(list(range(20)))
    
    cfg.train = ConfigDict()
    cfg.train.start_idx = 0
    cfg.train.end_idx = 50
    cfg.train.num_steps = 500
    cfg.train.num_parallel_nefs = 5000
    cfg.train.masked_portion = 1.0
    cfg.train.multi_gpu = False
    cfg.train.train_to_target_psnr = config_dict.placeholder(float)
    cfg.train.check_every = 10
    cfg.train.fixed_init = True
    cfg.train.verbose = True
    
    return cfg