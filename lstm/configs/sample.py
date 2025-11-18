from ml_collections import ConfigDict, config_dict

def get_config() -> ConfigDict:
    cfg = ConfigDict()
    
    cfg.checkpoint_dir = config_dict.placeholder(str)
    cfg.num_samples = 16
    cfg.steps = 256
    cfg.temperature = 0.65
    cfg.greedy = False
    cfg.device = "cuda"
    cfg.seed = 7
    cfg.output_dir = "lstm/samples"
    cfg.prefix = "imitation-learning-sketch"

    return cfg