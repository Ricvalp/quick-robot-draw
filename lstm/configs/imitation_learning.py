from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    
    cfg = ConfigDict()
    cfg.seed = 2024
    cfg.data_root = config_dict.placeholder(str)
    cfg.split = "train"
    cfg.backend = "lmdb"
    cfg.K = 5
    cfg.batch_size = 64
    cfg.epochs = 100
    cfg.lr = 1e-3
    cfg.weight_decay = 0.0
    cfg.num_workers = 16
    cfg.max_seq_len = 512
    cfg.coordinate_mode = "delta"
    cfg.encoder_hidden = 256
    cfg.encoder_layers = 1
    cfg.decoder_hidden = 512
    cfg.decoder_layers = 1
    cfg.latent_dim = 128
    cfg.num_mixtures = 20
    cfg.dropout = 0.0
    cfg.kl_start = 0.1
    cfg.kl_end = 1.0
    cfg.kl_anneal_steps = 20000
    cfg.grad_clip = 1.0
    cfg.checkpoint_dir = "lstm/checkpoints"
    cfg.save_interval = 10
    cfg.eval_samples = 4
    cfg.eval_steps = 200
    cfg.eval_interval = 1
    cfg.eval_temperature = 0.65
    cfg.eval_seed = 42
    cfg.greedy_eval = False
    cfg.profile = False
    cfg.trace_dir = "profiling/lstm/"
    
    cfg.wandb_logging = ConfigDict()
    cfg.wandb_logging.use = False
    cfg.wandb_logging.project = "lstm-imitation-learning"
    cfg.wandb_logging.entity = "your_entity"
    cfg.wandb_logging.run = None
    cfg.wandb_logging.log_interval = 200
    cfg.wandb_logging.log_all = False
    
    return cfg