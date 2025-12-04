from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:

    cfg = ConfigDict()
    cfg.seed = 2024
    cfg.data_root = config_dict.placeholder(str)
    cfg.device = "cuda"  # or "cpu"
    cfg.split = "train"
    cfg.backend = "lmdb"
    cfg.K = 4
    cfg.batch_size = 64
    cfg.epochs = 100
    cfg.lr = 1e-3
    cfg.input_dim = 7
    cfg.weight_decay = 0.0
    cfg.num_workers = 16
    cfg.max_seq_len = 350
    cfg.coordinate_mode = "delta"
    cfg.encoder_hidden = 256
    cfg.encoder_num_layers = 1
    cfg.decoder_hidden = 512
    cfg.decoder_num_layers = 1
    cfg.latent_dim = 128
    cfg.num_mixtures = 20
    cfg.dropout = 0.0
    cfg.kl_start = 0.0
    cfg.kl_end = 1.0
    cfg.kl_anneal_steps = 20000
    cfg.grad_clip = 1.0
    cfg.checkpoint_dir = "lstm/checkpoints"
    cfg.save_interval = 10
    cfg.eval_samples = 4
    cfg.eval_steps = 1000
    cfg.eval_interval = 1
    cfg.eval_temperature = 0.65
    cfg.eval_seed = 42
    cfg.greedy_eval = False
    cfg.profile = False
    cfg.trace_dir = "profiling/lstm/"

    cfg.wandb = ConfigDict()
    cfg.wandb.use = False
    cfg.wandb.project = "lstm-imitation-learning"
    cfg.wandb.entity = "your_entity"
    cfg.wandb.run = None
    cfg.wandb.log_interval = 200
    cfg.wandb.log_all = False

    return cfg
