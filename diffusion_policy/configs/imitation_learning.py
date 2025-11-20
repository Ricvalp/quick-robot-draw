from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.seed = 0
    cfg.device = "cuda"
    cfg.data_root = config_dict.placeholder(str)
    cfg.split = "train"
    cfg.backend = "lmdb"
    cfg.K = 1
    cfg.max_seq_len = 512
    cfg.horizon = 8
    cfg.batch_size = 128
    cfg.epochs = 200
    cfg.lr = 1e-4
    cfg.weight_decay = 0.0
    cfg.num_workers = 4
    cfg.num_train_timesteps = 1000
    cfg.beta_start = 1e-4
    cfg.beta_end = 2e-2
    cfg.beta_schedule = "scaled_linear"
    cfg.hidden_dim = 512
    cfg.num_layers = 4
    cfg.num_heads = 4
    cfg.mlp_dim = 1024
    cfg.dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.num_inference_steps = 50
    cfg.checkpoint_dir = "diffusion_policy/checkpoints"
    cfg.wandb_use=True
    cfg.wandb_project = "in-context imitation learning diffusion"
    cfg.wandb_entity = "ricvalp"
    cfg.loss_log_every = 10
    cfg.eval_samples = 4
    cfg.eval_max_tokens = 128
    cfg.eval_interval = 500
    cfg.eval_sample_seed = 42

    return cfg
