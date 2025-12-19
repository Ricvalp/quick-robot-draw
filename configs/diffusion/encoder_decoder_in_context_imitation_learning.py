from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:

    cfg = ConfigDict()
    cfg.run = ConfigDict()
    cfg.run.seed = 2026
    cfg.run.device = "cuda"  # or "cpu"

    cfg.data = ConfigDict()
    cfg.data.root = config_dict.placeholder(str)
    cfg.data.split = "train"
    cfg.data.backend = "lmdb"
    cfg.data.K = 4
    cfg.data.max_query_len = 60
    cfg.data.max_context_len = 300
    cfg.data.max_seq_len = 360
    cfg.data.coordinate_mode = "absolute"

    cfg.loader = ConfigDict()
    cfg.loader.batch_size = 128
    cfg.loader.num_workers = 16

    cfg.training = ConfigDict()
    cfg.training.epochs = 100
    cfg.training.lr = 1e-4
    cfg.training.weight_decay = 0.0

    cfg.logging = ConfigDict()
    cfg.logging.loss_log_every = 100

    cfg.model = ConfigDict()
    cfg.model.input_dim = 6
    cfg.model.output_dim = 6
    cfg.model.num_train_timesteps = 1000
    cfg.model.beta_start = 1e-4
    cfg.model.beta_end = 2e-2
    cfg.model.beta_schedule = "scaled_linear"
    cfg.model.hidden_dim = 512
    cfg.model.num_layers = 4
    cfg.model.num_heads = 4
    cfg.model.mlp_dim = 1024
    cfg.model.dropout = 0.0
    cfg.model.attention_dropout = 0.0
    cfg.model.horizon = 8

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "diffusion/checkpoints"
    cfg.checkpoint.save_interval = 10

    cfg.eval = ConfigDict()
    cfg.eval.samples = 64
    cfg.eval.seed = 42
    cfg.eval.num_inference_steps = 300

    cfg.profiling = ConfigDict()
    cfg.profiling.use = False
    cfg.profiling.trace_dir = "profiling/diffusion/"

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "diffusion-imitation-learning"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.log_interval = 200
    cfg.wandb.log_all = False

    return cfg
