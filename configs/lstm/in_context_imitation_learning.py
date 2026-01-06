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
    cfg.data.max_seq_len = 480
    cfg.data.max_query_len = 60
    cfg.data.max_context_len = 400
    cfg.data.coordinate_mode = "delta"
    cfg.data.index_dir = "metrics/index/faiss_index/"
    cfg.data.ids_dir = "metrics/index/ids_family/"

    cfg.loader = ConfigDict()
    cfg.loader.batch_size = 256
    cfg.loader.num_workers = 12

    cfg.training = ConfigDict()
    cfg.training.epochs = 1000
    cfg.training.lr = 1e-4
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0

    cfg.training.warmup_cosine_annealing = ConfigDict()
    cfg.training.warmup_cosine_annealing.use = False
    cfg.training.warmup_cosine_annealing.warmup_steps = 5000
    cfg.training.warmup_cosine_annealing.T_max = 20000
    cfg.training.warmup_cosine_annealing.max_lr = 1e-3
    cfg.training.warmup_cosine_annealing.min_lr = 1e-5

    cfg.training.cosine_annealing = ConfigDict()
    cfg.training.cosine_annealing.use = False
    cfg.training.cosine_annealing.T_max = 20000
    cfg.training.cosine_annealing.eta_min = 1e-6

    cfg.model = ConfigDict()
    cfg.model.input_dim = 7
    cfg.model.output_dim = 6
    cfg.model.latent_dim = 128
    cfg.model.encoder_hidden = 256
    cfg.model.encoder_num_layers = 1
    cfg.model.decoder_hidden = 512
    cfg.model.decoder_num_layers = 1
    cfg.model.num_mixtures = 20
    cfg.model.dropout = 0.0
    cfg.model.teacher_forcing_with_eos = False

    cfg.kl = ConfigDict()
    cfg.kl.start = 0.0
    cfg.kl.end = 1.0
    cfg.kl.anneal_steps = 20000

    cfg.logging = ConfigDict()
    cfg.logging.loss_log_every = 100

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "lstm/checkpoints"
    cfg.checkpoint.save_interval = 1

    cfg.eval = ConfigDict()
    cfg.eval.samples = 16
    cfg.eval.steps = 1000
    cfg.eval.temperature = 0.65
    cfg.eval.seed = 42
    cfg.eval.greedy = False
    cfg.eval.eval_on_train = False

    cfg.profiling = ConfigDict()
    cfg.profiling.use = False
    cfg.profiling.trace_dir = "profiling/lstm/"

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "lstm-in-context-imitation-learning-sweeps"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.log_interval = 200
    cfg.wandb.log_all = False

    return cfg
