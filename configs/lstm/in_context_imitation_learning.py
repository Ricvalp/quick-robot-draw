from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:

    cfg = ConfigDict()
    cfg.run = ConfigDict()
    cfg.run.seed = 2024
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

    cfg.loader = ConfigDict()
    cfg.loader.batch_size = 64
    cfg.loader.num_workers = 16

    cfg.training = ConfigDict()
    cfg.training.epochs = 100
    cfg.training.lr = 1e-3
    cfg.training.weight_decay = 0.0
    cfg.training.grad_clip = 1.0

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
    cfg.checkpoint.save_interval = 10

    cfg.eval = ConfigDict()
    cfg.eval.samples = 4
    cfg.eval.steps = 1000
    cfg.eval.interval = 5
    cfg.eval.temperature = 0.65
    cfg.eval.seed = 42
    cfg.eval.greedy = False
    cfg.eval.eval_on_train = False

    cfg.profiling = ConfigDict()
    cfg.profiling.use = False
    cfg.profiling.trace_dir = "profiling/lstm/"

    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "lstm-imitation-learning"
    cfg.wandb.entity = "ricvalp"
    cfg.wandb.log_interval = 200
    cfg.wandb.log_all = False

    return cfg
