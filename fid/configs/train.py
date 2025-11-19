from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:

    cfg = ConfigDict()
    cfg.seed = 42
    cfg.data_dir = config_dict.placeholder(str)
    cfg.checkpoint_dir = config_dict.placeholder(str)
    cfg.batch_size = 256
    cfg.num_workers = 4
    cfg.learning_rate = 1e-3
    cfg.num_epochs = 10
    cfg.log_interval = 100
    cfg.val_steps_per_epoch = 1000
    cfg.save_interval = 10000
    cfg.checkpoint_dir = "fid/checkpoints/"

    cfg.wandb_logging = ConfigDict()
    cfg.wandb_logging.use = False
    cfg.wandb_logging.project = "resnet18-training"
    cfg.wandb_logging.entity = "ricvalp"
    cfg.wandb_logging.log_interval = 200
    cfg.wandb_logging.log_all = False

    cfg.fid = ConfigDict()
    cfg.fid.checkpoint_filename = "resnet18_step100000.pt"

    return cfg
