from datetime import datetime

from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:

    cfg = ConfigDict()
    cfg.run = ConfigDict()
    cfg.run.seed = 2026
    cfg.run.device = "cuda"

    cfg.data = ConfigDict()
    cfg.data.root = config_dict.placeholder(str)
    cfg.data.split = "train"
    cfg.data.backend = "lmdb"
    cfg.data.K = 4
    cfg.data.max_seq_len = 480
    cfg.data.max_query_len = 60
    cfg.data.max_context_len = 400
    cfg.data.coordinate_mode = "absolute"
    cfg.data.index_dir = "metrics/index/faiss_index/"
    cfg.data.ids_dir = "metrics/index/ids_family/"

    cfg.loader = ConfigDict()
    cfg.loader.batch_size = 256

    cfg.logging = ConfigDict()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.logging.dir = f"figures/diffusion/decoder_only/{timestamp}"

    cfg.checkpoint = ConfigDict()
    cfg.checkpoint.dir = "diffusion/checkpoints/decoder_only"
    cfg.checkpoint.name = config_dict.placeholder(str)
    cfg.checkpoint.epoch = config_dict.placeholder(int)

    cfg.eval = ConfigDict()
    cfg.eval.tasks = ["fid"]
    cfg.eval.samples = 256
    cfg.eval.num_many_samples = 16
    cfg.eval.seed = 42
    cfg.eval.num_inference_steps = 300

    cfg.eval.fid = ConfigDict()
    cfg.eval.fid.num_samples = 5000
    cfg.eval.fid.num_gt_samples = 5000
    cfg.eval.fid.resnet_checkpoint_path = "metrics/checkpoints/resnet18_step40000.pt"

    return cfg
