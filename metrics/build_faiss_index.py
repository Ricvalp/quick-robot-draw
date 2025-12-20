import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ml_collections import config_flags
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset import QuickDrawEpisodes, RasterizerConfig
from metrics import EpisodeToImageCollate, ResNet18FeatureExtractor


def compute_embeddings(
    raw_dataset: Dataset,
    embedding_model: nn.Module,
    rasterizer_config: RasterizerConfig,
    family_to_label_dict: Dict[str, int],
    out_dir: str,
    device: str = "cuda",
    num_workers: int = 0,
):

    episode_to_image_collate = EpisodeToImageCollate(
        rasterizer_config=rasterizer_config
    )
    loader = torch.utils.data.DataLoader(
        raw_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=episode_to_image_collate,
    )

    embeddings = defaultdict(list)
    ids = defaultdict(list)

    with torch.no_grad():
        for imgs, families, sketch_ids in tqdm(loader):
            imgs = imgs.to(device)  # (B,1,64,64)
            emb = embedding_model(imgs)  # (B,512)
            emb = F.normalize(emb, dim=1)  # cosine similarity

            emb = emb.cpu().numpy()
            families = families.numpy()
            sketch_ids = sketch_ids.numpy()

            for e, f, sid in zip(emb, families, sketch_ids):
                embeddings[f].append(e)
                ids[f].append(sid)

    os.makedirs(out_dir, exist_ok=True)
    for f in embeddings:
        np.save(f"{out_dir}/embeddings_family_{f}.npy", np.vstack(embeddings[f]))
        np.save(f"{out_dir}/ids_family_{f}.npy", np.array(ids[f]))


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="configs/metrics/build_faiss_index.py"
)
_RASTERIZER_CONFIG = config_flags.DEFINE_config_file(
    "rasterizer_config", default="configs/metrics/cache.py"
)


def main(_):
    cfg = load_cfgs(_CONFIG_FILE)

    rasterizer_config = load_cfgs(_RASTERIZER_CONFIG).rasterizer_config
    rasterizer_config = RasterizerConfig(**rasterizer_config)

    with open(cfg.family_to_label_path, "r") as f:
        family_to_label_dict = yaml.safe_load(f)

    raw_dataset = QuickDrawEpisodes(
        root=cfg.dataset_path,
        split="train",
        K=0,
        augment=False,
        coordinate_mode="absolute",
    )

    embedding_model = ResNet18FeatureExtractor(
        prertained_checkpoint_path=cfg.checkpoint_path
    )
    embedding_model.eval()

    compute_embeddings(
        raw_dataset=raw_dataset,
        embedding_model=embedding_model.to("cuda"),
        rasterizer_config=rasterizer_config,
        family_to_label_dict=family_to_label_dict,
        out_dir=cfg.out_dir,
        device="cuda",
        num_workers=cfg.num_workers,
    )


def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg


if __name__ == "__main__":
    import absl.app as app

    app.run(main)
