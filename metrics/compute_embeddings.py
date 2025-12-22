import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import config_flags
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset import QuickDrawSketches, RasterizerConfig, SketchStorage, StorageConfig
from metrics import ResNet18FeatureExtractor, SketchToImageCollate


def compute_embeddings(
    raw_dataset: Dataset,
    embedding_model: nn.Module,
    family: str,
    rasterizer_config: RasterizerConfig,
    out_dir: str,
    device: str = "cuda",
    num_workers: int = 0,
):

    sketch_to_image_collate = SketchToImageCollate(rasterizer_config=rasterizer_config)
    loader = torch.utils.data.DataLoader(
        raw_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sketch_to_image_collate,
    )

    embeddings = []
    ids = []

    with torch.no_grad():
        for batch in tqdm(loader):

            imgs = batch["imgs"]
            sketch_ids = batch["sketch_ids"]

            imgs = imgs.unsqueeze(1).to(device)
            emb = embedding_model(imgs)
            emb = F.normalize(emb, dim=1)

            emb = emb.cpu().numpy()

            for e, sid in zip(emb, sketch_ids):
                embeddings.append(e)
                ids.append(int(sid))

    os.makedirs(out_dir + "/embeddings", exist_ok=True)
    os.makedirs(out_dir + "/ids_family", exist_ok=True)

    np.save(f"{out_dir}/embeddings/family_{family}.npy", np.vstack(embeddings))
    np.save(f"{out_dir}/ids_family/{family}.npy", np.array(ids))


def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg


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

    embedding_model = ResNet18FeatureExtractor(
        prertained_checkpoint_path=cfg.checkpoint_path
    )
    embedding_model.eval()

    root = cfg.dataset_path

    storage_config = StorageConfig(root=root, backend=cfg.backend)
    tmp_sketch_storage = SketchStorage(storage_config, mode="r")
    all_families = tmp_sketch_storage.families()
    family_to_samples: Dict[str, List[str]] = {}

    assigned_families = all_families
    for family in assigned_families:
        samples = tmp_sketch_storage.samples_for_family(family)
        if len(samples) > 0:
            family_to_samples[family] = samples

    tmp_sketch_storage.close()
    family_ids = sorted(family_to_samples.keys())

    for family in family_ids:

        raw_dataset = QuickDrawSketches(
            family=family,
            family_samples=family_to_samples[family],
            storage_config=storage_config,
        )

        compute_embeddings(
            raw_dataset=raw_dataset,
            family=family,
            embedding_model=embedding_model.to("cuda"),
            rasterizer_config=rasterizer_config,
            out_dir=cfg.out_dir,
            device="cuda",
            num_workers=cfg.num_workers,
        )


if __name__ == "__main__":
    import absl.app as app

    app.run(main)
