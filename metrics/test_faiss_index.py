import random

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from absl import app
from ml_collections import config_flags

from dataset import SketchStorage, StorageConfig
from metrics import ResNet18FeatureExtractor, SketchToImage


def load_config(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg


def load_family_index(out_dir, family):
    index = faiss.read_index(f"{out_dir}faiss_index/family_{family}.index")
    ids = np.load(f"{out_dir}ids_family/{family}.npy")
    return index, ids


def embed_query(img, embedding_model, device="cuda"):
    with torch.no_grad():
        img = img.unsqueeze(1).to(device)  # (1,1,64,64)
        emb = embedding_model(img)
        emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy()  # (1,512)


def plot_sketches(key_sketch, closest_sketches, name=None):

    import matplotlib.pyplot as plt

    n = len(closest_sketches) + 1
    plt.figure(figsize=(3 * n, 3))

    plt.subplot(1, n, 1)
    plt.title("Query Sketch")
    plt.imshow(key_sketch[0], cmap="gray")
    plt.axis("off")

    for i, sketch in enumerate(closest_sketches):
        plt.subplot(1, n, i + 2)
        plt.title(f"Closest #{i+1}")
        plt.imshow(sketch[0], cmap="gray")
        plt.axis("off")

    if name:
        plt.savefig(name)
    else:
        plt.show()


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="configs/metrics/build_faiss_index.py"
)
_RASTERIZER_CONFIG = config_flags.DEFINE_config_file(
    "rasterizer_config", default="configs/metrics/cache.py"
)


def main(_):

    cfg = load_config(_CONFIG_FILE)
    rasterizer_config = load_config(_RASTERIZER_CONFIG).rasterizer_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    family = "axe"
    index, ids = load_family_index(cfg.out_dir, family)

    sketch_to_image = SketchToImage(rasterizer_config=rasterizer_config)

    embedding_model = ResNet18FeatureExtractor(
        pretrained_checkpoint_path=cfg.checkpoint_path
    ).to(device)
    embedding_model.eval()

    storage_config = StorageConfig(
        root=cfg.dataset_path,
        backend=cfg.backend,
    )
    sketch_storage = SketchStorage(config=storage_config)
    sketches_ids = sketch_storage.samples_for_family(family)

    for sketch_id in sketches_ids[:10]:
        key_sketch = sketch_storage.get(family, sketch_id)
        absolute = torch.tensor(key_sketch.absolute)
        pen = torch.tensor(key_sketch.pen)
        tokens = torch.cat([absolute, pen.unsqueeze(-1)], dim=-1)
        key_img = sketch_to_image(
            {
                "tokens": tokens,
                "family_id": "",
                "sketch_id": "",
            }
        )
        key_img = key_img["img"].to(device)
        q = embed_query(key_img, embedding_model)
        _, idxs = index.search(q, k=5)
        closest_sketch_ids = ids[idxs[0]]

        closest_sketches = [
            sketch_storage.get(family, sid) for sid in closest_sketch_ids
        ]

        closest_imgs = []
        for s in closest_sketches:
            absolute = torch.tensor(s.absolute)
            pen = torch.tensor(s.pen)
            tokens = torch.cat([absolute, pen.unsqueeze(-1)], dim=-1)
            img = sketch_to_image(
                {
                    "tokens": tokens,
                    "family_id": "",
                    "sketch_id": "",
                }
            )
            closest_imgs.append(img["img"].cpu())

        plot_sketches(
            key_img.cpu(),
            closest_imgs,
            name=f"figures/closest_sketches_{sketch_id}.png",
        )

    for sketch_id in sketches_ids[:10]:
        key_sketch = sketch_storage.get(family, sketch_id)

        absolute = torch.tensor(key_sketch.absolute)
        pen = torch.tensor(key_sketch.pen)
        tokens = torch.cat([absolute, pen.unsqueeze(-1)], dim=-1)
        key_sketch = {
            "tokens": tokens,
            "family_id": "",
            "sketch_id": "",
        }
        key_img = sketch_to_image(key_sketch)["img"]

        random_sketches = [
            sketch_storage.get(family, sid) for sid in random.sample(sketches_ids, 5)
        ]

        random_imgs = []
        for s in random_sketches:
            absolute = torch.tensor(s.absolute)
            pen = torch.tensor(s.pen)
            tokens = torch.cat([absolute, pen.unsqueeze(-1)], dim=-1)
            img = sketch_to_image(
                {
                    "tokens": tokens,
                    "family_id": "",
                    "sketch_id": "",
                }
            )
            random_imgs.append(img["img"].cpu())
        plot_sketches(
            key_img.cpu(), random_imgs, name=f"figures/random_sketches_{sketch_id}.png"
        )


if __name__ == "__main__":
    app.run(main)
