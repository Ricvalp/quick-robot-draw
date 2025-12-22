import os

import faiss
import numpy as np
from absl import app
from ml_collections import config_flags
from tqdm import tqdm


def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="configs/metrics/build_faiss_index.py"
)


def main(_):

    cfg = load_cfgs(_CONFIG_FILE)

    out_dir = cfg.out_dir
    emb_dir = out_dir + "/embeddings"
    index_dir = out_dir + "/faiss_index"

    os.makedirs(index_dir, exist_ok=True)

    for fname in tqdm(os.listdir(emb_dir)):
        if not fname.endswith(".npy"):
            continue

        family = fname.replace("family_", "").replace(".npy", "")
        emb_path = os.path.join(emb_dir, fname)

        X = np.load(emb_path).astype("float32")  # (N, 512)

        d = X.shape[1]

        index = faiss.IndexFlatIP(d)
        index.add(X)

        faiss.write_index(
            index,
            os.path.join(index_dir, f"family_{family}.index"),
        )

        print(f"[OK] family={family}, n={len(X)}")

    print("All done.")


if __name__ == "__main__":
    app.run(main)
