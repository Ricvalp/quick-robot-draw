from ml_collections import ConfigDict, config_flags
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm
from pathlib import Path
import numpy as np

from fid import get_cached_loader
from fid.fid_resnet18 import ResNet18FeatureExtractor, compute_fid


def load_cfgs(_CONFIG_FILE: str) -> ConfigDict:
    cfg = _CONFIG_FILE.value
    return cfg


_CONFIG_FILE = config_flags.DEFINE_config_file("config", default="fid/configs/train.py")


def main(_):
    
    cfg = load_cfgs(_CONFIG_FILE)
    
    val_dataloader = get_cached_loader(
    shard_glob=cfg.data_dir + "/val/*",
    batch_size=512,
    num_workers=3,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet_checkpoint = Path(cfg.checkpoint_dir) / cfg.fid.checkpoint_filename
    model = ResNet18FeatureExtractor(
        prertained_checkpoint_path=str(resnet_checkpoint)
    )

    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        all_features = []
        all_features_fid = []
        labels = []
        for i, batch in tqdm(enumerate(val_dataloader)):
            images = batch["img"].unsqueeze(1).to(device)  # add channel dimension
            features = model(images)
            if i < 20:
                all_features.append(features)
                labels.append(batch["label"])
            if i >= 20:
                all_features_fid.append(features)
                if i >= 40:
                    break
    
    all_features = torch.cat(all_features, dim=0).cpu().numpy()
    all_features_fid = torch.cat(all_features_fid, dim=0).cpu().numpy()
    
    labels = torch.cat(labels, dim=0)
    tsne_results = TSNE(n_components=2, init="pca", perplexity=10.0).fit_transform(all_features)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.title(f"ResNet18 Feature TSNE on Validation Set: dim={all_features.shape[1]}")
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels.numpy(), cmap='tab10', alpha=0.7)
    plt.savefig("ResNet18-val-tsne.png")
    
    FID = compute_fid(
        generated_features=all_features_fid,
        statistics={"mu": np.mean(all_features, axis=0), "sigma": np.cov(all_features, rowvar=False)},
    )

    print(f"FID={FID}")
    
if __name__ == "__main__":
    from absl import app
    app.run(main)