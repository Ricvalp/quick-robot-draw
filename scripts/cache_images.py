import io
import os

import torch
import webdataset as wds
import yaml
from ml_collections import config_flags
from tqdm import tqdm

from dataset import QuickDrawEpisodes, RasterizerConfig
from fid import EpisodeToImage


def write_shards(
    raw_dataset,
    image_fn,
    family_to_label_dict,
    out_dir,
    shard_size=5000,
    split_fracs={"train": 0.9, "val": 0.1},
    seed=42,
):
    """
    Convert a raw dataset into preprocessed WebDataset shards.

    raw_dataset: dataset where __getitem__ returns a *raw* unprocessed sample
    collate_fn : your full collator (will be applied to each sample individually)
    out_dir    : directory where shards are saved
    shard_size : number of samples per .tar shard
    """

    for split in split_fracs.keys():
        os.makedirs(out_dir + f"/{split}", exist_ok=True)

    writers = {
        split: wds.ShardWriter(f"{out_dir}/{split}/shard-%06d.tar", maxcount=shard_size)
        for split in split_fracs
    }
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(raw_dataset), generator=rng).tolist()
    fam_counts = {
        fam: len(samples) for fam, samples in raw_dataset.family_to_samples.items()
    }
    val_budget = {
        fam: max(1, int(fam_counts[fam] * split_fracs["val"])) for fam in fam_counts
    }

    for idx in tqdm(perm):
        processed = image_fn(raw_dataset[idx])
        fam = processed["family"]
        split = "val" if val_budget[fam] > 0 else "train"
        if split == "val":
            val_budget[fam] -= 1

        key = f"{idx:08d}"
        wds_sample = {"__key__": key}

        buf = io.BytesIO()
        img = processed["img"]
        torch.save(img, buf)
        wds_sample["img.pt"] = buf.getvalue()

        buf = io.BytesIO()
        label = family_to_label_dict[processed["family"]]
        torch.save(torch.tensor(label).unsqueeze(0), buf)
        wds_sample["label.pt"] = buf.getvalue()

        writers[split].write(wds_sample)

    for split in writers:
        writers[split].close()


_CONFIG_FILE = config_flags.DEFINE_config_file("config", default="fid/config.py")


def main(_):
    cfg = load_cfgs(_CONFIG_FILE)

    rasterizer_config = RasterizerConfig(**cfg.rasterizer_config)

    with open(cfg.family_to_label_path, "r") as f:
        family_to_label_dict = yaml.safe_load(f)

    raw_dataset = QuickDrawEpisodes(
        root=cfg.dataset_path,
        split="train",
        K=0,
        augment=False,
        coordinate_mode="absolute",
    )
    collate_fn = EpisodeToImage(rasterizer_config=rasterizer_config)

    write_shards(
        raw_dataset,
        image_fn=collate_fn,
        family_to_label_dict=family_to_label_dict,
        out_dir=cfg.output_dir,
        shard_size=cfg.shard_size,
        split_fracs=cfg.split_fracs,
        seed=cfg.seed,
    )


def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg


if __name__ == "__main__":
    import absl.app as app

    app.run(main)
