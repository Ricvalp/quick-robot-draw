import os
import io
import torch
import webdataset as wds
import tqdm

from ml_collections import config_flags

from glob import glob
from torch.utils.data import DataLoader

from fid import EpisodeToImage
from dataset import QuickDrawEpisodes


def write_shards(raw_dataset, collate_fn, out_dir, shard_size=5000):
    """
    Convert a raw dataset into preprocessed WebDataset shards.

    raw_dataset: dataset where __getitem__ returns a *raw* unprocessed sample
    collate_fn : your full collator (will be applied to each sample individually)
    out_dir    : directory where shards are saved
    shard_size : number of samples per .tar shard
    """
    os.makedirs(out_dir, exist_ok=True)

    sink = wds.ShardWriter(
        pattern=f"{out_dir}/shard-%06d.tar",
        maxcount=shard_size
    )

    for i in tqdm.trange(len(raw_dataset)):
        raw = raw_dataset[i]

        # Your collate_fn expects a *list* (batch)
        processed = collate_fn([raw])   # returns dict with shapes like [1, T, C]

        key = f"{i:08d}"
        wds_sample = {"__key__": key}

        # Save each tensor field as a .pt buffer
        for k, v in processed.items():
            buf = io.BytesIO()
            torch.save(v, buf)
            wds_sample[f"{k}.pt"] = buf.getvalue()

        sink.write(wds_sample)

    sink.close()


_CONFIG_FILE = config_flags.DEFINE_config_file("task", default="fid/config.py")

def main(_):
    cfg = load_cfgs(_CONFIG_FILE)

    raw_dataset = QuickDrawEpisodes(split="train")
    collate_fn = EpisodeToImage(image_size=64)

    out_dir = "data/cached_quickdraw/train"
    write_shards(raw_dataset, collate_fn, out_dir, shard_size=5000)


def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg


if __name__ == "__main__":
    import absl.app as app

    app.run(main)