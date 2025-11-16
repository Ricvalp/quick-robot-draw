import os
import io
import torch
import webdataset as wds
import tqdm
import yaml

from ml_collections import config_flags

from glob import glob
from torch.utils.data import DataLoader

from fid import EpisodeToImage
from dataset import QuickDrawEpisodes, RasterizerConfig


def write_shards(raw_dataset, image_fn, family_to_label_dict, out_dir, shard_size=5000):
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

        processed = image_fn(raw)

        key = f"{i:08d}"
        wds_sample = {"__key__": key}
        
        buf = io.BytesIO()
        img = processed['img']
        torch.save(img, buf)
        wds_sample['img.pt'] = buf.getvalue()
        
        buf = io.BytesIO()
        label = family_to_label_dict[processed['family']]
        torch.save(torch.tensor(label).unsqueeze(0), buf)
        wds_sample['label.pt'] = buf.getvalue()

        sink.write(wds_sample)

    sink.close()

_CONFIG_FILE = config_flags.DEFINE_config_file("config", default="fid/config.py")

def main(_):
    cfg = load_cfgs(_CONFIG_FILE)

    rasterizer_config = RasterizerConfig(
        **cfg.rasterizer_config
        )

    with open(cfg.family_to_label_path, 'r') as f:
        family_to_label_dict = yaml.safe_load(f)
    
    raw_dataset = QuickDrawEpisodes(
        root=cfg.dataset_path,
        split="train",
        K=0,
        augment=False,
        coordinate_mode="absolute",
        )
    collate_fn = EpisodeToImage(
        rasterizer_config=rasterizer_config
    )

    write_shards(
        raw_dataset,
        image_fn=collate_fn,
        family_to_label_dict=family_to_label_dict,
        out_dir=cfg.output_dir,
        shard_size=cfg.shard_size,
    )

def load_cfgs(
    _CONFIG_FILE,
):
    cfg = _CONFIG_FILE.value

    return cfg


if __name__ == "__main__":
    import absl.app as app

    app.run(main)