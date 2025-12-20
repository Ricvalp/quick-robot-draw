import io
from glob import glob
from typing import Dict

import torch
import webdataset as wds
from torch.utils.data import DataLoader

from dataset import RasterizerConfig, rasterize_absolute_points

__all__ = [
    "EpisodeToImage",
    "EpisodeToImageCollate",
    "get_cached_loader",
]


class EpisodeToImage:
    """
    Unconditional diffusion collator: treats the entire sketch history as context and
    learns to denoise a future horizon chunk.
    """

    def __init__(self, rasterizer_config) -> None:
        self.rasterizer_config = rasterizer_config

    def __call__(self, sketch: Dict[str, torch.Tensor]):  # -> Dict[str, torch.Tensor]:

        tokens = sketch["tokens"]
        filtered = self._filter_tokens(tokens)
        img = rasterize_absolute_points(
            sketch=filtered.numpy(), config=self.rasterizer_config
        )

        return {
            "img": torch.from_numpy(img).unsqueeze(0),
            "family": sketch["family_id"],
            "sketch_id": sketch["query_id"],
        }

    @staticmethod
    def _filter_tokens(tokens: torch.Tensor) -> torch.Tensor:
        """Drop special tokens, keeping only actual sketch."""
        reset_idx = (tokens[:, 5] == 1.0).nonzero(as_tuple=True)[0]
        if reset_idx.numel() > 0:
            filtered = tokens[reset_idx[0] + 1 : -1]
        else:
            raise ValueError("No reset token found in sketch tokens.")
        return filtered[1:, :3]  # Keep only x, y, pen_state


class EpisodeToImageCollate:

    def __init__(self, rasterizer_config: RasterizerConfig) -> None:
        self.rasterizer_config = rasterizer_config

        self.episode_to_image = EpisodeToImage(rasterizer_config=rasterizer_config)

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        imgs = []
        families = []
        sketch_ids = []

        for sketch in batch:
            processed = self.episode_to_image(sketch)
            imgs.append(processed["img"])
            families.append(sketch["family_id"].unsqueeze(0))
            sketch_ids.append(sketch["sketch_id"].unsqueeze(0))

        return {
            "img": torch.cat(imgs, dim=0),
            "family": torch.cat(families, dim=0),
            "sketch_id": torch.cat(sketch_ids, dim=0),
        }


def decode_pt(sample):
    """
    sample: a tuple (img_bytes, label_bytes, ...)
    This function converts bytes to tensors.
    """

    decoded = {}
    for key, value in sample.items():
        if key.endswith(".pt"):
            buf = io.BytesIO(value)
            decoded[key[:-3]] = torch.load(buf, weights_only=False)  # remove ".pt"
    return decoded


def cached_collate(samples):

    batch = {}
    batch["img"] = torch.cat([s["img"] for s in samples], dim=0)
    batch["label"] = torch.cat([s["label"] for s in samples], dim=0)

    # for key in samples[0].keys():
    #     batch[key] = torch.cat([s[key] for s in samples], dim=0)

    return batch


def get_cached_loader(shard_glob, batch_size, num_workers=4):
    shards = sorted(glob(shard_glob))

    dataset = (
        wds.WebDataset(shards)
        .decode()  # identity, we decode manually
        .to_tuple("img.pt", "label.pt")
        .map(lambda tup: {"img.pt": tup[0], "label.pt": tup[1]})
        .map(decode_pt)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=cached_collate,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
