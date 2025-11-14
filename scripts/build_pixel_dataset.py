#!/usr/bin/env python3
"""Convert cached QuickDraw sketches into raster WebDataset shards."""

from __future__ import annotations

import argparse
import atexit
import io
import json
import os
import re
import tarfile
import time
from dataclasses import asdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is missing
    tqdm = None

from dataset.rasterize import RasterizerConfig, rasterize_processed_sketch
from dataset.storage import DatasetManifest, SketchStorage, StorageConfig
from scripts.build_dataset import load_config


_WORKER_STORAGE: Optional[SketchStorage] = None
_WORKER_RASTERIZER: Optional[RasterizerConfig] = None

def _worker_init(storage_kwargs: Dict[str, object], raster_kwargs: Dict[str, object]) -> None:
    global _WORKER_STORAGE, _WORKER_RASTERIZER
    config = StorageConfig(**storage_kwargs)
    _WORKER_STORAGE = SketchStorage(config, mode="r")
    _WORKER_RASTERIZER = RasterizerConfig(**raster_kwargs)

    def _cleanup() -> None:
        if _WORKER_STORAGE is not None:
            _WORKER_STORAGE.close()

    atexit.register(_cleanup)


def _worker_render(task: Tuple[str, str]) -> Tuple[str, str, np.ndarray, Dict[str, object]]:
    assert _WORKER_STORAGE is not None and _WORKER_RASTERIZER is not None
    family_id, sample_id = task
    sketch = _WORKER_STORAGE.get(family_id, sample_id)
    image = rasterize_processed_sketch(sketch, config=_WORKER_RASTERIZER)
    pixels = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    metadata = {
        "family_id": family_id,
        "sample_id": sample_id,
        "length": int(sketch.length),
    }
    return family_id, sample_id, pixels, metadata


class WebDatasetShardWriter:
    """Append image/metadata pairs into rolling tar shards."""

    def __init__(self, root: str, split: str, samples_per_shard: int) -> None:
        if samples_per_shard <= 0:
            raise ValueError("samples_per_shard must be positive.")
        self.base_dir = os.path.join(root, split)
        os.makedirs(self.base_dir, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self._current_shard: Optional[tarfile.TarFile] = None
        self._samples_in_shard = 0
        self._next_index = 0
        self.total_samples = 0

    def _open_new_shard(self) -> None:
        if self._current_shard is not None:
            self._current_shard.close()
        shard_path = os.path.join(self.base_dir, f"shard_{self._next_index:05d}.tar")
        self._next_index += 1
        self._samples_in_shard = 0
        self._current_shard = tarfile.open(shard_path, mode="w")

    def write(self, key: str, image: np.ndarray, metadata: Dict[str, object]) -> None:
        if self._current_shard is None or self._samples_in_shard >= self.samples_per_shard:
            self._open_new_shard()

        png_bytes = _encode_png(image)
        meta_bytes = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

        now = time.time()
        image_info = tarfile.TarInfo(name=f"{key}.png")
        image_info.size = len(png_bytes)
        image_info.mtime = now
        meta_info = tarfile.TarInfo(name=f"{key}.json")
        meta_info.size = len(meta_bytes)
        meta_info.mtime = now

        assert self._current_shard is not None
        self._current_shard.addfile(image_info, io.BytesIO(png_bytes))
        self._current_shard.addfile(meta_info, io.BytesIO(meta_bytes))

        self._samples_in_shard += 1
        self.total_samples += 1

    def close(self) -> None:
        if self._current_shard is not None:
            self._current_shard.close()
            self._current_shard = None

    @property
    def num_shards(self) -> int:
        if self._current_shard is None:
            return self._next_index
        return self._next_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a raster WebDataset from cached sketches.")
    parser.add_argument("--config", required=True, help="Path to the preprocessing config used for vectors.")
    parser.add_argument("--output-root", required=True, help="Directory where pixel shards will be written.")
    parser.add_argument("--img-size", type=int, default=64, help="Target square image size.")
    parser.add_argument("--antialias", type=int, default=2, help="Supersampling factor before downsampling.")
    parser.add_argument("--line-width", type=float, default=1.6, help="Line width in pixels at base resolution.")
    parser.add_argument("--samples-per-shard", type=int, default=4096, help="Number of images per tar shard.")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated list of splits to build.")
    parser.add_argument("--num-workers", type=int, default=0, help="Process pool size for rasterisation.")
    parser.add_argument("--chunksize", type=int, default=32, help="Chunk size for worker task dispatch.")
    parser.add_argument("--max-per-split", type=int, default=None, help="Optional cap on samples per split (debugging).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing shards under output_root.")
    return parser.parse_args()


def _encode_png(image: np.ndarray) -> bytes:
    if image.dtype != np.uint8:
        arr = np.clip(image, 0, 255).astype(np.uint8)
    else:
        arr = image
    with io.BytesIO() as buffer:
        Image.fromarray(arr, mode="L").save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def _sanitize_token(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text)


def _make_sample_key(family_id: str, sample_id: str) -> str:
    family = _sanitize_token(family_id)
    sample = _sanitize_token(str(sample_id))
    return f"{family}-{sample}"


def _ensure_output_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            # Abort if directory already contains files.
            if any(os.scandir(path)):
                raise RuntimeError(
                    f"Output directory '{path}' already contains files. Pass --overwrite to replace them."
                )
    else:
        os.makedirs(path, exist_ok=True)


def _resolve_split_families(
    manifest: DatasetManifest, requested_splits: List[str]
) -> Dict[str, List[str]]:
    split_map = manifest.config.get("family_split_map", {})
    result = {split: [] for split in requested_splits}
    for family_id in sorted(manifest.sketch_counts.keys()):
        assigned = split_map.get(family_id, "train")
        if assigned in result:
            result[assigned].append(family_id)
    return result


def _iter_split_samples(
    listing_storage: SketchStorage,
    families: Iterable[str],
    limit: Optional[int],
) -> Iterator[Tuple[str, str]]:
    emitted = 0
    for family_id in families:
        sample_ids = listing_storage.samples_for_family(family_id)
        for sample_id in sample_ids:
            yield family_id, sample_id
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _progress(iterator: Iterable, total: Optional[int], desc: str) -> Iterator:
    if tqdm is None:
        yield from iterator
    else:  # pragma: no cover - tqdm visual output not exercised in tests
        yield from tqdm(iterator, total=total, desc=desc, unit="img")


def _render_stream(
    tasks: Iterable[Tuple[str, str]],
    *,
    total: Optional[int],
    desc: str,
    storage_config: StorageConfig,
    raster_config: RasterizerConfig,
    num_workers: int,
    chunksize: int,
) -> Iterator[Tuple[str, str, np.ndarray, Dict[str, object]]]:
    if num_workers > 0:
        from concurrent.futures import ProcessPoolExecutor

        storage_kwargs = asdict(storage_config)
        raster_kwargs = asdict(raster_config)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
            initargs=(storage_kwargs, raster_kwargs),
        ) as executor:
            mapped = executor.map(_worker_render, tasks, chunksize=chunksize)
            yield from _progress(mapped, total, desc)
    else:
        storage = SketchStorage(storage_config, mode="r")
        try:
            def iterator() -> Iterator[Tuple[str, str, np.ndarray, Dict[str, object]]]:
                for family_id, sample_id in tasks:
                    sketch = storage.get(family_id, sample_id)
                    image = rasterize_processed_sketch(sketch, config=raster_config)
                    pixels = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    metadata = {
                        "family_id": family_id,
                        "sample_id": sample_id,
                        "length": int(sketch.length),
                    }
                    yield family_id, sample_id, pixels, metadata

            yield from _progress(iterator(), total, desc)
        finally:
            storage.close()


def main() -> None:
    args = parse_args()
    _ensure_output_dir(args.output_root, overwrite=args.overwrite)

    config = load_config(args.config)
    storage_root = config.get("root", "data/")
    manifest_path = os.path.join(storage_root, "DatasetManifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Processed dataset manifest not found at '{manifest_path}'. Run build_dataset.py first."
        )
    manifest = DatasetManifest.load(manifest_path)

    storage_config = StorageConfig(
        root=storage_root,
        backend=config.get("backend", "lmdb"),
        map_size_bytes=int(config.get("map_size_bytes", 1 << 40)),
        compression=config.get("storage", {}).get("compression"),
        shards=int(config.get("storage", {}).get("shards", 64)),
        items_per_shard=int(config.get("storage", {}).get("items_per_shard", 2048)),
    )

    raster_config = RasterizerConfig(
        img_size=int(args.img_size),
        antialias=int(args.antialias),
        line_width=float(args.line_width),
        background_value=0.0,
        stroke_value=1.0,
        normalize_inputs=False,
    )

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        raise ValueError("No splits specified.")

    split_families = _resolve_split_families(manifest, splits)
    listing_storage = SketchStorage(storage_config, mode="r")

    stats: Dict[str, Dict[str, object]] = {}
    try:
        for split in splits:
            families = split_families.get(split, [])
            if not families:
                print(f"[WARN] No families assigned to split '{split}'. Skipping.")
                continue

            writer = WebDatasetShardWriter(args.output_root, split, args.samples_per_shard)
            limit = args.max_per_split
            expected = sum(manifest.sketch_counts.get(fam, 0) for fam in families)
            if limit is not None:
                expected = min(expected, limit)

            tasks = _iter_split_samples(listing_storage, families, limit)
            desc = f"Rasterising {split}"

            processed = 0
            for family_id, sample_id, image, metadata in _render_stream(
                tasks,
                total=expected,
                desc=desc,
                storage_config=storage_config,
                raster_config=raster_config,
                num_workers=int(args.num_workers),
                chunksize=int(args.chunksize),
            ):
                key = _make_sample_key(family_id, sample_id)
                metadata.update({"split": split})
                writer.write(key, image, metadata)
                processed += 1

            writer.close()
            stats[split] = {
                "num_families": len(families),
                "num_samples": processed,
                "num_shards": writer.num_shards,
            }
            print(f"[{split}] Wrote {processed} images across {writer.num_shards} shards.")
    finally:
        listing_storage.close()

    manifest_payload = {
        "version": "1.0.0",
        "source_manifest": manifest_path,
        "rasterizer": asdict(raster_config),
        "output_root": args.output_root,
        "samples_per_shard": args.samples_per_shard,
        "splits": stats,
        "max_per_split": args.max_per_split,
    }

    manifest_out = os.path.join(args.output_root, "PixelDatasetManifest.json")
    with open(manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
    print(f"Wrote pixel dataset manifest to {manifest_out}.")


if __name__ == "__main__":
    main()
