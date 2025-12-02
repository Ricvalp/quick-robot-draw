#!/usr/bin/env python3
"""
Build the QuickDraw imitation learning dataset.

Example usage:
    python scripts/build_dataset.py --config config/data_config.yaml --num-workers 4
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
from ml_collections import config_flags

from dataset.episode_builder import EpisodeBuilder
from dataset.preprocess import (
    ProcessedSketch,
    QuickDrawPreprocessor,
    RawSketch,
    load_binary_sketches,
    load_ndjson_sketches,
)
from dataset.storage import (
    DatasetManifest,
    EpisodeStorage,
    SketchStorage,
    StorageConfig,
    hash_config,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


class SimpleProgressBar:
    """Fallback progress indicator when tqdm is unavailable."""

    def __init__(self, iterable, *, desc: str = "", unit: str = "item"):
        """Store metadata about the wrapped iterable for manual progress output.

        The iterable is eagerly materialised so we know its length for ETA estimates.
        """
        self.items = list(iterable)
        self.desc = desc or "Progress"
        self.unit = unit or "item"

    def __iter__(self):
        """Yield items while printing periodic ETA/throughput updates.

        Updates are throttled to avoid spamming the terminal while still providing ETAs.
        """
        total = len(self.items)
        if total == 0:
            print(f"{self.desc}: 0 {self.unit}s", file=sys.stderr)
            return
        start = time.time()
        last_refresh = start
        for idx, item in enumerate(self.items, start=1):
            yield item
            now = time.time()
            if idx == 1 or now - last_refresh >= 0.5 or idx == total:
                elapsed = max(now - start, 1e-8)
                rate = idx / elapsed
                remaining = total - idx
                eta = remaining / rate if rate > 0 else float("inf")
                eta_min = int(eta // 60) if math.isfinite(eta) else "?"
                eta_sec = int(eta % 60) if math.isfinite(eta) else "?"
                msg = (
                    f"{self.desc}: {idx}/{total} {self.unit}s | "
                    f"{rate:.1f} {self.unit}s/s | ETA {eta_min}m {eta_sec}s"
                )
                print(f"\r{msg}", end="", file=sys.stderr, flush=True)
                last_refresh = now
        print(file=sys.stderr)

    def __len__(self) -> int:
        """Return the number of buffered items.

        This mirrors the standard container protocol so helper utilities can introspect.
        """
        return len(self.items)


def progress_iterable(iterable, *, desc: str, unit: str):
    """Return an iterator wrapped with a progress indicator.

    Uses tqdm when available, otherwise falls back to `SimpleProgressBar`.
    """
    if tqdm is not None:
        return tqdm(iterable, desc=desc, unit=unit)
    return SimpleProgressBar(iterable, desc=desc, unit=unit)


def load_cfgs(_CONFIG_FILE) -> Dict:
    cfg = _CONFIG_FILE.value
    return cfg


def _normalize_families_filter(value) -> Optional[List[str]]:
    """Normalize user-provided family filters into a lowercase list."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [token.strip() for token in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(token).strip() for token in value]
    else:
        raise ValueError("families filter must be a list or comma-separated string.")
    normalized = [item for item in items if item]
    return [item.lower() for item in normalized] or None


def _family_from_path(path: str) -> str:
    """Infer the QuickDraw family name from a raw filename."""
    name = os.path.basename(path)
    for suffix in (".ndjson", ".bin"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def discover_raw_files(
    root: str,
    limit: Optional[int] = None,
    allowed_families: Optional[List[str]] = None,
) -> List[str]:
    """Recursively collect `.ndjson`/`.bin` files under `root`, respecting an optional cap.

    The returned list is lexicographically sorted to keep builds deterministic.
    """
    raw_files: List[str] = []
    allowed: Optional[set] = set(allowed_families) if allowed_families else None
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith(".ndjson") or name.endswith(".bin"):
                path = os.path.join(dirpath, name)
                family_name = _family_from_path(path).lower()
                if allowed and family_name not in allowed:
                    continue
                raw_files.append(path)
                if limit is not None and len(raw_files) >= limit:
                    return raw_files
    return raw_files


def _raw_to_payload(raw: RawSketch) -> Dict[str, object]:
    """Convert a RawSketch object into a lightweight, serialisable payload.

    Payloads are safe to send across process boundaries without numpy objects.
    """
    return {
        "family_id": raw.family_id,
        "strokes": [stroke.tolist() for stroke in raw.strokes],
        "metadata": raw.metadata,
    }


_WORKER_PREPROCESSOR: Optional[QuickDrawPreprocessor] = None


def _worker_init(preprocessor_kwargs: Dict[str, object]) -> None:
    """Initialise the per-process QuickDrawPreprocessor instance.

    Stashing the object globally avoids expensive re-instantiation per sketch.
    """
    global _WORKER_PREPROCESSOR
    _WORKER_PREPROCESSOR = QuickDrawPreprocessor(**preprocessor_kwargs)


def _worker_preprocess(payload: Dict[str, object]) -> Optional[ProcessedSketch]:
    """Preprocess a sketch payload inside a worker process.

    Returns `None` when the sketch gets filtered out by preprocessing rules.
    """
    assert _WORKER_PREPROCESSOR is not None
    strokes = [np.asarray(stroke, dtype=np.float32) for stroke in payload["strokes"]]
    raw = RawSketch(
        family_id=payload["family_id"],
        strokes=strokes,
        metadata=payload["metadata"],
    )
    return _WORKER_PREPROCESSOR.preprocess(raw)


def preprocess_dataset(
    *,
    raw_files: Iterable[str],
    storage: SketchStorage,
    preprocessor_kwargs: Dict[str, object],
    num_workers: int,
    max_sketches_per_file: Optional[int],
    seed: int,
) -> Dict[str, object]:
    """
    Preprocess all sketches and store them in the provided storage backend.

    Returns
    -------
    Dict[str, object]
        Summary statistics: counts, normalization bounds, etc.
    """
    counts: Dict[str, int] = defaultdict(int)
    bbox_min = None
    bbox_max = None

    if num_workers > 0:
        worker_args = (preprocessor_kwargs,)
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_worker_init,
            initargs=worker_args,
        )
        payload_iter = iterate_raw_payloads(
            raw_files,
            max_sketches_per_file=max_sketches_per_file,
            seed=seed,
        )
        preprocess_iter = executor.map(
            _worker_preprocess,
            payload_iter,
            chunksize=8,
        )
        iterator = preprocess_iter
    else:
        preprocessor = QuickDrawPreprocessor(**preprocessor_kwargs)

        def sequential_iterator() -> Iterator[Optional[ProcessedSketch]]:
            for raw in iterate_raw_sketches(
                raw_files,
                max_sketches_per_file=max_sketches_per_file,
                seed=seed,
            ):
                processed = preprocessor.preprocess(raw)
                yield processed

        iterator = sequential_iterator()

    progress = progress_iterable(iterator, desc="Preprocessing", unit="sketch")

    processed_count = 0
    for processed in progress:
        if processed is None:
            continue
        if storage.exists(processed.family_id, processed.sample_id):
            continue
        storage.put(processed)
        counts[processed.family_id] += 1
        norm_meta = processed.metadata.get("normalization")
        if norm_meta:
            bbox_min = (
                np.minimum(bbox_min, norm_meta["bbox_min"])
                if bbox_min is not None
                else np.asarray(norm_meta["bbox_min"], dtype=np.float32)
            )
            bbox_max = (
                np.maximum(bbox_max, norm_meta["bbox_max"])
                if bbox_max is not None
                else np.asarray(norm_meta["bbox_max"], dtype=np.float32)
            )
        processed_count += 1

    if num_workers > 0:
        executor.shutdown()

    normalization = {}
    if bbox_min is not None and bbox_max is not None:
        normalization = {
            "global_bbox_min": np.asarray(bbox_min, dtype=np.float32).tolist(),
            "global_bbox_max": np.asarray(bbox_max, dtype=np.float32).tolist(),
        }

    return {
        "counts": dict(counts),
        "normalization": normalization,
        "processed_count": processed_count,
    }


def iterate_raw_sketches(
    raw_files: Iterable[str],
    max_sketches_per_file: Optional[int],
    seed: int,
) -> Iterator[RawSketch]:
    """Yield RawSketch objects from a list of ndjson/bin files.

    Each generator instance walks the files sequentially to preserve ordering.
    """
    base_rng = np.random.RandomState(seed)
    for path in raw_files:
        file_seed = int(base_rng.randint(0, 2**32 - 1))
        file_rng = np.random.RandomState(file_seed)
        yield from _load_sketches_from_file(path, max_sketches_per_file, file_rng)


def _load_sketches_from_file(
    path: str,
    max_sketches_per_file: Optional[int],
    rng: np.random.RandomState,
) -> Iterator[RawSketch]:
    """Load sketches from `path`, optionally sampling at most `max_sketches_per_file`."""
    loader = (
        load_ndjson_sketches(path)
        if path.endswith(".ndjson")
        else load_binary_sketches(path)
    )
    if max_sketches_per_file is None or max_sketches_per_file <= 0:
        yield from loader
        return

    reservoir: List[RawSketch] = []
    for idx, sketch in enumerate(loader):
        if len(reservoir) < max_sketches_per_file:
            reservoir.append(sketch)
        else:
            j = rng.randint(0, idx + 1)
            if j < max_sketches_per_file:
                reservoir[j] = sketch
    if not reservoir:
        return
    order = rng.permutation(len(reservoir))
    for pos in order:
        yield reservoir[int(pos)]


def iterate_raw_payloads(
    raw_files: Iterable[str],
    max_sketches_per_file: Optional[int],
    seed: int,
) -> Iterator[Dict[str, object]]:
    """Yield serialisable payloads for multiprocessing from raw sketch files.

    This avoids sharing numpy arrays through pickling, which can be slow.
    """
    for raw in iterate_raw_sketches(
        raw_files,
        max_sketches_per_file=max_sketches_per_file,
        seed=seed,
    ):
        yield _raw_to_payload(raw)


def assign_family_splits(
    family_ids: List[str],
    split_config: Dict[str, float],
) -> Dict[str, str]:
    """Partition family IDs into train/val/test splits based on ratio config.

    The result is a dict mapping each family to its split label.
    """
    ratios = {
        "train": float(split_config.get("train_ratio", 0.8)),
        "val": float(split_config.get("val_ratio", 0.1)),
        "test": float(split_config.get("test_ratio", 0.1)),
    }
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    ratios = {k: v / total for k, v in ratios.items()}
    counts = {k: int(round(ratios[k] * len(family_ids))) for k in ratios}

    # Adjust counts to ensure they sum to total families.
    allocated = sum(counts.values())
    while allocated < len(family_ids):
        counts["train"] += 1
        allocated += 1
    while allocated > len(family_ids):
        counts["train"] -= 1
        allocated -= 1

    assignments: Dict[str, str] = {}
    idx = 0
    families_sorted = sorted(family_ids)
    for split in ("train", "val", "test"):
        for _ in range(counts[split]):
            if idx >= len(families_sorted):
                break
            assignments[families_sorted[idx]] = split
            idx += 1
    # Any remaining families default to train.
    for fam in families_sorted:
        assignments.setdefault(fam, "train")
    return assignments


def prebuild_episodes(
    *,
    storage_config: StorageConfig,
    manifest: DatasetManifest,
    num_episodes: int,
    k_shot: int,
    max_seq_len: int,
    augment_config: Dict[str, object],
    seed: int,
) -> int:
    """Optionally precompute and store a fixed number of full episodes.

    Returns the number of successfully stored episodes for manifest bookkeeping.
    """
    sketch_storage = SketchStorage(storage_config, mode="r")
    episode_storage = EpisodeStorage(storage_config, mode="w")
    builder = EpisodeBuilder(
        fetch_family=sketch_storage.samples_for_family,
        fetch_sketch=sketch_storage.get,
        family_ids=sketch_storage.families(),
        k_shot=k_shot,
        max_seq_len=max_seq_len,
        seed=seed,
        augment_config=augment_config,
    )

    rng = np.random.RandomState(seed)
    iterator = range(num_episodes)
    progress = tqdm(iterator, desc="Episodes", unit="episode") if tqdm else iterator
    stored = 0
    for idx in progress:
        episode = builder.build_episode(
            augment=False, rng=np.random.RandomState(rng.randint(0, 2**32))
        )
        episode_storage.put(episode)
        stored += 1

    episode_storage.close()
    sketch_storage.close()
    return stored


_CONFIG_FILE = config_flags.DEFINE_config_file(
    "config", default="configs/dataset/build_dataset.py"
)


def main(_) -> None:
    """
    Coordinates preprocessing, manifest writing, and optional episode caching.
    """
    config = load_cfgs(_CONFIG_FILE)

    def _plain(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_plain(v) for v in obj)
        return obj

    config_plain = _plain(config)

    config_hash = hash_config(config_plain)

    storage_root = config.get("root", "data/")
    os.makedirs(storage_root, exist_ok=True)
    backend = config.get("backend", "lmdb")
    storage_config = StorageConfig(
        root=storage_root,
        backend=backend,
        map_size_bytes=int(config.get("map_size_bytes", 1 << 40)),
        compression=config.get("storage", {}).get("compression"),
        shards=int(config.get("storage", {}).get("shards", 64)),
        items_per_shard=int(config.get("storage", {}).get("items_per_shard", 2048)),
    )

    manifest_path = os.path.join(storage_root, "DatasetManifest.json")

    raw_root = config.get("raw_root")
    if raw_root is None or not os.path.exists(raw_root):
        raise FileNotFoundError(
            f"Raw data root '{raw_root}' does not exist. Update config."
        )

    family_filter = _normalize_families_filter(config.get("families"))

    raw_files = discover_raw_files(
        raw_root,
        limit=config.max_files,
        allowed_families=family_filter,
    )
    if not raw_files:
        raise RuntimeError(f"No .ndjson or .bin files found under '{raw_root}'.")

    print(f"Found {len(raw_files)} raw files. Starting preprocessing...")

    preprocessor_kwargs = {
        "normalize": bool(config.get("normalize", True)),
        "resample_points": config.get("resample", {}).get("points"),
        "resample_spacing": config.get("resample", {}).get("spacing"),
        "keep_zero_length": config.get("resample", {}).get("keep_zero_length", True),
        "simplify_enabled": config.get("simplify", {}).get("enabled", False),
        "simplify_epsilon": config.get("simplify", {}).get("epsilon", 2.0),
    }

    sketch_storage = SketchStorage(storage_config, mode="w")
    max_sketches_per_file = config.get("max_sketches_per_file")
    if max_sketches_per_file is not None:
        max_sketches_per_file = int(max_sketches_per_file)

    summary = preprocess_dataset(
        raw_files=raw_files,
        storage=sketch_storage,
        preprocessor_kwargs=preprocessor_kwargs,
        num_workers=config.num_workers,
        max_sketches_per_file=max_sketches_per_file,
        seed=int(config.get("seed", 0)),
    )
    sketch_storage.close()

    storage_reader = SketchStorage(storage_config, mode="r")
    existing_families = storage_reader.families()
    counts = {
        fam: len(storage_reader.samples_for_family(fam)) for fam in existing_families
    }
    storage_reader.close()

    if not counts:
        raise RuntimeError("No sketches were stored; aborting.")

    family_ids = list(counts.keys())
    split_map = assign_family_splits(family_ids, config.get("split", {}))

    manifest = DatasetManifest(
        version="1.0.0",
        backend=backend,
        config_hash=config_hash,
        sketch_counts=counts,
        total_sketches=sum(counts.values()),
        total_episodes=0,
        config={**config_plain, "family_split_map": split_map},
        normalization=summary["normalization"],
    )

    num_prebuilt = int(config.get("num_prebuilt_episodes", 0))
    if num_prebuilt > 0:
        stored = prebuild_episodes(
            storage_config=storage_config,
            manifest=manifest,
            num_episodes=num_prebuilt,
            k_shot=int(config.get("num_prompts", 5)),
            max_seq_len=int(config.get("max_seq_len", 512)),
            augment_config=config.get("augmentations", {}),
            seed=int(config.get("seed", 0)),
        )
        manifest.total_episodes = stored
        print(f"Stored {stored} pre-built episodes.")

    manifest.save(manifest_path)

    print("Dataset build complete.")
    print(json.dumps(manifest.to_dict(), indent=2))


if __name__ == "__main__":
    from absl import app

    app.run(main)
