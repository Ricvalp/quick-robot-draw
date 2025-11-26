"""
Storage backends and manifest utilities for the QuickDraw dataset pipeline.

This module offers pluggable backends (LMDB, WebDataset-style tar shards, and
HDF5) for storing processed sketches and pre-built episodes. It also manages
dataset manifests that capture preprocessing configuration and dataset stats.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .episode_builder import Episode
from .preprocess import ProcessedSketch

try:
    import lmdb  # type: ignore
except ImportError:  # pragma: no cover - backend optional
    lmdb = None

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - backend optional
    h5py = None

import msgpack

__all__ = [
    "StorageConfig",
    "DatasetManifest",
    "SketchStorage",
    "EpisodeStorage",
    "hash_config",
]


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def _array_to_bytes(array: np.ndarray) -> bytes:
    """Serialise a numpy array into a compact binary buffer."""
    with io.BytesIO() as buffer:
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()


def _bytes_to_array(data: bytes) -> np.ndarray:
    """Deserialize a numpy array from a binary buffer produced by `_array_to_bytes`."""
    with io.BytesIO(data) as buffer:
        return np.load(buffer, allow_pickle=False)


def _serialize_sketch(sketch: ProcessedSketch) -> bytes:
    """Encode a processed sketch into msgpack bytes."""
    payload = {
        "family_id": sketch.family_id,
        "sample_id": sketch.sample_id,
        "length": sketch.length,
        "metadata": sketch.metadata,
        "absolute": _array_to_bytes(sketch.absolute.astype(np.float32, copy=False)),
        "deltas": _array_to_bytes(sketch.deltas.astype(np.float32, copy=False)),
        "pen": _array_to_bytes(sketch.pen.astype(np.float32, copy=False)),
    }
    return msgpack.packb(payload, use_bin_type=True)


def _deserialize_sketch(data: bytes) -> ProcessedSketch:
    """Decode msgpack bytes into a `ProcessedSketch`."""
    payload = msgpack.unpackb(data, raw=False)
    absolute = _bytes_to_array(payload["absolute"])
    deltas = _bytes_to_array(payload["deltas"])
    pen = _bytes_to_array(payload["pen"])
    return ProcessedSketch(
        family_id=payload["family_id"],
        sample_id=payload["sample_id"],
        absolute=absolute,
        deltas=deltas,
        pen=pen,
        length=int(payload["length"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _serialize_episode(episode: Episode) -> bytes:
    """Encode an episode (without duplicating full sketch data)."""
    payload = {
        "episode_id": episode.episode_id,
        "family_id": episode.family_id,
        "lengths": episode.lengths,
        "metadata": episode.metadata,
        "tokens": _array_to_bytes(episode.tokens.astype(np.float32, copy=False)),
        "prompt_ids": [sk.sample_id for sk in episode.prompt],
        "query_id": episode.query.sample_id,
    }
    return msgpack.packb(payload, use_bin_type=True)


def _deserialize_episode(data: bytes) -> Dict[str, object]:
    """Return the lightweight episode data stored in msgpack."""
    payload = msgpack.unpackb(data, raw=False)
    payload["tokens"] = _bytes_to_array(payload["tokens"])
    return payload


def hash_config(config: Dict[str, object]) -> str:
    """Create a stable SHA-1 hash for a configuration dictionary."""
    serialized = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha1(serialized).hexdigest()


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass
class DatasetManifest:
    """
    Store metadata describing the processed dataset.

    Fields
    ------
    version : str
        Semantic version of the preprocessing pipeline.
    backend : str
        Storage backend used (lmdb, webdataset, hdf5).
    config_hash : str
        Hash of the preprocessing configuration.
    sketch_counts : Dict[str, int]
        Number of sketches per family.
    total_sketches : int
        Total number of sketches.
    total_episodes : int
        Number of pre-built episodes stored (can be zero).
    config : Dict[str, object]
        Full configuration dictionary (optional but handy).
    normalization : Dict[str, object]
        Additional normalization statistics (e.g. global bounds).
    """

    version: str
    backend: str
    config_hash: str
    sketch_counts: Dict[str, int] = field(default_factory=dict)
    total_sketches: int = 0
    total_episodes: int = 0
    config: Dict[str, object] = field(default_factory=dict)
    normalization: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "backend": self.backend,
            "config_hash": self.config_hash,
            "sketch_counts": self.sketch_counts,
            "total_sketches": self.total_sketches,
            "total_episodes": self.total_episodes,
            "config": self.config,
            "normalization": self.normalization,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "DatasetManifest":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            version=data["version"],
            backend=data["backend"],
            config_hash=data["config_hash"],
            sketch_counts=data.get("sketch_counts", {}),
            total_sketches=int(data.get("total_sketches", 0)),
            total_episodes=int(data.get("total_episodes", 0)),
            config=data.get("config", {}),
            normalization=data.get("normalization", {}),
        )


# --------------------------------------------------------------------------- #
# Storage configuration
# --------------------------------------------------------------------------- #


@dataclass
class StorageConfig:
    """Configuration describing a storage backend."""

    root: str
    backend: str = "lmdb"
    map_size_bytes: int = 1 << 40  # 1 TB logical for LMDB
    compression: Optional[str] = None
    shards: int = 64
    items_per_shard: int = 2048

    def sketch_path(self) -> str:
        """Return the directory path used to store processed sketches.

        Keeping this helper centralises path construction for any backend.
        """
        return os.path.join(self.root, "sketches")

    def episode_path(self) -> str:
        """Return the directory path used to store prebuilt episodes.

        This ensures every component writes episodes under a consistent root.
        """
        return os.path.join(self.root, "episodes")


# --------------------------------------------------------------------------- #
# Backend primitives
# --------------------------------------------------------------------------- #


class _BaseBackend:
    """Abstract base class for storage backend adapters."""

    def put(self, key: str, value: bytes) -> None:
        """Persist a value under the provided key.

        Concrete backends must override this with their own write semantics.
        """
        raise NotImplementedError

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve a serialized value by key.

        Returning `None` indicates the requested record does not exist.
        """
        raise NotImplementedError

    def keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Iterate over stored keys, optionally filtered by prefix.

        The prefix mechanism enables hierarchical keys like `sketch/<family>/...`.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources held by the backend.

        Backends that manage files or processes should override this hook.
        """
        pass


class _LMDBBackend(_BaseBackend):
    """LMDB-backed key-value store."""

    def __init__(self, path: str, *, readonly: bool, map_size: int) -> None:
        """Initialise the LMDB environment at the target path.

        This eagerly creates the directory so subsequent writers can map the DB.
        """
        if lmdb is None:
            raise RuntimeError("lmdb package is required for LMDB backend.")
        os.makedirs(path, exist_ok=True)
        self.readonly = readonly
        self.env = lmdb.open(
            path,
            map_size=map_size,
            subdir=True,
            readonly=readonly,
            lock=not readonly,
            readahead=not readonly,
            max_readers=512,
        )

    def put(self, key: str, value: bytes) -> None:
        """Write a key/value pair into the LMDB database.

        The operation is performed inside a short-lived write transaction.
        """
        encoded = key.encode("utf-8")
        with self.env.begin(write=True) as txn:
            txn.put(encoded, value)

    def get(self, key: str) -> Optional[bytes]:
        """Fetch a serialized value by key from LMDB.

        Returns `None` when the key is not present to mirror dict semantics.
        """
        encoded = key.encode("utf-8")
        with self.env.begin(write=False) as txn:
            data = txn.get(encoded)
        return data

    def keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Iterate over keys stored in LMDB, optionally filtered by prefix.

        Keys are decoded to UTF-8 strings so callers can safely split them.
        """
        with self.env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            if prefix is None:
                for key, _ in cursor:
                    raw_key = key
                    if isinstance(raw_key, memoryview):
                        raw_key = raw_key.tobytes()
                    else:
                        raw_key = bytes(raw_key)
                    yield raw_key.decode("utf-8")
            else:
                prefix_bytes = prefix.encode("utf-8")
                if cursor.set_range(prefix_bytes):
                    do_continue = True
                    while do_continue:
                        raw_key = cursor.key()
                        if isinstance(raw_key, memoryview):
                            key_bytes = raw_key.tobytes()
                        else:
                            key_bytes = bytes(raw_key)
                        if not key_bytes.startswith(prefix_bytes):
                            break
                        yield key_bytes.decode("utf-8")
                        do_continue = cursor.next()

    def close(self) -> None:
        """Flush pending writes (if any) and close the LMDB environment.

        Read-only handles simply close without syncing to avoid permission issues.
        """
        if not self.readonly:
            try:
                self.env.sync()
            except lmdb.Error:
                pass
        self.env.close()


class _HDF5Backend(_BaseBackend):
    """HDF5-backed store mapping keys to byte arrays."""

    def __init__(self, path: str, *, readonly: bool) -> None:
        """Open or create an HDF5 file to store serialized arrays.

        The directory is created automatically so that repeated runs succeed.
        """
        if h5py is None:
            raise RuntimeError("h5py is required for HDF5 backend.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mode = "r" if readonly else "a"
        self.file = h5py.File(path, mode)

    def put(self, key: str, value: bytes) -> None:
        """Store a value as a compressed dataset under the provided key.

        Pre-existing datasets are removed to keep the file consistent.
        """
        if key in self.file:
            del self.file[key]
        data = np.frombuffer(value, dtype=np.uint8)
        self.file.create_dataset(key, data=data, compression="gzip")

    def get(self, key: str) -> Optional[bytes]:
        """Return the serialized bytes for a dataset key, if present.

        Data is returned as a contiguous bytes object suitable for msgpack decoding.
        """
        if key not in self.file:
            return None
        data = self.file[key][:]
        return bytes(data)

    def keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield dataset keys within the HDF5 file, optionally filtered by prefix.

        Nested groups are traversed recursively to emulate a hierarchical store.
        """

        def recurse(group, current_path=""):
            for name, item in group.items():
                path = f"{current_path}/{name}"
                if isinstance(item, h5py.Dataset):
                    yield path.lstrip("/")
                else:
                    yield from recurse(item, path)

        for key in recurse(self.file):
            if prefix is None or key.startswith(prefix):
                yield key

    def close(self) -> None:
        """Flush and close the underlying HDF5 file.

        Explicit flushing reduces the risk of corrupted files if the process stops.
        """
        self.file.flush()
        self.file.close()


class _WebDatasetBackend(_BaseBackend):
    """
    Minimal WebDataset-style tar shard backend.

    Data is appended to shards named `shard_{index:05d}.tar`. A simple index
    maps keys to the tar member storing their payload. While not as feature rich
    as the official WebDataset tooling, it provides compatible shard layouts.
    """

    def __init__(
        self,
        path: str,
        *,
        readonly: bool,
        items_per_shard: int,
    ) -> None:
        """Initialise shard directory and optionally build the read index."""
        self.path = path
        self.readonly = readonly
        self.items_per_shard = items_per_shard
        os.makedirs(path, exist_ok=True)
        self.index: Dict[str, Tuple[str, str]] = {}

        if readonly:
            self._build_index()
            self.tar = None
        else:
            self.current_shard = 0
            self.items_in_shard = 0
            self.tar = None
            self._open_new_shard()

    def _shard_path(self, idx: int) -> str:
        """Return the absolute file path for a shard index.

        Shards are numbered deterministically to ease downstream shuffling.
        """
        return os.path.join(self.path, f"shard_{idx:05d}.tar")

    def _open_new_shard(self) -> None:
        """Close the active shard (if any) and open a new writable tar.

        Writers call this whenever the previous shard reaches capacity.
        """
        if self.tar is not None:
            self.tar.close()
        shard_path = self._shard_path(self.current_shard)
        self.tar = tarfile.open(shard_path, mode="w")
        self.items_in_shard = 0

    def _build_index(self) -> None:
        """Scan existing shards and build the in-memory key → member index.

        The index lets readers answer random lookups without scanning tar files.
        """
        for name in sorted(os.listdir(self.path)):
            if not name.endswith(".tar"):
                continue
            shard_path = os.path.join(self.path, name)
            with tarfile.open(shard_path, "r") as tar:
                for member in tar.getmembers():
                    key = member.name.replace(".msgpack", "")
                    self.index[key] = (shard_path, member.name)

    def put(self, key: str, value: bytes) -> None:
        """Append a record to the current shard, rolling when it fills.

        Metadata is updated immediately so subsequent reads can see the record.
        """
        if self.readonly:
            raise RuntimeError("Cannot write using readonly backend.")

        if self.items_in_shard >= self.items_per_shard:
            self.current_shard += 1
            self._open_new_shard()

        assert self.tar is not None
        tarinfo = tarfile.TarInfo(name=f"{key}.msgpack")
        tarinfo.size = len(value)
        tarinfo.mtime = time.time()
        self.tar.addfile(tarinfo, io.BytesIO(value))
        self.items_in_shard += 1
        shard_path = self._shard_path(self.current_shard)
        self.index[key] = (shard_path, tarinfo.name)

    def get(self, key: str) -> Optional[bytes]:
        """Retrieve a value by key by reading the appropriate shard/member.

        Returns `None` if the key has not been written or the member is missing.
        """
        entry = self.index.get(key)
        if entry is None:
            return None
        shard_path, member_name = entry
        with tarfile.open(shard_path, "r") as tar:
            member = tar.getmember(member_name)
            extracted = tar.extractfile(member)
            if extracted is None:
                return None
            return extracted.read()

    def keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield available keys, optionally filtering by a prefix string.

        The index dictionary makes this a constant-time operation per key.
        """
        for key in self.index:
            if prefix is None or key.startswith(prefix):
                yield key

    def close(self) -> None:
        """Close the active shard file when writing mode is used.

        Readers simply drop references because they only open shards on demand.
        """
        if self.tar is not None:
            self.tar.close()


# --------------------------------------------------------------------------- #
# High-level storage APIs
# --------------------------------------------------------------------------- #


class SketchStorage:
    """
    Store and retrieve processed sketches.

    Parameters
    ----------
    config : StorageConfig
        Backend configuration.
    mode : str
        `"r"` for read-only access, `"w"` for read/write.
    """

    def __init__(self, config: StorageConfig, mode: str = "r") -> None:
        self.config = config
        self.mode = mode
        readonly = mode == "r"
        backend = config.backend.lower()
        path = config.sketch_path()
        if backend == "lmdb":
            backend_obj = _LMDBBackend(
                path, readonly=readonly, map_size=config.map_size_bytes
            )
        elif backend == "hdf5":
            backend_obj = _HDF5Backend(
                os.path.join(path, "sketches.h5"), readonly=readonly
            )
        elif backend == "webdataset":
            backend_obj = _WebDatasetBackend(
                path,
                readonly=readonly,
                items_per_shard=config.items_per_shard,
            )
        else:
            raise ValueError(f"Unsupported backend '{config.backend}'.")
        self.backend = backend_obj

    def close(self) -> None:
        """Release backend resources."""
        self.backend.close()

    def put(self, sketch: ProcessedSketch) -> None:
        """Store a processed sketch under its `(family, sample)` key."""
        if self.mode == "r":
            raise RuntimeError("Cannot write to storage opened in read-only mode.")
        key = self._sketch_key(sketch.family_id, sketch.sample_id)
        self.backend.put(key, _serialize_sketch(sketch))

    def get(self, family_id: str, sample_id: str) -> ProcessedSketch:
        """Retrieve a stored sketch."""
        key = self._sketch_key(family_id, sample_id)
        data = self.backend.get(key)
        if data is None:
            raise KeyError(f"Sketch '{family_id}/{sample_id}' not found.")
        return _deserialize_sketch(data)

    def exists(self, family_id: str, sample_id: str) -> bool:
        """Return True when a sketch is already cached."""
        key = self._sketch_key(family_id, sample_id)
        data = self.backend.get(key)
        return data is not None

    def families(self) -> List[str]:
        """List all family identifiers stored in the backend."""
        families: set = set()
        prefix = "sketch/"
        for key in self.backend.keys(prefix=prefix):
            _, family, _ = key.split("/", 2)
            families.add(family)
        return sorted(families)

    def samples_for_family(self, family_id: str) -> List[str]:
        """Return sorted sample identifiers for the given family."""
        prefix = f"sketch/{family_id}/"
        samples: List[str] = []
        for key in self.backend.keys(prefix=prefix):
            _, _, sample_id = key.split("/", 2)
            samples.append(sample_id)
        return sorted(samples)

    @staticmethod
    def _sketch_key(family_id: str, sample_id: str) -> str:
        return f"sketch/{family_id}/{sample_id}"


class EpisodeStorage:
    """Store lightweight episode representations."""

    def __init__(self, config: StorageConfig, mode: str = "r") -> None:
        self.config = config
        self.mode = mode
        readonly = mode == "r"
        backend = config.backend.lower()
        path = config.episode_path()
        if backend == "lmdb":
            backend_obj = _LMDBBackend(
                path, readonly=readonly, map_size=config.map_size_bytes
            )
        elif backend == "hdf5":
            backend_obj = _HDF5Backend(
                os.path.join(path, "episodes.h5"), readonly=readonly
            )
        elif backend == "webdataset":
            backend_obj = _WebDatasetBackend(
                path,
                readonly=readonly,
                items_per_shard=config.items_per_shard,
            )
        else:
            raise ValueError(f"Unsupported backend '{config.backend}'.")
        self.backend = backend_obj

    def close(self) -> None:
        """Release backend resources."""
        self.backend.close()

    def put(self, episode: Episode) -> None:
        """Store a lightweight episode representation."""
        if self.mode == "r":
            raise RuntimeError("Cannot write to storage opened in read-only mode.")
        key = self._episode_key(episode.family_id, episode.episode_id)
        self.backend.put(key, _serialize_episode(episode))

    def get(self, family_id: str, episode_id: str) -> Dict[str, object]:
        """Retrieve an encoded episode payload."""
        key = self._episode_key(family_id, episode_id)
        data = self.backend.get(key)
        if data is None:
            raise KeyError(f"Episode '{family_id}/{episode_id}' not found.")
        return _deserialize_episode(data)

    def list_episode_ids(self, family_id: Optional[str] = None) -> List[str]:
        """Return stored episode identifiers, optionally filtered by family."""
        if family_id is None:
            prefix = "episode/"
        else:
            prefix = f"episode/{family_id}/"
        ids = []
        for key in self.backend.keys(prefix=prefix):
            _, fam, episode_id = key.split("/", 2)
            if family_id is None or fam == family_id:
                ids.append(episode_id)
        return sorted(ids)

    @staticmethod
    def _episode_key(family_id: str, episode_id: str) -> str:
        return f"episode/{family_id}/{episode_id}"
