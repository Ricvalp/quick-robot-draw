"""
Preprocessing utilities for the Quick, Draw! dataset.

This module loads raw sketches from the Google Quick, Draw! dataset (either
`.ndjson` or `.bin` format), converts them to normalized absolute coordinates,
computes delta representations, attaches pen-up/pen-down flags, and prepares
metadata required for downstream K-shot episode construction.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RawSketch",
    "ProcessedSketch",
    "QuickDrawPreprocessor",
    "load_ndjson_sketches",
    "load_binary_sketches",
]


def _to_int_array(values: Sequence[int]) -> np.ndarray:
    """Return a numpy array of `int32` from the given sequence."""
    return np.asarray(list(values), dtype=np.int32)


def _ensure_float32(points: np.ndarray) -> np.ndarray:
    """Cast point array to float32 if required."""
    if points.dtype == np.float32:
        return points
    return points.astype(np.float32, copy=False)


@dataclass
class RawSketch:
    """Container for a raw sketch before preprocessing."""

    family_id: str
    strokes: List[np.ndarray]
    metadata: Dict[str, object] = field(default_factory=dict)

    def total_points(self) -> int:
        """Return the total number of points across all strokes."""
        return int(sum(len(stroke[0]) for stroke in self.strokes))


@dataclass
class ProcessedSketch:
    """Normalized sketch ready for storage or episode composition."""

    family_id: str
    sample_id: str
    absolute: np.ndarray
    deltas: np.ndarray
    pen: np.ndarray
    length: int
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """
        Convert the sketch to a serialisable dictionary.

        Arrays are returned as numpy arrays; callers may choose how
        to serialise them (e.g., msgpack, HDF5, NumPy npz).
        """
        return {
            "family_id": self.family_id,
            "sample_id": self.sample_id,
            "absolute": self.absolute,
            "deltas": self.deltas,
            "pen": self.pen,
            "length": self.length,
            "metadata": self.metadata,
        }


class QuickDrawPreprocessor:
    """
    Preprocess Quick, Draw! sketches into a token-ready format.

    Parameters
    ----------
    normalize : bool
        If True, normalise sketches to fit within [-1, 1]^2.
    resample_points : Optional[int]
        If provided, resample each stroke to a fixed number of points
        (minimum of 2). Takes precedence over `resample_spacing`.
    resample_spacing : Optional[float]
        Target arclength spacing between consecutive points for resampling.
        Only used when `resample_points` is None.
    keep_zero_length : bool
        Retain zero-length strokes (single points). When False,
        strokes shorter than two unique points are discarded.
    dtype : np.dtype
        Floating dtype for the output arrays.
    """

    def __init__(
        self,
        *,
        normalize: bool = True,
        resample_points: Optional[int] = None,
        resample_spacing: Optional[float] = None,
        keep_zero_length: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.normalize = normalize
        self.resample_points = resample_points
        self.resample_spacing = resample_spacing
        self.keep_zero_length = keep_zero_length
        self.dtype = dtype

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def preprocess(self, sketch: RawSketch) -> Optional[ProcessedSketch]:
        """
        Convert a raw sketch into the canonical representation.

        Returns
        -------
        ProcessedSketch or None
            The processed sketch. Returns None when the sketch contains
            insufficient data after preprocessing (e.g., filtered out).
        """
        strokes = self._prepare_strokes(sketch.strokes)
        if not strokes:
            return None

        absolute, pen = self._flatten_strokes(strokes)
        if absolute.size == 0:
            return None

        absolute = _ensure_float32(absolute).astype(self.dtype, copy=False)
        pen = np.asarray(pen, dtype=np.float32)

        norm_meta: Dict[str, object] = {}
        if self.normalize:
            absolute, norm_meta = self._normalize_points(absolute)

        deltas = self._compute_deltas(absolute)

        sample_id = self._resolve_sample_id(sketch.metadata)

        metadata = dict(sketch.metadata)
        metadata.update(
            {
                "normalization": norm_meta,
                "original_num_strokes": len(sketch.strokes),
                "post_num_strokes": len(strokes),
                "original_total_points": sketch.total_points(),
                "post_total_points": int(absolute.shape[0]),
            }
        )

        return ProcessedSketch(
            family_id=sketch.family_id,
            sample_id=sample_id,
            absolute=absolute,
            deltas=deltas,
            pen=pen,
            length=int(absolute.shape[0]),
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _prepare_strokes(self, strokes: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Optionally resample strokes and drop zero-length segments."""
        prepared: List[np.ndarray] = []
        for stroke in strokes:
            if stroke.shape[1] == 0:
                continue
            unique = np.unique(stroke, axis=1)
            if unique.shape[1] < 2 and not self.keep_zero_length:
                continue
            resampled = self._resample_stroke(stroke)
            prepared.append(resampled)
        return prepared

    def _resample_stroke(self, stroke: np.ndarray) -> np.ndarray:
        """
        Resample a single stroke to match the configured spacing/points.

        The stroke array has shape (2, N) with absolute coordinates.
        """
        if stroke.shape[1] <= 1:
            return stroke.astype(self.dtype, copy=False)

        if self.resample_points is not None and self.resample_points >= 2:
            num_points = int(self.resample_points)
        else:
            num_points = None

        spacing = self.resample_spacing

        if num_points is None and spacing is None:
            return stroke.astype(self.dtype, copy=False)

        points = stroke.T.astype(np.float64, copy=False)
        diffs = np.diff(points, axis=0)
        seg_length = np.linalg.norm(diffs, axis=1)
        cumulative = np.insert(np.cumsum(seg_length), 0, 0.0)
        total_length = cumulative[-1]

        if total_length == 0 or points.shape[0] == 1:
            return stroke.astype(self.dtype, copy=False)

        if num_points is None:
            assert spacing is not None
            num_points = max(2, int(math.floor(total_length / spacing)) + 1)

        target = np.linspace(0.0, total_length, num=num_points, dtype=np.float64)
        resampled = np.zeros((num_points, 2), dtype=np.float64)
        for dim in range(2):
            resampled[:, dim] = np.interp(target, cumulative, points[:, dim])

        return resampled.T.astype(self.dtype, copy=False)

    def _flatten_strokes(
        self, strokes: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Concatenate strokes into a contiguous array and compute pen flags.

        Returns
        -------
        absolute : np.ndarray of shape (N, 2)
        pen : np.ndarray of shape (N,)
            Pen flag uses 1.0 to indicate "pen-down drawing" for the move
            leading *into* the point, and 0.0 for pen-up / stroke restart.
        """
        abs_points: List[np.ndarray] = []
        pen_flags: List[float] = []
        for stroke in strokes:
            stroke = stroke.astype(self.dtype, copy=False)
            if stroke.shape[1] == 0:
                continue
            points = stroke.T
            for idx, point in enumerate(points):
                abs_points.append(point)
                pen_flags.append(0.0 if idx == 0 else 1.0)
        if not abs_points:
            return np.empty((0, 2), dtype=self.dtype), np.empty((0,), dtype=np.float32)
        absolute = np.stack(abs_points, axis=0)
        pen = np.asarray(pen_flags, dtype=np.float32)
        return absolute, pen

    def _normalize_points(
        self, absolute: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Normalise sketch coordinates into [-1, 1]^2."""
        mins = absolute.min(axis=0)
        maxs = absolute.max(axis=0)
        center = (mins + maxs) / 2.0
        scale = max(maxs - mins)
        if scale <= 0:
            scale = 1.0
        normalized = (absolute - center) / (scale / 2.0)
        normalized = normalized.astype(self.dtype, copy=False)
        metadata = {
            "bbox_min": mins.tolist(),
            "bbox_max": maxs.tolist(),
            "center": center.tolist(),
            "scale": float(scale / 2.0),
        }
        return normalized, metadata

    def _compute_deltas(self, absolute: np.ndarray) -> np.ndarray:
        """Compute successive deltas (dx, dy) for absolute positions."""
        if absolute.shape[0] == 0:
            return np.empty((0, 2), dtype=self.dtype)
        deltas = np.zeros_like(absolute)
        deltas[0] = absolute[0]
        deltas[1:] = absolute[1:] - absolute[:-1]
        return deltas.astype(self.dtype, copy=False)

    @staticmethod
    def _resolve_sample_id(metadata: Dict[str, object]) -> str:
        """Derive a stable sample identifier from metadata."""
        if "key_id" in metadata:
            return str(metadata["key_id"])
        digest = hashlib.md5(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest


# --------------------------------------------------------------------------- #
# Raw data loaders
# --------------------------------------------------------------------------- #


def load_ndjson_sketches(
    path: str,
    *,
    family_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Generator[RawSketch, None, None]:
    """
    Yield sketches from a QuickDraw `.ndjson` file.

    Parameters
    ----------
    path : str
        Path to the `.ndjson` file.
    family_id : Optional[str]
        Name of the drawing category. Defaults to filename stem.
    limit : Optional[int]
        If provided, stop after emitting this many sketches.
    """
    resolved_family = family_id or _infer_family_from_path(path)
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            raw = json.loads(line)
            strokes = [
                np.vstack((_to_int_array(stroke[0]), _to_int_array(stroke[1])))
                for stroke in raw["drawing"]
            ]
            metadata = {
                "key_id": raw.get("key_id"),
                "word": raw.get("word", resolved_family),
                "recognized": raw.get("recognized"),
                "countrycode": raw.get("countrycode"),
                "timestamp": raw.get("timestamp"),
            }
            yield RawSketch(
                family_id=resolved_family,
                strokes=strokes,
                metadata=metadata,
            )


def load_binary_sketches(
    path: str,
    *,
    family_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Generator[RawSketch, None, None]:
    """
    Yield sketches from a QuickDraw binary `.bin` file.

    The binary format is documented in the QuickDraw dataset repository.
    Each sketch includes metadata such as key id, country code, recognised flag,
    and timestamp.
    """
    resolved_family = family_id or _infer_family_from_path(path)
    with open(path, "rb") as handle:
        idx = 0
        while True:
            if limit is not None and idx >= limit:
                break
            sketch = _unpack_binary_drawing(handle)
            if sketch is None:
                break
            strokes = [
                np.vstack((_to_int_array(xs), _to_int_array(ys)))
                for (xs, ys) in sketch["image"]
            ]
            metadata = {
                "key_id": sketch["key_id"],
                "countrycode": sketch["country_code"].decode("ascii", errors="ignore"),
                "recognized": bool(sketch["recognized"]),
                "timestamp": sketch["timestamp"],
            }
            yield RawSketch(
                family_id=resolved_family,
                strokes=strokes,
                metadata=metadata,
            )
            idx += 1


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #


def _infer_family_from_path(path: str) -> str:
    """Infer family/category name from a file path."""
    stem = os.path.basename(path)
    stem = stem.replace(".ndjson", "").replace(".bin", "")
    return stem


def _unpack_binary_drawing(handle: io.BufferedReader) -> Optional[Dict[str, object]]:
    """
    Parse a single drawing from a binary QuickDraw stream.

    Returns
    -------
    Dict[str, object] or None
        Returns None when EOF is reached.
    """
    header = handle.read(8 + 2 + 1 + 4 + 2)
    if len(header) < 17:
        return None
    (key_id,) = struct.unpack("<Q", header[:8])
    country_code = struct.unpack("<2s", header[8:10])[0]
    recognized = struct.unpack("<b", header[10:11])[0]
    (timestamp,) = struct.unpack("<I", header[11:15])
    (n_strokes,) = struct.unpack("<H", header[15:17])

    image: List[Tuple[Sequence[int], Sequence[int]]] = []
    for _ in range(n_strokes):
        n_points_bytes = handle.read(2)
        if len(n_points_bytes) < 2:
            return None
        (n_points,) = struct.unpack("<H", n_points_bytes)
        fmt = f"<{n_points}B"
        xs = struct.unpack(fmt, handle.read(n_points))
        ys = struct.unpack(fmt, handle.read(n_points))
        image.append((xs, ys))

    return {
        "key_id": key_id,
        "country_code": country_code,
        "recognized": recognized,
        "timestamp": timestamp,
        "image": image,
    }
