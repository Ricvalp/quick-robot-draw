"""
Episode construction utilities for Quick, Draw! sketches.

This module defines helpers to compose K-shot imitation learning episodes with
prompt sketches and a query sketch. Special tokens provide structural cues
for downstream Transformer or state-space models.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .preprocess import ProcessedSketch

__all__ = ["Episode", "EpisodeBuilder"]


@dataclass
class Episode:
    """Lightweight container describing a single K-shot episode."""

    episode_id: str
    family_id: str
    prompt: List[ProcessedSketch]
    query: ProcessedSketch
    tokens: np.ndarray
    lengths: Dict[str, int]
    metadata: Dict[str, object] = field(default_factory=dict)


class EpisodeBuilder:
    """
    Assemble K-shot prompt/query episodes from processed sketches.

    Parameters
    ----------
    fetch_family : callable
        Function returning a list of sample identifiers for a given family.
    fetch_sketch : callable
        Function fetching a `ProcessedSketch` by `(family_id, sample_id)`.
    family_ids : Sequence[str]
        Set of available families/classes.
    k_shot : int
        Number of prompt sketches per episode.
    max_seq_len : Optional[int]
        Optional guard ensuring total token length does not exceed the limit.
    seed : Optional[int]
        Seed for the internal random state, ensuring reproducibility.
    augment_config : Optional[dict]
        Configuration controlling geometric augmentations.
    dtype : np.dtype
        dtype for the generated token matrix.
    """

    def __init__(
        self,
        *,
        fetch_family,
        fetch_sketch,
        family_ids: Sequence[str],
        k_shot: int,
        max_seq_len: Optional[int] = None,
        seed: Optional[int] = None,
        augment_config: Optional[Dict[str, object]] = None,
        dtype=np.float32,
    ) -> None:
        self.fetch_family = fetch_family
        self.fetch_sketch = fetch_sketch
        self.family_ids = list(family_ids)
        self.k_shot = int(k_shot)
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.random = np.random.RandomState(seed)
        self.augment_config = self._resolve_augment_config(augment_config or {})
        self.token_dim = 7  # dx, dy, pen, start, sep, reset, stop

        self.special_tokens = {
            "start": self._special_token(start=1.0),
            "separator": self._special_token(sep=1.0),
            "reset": self._special_token(reset=1.0),
            "stop": self._special_token(stop=1.0),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build_episode(
        self,
        *,
        family_id: Optional[str] = None,
        augment: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ) -> Episode:
        """
        Compose a single K-shot episode.

        Parameters
        ----------
        family_id : Optional[str]
            Optionally force sampling from a specific family.
        augment : bool
            Apply random augmentations to each sketch when True. The augmenter
            settings are controlled by `augment_config`.
        """
        rng = rng or self.random
        resolved_family = family_id or self._sample_family(rng)
        sample_ids = list(self.fetch_family(resolved_family))
        if len(sample_ids) < self.k_shot + 1:
            raise ValueError(
                f"Family '{resolved_family}' does not have enough sketches "
                f"for {self.k_shot}-shot episodes."
            )
        rng.shuffle(sample_ids)
        prompt_ids = sample_ids[: self.k_shot]
        query_id = sample_ids[self.k_shot]

        prompt_sketches = [
            self._maybe_augment(self.fetch_sketch(resolved_family, sid), augment, rng)
            for sid in prompt_ids
        ]
        query_sketch = self._maybe_augment(
            self.fetch_sketch(resolved_family, query_id), augment, rng
        )

        episode_tokens = self._compose_tokens(prompt_sketches, query_sketch)
        total_len = episode_tokens.shape[0]
        if self.max_seq_len is not None and total_len > self.max_seq_len:
            raise ValueError(
                f"Episode length {total_len} exceeds limit {self.max_seq_len}."
            )

        episode_id = uuid.uuid4().hex
        metadata = {
            "prompt_ids": prompt_ids,
            "query_id": query_id,
            "family_id": resolved_family,
            "k_shot": self.k_shot,
            "length": total_len,
        }
        return Episode(
            episode_id=episode_id,
            family_id=resolved_family,
            prompt=prompt_sketches,
            query=query_sketch,
            tokens=episode_tokens,
            lengths={
                "prompt": sum(sk.length for sk in prompt_sketches),
                "query": query_sketch.length,
                "total": total_len,
            },
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _sample_family(self, rng: np.random.RandomState) -> str:
        """Sample a family identifier using the provided RNG."""
        idx = rng.randint(0, len(self.family_ids))
        return self.family_ids[idx]

    def _special_token(
        self,
        start: float = 0.0,
        sep: float = 0.0,
        reset: float = 0.0,
        stop: float = 0.0,
    ) -> np.ndarray:
        """
        Create a special control token with the requested indicator bits.

        Channels 0-2 remain zero; channels 3-6 encode the structural markers
        (start/sep/reset/stop) used by downstream policies.
        """
        token = np.zeros(self.token_dim, dtype=self.dtype)
        token[3] = start
        token[4] = sep
        token[5] = reset
        token[6] = stop
        return token

    def _compose_tokens(
        self, prompt_sketches: List[ProcessedSketch], query_sketch: ProcessedSketch
    ) -> np.ndarray:
        """
        Concatenate prompt and query sketches into a single token matrix.

        The sequence adheres to the `[START, prompt₁, SEP, …, RESET, START, query, STOP]`
        convention required for in-context imitation learning.
        """
        segments: List[np.ndarray] = [self.special_tokens["start"]]
        for sketch in prompt_sketches:
            segments.append(self._sketch_to_tokens(sketch))
            segments.append(self.special_tokens["separator"])
        segments.append(self.special_tokens["reset"])
        segments.append(self.special_tokens["start"])
        segments.append(self._sketch_to_tokens(query_sketch))
        segments.append(self.special_tokens["stop"])
        return np.vstack(segments).astype(self.dtype, copy=False)

    def _sketch_to_tokens(self, sketch: ProcessedSketch) -> np.ndarray:
        """
        Convert a processed sketch into its `(dx, dy, pen, start, sep, reset, stop)` tokens.

        Only the sketch-level start flag is set here; higher-level separators are appended
        by `_compose_tokens`.
        """
        tokens = np.zeros((sketch.length, self.token_dim), dtype=self.dtype)
        tokens[:, 0:2] = sketch.deltas
        tokens[:, 2] = sketch.pen
        tokens[0, 3] = 1.0  # mark new sketch start
        return tokens

    def _maybe_augment(
        self,
        sketch: ProcessedSketch,
        enabled: bool,
        rng: np.random.RandomState,
    ) -> ProcessedSketch:
        """
        Apply random geometric augmentations to a sketch when requested.

        Augmentations include rotation, isotropic scaling, translation, and Gaussian
        jitter, each controlled by the configuration dictionary. Returns a new
        `ProcessedSketch` with recomputed deltas and augmented metadata.
        """
        if not enabled:
            return sketch
        cfg = self.augment_config
        absolute = sketch.absolute.copy()
        transforms_meta = {}
        if cfg["rotation"]["enabled"]:
            angle = rng.uniform(*cfg["rotation"]["range"])
            transforms_meta["rotation"] = angle
            c, s = math.cos(angle), math.sin(angle)
            rot = np.array([[c, -s], [s, c]], dtype=absolute.dtype)
            absolute = absolute @ rot.T
        if cfg["scale"]["enabled"]:
            scale = rng.uniform(*cfg["scale"]["range"])
            transforms_meta["scale"] = scale
            absolute *= scale
        if cfg["translation"]["enabled"]:
            tx = rng.uniform(*cfg["translation"]["range"])
            ty = rng.uniform(*cfg["translation"]["range"])
            transforms_meta["translation"] = (tx, ty)
            absolute += np.array([tx, ty], dtype=absolute.dtype)
        if cfg["jitter"]["enabled"]:
            std = cfg["jitter"]["std"]
            absolute += rng.normal(scale=std, size=absolute.shape).astype(
                absolute.dtype, copy=False
            )
            transforms_meta["jitter"] = std

        deltas = np.zeros_like(absolute)
        deltas[0] = absolute[0]
        if absolute.shape[0] > 1:
            deltas[1:] = absolute[1:] - absolute[:-1]

        metadata = dict(sketch.metadata)
        metadata["augmentations"] = transforms_meta

        return ProcessedSketch(
            family_id=sketch.family_id,
            sample_id=sketch.sample_id,
            absolute=absolute,
            deltas=deltas.astype(self.dtype, copy=False),
            pen=sketch.pen.copy(),
            length=sketch.length,
            metadata=metadata,
        )

    @staticmethod
    def _resolve_augment_config(config: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        """Merge user-provided augmentation configuration with defaults."""
        default = {
            "rotation": {"enabled": True, "range": (-math.pi, math.pi)},
            "scale": {"enabled": True, "range": (0.8, 1.2)},
            "translation": {"enabled": True, "range": (-0.1, 0.1)},
            "jitter": {"enabled": False, "std": 0.01},
        }
        merged = {}
        for key, params in default.items():
            user = config.get(key, {})
            merged[key] = {**params, **user}
            merged[key]["enabled"] = bool(merged[key].get("enabled", params["enabled"]))
        return merged
