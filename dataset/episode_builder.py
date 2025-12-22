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
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np

from .preprocess import ProcessedSketch

__all__ = ["Episode", "EpisodeBuilder", "EpisodeBuilderSimilar"]


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
    coordinate_mode : str
        Either `"delta"` (default) or `"absolute"`, controlling whether
        token coordinates use successive differences or absolute positions.
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
        max_query_len: Optional[int] = None,
        max_context_len: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float32,
        coordinate_mode: str = "delta",
    ) -> None:
        self.fetch_family = fetch_family
        self.fetch_sketch = fetch_sketch
        self.family_ids = list(family_ids)
        self.k_shot = int(k_shot)
        self.max_seq_len = max_seq_len
        self.max_query_len = max_query_len
        self.max_context_len = max_context_len
        self.dtype = dtype
        self.random = np.random.RandomState(seed)
        self.token_dim = 7  # dx, dy, pen, start, sep, reset, stop
        self.coordinate_mode = coordinate_mode.lower()
        if self.coordinate_mode not in {"delta", "absolute"}:
            raise ValueError(
                f"Unsupported coordinate_mode '{coordinate_mode}'. "
                "Expected 'delta' or 'absolute'."
            )

        self.special_tokens = {
            "separator": self._special_token(sep=1.0),
            "reset": self._special_token(reset=1.0),
            "stop": self._special_token(stop=1.0),
        }

    def build_episode(
        self,
        *,
        family_id: Optional[str] = None,
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
            self.fetch_sketch(resolved_family, sid) for sid in prompt_ids
        ]
        query_sketch = self.fetch_sketch(resolved_family, query_id)

        episode_tokens = self._compose_tokens(prompt_sketches, query_sketch)
        total_len = episode_tokens.shape[0]
        if self.max_seq_len is not None and total_len > self.max_seq_len:
            raise ValueError(
                f"Episode length {total_len} exceeds limit {self.max_seq_len}."
            )
        if self.max_query_len is not None and query_sketch.length > self.max_query_len:
            raise ValueError(
                f"Query length {query_sketch.length} exceeds limit {self.max_query_len}."
            )
        if (
            self.max_context_len is not None
            and sum(sk.length for sk in prompt_sketches) + self.k_shot + 2
            > self.max_context_len
        ):
            raise ValueError(
                f"Context length {sum(sk.length for sk in prompt_sketches)} "
                f"exceeds limit {self.max_context_len}."
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

    def _sample_family(self, rng: np.random.RandomState) -> str:
        """Sample a family identifier using the provided RNG."""
        idx = rng.randint(0, len(self.family_ids))
        return self.family_ids[idx]

    def _special_token(
        self,
        sep: float = 0.0,
        reset: float = 0.0,
        stop: float = 0.0,
    ) -> np.ndarray:
        """
        Create a special control token with the requested indicator bits.

        Channels 0-2 remain zero; channels 3-6 encode the structural markers
        (sep/reset/stop) used by downstream policies.
        """
        token = np.zeros(self.token_dim, dtype=self.dtype)
        token[4] = sep
        token[5] = reset
        token[6] = stop
        return token

    def _compose_tokens(
        self, prompt_sketches: List[ProcessedSketch], query_sketch: ProcessedSketch
    ) -> np.ndarray:
        """
        Concatenate prompt and query sketches into a single token matrix.

        The sequence adheres to the `[START, prompt₁, SEP, …, RESET, query, STOP]`
        convention required for in-context imitation learning.
        """
        segments: List[np.ndarray] = [self.special_tokens["separator"]]  # Dummy start
        for sketch in prompt_sketches:
            segments.append(self._sketch_to_tokens(sketch))
            segments.append(self.special_tokens["separator"])
        segments.append(self.special_tokens["reset"])
        segments.append(self.special_tokens["separator"])
        segments.append(self._sketch_to_tokens(query_sketch))
        segments.append(self.special_tokens["stop"])
        return np.vstack(segments).astype(self.dtype, copy=False)

    def _sketch_to_tokens(self, sketch: ProcessedSketch) -> np.ndarray:
        """
        Convert a processed sketch into its `(dx, dy, pen-up, pen-down, sep, reset, stop)` tokens.

        """
        tokens = np.zeros((sketch.length, self.token_dim), dtype=self.dtype)
        if self.coordinate_mode == "delta":
            tokens[:, 0:2] = sketch.deltas
        else:
            tokens[:, 0:2] = sketch.absolute
        tokens[:, 2] = sketch.pen
        tokens[:, 3] = 1 - sketch.pen
        return tokens

    @staticmethod
    def _resolve_augment_config(
        config: Dict[str, object],
    ) -> Dict[str, Dict[str, object]]:
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


class EpisodeBuilderSimilar(EpisodeBuilder):

    def __init__(
        self,
        *,
        fetch_family,
        fetch_sketch,
        family_ids: Sequence[str],
        k_shot: int,
        max_seq_len: Optional[int] = None,
        max_query_len: Optional[int] = None,
        max_context_len: Optional[int] = None,
        seed: Optional[int] = None,
        dtype=np.float32,
        coordinate_mode: str = "delta",
        faiss_indices: Dict[str, faiss.IndexFlatL2] = None,
        ids: Dict[str, np.ndarray] = None,
    ) -> None:
        super().__init__(
            fetch_family=fetch_family,
            fetch_sketch=fetch_sketch,
            family_ids=family_ids,
            k_shot=k_shot,
            max_seq_len=max_seq_len,
            max_query_len=max_query_len,
            max_context_len=max_context_len,
            seed=seed,
            dtype=dtype,
            coordinate_mode=coordinate_mode,
        )
        self.faiss_indices = faiss_indices
        self.ids = ids

    def build_episode(
        self,
        *,
        family_id: Optional[str] = None,
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
        query_id = sample_ids[0]
        query_sketch = self.fetch_sketch(resolved_family, query_id)

        # index, ids = self._load_family_index(resolved_family)

        faiss_idx = self.ids[resolved_family].tolist().index(int(query_id))
        q = self.faiss_indices[resolved_family].reconstruct(faiss_idx).reshape(1, -1)
        _, idxs = self.faiss_indices[resolved_family].search(q, k=self.k_shot + 1)
        closest_sketch_ids = self.ids[resolved_family][idxs[0]]
        prompt_ids = [sid for sid in closest_sketch_ids if int(sid) != int(query_id)]
        prompt_sketches = [
            self.fetch_sketch(resolved_family, sid) for sid in prompt_ids
        ]

        if len(prompt_sketches) != self.k_shot:
            raise ValueError(
                "For some reason the query sketch was included in the prompt sketches."
            )

        # prompt_ids = sample_ids[: self.k_shot]
        # query_id = sample_ids[self.k_shot]

        # prompt_sketches = [
        #     self.fetch_sketch(resolved_family, sid) for sid in prompt_ids
        # ]
        # query_sketch = self.fetch_sketch(resolved_family, query_id)

        episode_tokens = self._compose_tokens(prompt_sketches, query_sketch)
        total_len = episode_tokens.shape[0]
        if self.max_seq_len is not None and total_len > self.max_seq_len:
            raise ValueError(
                f"Episode length {total_len} exceeds limit {self.max_seq_len}."
            )
        if self.max_query_len is not None and query_sketch.length > self.max_query_len:
            raise ValueError(
                f"Query length {query_sketch.length} exceeds limit {self.max_query_len}."
            )
        if (
            self.max_context_len is not None
            and sum(sk.length for sk in prompt_sketches) + self.k_shot + 2
            > self.max_context_len
        ):
            raise ValueError(
                f"Context length {sum(sk.length for sk in prompt_sketches)} "
                f"exceeds limit {self.max_context_len}."
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

    def _load_family_index(self, family):
        index = faiss.read_index(f"{self.index_dir}family_{family}.index")
        ids = np.load(f"{self.ids_dir}{family}.npy")
        return index, ids
