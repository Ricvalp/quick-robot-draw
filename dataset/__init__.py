"""
QuickDraw dataset processing package.

This package provides utilities to preprocess Quick, Draw! sketches, build
K-shot imitation learning episodes, store processed data efficiently, and
serve it through PyTorch `Dataset` interfaces.
"""

from .preprocess import (
    QuickDrawPreprocessor,
    ProcessedSketch,
    RawSketch,
    load_ndjson_sketches,
    load_binary_sketches,
)
from .episode_builder import EpisodeBuilder, Episode
from .storage import (
    StorageConfig,
    DatasetManifest,
    SketchStorage,
    EpisodeStorage,
)
from .loader import QuickDrawEpisodes, QuickDrawEpisodesAbsolute, quickdraw_collate_fn
from .diffusion import DiffusionCollator
from . import visualize

__all__ = [
    "QuickDrawPreprocessor",
    "ProcessedSketch",
    "RawSketch",
    "load_ndjson_sketches",
    "load_binary_sketches",
    "EpisodeBuilder",
    "Episode",
    "StorageConfig",
    "DatasetManifest",
    "SketchStorage",
    "EpisodeStorage",
    "QuickDrawEpisodes",
    "QuickDrawEpisodesAbsolute",
    "quickdraw_collate_fn",
    "DiffusionCollator",
    "visualize",
]
