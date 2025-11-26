"""
QuickDraw dataset processing package.

This package provides utilities to preprocess Quick, Draw! sketches, build
K-shot imitation learning episodes, store processed data efficiently, and
serve it through PyTorch `Dataset` interfaces.
"""

from . import visualize
from .diffusion import DiffusionCollator
from .episode_builder import Episode, EpisodeBuilder
from .loader import QuickDrawEpisodes
from .preprocess import (
    ProcessedSketch,
    QuickDrawPreprocessor,
    RawSketch,
    load_binary_sketches,
    load_ndjson_sketches,
)
from .rasterize import RasterizerConfig, rasterize_absolute_points
from .storage import DatasetManifest, EpisodeStorage, SketchStorage, StorageConfig

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
    "DiffusionCollator",
    "InContextDiffusionCollator",
    "visualize",
    "RasterizerConfig",
    "rasterize_absolute_points",
    "rasterize_processed_sketch",
]
