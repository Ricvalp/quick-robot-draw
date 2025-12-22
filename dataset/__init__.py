"""
QuickDraw dataset processing package.

This package provides utilities to preprocess Quick, Draw! sketches, build
K-shot imitation learning episodes, store processed data efficiently, and
serve it through PyTorch `Dataset` interfaces.
"""

from . import visualize
from .diffusion import (
    ContextQueryInContextDiffusionCollator,
    DiffusionCollator,
    InContextDiffusionCollator,
)
from .episode_builder import Episode, EpisodeBuilder, EpisodeBuilderSimilar
from .loader import QuickDrawEpisodes, QuickDrawSketches
from .lstm import ILRNNCollator, InContextSketchRNNCollator
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
    "EpisodeBuilderSimilar",
    "Episode",
    "StorageConfig",
    "DatasetManifest",
    "SketchStorage",
    "EpisodeStorage",
    "QuickDrawEpisodes",
    "QuickDrawSketches",
    "DiffusionCollator",
    "InContextDiffusionCollator",
    "visualize",
    "RasterizerConfig",
    "rasterize_absolute_points",
    "rasterize_processed_sketch",
    "ILRNNCollator",
    "InContextSketchRNNCollator",
    "ContextQueryInContextDiffusionCollator",
]
