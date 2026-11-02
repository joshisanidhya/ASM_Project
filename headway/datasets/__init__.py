"""Dataset loaders registry for the 2 supported datasets:
1. highD: German highway drone dataset (25 Hz -> 1 Hz, uninterrupted highway flow)
2. NGSIM: FHWA I-80 trajectory dataset (10 Hz -> 1 Hz, interrupted/uninterrupted highway flow)

Each loader extracts 1 Hz car-following time headways in [0.5, 25.0] seconds.
"""
from typing import Callable, Dict

from .common import DatasetMetadata, clip_headways, compute_dataset_metadata
from .highd import load_highd
from .ngsim import load_ngsim

LOADERS: Dict[str, Callable] = {
    "highd": load_highd,
    "ngsim": load_ngsim,
}

__all__ = [
    "LOADERS",
    "load_highd",
    "load_ngsim",
    "clip_headways",
    "DatasetMetadata",
    "compute_dataset_metadata",
]
