"""Dataset loaders registry for the 3 supported datasets:
1. highD: German highway drone dataset (25 Hz -> 1 Hz, uninterrupted highway flow)
2. NGSIM: FHWA I-80 trajectory dataset (10 Hz -> 1 Hz, interrupted/uninterrupted highway flow)
3. Lyft: Level 5 AV prediction dataset (10 Hz -> 1 Hz, urban mixed traffic)

Each loader extracts 1 Hz car-following time headways in [0.5, 25.0] seconds.
"""
from typing import Callable, Dict

from .common import DatasetMetadata, clip_headways, compute_dataset_metadata
from .highd import load_highd
from .lyft import load_lyft
from .ngsim import load_ngsim

LOADERS: Dict[str, Callable] = {
    "highd": load_highd,
    "ngsim": load_ngsim,
    "lyft": load_lyft,
}

__all__ = [
    "LOADERS",
    "load_highd",
    "load_ngsim",
    "load_lyft",
    "clip_headways",
    "DatasetMetadata",
    "compute_dataset_metadata",
]
