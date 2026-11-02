"""Standardized dataset utilities and headway clipping.

Supports exclusively:
1. highD (25 Hz -> 1 Hz, uninterrupted highway flow)
2. NGSIM (10 Hz -> 1 Hz, I-80 interrupted/uninterrupted highway flow)
3. Lyft Level 5 (10 Hz -> 1 Hz, urban mixed traffic)
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..distributions import ALPHA, T_MAX


@dataclass
class DatasetMetadata:
    name: str
    original_hz: int
    target_hz: int = 1
    total_records_processed: int = 0
    valid_headways_count: int = 0
    min_headway: float = 0.0
    max_headway: float = 0.0
    mean_headway: float = 0.0
    median_headway: float = 0.0
    std_headway: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "original_hz": self.original_hz,
            "target_hz": self.target_hz,
            "valid_headways_count": self.valid_headways_count,
            "min_headway": self.min_headway,
            "max_headway": self.max_headway,
            "mean_headway": self.mean_headway,
            "median_headway": self.median_headway,
            "std_headway": self.std_headway,
        }


def clip_headways(h: np.ndarray) -> np.ndarray:
    """Filter headways to the paper's valid car-following window [ALPHA, T_MAX].

    Paper section 4:
    - Excludes headways < 0.5 s (below physical safe human reaction time threshold)
    - Excludes headways > 25.0 s (free flow, not representing car-following interactions)
    """
    h_arr = np.asarray(h, dtype=float)
    valid = np.isfinite(h_arr) & (h_arr >= ALPHA) & (h_arr <= T_MAX)
    return h_arr[valid]


def compute_dataset_metadata(name: str, original_hz: int, headways: np.ndarray) -> DatasetMetadata:
    """Compute summary statistics for a validated headway array."""
    if len(headways) == 0:
        return DatasetMetadata(name=name, original_hz=original_hz)
    return DatasetMetadata(
        name=name,
        original_hz=original_hz,
        target_hz=1,
        total_records_processed=len(headways),
        valid_headways_count=len(headways),
        min_headway=float(np.min(headways)),
        max_headway=float(np.max(headways)),
        mean_headway=float(np.mean(headways)),
        median_headway=float(np.median(headways)),
        std_headway=float(np.std(headways)),
    )
