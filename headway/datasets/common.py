import numpy as np

from ..distributions import ALPHA, T_MAX


def clip_headways(h):
    """Paper section 4: drop headways < 0.5 s (unsafe) and > 25 s (not following)."""
    h = np.asarray(h, float)
    return h[(h >= ALPHA) & (h <= T_MAX)]
