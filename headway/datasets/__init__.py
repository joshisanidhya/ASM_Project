"""Loaders: loader(path, workers=None, **kw) -> 1 Hz car-following headways [s].

Add exiD / Waymo by writing another loader with this contract and registering it here.
"""
from .highd import load_highd
from .lyft import load_lyft
from .ngsim import load_ngsim

LOADERS = {"ngsim": load_ngsim, "highd": load_highd, "lyft": load_lyft}
