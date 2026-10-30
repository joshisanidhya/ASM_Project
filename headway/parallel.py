"""One shared process pool used by every loader and by the MCMC fits."""
import atexit
from concurrent.futures import ProcessPoolExecutor

_pools = {}


def pmap(fn, items, workers=None):
    """Map fn over items on all cores (workers=None) or inline (workers=1, for debugging).

    fn must be a module-level function and items picklable (Windows spawns workers).
    The pool is created once and reused: spawning a worker re-imports numpy/scipy/pandas,
    which costs seconds on Windows, so it must not be paid per call.
    """
    items = list(items)
    if workers == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    if workers not in _pools:
        _pools[workers] = ProcessPoolExecutor(workers)
    return list(_pools[workers].map(fn, items))


@atexit.register
def _shutdown():
    for p in _pools.values():
        p.shutdown(cancel_futures=True)
