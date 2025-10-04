from typing import Sequence
import numpy as np

def union_grid(xs: Sequence[np.ndarray]) -> np.ndarray:
    """Create union of multiple grids by concatenating and removing duplicates."""
    if not xs:
        return np.empty((0,), dtype=np.float64)
    cat = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in xs], axis=0)
    uni = np.unique(cat)
    return np.ascontiguousarray(uni, dtype=np.float64)
