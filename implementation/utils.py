
from typing import Sequence
import numpy as np
from .types import DistKind, DiscreteDist

def reconcile_cdf(lower: np.ndarray, upper: np.ndarray) -> None:
    np.minimum(lower, upper, out=lower)

def reconcile_ccdf(lower: np.ndarray, upper: np.ndarray) -> None:
    np.maximum(upper, lower, out=upper)

def budget_correction_last_bin(dist: "DiscreteDist", expected_total: float = 1.0, tol: float = 1e-12) -> None:
    """Correct budget in last bin of PMF (modifies dist.vals in-place)."""
    if dist.kind != DistKind.PMF:
        raise ValueError(f"budget_correction_last_bin expects PMF, got {dist.kind}")
    pmf = dist.vals
    p_neg = dist.p_neg_inf
    p_pos = dist.p_pos_inf
    eps = (p_neg + float(pmf.sum()) + p_pos) - expected_total
    if abs(eps) <= tol and pmf.size:
        pmf[-1] -= eps

def union_grid(xs: Sequence[np.ndarray]) -> np.ndarray:
    if not xs:
        return np.empty((0,), dtype=np.float64)
    cat = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in xs], axis=0)
    uni = np.unique(cat)
    return np.ascontiguousarray(uni, dtype=np.float64)

def identity_index_for_grid(t: np.ndarray) -> int:
    if t.ndim != 1 or t.size == 0:
        return 0
    if t[0] > 0.0:
        return 0
    i = int(np.searchsorted(t, 0.0, side="right")) - 1
    return max(i, 0)
