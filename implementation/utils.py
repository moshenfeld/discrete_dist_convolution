
from typing import Sequence
import numpy as np

def ensure_f64_c(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)

def running_max_inplace(a: np.ndarray) -> None:
    if a.ndim != 1:
        raise ValueError("running_max_inplace expects 1-D")
    cur = -np.inf
    for i in range(a.size):
        v = a[i]
        if v < cur:
            a[i] = cur
        else:
            cur = v

def running_min_reverse_inplace(a: np.ndarray) -> None:
    if a.ndim != 1:
        raise ValueError("running_min_reverse_inplace expects 1-D")
    cur = np.inf
    for i in range(a.size - 1, -1, -1):
        v = a[i]
        if v > cur:
            a[i] = cur
        else:
            cur = v

def clip_to_feasible_cdf(F: np.ndarray, p_neg_x: float, mX: float, p_pos_y: float) -> None:
    lo = p_neg_x
    hi = p_neg_x + mX * (1.0 - p_pos_y)
    np.clip(F, lo, hi, out=F)

def clip_to_feasible_ccdf(S: np.ndarray, p_pos_x: float) -> None:
    np.clip(S, p_pos_x, 1.0, out=S)

def reconcile_cdf(lower: np.ndarray, upper: np.ndarray) -> None:
    np.minimum(lower, upper, out=lower)

def reconcile_ccdf(lower: np.ndarray, upper: np.ndarray) -> None:
    np.maximum(upper, lower, out=upper)

def budget_correction_last_bin(pmf: np.ndarray, p_neg: float, p_pos: float, expected_total: float = 1.0, tol: float = 1e-12) -> None:
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
