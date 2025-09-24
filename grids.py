
from typing import Sequence, Tuple, Optional, Literal
import numpy as np

def _strict_f64(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    return x

def _decimate_uniform(t: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or t.size <= max_points:
        return t
    idx = np.linspace(0, t.size - 1, max_points, dtype=int)
    return t[idx]

def minkowski_sum_unique(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = _strict_f64(x); y = _strict_f64(y)
    sums = np.add.outer(x, y).ravel()
    t = np.unique(sums)
    return np.ascontiguousarray(t, dtype=np.float64)

def support_bounds(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = _strict_f64(x); y = _strict_f64(y)
    return float(x[0] + y[0]), float(x[-1] + y[-1])

def right_quantile_from_cdf_grid(x: np.ndarray, F: np.ndarray, q: float) -> float:
    x = _strict_f64(x); F = _strict_f64(F)
    if x.size == 0:
        return float("nan")
    lo = float(F[0]); hi = float(F[-1])
    q = float(np.clip(q, lo, hi))
    idx = int(np.searchsorted(F, q, side="left"))
    if idx >= x.size:
        idx = x.size - 1
    return float(x[idx])

def _cdf_array_from_dist_like(kind: str, x: np.ndarray, vals: np.ndarray, p_neg: float, p_pos: float) -> np.ndarray:
    x = _strict_f64(x); vals = _strict_f64(vals)
    if kind == "cdf":
        F = vals.copy()
    elif kind == "pmf":
        F = float(p_neg) + np.cumsum(vals, dtype=np.float64)
        np.minimum(F, 1.0 - float(p_pos), out=F)
    elif kind == "ccdf":
        F = 1.0 - vals
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return np.ascontiguousarray(F, dtype=np.float64)

def build_grid_trim_log_from_cdfs(xX: np.ndarray, FX: np.ndarray, xY: np.ndarray, FY: np.ndarray, *, beta: float = 1e-6, z_size: int | None = None, fallback: Literal["range-linear","minkowski"] = "range-linear") -> np.ndarray:
    xX = _strict_f64(xX); FX = _strict_f64(FX)
    xY = _strict_f64(xY); FY = _strict_f64(FY)
    if z_size is None:
        z_size = max(xX.size, xY.size)
    z_size = max(int(z_size), 2)

    q = float(np.sqrt(beta / 2.0))
    x_lo = right_quantile_from_cdf_grid(xX, FX, q)
    y_lo = right_quantile_from_cdf_grid(xY, FY, q)
    x_hi = right_quantile_from_cdf_grid(xX, FX, 1.0 - q)
    y_hi = right_quantile_from_cdf_grid(xY, FY, 1.0 - q)

    z_min = x_lo + y_lo
    z_max = x_hi + y_hi

    if (not np.isfinite(z_min)) or (not np.isfinite(z_max)) or z_min <= 0.0 or z_max <= 0.0 or z_max <= z_min:
        if fallback == "range-linear":
            lo, hi = support_bounds(xX, xY)
            lo = min(lo, z_min) if np.isfinite(z_min) else lo
            hi = max(hi, z_max) if np.isfinite(z_max) else hi
            t = np.linspace(lo, hi, z_size, dtype=np.float64)
            if t.size > 1:
                t[1:] += np.linspace(1e-12, 1e-9, t.size-1)
            return np.ascontiguousarray(t, dtype=np.float64)
        else:
            t = minkowski_sum_unique(xX, xY)
            return _decimate_uniform(t, z_size)

    t = np.geomspace(z_min, z_max, z_size, dtype=np.float64)
    if t.size > 1:
        t[1:] += np.linspace(1e-12, 1e-9, t.size-1)
    return np.ascontiguousarray(t, dtype=np.float64)

def build_grid_trim_log_from_dists(X, Y, *, beta: float = 1e-6, z_size: int | None = None, fallback: Literal["range-linear","minkowski"] = "range-linear") -> np.ndarray:
    FX = _cdf_array_from_dist_like(X.kind, X.x, X.vals, X.p_neg_inf, X.p_pos_inf)
    FY = _cdf_array_from_dist_like(Y.kind, Y.x, Y.vals, Y.p_neg_inf, Y.p_pos_inf)
    return build_grid_trim_log_from_cdfs(X.x, FX, Y.x, FY, beta=beta, z_size=z_size, fallback=fallback)

def build_grid_pmf_pmf(xX: np.ndarray, xY: np.ndarray, *, max_points: Optional[int] = None, strategy: Literal["trim-log","minkowski","range-linear"] = "trim-log", beta: float = 1e-6) -> np.ndarray:
    if strategy == "trim-log":
        # Without full CDFs here; fallback to exact Minkowski then decimate
        t = minkowski_sum_unique(xX, xY)
        t = _decimate_uniform(t, max_points if max_points is not None else t.size)
        return t
    if strategy == "minkowski":
        t = minkowski_sum_unique(xX, xY)
        t = _decimate_uniform(t, max_points if max_points is not None else t.size)
        return t
    else:
        if max_points is None or max_points < 2:
            raise ValueError("range-linear strategy requires max_points >= 2")
        lo, hi = support_bounds(xX, xY)
        t = np.linspace(lo, hi, max_points, dtype=np.float64)
        if t.size > 1:
            t[1:] += np.linspace(1e-12, 1e-9, t.size-1)
        return np.ascontiguousarray(t, dtype=np.float64)

def build_grid_pmf_cdf(xX: np.ndarray, xY: np.ndarray, *, max_points: Optional[int] = None, strategy: Literal["trim-log","minkowski","range-linear"] = "trim-log", beta: float = 1e-6) -> np.ndarray:
    return build_grid_pmf_pmf(xX, xY, max_points=max_points, strategy=strategy, beta=beta)

def build_grid_pmf_ccdf(xX: np.ndarray, xY: np.ndarray, *, max_points: Optional[int] = None, strategy: Literal["trim-log","minkowski","range-linear"] = "trim-log", beta: float = 1e-6) -> np.ndarray:
    return build_grid_pmf_pmf(xX, xY, max_points=max_points, strategy=strategy, beta=beta)

def build_grid_self_convolution(x_base: np.ndarray, T: int, *, max_points: Optional[int] = None, strategy: Literal["minkowski-repeated","range-linear"] = "range-linear") -> np.ndarray:
    x = _strict_f64(x_base)
    if T <= 0:
        raise ValueError("T must be positive")
    if strategy == "minkowski-repeated":
        t = x.copy()
        reps = T - 1
        while reps > 0:
            t = minkowski_sum_unique(t, x)
            if max_points is not None:
                t = _decimate_uniform(t, max_points)
            reps -= 1
        return t
    else:
        if max_points is None or max_points < 2:
            max_points = max(257, int(4 * np.sqrt(x.size) * np.log2(max(T,2))))
        lo = float(T * x[0]); hi = float(T * x[-1])
        t = np.linspace(lo, hi, max_points, dtype=np.float64)
        if t.size > 1:
            t[1:] += np.linspace(1e-12, 1e-9, t.size-1)
        return np.ascontiguousarray(t, dtype=np.float64)
