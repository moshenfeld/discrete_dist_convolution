
from typing import TYPE_CHECKING
import numpy as np
from .types import DistKind

if TYPE_CHECKING:
    from .types import DiscreteDist

def step_cdf_left(dist: "DiscreteDist", q: float) -> float:
    """Evaluate left-continuous CDF step function at q."""
    if dist.kind != DistKind.CDF:
        raise ValueError(f"step_cdf_left expects CDF, got {dist.kind}")
    x = dist.x
    F = dist.vals
    p_neg_inf = dist.p_neg_inf
    p_pos_inf = dist.p_pos_inf
    
    if q < x[0]:
        return float(p_neg_inf)
    if q >= x[-1]:
        return float(1.0 - p_pos_inf)
    idx = int(np.searchsorted(x, q, side="left")) - 1
    if idx < 0:
        return float(p_neg_inf)
    return float(F[idx])

def step_cdf_right(dist: "DiscreteDist", q: float) -> float:
    """Evaluate right-continuous CDF step function at q."""
    if dist.kind != DistKind.CDF:
        raise ValueError(f"step_cdf_right expects CDF, got {dist.kind}")
    x = dist.x
    F = dist.vals
    p_neg_inf = dist.p_neg_inf
    p_pos_inf = dist.p_pos_inf
    
    if q < x[0]:
        return float(p_neg_inf)
    if q >= x[-1]:
        return float(1.0 - p_pos_inf)
    idx = int(np.searchsorted(x, q, side="right")) - 1
    if idx < 0:
        return float(p_neg_inf)
    return float(F[idx])

def step_ccdf_left(dist: "DiscreteDist", q: float) -> float:
    """Evaluate left-continuous CCDF step function at q."""
    if dist.kind != DistKind.CCDF:
        raise ValueError(f"step_ccdf_left expects CCDF, got {dist.kind}")
    x = dist.x
    S = dist.vals
    p_neg_inf = dist.p_neg_inf
    p_pos_inf = dist.p_pos_inf
    
    if q < x[0]:
        return float(1.0 - p_neg_inf)
    if q >= x[-1]:
        return float(p_pos_inf)
    idx = int(np.searchsorted(x, q, side="left")) - 1
    if idx < 0:
        return float(1.0 - p_neg_inf)
    return float(S[idx])

def step_ccdf_right(dist: "DiscreteDist", q: float) -> float:
    """Evaluate right-continuous CCDF step function at q."""
    if dist.kind != DistKind.CCDF:
        raise ValueError(f"step_ccdf_right expects CCDF, got {dist.kind}")
    x = dist.x
    S = dist.vals
    p_neg_inf = dist.p_neg_inf
    p_pos_inf = dist.p_pos_inf
    
    if q < x[0]:
        return float(1.0 - p_neg_inf)
    if q >= x[-1]:
        return float(p_pos_inf)
    idx = int(np.searchsorted(x, q, side="right")) - 1
    if idx < 0:
        return float(1.0 - p_neg_inf)
    return float(S[idx])
