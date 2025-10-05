
from typing import Sequence, Tuple, Optional, Literal, Protocol, Union, Callable
import numpy as np
from scipy import stats
from .types import Mode, Spacing

def _strict_f64(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    return x


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


def build_grid_from_support_bounds(dist_1, dist_2, spacing, beta):
    """
    Build grid for convolution using support bounds based on probability mass thresholds.
    
    Given two distributions dist_1, dist_2, spacing type, and beta:
    1. Ensure dist_1 size = dist_2 size
    2. Compute x_min, x_max using beta (probability mass thresholds)
    3. Compute y_min, y_max using beta (probability mass thresholds)
    4. Compute z_min = x_min + y_min, z_max = x_max + y_max
    5. Compute new grid of same size as input distributions, linearly/geometrically spaced between z_min and z_max
    
    Parameters:
    -----------
    dist_1, dist_2 : DiscreteDist objects
        Input distributions
    spacing : Spacing
        Spacing.LINEAR for linear spacing, Spacing.GEOMETRIC for geometric spacing
    beta : float
        Parameter controlling grid range based on probability mass thresholds
        
    Returns:
    --------
    t : np.ndarray
        Output grid for convolution result
    """
    x1 = _strict_f64(dist_1.x)
    x2 = _strict_f64(dist_2.x)
    p1 = _strict_f64(dist_1.vals)
    p2 = _strict_f64(dist_2.vals)
    
    # Determine output grid size (same as inputs, take max if they differ)
    out_size = max(x1.size, x2.size)
    if out_size < 2:
        raise ValueError(f"Grid size must be >= 2, got {out_size}")
    
    # Compute probability mass threshold
    # threshold = np.sqrt(beta/2)
    threshold = beta/2
    
    # Find x_min and x_max using beta
    x1_min_ind_arr = np.where(np.cumsum(p1) <= threshold)[0]
    x1_min_ind = x1_min_ind_arr[-1] if len(x1_min_ind_arr) > 0 else 0
    x1_min = x1[x1_min_ind]

    x1_max_ind_arr = np.where(np.cumsum(p1[::-1]) <= threshold)[0]
    x1_max_ind = np.size(p1) - 1 - (x1_max_ind_arr[-1] if len(x1_max_ind_arr) > 0 else 0)
    x1_max = x1[x1_max_ind]
    
    # Find y_min and y_max using beta
    x2_min_ind_arr = np.where(np.cumsum(p2) <= threshold)[0]
    x2_min_ind = x2_min_ind_arr[-1] if len(x2_min_ind_arr) > 0 else 0
    x2_min = x2[x2_min_ind]

    x2_max_ind_arr = np.where(np.cumsum(p2[::-1]) <= threshold)[0]
    x2_max_ind = np.size(p2) - 1 - (x2_max_ind_arr[-1] if len(x2_max_ind_arr) > 0 else 0)
    x2_max = x2[x2_max_ind]
    
    # Compute support bounds
    x_out_min = x1_min + x2_min
    x_out_max = x1_max + x2_max
    
    if not np.isfinite(x_out_min) or not np.isfinite(x_out_max):
        raise ValueError(f"Support bounds not finite: x_out_min={x_out_min}, x_out_max={x_out_max}")
    
    if x_out_max <= x_out_min:
        raise ValueError(f"Invalid support bounds: x_out_min={x_out_min} >= x_out_max={x_out_max}")
    
    # Create grid based on spacing type
    return _sample_from_range(x_out_min, x_out_max, out_size, spacing)

def discretize_continuous_to_pmf(dist: stats.rv_continuous,
                                  n_grid: int,
                                  beta: float,
                                  mode: Mode,
                                  spacing: Spacing) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Discretize a continuous distribution onto a grid using quantile-based spacing.
    
    Parameters:
    -----------
    dist : scipy.stats.rv_continuous
        Continuous distribution object with cdf(), ppf(), and sf() methods
    n_grid : int
        Number of grid points
    beta : float
        Tail probability to trim
    mode : Mode
        Mode.DOMINATES for upper bound, Mode.IS_DOMINATED for lower bound
    spacing : Spacing
        Spacing.LINEAR for linspace, Spacing.GEOMETRIC for geomspace
        
    Returns:
    --------
    x : np.ndarray
        Grid points
    pmf : np.ndarray
        PMF values at grid points
    p_neg_inf : float
        Probability mass at -∞
    p_pos_inf : float
        Probability mass at +∞
        
    Algorithm per IMPLEMENTATION_GUIDE_NUMBA.md:
    1. Determine range via quantiles: [q_min, q_max] = [beta/2, 1-beta/2]
    2. Create geometric spacing in this range
    3. Discretize using CDF differences
    4. Assign tail masses to ±∞ based on mode
    """
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    if not (0 < beta < 1):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    
    # Step 1: Determine range via quantiles
    x_min = dist.ppf(beta / 2)
    x_max = dist.ppf(1 - beta / 2)
    
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError(f"Quantiles not finite: x_min={x_min}, x_max={x_max}")
    
    # Step 2: Create spacing based on spacing parameter
    x = _sample_from_range(x_min, x_max, n_grid, spacing)

    # Step 3: Discretize to PMF using CDF
    # Use CDF to sample the first half of the grid and CCDF to sample the second half
    x_center_ind = np.searchsorted(x, dist.ppf(0.5))
    x_left = x[:x_center_ind+1]
    x_right = x[x_center_ind:]
    CDF_left = dist.cdf(x_left)
    CCDF_right = dist.sf(x_right)
    combined_PMF = np.concatenate([np.diff(CDF_left), -np.diff(CCDF_right)])
    left = CDF_left[0]
    right = CCDF_right[-1]
    
    # Step 4: Assign tail masses to ±∞ based on mode
    if mode == Mode.DOMINATES:
        pmf = np.concatenate(([left], combined_PMF))
        p_neg_inf = 0.0
        p_pos_inf = right
    else:
        pmf = np.concatenate([combined_PMF, [right]])
        p_neg_inf = left
        p_pos_inf = 0.0
        
    return x, pmf, float(p_neg_inf), float(p_pos_inf)

def _sample_from_range(x_min: float, x_max: float, n_grid: int, spacing: Spacing) -> np.ndarray:
    if spacing == Spacing.GEOMETRIC:
        if x_min <= 0:
            raise ValueError(f"Cannot use geometric spacing when range [{x_min:.6f}, {x_max:.6f}] contains negative values.")
        else:
            x = np.geomspace(x_min, x_max, n_grid, dtype=np.float64)
    else:
        x = np.linspace(x_min, x_max, n_grid, dtype=np.float64)
    return x
