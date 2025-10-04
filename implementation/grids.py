
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


def build_grid_from_support_bounds(dist_x, dist_y, spacing, beta):
    """
    Build grid for convolution using support bounds based on probability mass thresholds.
    
    Given two distributions dist_x, dist_y, spacing type, and beta:
    1. Ensure dist_x size = dist_y size
    2. Compute x_min, x_max using beta (probability mass thresholds)
    3. Compute y_min, y_max using beta (probability mass thresholds)
    4. Compute z_min = x_min + y_min, z_max = x_max + y_max
    5. Compute new grid of same size as input distributions, linearly/geometrically spaced between z_min and z_max
    
    Parameters:
    -----------
    dist_x, dist_y : DiscreteDist objects
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
    xX = _strict_f64(dist_x.x)
    xY = _strict_f64(dist_y.x)
    pX = _strict_f64(dist_x.vals)
    pY = _strict_f64(dist_y.vals)
    
    # Determine output grid size (same as inputs, take max if they differ)
    z_size = max(xX.size, xY.size)
    if z_size < 2:
        raise ValueError(f"Grid size must be >= 2, got {z_size}")
    
    # Compute probability mass threshold
    threshold = np.sqrt(beta/2)
    
    # Find x_min and x_max using beta
    where_pX = np.where(np.cumsum(pX) <= threshold)[0]
    iXmin = where_pX[-1] if len(where_pX) > 0 else 0
    
    where_pX_rev = np.where(np.cumsum(pX[::-1]) <= threshold)[0]
    iXmax = np.size(pX) - 1 - (where_pX_rev[-1] if len(where_pX_rev) > 0 else 0)
    
    x_min = xX[iXmin]
    x_max = xX[iXmax]
    
    # Find y_min and y_max using beta
    where_pY = np.where(np.cumsum(pY) <= threshold)[0]
    iYmin = where_pY[-1] if len(where_pY) > 0 else 0
    
    where_pY_rev = np.where(np.cumsum(pY[::-1]) <= threshold)[0]
    iYmax = np.size(pY) - 1 - (where_pY_rev[-1] if len(where_pY_rev) > 0 else 0)
    
    y_min = xY[iYmin]
    y_max = xY[iYmax]
    
    # Compute support bounds
    z_min = x_min + y_min
    z_max = x_max + y_max
    
    if not np.isfinite(z_min) or not np.isfinite(z_max):
        raise ValueError(f"Support bounds not finite: z_min={z_min}, z_max={z_max}")
    
    if z_max <= z_min:
        raise ValueError(f"Invalid support bounds: z_min={z_min} >= z_max={z_max}")
    
    # Create grid based on spacing type
    if spacing == Spacing.GEOMETRIC:
        # Geometric spacing
        if z_min > 0:
            # Positive support: use geomspace directly
            t = np.geomspace(z_min, z_max, z_size, dtype=np.float64)
        elif z_max < 0:
            # Negative support: geomspace on absolute values, then negate and reverse
            t = -np.geomspace(-z_max, -z_min, z_size, dtype=np.float64)[::-1]
        else:
            # Support contains 0: cannot use geometric spacing
            raise ValueError(
                f"Cannot use geometric spacing when range [{z_min:.6f}, {z_max:.6f}] contains 0. "
                f"Use spacing=Spacing.LINEAR instead."
            )
    else:
        # Linear spacing
        t = np.linspace(z_min, z_max, z_size, dtype=np.float64)
    
    # Add small perturbation to avoid exact collisions (except first point)
    if t.size > 1:
        t[1:] += np.linspace(1e-12, 1e-9, t.size-1)
    
    return np.ascontiguousarray(t, dtype=np.float64)

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
    q_min = dist.ppf(beta / 2)
    q_max = dist.ppf(1 - beta / 2)
    
    if not np.isfinite(q_min) or not np.isfinite(q_max):
        raise ValueError(f"Quantiles not finite: q_min={q_min}, q_max={q_max}")
    
    # Step 2: Create spacing based on spacing parameter
    if spacing == Spacing.GEOMETRIC:
        if q_min <= 0:
            raise ValueError(f"Cannot use geometric spacing when range [{q_min:.6f}, {q_max:.6f}] contains negative values.")
        else:
            x = np.geomspace(q_min, q_max, n_grid, dtype=np.float64)
    else:
        x = np.linspace(q_min, q_max, n_grid, dtype=np.float64)
        
    # Step 3: Discretize to PMF using CDF
    # Step 4: Assign tail masses to ±∞ based on mode
    if mode == Mode.DOMINATES:
        F = dist.cdf(x)
        pmf = np.concatenate(([F[0]], np.diff(F)))
        p_neg_inf = 0.0
        p_pos_inf = dist.sf(x[-1])
    else:
        F = dist.sf(x)
        pmf = np.concatenate([-np.diff(F), [F[-1]]])
        p_neg_inf = dist.cdf(x[0])
        p_pos_inf = 0.0
        
    return x, pmf, float(p_neg_inf), float(p_pos_inf)
