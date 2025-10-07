"""
Enhanced implementation module for discrete distribution convolution.

Improvements over v1:
- Consistent use of DiscreteDist throughout
- Fixed bin width strategy option
- Better API design
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np
from scipy import stats
from numba import njit

# =============================================================================
# TYPES AND ENUMS
# =============================================================================

class BoundType(Enum):
    """Tie-breaking bound_type for discretization."""
    DOMINATES = "DOMINATES"
    IS_DOMINATED = "IS_DOMINATED"

class SpacingType(Enum):
    """Grid spacing_type strategy."""
    LINEAR = "linear"
    GEOMETRIC = "geometric"

class GridStrategy(Enum):
    """Grid generation strategy for convolution."""
    FIXED_NUM_POINTS = "fixed_points"  # Maintain fixed number of grid points
    FIXED_WIDTH = "fixed_width"    # Maintain fixed bin width

class SummationMethod(Enum):
    """Summation method for numerical stability."""
    STANDARD = "standard"
    KAHAN = "kahan"
    SORTED = "sorted"

@dataclass
class DiscreteDist:
    x: np.ndarray
    PMF: np.ndarray
    p_neg_inf: float = 0.0
    p_pos_inf: float = 0.0
    
    def __post_init__(self):
        tolerance = 1e-13
        if self.x.ndim != 1 or self.PMF.ndim != 1 or self.x.shape != self.PMF.shape:
            raise ValueError("x and PMF must be 1-D arrays of equal length")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x must be strictly increasing")
        if any(self.PMF < -tolerance):
            raise ValueError("PMF must be nonnegative")
        if self.p_neg_inf < 0:
            raise ValueError("p_neg_inf must be nonnegative")
        if self.p_pos_inf < 0:
            raise ValueError("p_pos_inf must be nonnegative")

        sorted_vals = np.sort(np.abs(self.PMF)) * np.sign(self.PMF[np.argsort(np.abs(self.PMF))])
        pmf_sum = np.sum(sorted_vals)
        
        total_mass = pmf_sum + self.p_neg_inf + self.p_pos_inf
        mass_error = abs(total_mass - 1.0)
        if mass_error > tolerance:
            error_msg = f"MASS CONSERVATION ERROR"
            error_msg += f": Error={mass_error:.2e} (tolerance={tolerance:.0e})"
            error_msg += f", PMF sum={pmf_sum:.15f}"
            error_msg += f", p_neg_inf={self.p_neg_inf:.2e}"
            error_msg += f", p_pos_inf={self.p_pos_inf:.2e}"
            error_msg += f", Total mass={total_mass:.15f}"
            raise ValueError(error_msg)


# =============================================================================
# GRID GENERATION AND DISCRETIZATION
# =============================================================================

def discretize_continuous_to_pmf(dist: stats.rv_continuous,
                                 n_grid: int,
                                 beta: float,
                                 bound_type: BoundType,
                                 spacing_type: SpacingType = SpacingType.GEOMETRIC,
                                 tail_switch: float = 1e-12
                                 ) -> DiscreteDist:
    """
    Discretize a continuous distribution onto a grid.
    
    Returns DiscreteDist object directly (changed from v1 which returned tuple).
    """
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    if not (0 < beta < 1):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    
    # Determine range via quantiles
    x_min = dist.ppf(beta / 2)
    x_max = dist.isf(beta / 2)
    
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError(f"Quantiles not finite: x_min={x_min}, x_max={x_max}")
    
    # Create grid
    x = _sample_from_range(x_min, x_max, n_grid, spacing_type)

    # Create bin edges
    edges = np.empty(n_grid + 1)
    edges[0] = x[0] - (x[1] - x[0]) / 2
    edges[1:-1] = (x[:-1] + x[1:]) / 2
    edges[-1] = x[-1] + (x[-1] - x[-2]) / 2
    
    # Compute bin masses stably
    a, b = edges[:-1], edges[1:]
    
    F_a, F_b = dist.cdf(a), dist.cdf(b)
    S_a, S_b = dist.sf(a), dist.sf(b)
    
    # Classify bins into three regions
    right_tail = (S_a < tail_switch) & (S_b < tail_switch)
    left_tail = (F_a < tail_switch) & (F_b < tail_switch)
    middle = ~(right_tail | left_tail)
    
    pmf = np.empty(n_grid, dtype=np.float64)
    pmf[middle] = F_b[middle] - F_a[middle]
    
    if np.any(right_tail):
        logSa = dist.logsf(a[right_tail])
        logSb = dist.logsf(b[right_tail])
        pmf[right_tail] = np.exp(logSa) * (-np.expm1(logSb - logSa))
    
    if np.any(left_tail):
        logFa = dist.logcdf(a[left_tail])
        logFb = dist.logcdf(b[left_tail])
        pmf[left_tail] = np.exp(logFb) * (-np.expm1(logFa - logFb))
    
    pmf = np.maximum(pmf, 0.0)
    
    # Compute infinity masses
    if F_a[0] < tail_switch:
        p_neg_mass = np.exp(dist.logcdf(edges[0]))
    else:
        p_neg_mass = F_a[0]
    
    if S_b[-1] < tail_switch:
        p_pos_mass = np.exp(dist.logsf(edges[-1]))
    else:
        p_pos_mass = S_b[-1]
    
    # Normalize
    finite_mass = pmf.sum(dtype=np.float64)
    total_mass = finite_mass + p_neg_mass + p_pos_mass
    
    if not np.isclose(total_mass, 1.0, rtol=0, atol=1e-13):
        if finite_mass > 0.0:
            pmf *= (1.0 - p_neg_mass - p_pos_mass) / finite_mass
    
    # Assign infinity masses based on bound_type
    if bound_type == BoundType.DOMINATES:
        p_neg_inf = 0.0
        p_pos_inf = p_neg_mass + p_pos_mass
    else:  # IS_DOMINATED
        p_neg_inf = p_neg_mass + p_pos_mass
        p_pos_inf = 0.0
    
    return DiscreteDist(x=x, PMF=pmf, p_neg_inf=p_neg_inf, p_pos_inf=p_pos_inf)

# =============================================================================
# CONVOLUTION KERNEL
# =============================================================================
def _sample_from_range(x_min: float, x_max: float, n_grid: int, spacing_type: SpacingType) -> np.ndarray:
    """Sample points from a range using specified spacing_type strategy."""
    if spacing_type == SpacingType.GEOMETRIC:
        if x_min <= 0:
            raise ValueError(f"Cannot use geometric spacing_type when range [{x_min:.6f}, {x_max:.6f}] contains negative values.")
        else:
            x = np.geomspace(x_min, x_max, n_grid, dtype=np.float64)
    else:
        x = np.linspace(x_min, x_max, n_grid, dtype=np.float64)
    return x

def _compute_bin_width_or_ratio(dist: DiscreteDist, spacing_type: SpacingType) -> float:
    """
    Compute the characteristic bin width/ratio of a distribution.
    
    For linear spacing_type: median of differences
    For geometric spacing_type: median of log-differences (ratio - 1)
    """
    if len(dist.x) < 2:
        raise ValueError("Distribution must have at least 2 points to compute bin width")
    
    if spacing_type == SpacingType.LINEAR:
        diffs = np.diff(dist.x)
        median_diff = np.median(diffs)
        if any(np.abs(diffs - median_diff) > 1e-12):
            raise ValueError(f"Distribution has non-uniform bin widths: median_diff={median_diff}, diffs={diffs}")
        return median_diff
    else:  # GEOMETRIC
        if np.any(dist.x <= 0):
            raise ValueError("Cannot compute geometric bin width for non-positive values")
        log_ratios = np.log(dist.x[1:] / dist.x[:-1])
        med_log_ratio = np.median(log_ratios)
        if any(np.abs(log_ratios - med_log_ratio) > 1e-12):
            raise ValueError(f"Distribution has non-uniform bin widths: median_ratio={np.median(log_ratios)}, log_ratios={log_ratios}")
        return np.exp(med_log_ratio)

def _calc_new_grid_from_distributions(dist_1: DiscreteDist, dist_2: DiscreteDist, beta: float,
                                      spacing_type: SpacingType, grid_strategy: GridStrategy) -> np.ndarray:
    """
    Build grid for convolution using support bounds.
    
    Parameters:
    -----------
    dist_1, dist_2 : DiscreteDist
        Input distributions
    spacing_type : SpacingType
        LINEAR or GEOMETRIC spacing_type
    beta : float
        Probability mass threshold for trimming
    grid_strategy : GridStrategy
        FIXED_NUM_POINTS: maintain number of grid points (original behavior)
        FIXED_WIDTH: maintain bin width
    target_bin_width : Optional[float]
        Bin width to maintain (if None, computed from inputs)
    """
    x1, p1 = dist_1.x, dist_1.PMF
    x2, p2 = dist_2.x, dist_2.PMF
    
    # Compute support bounds using cumulative mass threshold
    threshold = beta / 2

    x1_min_ind_arr = np.where(np.cumsum(p1) <= threshold)[0]
    x1_min_ind = x1_min_ind_arr[-1] if len(x1_min_ind_arr) > 0 else 0
    x1_min = x1[x1_min_ind]

    x1_max_ind_arr = np.where(np.cumsum(p1[::-1]) <= threshold)[0]
    x1_max_ind = len(p1) - 1 - (x1_max_ind_arr[-1] if len(x1_max_ind_arr) > 0 else 0)
    x1_max = x1[x1_max_ind]
    
    x2_min_ind_arr = np.where(np.cumsum(p2) <= threshold)[0]
    x2_min_ind = x2_min_ind_arr[-1] if len(x2_min_ind_arr) > 0 else 0
    x2_min = x2[x2_min_ind]

    x2_max_ind_arr = np.where(np.cumsum(p2[::-1]) <= threshold)[0]
    x2_max_ind = len(p2) - 1 - (x2_max_ind_arr[-1] if len(x2_max_ind_arr) > 0 else 0)
    x2_max = x2[x2_max_ind]
    
    x_out_min = x1_min + x2_min
    x_out_max = x1_max + x2_max
    if not np.isfinite(x_out_min) or not np.isfinite(x_out_max):
        raise ValueError(f"Support bounds not finite: x_out_min={x_out_min}, x_out_max={x_out_max}")
    if x_out_max <= x_out_min:
        raise ValueError(f"Invalid support bounds: x_out_min={x_out_min} >= x_out_max={x_out_max}")
    
    # Determine number of grid points based on strategy
    if grid_strategy == GridStrategy.FIXED_NUM_POINTS:
        out_size = max(x1.size, x2.size)
    else:  # FIXED_WIDTH
        width_or_ratio_1 = _compute_bin_width_or_ratio(dist_1, spacing_type)
        width_or_ratio_2 = _compute_bin_width_or_ratio(dist_2, spacing_type)
        target_width_or_ratio = min(width_or_ratio_1, width_or_ratio_2)  # Use finer resolution
        
        if spacing_type == SpacingType.LINEAR:
            out_size = int(np.ceil((x_out_max - x_out_min) / target_width_or_ratio)) + 1
        else:  # GEOMETRIC
            # target_width_or_ratio is the relative width ratio (x_out_max / x_out_min)
            ratio =  target_width_or_ratio
            out_size = int(np.ceil(np.log(x_out_max / x_out_min) / np.log(ratio))) + 1    
    return _sample_from_range(x_out_min, x_out_max, max(out_size, 2), spacing_type)

@njit(cache=True)
def _pmf_convolution_kernel(x1: np.ndarray, p1: np.ndarray, 
                            x2: np.ndarray, p2: np.ndarray,
                            x_out: np.ndarray, dominates: bool):
    """Kahan-compensated PMF×PMF kernel."""
    pmf_out = np.zeros(x_out.size, dtype=np.float64)
    compensations = np.zeros(x_out.size, dtype=np.float64)
    pneg_extra = 0.0
    ppos_extra = 0.0
    pneg_comp = 0.0
    ppos_comp = 0.0
    
    for i in range(x1.size):
        last_idx = 0
        for j in range(x2.size):
            z = x1[i] + x2[j]
            mass = p1[i] * p2[j]
            
            if dominates:
                while last_idx < x_out.size and x_out[last_idx] < z:
                    last_idx += 1
                
                if last_idx >= x_out.size:
                    y = mass - ppos_comp
                    t = ppos_extra + y
                    ppos_comp = (t - ppos_extra) - y
                    ppos_extra = t
                else:
                    y = mass - compensations[last_idx]
                    t = pmf_out[last_idx] + y
                    compensations[last_idx] = (t - pmf_out[last_idx]) - y
                    pmf_out[last_idx] = t
            else:
                while last_idx < x_out.size and x_out[last_idx] <= z:
                    last_idx += 1
                
                idx = last_idx - 1
                if idx < 0:
                    y = mass - pneg_comp
                    t = pneg_extra + y
                    pneg_comp = (t - pneg_extra) - y
                    pneg_extra = t
                else:
                    y = mass - compensations[idx]
                    t = pmf_out[idx] + y
                    compensations[idx] = (t - pmf_out[idx]) - y
                    pmf_out[idx] = t
    
    return pmf_out, pneg_extra, ppos_extra


# =============================================================================
# MAIN API FUNCTIONS
# =============================================================================
def self_convolve_discrete_distributions(dist: DiscreteDist, T: int, beta: float, bound_type: BoundType,
                                         spacing_type: SpacingType = SpacingType.GEOMETRIC, grid_strategy: GridStrategy = GridStrategy.FIXED_WIDTH
                                         ) -> DiscreteDist:
    """
    Self-convolve a PMF T times with configurable numerical precision and grid strategy.
    """    
    if T == 1:
        return dist
    
    base_dist = dist
    acc_dist = None
    while T > 0:
        if T & 1:
            if acc_dist is None:
                acc_dist = base_dist
            else:
                acc_dist = convolve_discrete_distributions(dist_1=acc_dist, dist_2=base_dist, beta=beta, 
                                                           bound_type=bound_type, spacing_type=spacing_type, grid_strategy=grid_strategy)
        T >>= 1
        if T > 0:
            base_dist = convolve_discrete_distributions(dist_1=base_dist, dist_2=base_dist, beta=beta, 
                                                        bound_type=bound_type, spacing_type=spacing_type, grid_strategy=grid_strategy)
    return acc_dist


def convolve_discrete_distributions(dist_1: DiscreteDist, dist_2: DiscreteDist, beta: float, bound_type: BoundType,
                                    spacing_type: SpacingType = SpacingType.GEOMETRIC, grid_strategy: GridStrategy = GridStrategy.FIXED_WIDTH
                                    ) -> DiscreteDist:
    """Convolve two PMFs with configurable numerical precision and grid strategy."""
    if bound_type == BoundType.DOMINATES and (dist_1.p_neg_inf > 0 or dist_2.p_neg_inf > 0):
        raise ValueError(f'DOMINATES bound_type requires p_neg_inf=0 for both inputs')
    if bound_type == BoundType.IS_DOMINATED and (dist_1.p_pos_inf > 0 or dist_2.p_pos_inf > 0):
        raise ValueError(f'IS_DOMINATED bound_type requires p_pos_inf=0 for both inputs')

    x_out = _calc_new_grid_from_distributions(dist_1=dist_1, dist_2=dist_2, beta=beta, spacing_type=spacing_type, grid_strategy=grid_strategy)
    PMF_out, p_neg_extra, p_pos_extra = _pmf_convolution_kernel(x1=dist_1.x, p1=dist_1.PMF, x2=dist_2.x, p2=dist_2.PMF, x_out=x_out, dominates=(bound_type == BoundType.DOMINATES))
    p_neg_inf = p_neg_extra + dist_1.p_neg_inf + dist_2.p_neg_inf - dist_1.p_neg_inf*dist_2.p_neg_inf
    p_pos_inf = p_pos_extra + dist_1.p_pos_inf + dist_2.p_pos_inf - dist_1.p_pos_inf*dist_2.p_pos_inf

    if bound_type == BoundType.DOMINATES and p_neg_inf > 0:
        raise ValueError(f'DOMINATES bound_type requires output p_neg_inf=0, got {p_neg_inf}')
    if bound_type == BoundType.IS_DOMINATED and p_pos_inf > 0:
        raise ValueError(f'IS_DOMINATED bound_type requires output p_pos_inf=0, got {p_pos_inf}')
    return DiscreteDist(x=x_out, PMF=PMF_out, p_neg_inf=p_neg_inf, p_pos_inf=p_pos_inf)
