"""
Enhanced implementation module for discrete distribution convolution.

Improvements over v1:
- Consistent use of DiscreteDist throughout
- Fixed bin width strategy option
- Better API design
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
from enum import Enum
import numpy as np
from scipy import stats
from numba import njit

# =============================================================================
# TYPES AND ENUMS
# =============================================================================

class Mode(Enum):
    """Tie-breaking mode for discretization."""
    DOMINATES = "DOMINATES"
    IS_DOMINATED = "IS_DOMINATED"

class Spacing(Enum):
    """Grid spacing strategy."""
    LINEAR = "linear"
    GEOMETRIC = "geometric"

class GridStrategy(Enum):
    """Grid generation strategy for convolution."""
    FIXED_POINTS = "fixed_points"  # Maintain fixed number of grid points
    FIXED_WIDTH = "fixed_width"    # Maintain fixed bin width

class DistKind(Enum):
    """Distribution kind."""
    PMF = "pmf"
    CDF = "cdf"
    CCDF = "ccdf"

class SummationMethod(Enum):
    """Summation method for numerical stability."""
    STANDARD = "standard"
    KAHAN = "kahan"
    SORTED = "sorted"

@dataclass
class DiscreteDist:
    x: np.ndarray
    kind: DistKind
    vals: np.ndarray
    p_neg_inf: float = 0.0
    p_pos_inf: float = 0.0
    name: Optional[str] = None
    debug_check: bool = False
    
    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.vals = np.ascontiguousarray(self.vals, dtype=np.float64)
        if self.x.ndim != 1 or self.vals.ndim != 1 or self.x.shape != self.vals.shape:
            raise ValueError("x and vals must be 1-D arrays of equal length")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x must be strictly increasing")
        if self.p_neg_inf < 0 or self.p_pos_inf < 0:
            raise ValueError("p_neg_inf and p_pos_inf must be nonnegative")
        if self.debug_check:
            if self.kind == DistKind.PMF:
                if np.any(self.vals < -1e-12):
                    raise ValueError("PMF must be nonnegative")
            elif self.kind == DistKind.CDF:
                if np.any(np.diff(self.vals) < -1e-12):
                    raise ValueError("CDF must be nondecreasing")
            elif self.kind == DistKind.CCDF:
                if np.any(np.diff(self.vals) > 1e-12):
                    raise ValueError("CCDF must be nonincreasing")

# =============================================================================
# NUMERICAL UTILITIES
# =============================================================================

def kahan_sum(arr: np.ndarray) -> float:
    """Kahan compensated summation for improved numerical accuracy."""
    s = 0.0
    c = 0.0
    for x in arr:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

def sorted_sum(arr: np.ndarray) -> float:
    """Sum array from smallest to largest absolute value."""
    return np.sum(np.sort(np.abs(arr)) * np.sign(arr[np.argsort(np.abs(arr))]))

def compensated_sum(arr: np.ndarray, method: SummationMethod = SummationMethod.STANDARD) -> float:
    """Sum array using specified method."""
    if method == SummationMethod.KAHAN:
        return kahan_sum(arr)
    elif method == SummationMethod.SORTED:
        return sorted_sum(arr)
    else:
        return np.sum(arr)

# =============================================================================
# GRID GENERATION AND DISCRETIZATION
# =============================================================================

def _sample_from_range(x_min: float, x_max: float, n_grid: int, spacing: Spacing) -> np.ndarray:
    """Sample points from a range using specified spacing strategy."""
    if spacing == Spacing.GEOMETRIC:
        if x_min <= 0:
            raise ValueError(f"Cannot use geometric spacing when range [{x_min:.6f}, {x_max:.6f}] contains negative values.")
        else:
            x = np.geomspace(x_min, x_max, n_grid, dtype=np.float64)
    else:
        x = np.linspace(x_min, x_max, n_grid, dtype=np.float64)
    return x

def _compute_bin_width(dist: DiscreteDist, spacing: Spacing) -> float:
    """
    Compute the characteristic bin width of a distribution.
    
    For linear spacing: median of differences
    For geometric spacing: median of log-differences (ratio - 1)
    """
    if len(dist.x) < 2:
        raise ValueError("Distribution must have at least 2 points to compute bin width")
    
    if spacing == Spacing.LINEAR:
        diffs = np.diff(dist.x)
        return np.median(diffs)
    else:  # GEOMETRIC
        if np.any(dist.x <= 0):
            raise ValueError("Cannot compute geometric bin width for non-positive values")
        ratios = dist.x[1:] / dist.x[:-1]
        return np.median(ratios) - 1.0  # Return relative width

def build_grid_from_support_bounds(dist_1: DiscreteDist, dist_2: DiscreteDist, 
                                   spacing: Spacing, beta: float,
                                   grid_strategy: GridStrategy = GridStrategy.FIXED_POINTS,
                                   target_bin_width: Optional[float] = None) -> np.ndarray:
    """
    Build grid for convolution using support bounds.
    
    Parameters:
    -----------
    dist_1, dist_2 : DiscreteDist
        Input distributions
    spacing : Spacing
        LINEAR or GEOMETRIC spacing
    beta : float
        Probability mass threshold for trimming
    grid_strategy : GridStrategy
        FIXED_POINTS: maintain number of grid points (original behavior)
        FIXED_WIDTH: maintain bin width
    target_bin_width : Optional[float]
        Bin width to maintain (if None, computed from inputs)
    """
    x1, p1 = dist_1.x, dist_1.vals
    x2, p2 = dist_2.x, dist_2.vals
    
    # Compute support bounds using cumulative mass threshold
    threshold = beta / 2
    
    cum1 = np.cumsum(p1)
    cum2 = np.cumsum(p2)
    
    x1_min_ind_arr = np.where(cum1 <= threshold)[0]
    x1_min_ind = x1_min_ind_arr[-1] if len(x1_min_ind_arr) > 0 else 0
    x1_min = x1[x1_min_ind]

    cum1_rev = np.cumsum(p1[::-1])
    x1_max_ind_arr = np.where(cum1_rev <= threshold)[0]
    x1_max_ind = len(p1) - 1 - (x1_max_ind_arr[-1] if len(x1_max_ind_arr) > 0 else 0)
    x1_max = x1[x1_max_ind]
    
    x2_min_ind_arr = np.where(cum2 <= threshold)[0]
    x2_min_ind = x2_min_ind_arr[-1] if len(x2_min_ind_arr) > 0 else 0
    x2_min = x2[x2_min_ind]

    cum2_rev = np.cumsum(p2[::-1])
    x2_max_ind_arr = np.where(cum2_rev <= threshold)[0]
    x2_max_ind = len(p2) - 1 - (x2_max_ind_arr[-1] if len(x2_max_ind_arr) > 0 else 0)
    x2_max = x2[x2_max_ind]
    
    x_out_min = x1_min + x2_min
    x_out_max = x1_max + x2_max
    
    if not np.isfinite(x_out_min) or not np.isfinite(x_out_max):
        raise ValueError(f"Support bounds not finite: x_out_min={x_out_min}, x_out_max={x_out_max}")
    
    if x_out_max <= x_out_min:
        raise ValueError(f"Invalid support bounds: x_out_min={x_out_min} >= x_out_max={x_out_max}")
    
    # Determine number of grid points based on strategy
    if grid_strategy == GridStrategy.FIXED_POINTS:
        out_size = max(x1.size, x2.size)
    else:  # FIXED_WIDTH
        if target_bin_width is None:
            # Compute bin width from inputs
            width1 = _compute_bin_width(dist_1, spacing)
            width2 = _compute_bin_width(dist_2, spacing)
            target_bin_width = min(width1, width2)  # Use finer resolution
        
        if spacing == Spacing.LINEAR:
            out_size = int(np.ceil((x_out_max - x_out_min) / target_bin_width)) + 1
        else:  # GEOMETRIC
            # target_bin_width is the relative width (ratio - 1)
            ratio = 1.0 + target_bin_width
            if x_out_min > 0:
                out_size = int(np.ceil(np.log(x_out_max / x_out_min) / np.log(ratio))) + 1
            else:
                # Fall back to linear
                out_size = max(x1.size, x2.size)
    
    out_size = max(out_size, 2)  # Ensure at least 2 points
    
    return _sample_from_range(x_out_min, x_out_max, out_size, spacing)

def discretize_continuous_to_pmf(dist: stats.rv_continuous,
                                  n_grid: int,
                                  beta: float,
                                  mode: Mode,
                                  spacing: Spacing,
                                  tail_switch: float = 1e-12,
                                  name: Optional[str] = None) -> DiscreteDist:
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
    x = _sample_from_range(x_min, x_max, n_grid, spacing)

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
    
    # Assign infinity masses based on mode
    if mode == Mode.DOMINATES:
        p_neg_inf = 0.0
        p_pos_inf = p_neg_mass + p_pos_mass
    else:  # IS_DOMINATED
        p_neg_inf = p_neg_mass + p_pos_mass
        p_pos_inf = 0.0
    
    if name is None:
        name = f"Discretized_{dist.__class__.__name__}"
    
    return DiscreteDist(
        x=x,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=float(p_neg_inf),
        p_pos_inf=float(p_pos_inf),
        name=name
    )

# =============================================================================
# CONVOLUTION KERNELS
# =============================================================================

def check_mass_conservation(dist: DiscreteDist, tolerance: float = 1e-13, 
                           sum_method: SummationMethod = SummationMethod.SORTED) -> None:
    """Check if a distribution conserves probability mass within tolerance."""
    if sum_method == SummationMethod.SORTED:
        sorted_vals = np.sort(np.abs(dist.vals)) * np.sign(dist.vals[np.argsort(np.abs(dist.vals))])
        pmf_sum = np.sum(sorted_vals)
    elif sum_method == SummationMethod.KAHAN:
        pmf_sum = kahan_sum(dist.vals)
    else:
        pmf_sum = dist.vals.sum()
    
    total_mass = pmf_sum + dist.p_neg_inf + dist.p_pos_inf
    mass_error = abs(total_mass - 1.0)
    
    if mass_error > tolerance:
        error_msg = f"MASS CONSERVATION ERROR"
        error_msg += f": Error={mass_error:.2e} (tolerance={tolerance:.0e})"
        error_msg += f", PMF sum={pmf_sum:.15f}"
        error_msg += f", p_neg_inf={dist.p_neg_inf:.2e}"
        error_msg += f", p_pos_inf={dist.p_pos_inf:.2e}"
        error_msg += f", Total mass={total_mass:.15f}"
        raise ValueError(error_msg)

@njit(cache=True)
def _pmf_pmf_kernel_standard(x1: np.ndarray, p1: np.ndarray, 
                             x2: np.ndarray, p2: np.ndarray,
                             x_out: np.ndarray, mode_val: int):
    """Standard PMF×PMF kernel without Kahan summation."""
    pmf_out = np.zeros(x_out.size, dtype=np.float64)
    pneg_extra = 0.0
    ppos_extra = 0.0
    
    for i in range(x1.size):
        last_idx = 0
        for j in range(x2.size):
            z = x1[i] + x2[j]
            mass = p1[i] * p2[j]
            
            if mode_val == 0:  # DOMINATES
                while last_idx < x_out.size and x_out[last_idx] < z:
                    last_idx += 1
                
                if last_idx >= x_out.size:
                    ppos_extra += mass
                else:
                    pmf_out[last_idx] += mass
            else:  # IS_DOMINATED
                while last_idx < x_out.size and x_out[last_idx] <= z:
                    last_idx += 1
                
                idx = last_idx - 1
                if idx < 0:
                    pneg_extra += mass
                else:
                    pmf_out[idx] += mass
    
    return pmf_out, pneg_extra, ppos_extra

@njit(cache=True)
def _pmf_pmf_kernel_kahan(x1: np.ndarray, p1: np.ndarray, 
                          x2: np.ndarray, p2: np.ndarray,
                          x_out: np.ndarray, mode_val: int):
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
            
            if mode_val == 0:  # DOMINATES
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
            else:  # IS_DOMINATED
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

def convolve_pmf_pmf_to_pmf_core(dist1: DiscreteDist, dist2: DiscreteDist, 
                                 mode: Mode, spacing: Spacing, beta: float,
                                 use_kahan: bool = True,
                                 sum_method: SummationMethod = SummationMethod.SORTED,
                                 grid_strategy: GridStrategy = GridStrategy.FIXED_POINTS,
                                 target_bin_width: Optional[float] = None) -> DiscreteDist:
    """PMF×PMF → PMF convolution with configurable precision and grid strategy."""
    if dist1.kind != DistKind.PMF or dist2.kind != DistKind.PMF:
        raise ValueError(f'convolve_pmf_pmf_to_pmf_core expects PMF inputs, got {dist1.kind}, {dist2.kind}')
    
    if mode == Mode.DOMINATES and (dist1.p_neg_inf > 0 or dist2.p_neg_inf > 0):
        raise ValueError(f'DOMINATES mode requires p_neg_inf=0 for both inputs')
    if mode == Mode.IS_DOMINATED and (dist1.p_pos_inf > 0 or dist2.p_pos_inf > 0):
        raise ValueError(f'IS_DOMINATED mode requires p_pos_inf=0 for both inputs')
    
    check_mass_conservation(dist1, sum_method=sum_method)
    check_mass_conservation(dist2, sum_method=sum_method)

    x_out = build_grid_from_support_bounds(dist1, dist2, spacing, beta, 
                                           grid_strategy, target_bin_width)
    
    mode_val = 0 if mode == Mode.DOMINATES else 1
    if use_kahan:
        pmf_out, pneg_extra, ppos_extra = _pmf_pmf_kernel_kahan(
            dist1.x, dist1.vals, dist2.x, dist2.vals, x_out, mode_val
        )
    else:
        pmf_out, pneg_extra, ppos_extra = _pmf_pmf_kernel_standard(
            dist1.x, dist1.vals, dist2.x, dist2.vals, x_out, mode_val
        )
    
    pnegZ = pneg_extra + dist1.p_neg_inf + dist2.p_neg_inf - dist1.p_neg_inf*dist2.p_neg_inf
    pposZ = ppos_extra + dist1.p_pos_inf + dist2.p_pos_inf - dist1.p_pos_inf*dist2.p_pos_inf
    
    if mode == Mode.DOMINATES and pnegZ > 0:
        raise ValueError(f'DOMINATES mode requires output p_neg_inf=0, got {pnegZ}')
    if mode == Mode.IS_DOMINATED and pposZ > 0:
        raise ValueError(f'IS_DOMINATED mode requires output p_pos_inf=0, got {pposZ}')
    
    result = DiscreteDist(x=x_out, kind=DistKind.PMF, vals=pmf_out, 
                         p_neg_inf=pnegZ, p_pos_inf=pposZ)
    check_mass_conservation(result, sum_method=sum_method)
    return result

# =============================================================================
# SELF-CONVOLUTION ALGORITHMS
# =============================================================================

def self_convolve_pmf_core(base: DiscreteDist, T: int, mode: Mode, spacing: Spacing, beta: float,
                          use_kahan: bool = True,
                          sum_method: SummationMethod = SummationMethod.SORTED,
                          grid_strategy: GridStrategy = GridStrategy.FIXED_POINTS,
                          target_bin_width: Optional[float] = None) -> DiscreteDist:
    """Self-convolve a PMF T times using exponentiation-by-squaring."""
    if base.kind != DistKind.PMF:
        raise ValueError(f'self_convolve_pmf_core expects PMF, got {base.kind}')
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if T == 1:
        return base
    
    check_mass_conservation(base, sum_method=sum_method)
    
    base_dist = base
    acc_dist = None
    while T > 0:
        if T & 1:
            if acc_dist is None:
                acc_dist = base_dist
            else:
                acc_dist = convolve_pmf_pmf_to_pmf_core(
                    acc_dist, base_dist, mode, spacing, beta,
                    use_kahan=use_kahan, sum_method=sum_method,
                    grid_strategy=grid_strategy, target_bin_width=target_bin_width
                )
        T >>= 1
        if T > 0:
            base_dist = convolve_pmf_pmf_to_pmf_core(
                base_dist, base_dist, mode, spacing, beta,
                use_kahan=use_kahan, sum_method=sum_method,
                grid_strategy=grid_strategy, target_bin_width=target_bin_width
            )
    
    check_mass_conservation(acc_dist, sum_method=sum_method)
    return acc_dist

# =============================================================================
# MAIN API FUNCTIONS
# =============================================================================

def cdf_to_pmf(dist: DiscreteDist) -> DiscreteDist:
    """Convert CDF to PMF."""
    if dist.kind != DistKind.CDF:
        raise ValueError(f"Expected CDF, got {dist.kind}")
    
    F = dist.vals
    pmf = np.diff(np.concatenate(([float(dist.p_neg_inf)], F))).astype(np.float64, copy=False)
    pmf[pmf < 0.0] = 0.0
    
    return DiscreteDist(
        x=dist.x,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=dist.p_neg_inf,
        p_pos_inf=dist.p_pos_inf,
        name=f"{dist.name}_pmf" if dist.name else "PMF"
    )

def ccdf_to_pmf(dist: DiscreteDist) -> DiscreteDist:
    """Convert CCDF to PMF."""
    if dist.kind != DistKind.CCDF:
        raise ValueError(f"Expected CCDF, got {dist.kind}")
    
    S = dist.vals
    pmf = np.empty_like(S)
    if S.size == 0:
        pmf = S.copy()
    else:
        pmf[0] = max(0.0, 1.0 - float(dist.p_neg_inf) - float(S[0]))
        if S.size > 1:
            pmf[1:] = S[:-1] - S[1:]
            pmf[1:][pmf[1:] < 0.0] = 0.0
    
    return DiscreteDist(
        x=dist.x,
        kind=DistKind.PMF,
        vals=pmf,
        p_neg_inf=dist.p_neg_inf,
        p_pos_inf=dist.p_pos_inf,
        name=f"{dist.name}_pmf" if dist.name else "PMF"
    )

def pmf_to_cdf(dist: DiscreteDist) -> DiscreteDist:
    """Convert PMF to CDF."""
    if dist.kind != DistKind.PMF:
        raise ValueError(f"Expected PMF, got {dist.kind}")
    
    cdf = np.cumsum(dist.vals) + dist.p_neg_inf
    
    return DiscreteDist(
        x=dist.x,
        kind=DistKind.CDF,
        vals=cdf,
        p_neg_inf=dist.p_neg_inf,
        p_pos_inf=dist.p_pos_inf,
        name=f"{dist.name}_cdf" if dist.name else "CDF"
    )

def self_convolve_pmf(base: DiscreteDist, T: int, mode: Mode = Mode.DOMINATES, 
                     spacing: Spacing = Spacing.LINEAR, beta: float = 1e-12,
                     use_kahan: bool = True,
                     sum_method: SummationMethod = SummationMethod.SORTED,
                     grid_strategy: GridStrategy = GridStrategy.FIXED_POINTS,
                     target_bin_width: Optional[float] = None) -> DiscreteDist:
    """
    Self-convolve a PMF T times with configurable numerical precision and grid strategy.
    """
    if base.kind != DistKind.PMF:
        raise ValueError('self_convolve_pmf expects base as PMF')
    if T < 1:
        raise ValueError(f'T must be >= 1, got {T}')
    
    result = self_convolve_pmf_core(base, int(T), mode, spacing, beta, 
                                   use_kahan=use_kahan, sum_method=sum_method,
                                   grid_strategy=grid_strategy, 
                                   target_bin_width=target_bin_width)
    result.name = f'{base.name or "X"}⊕^{T}'
    return result

def convolve_pmf_pmf_to_pmf(X: DiscreteDist, Y: DiscreteDist, mode: Mode = Mode.DOMINATES, 
                           spacing: Spacing = Spacing.LINEAR, beta: float = 1e-12,
                           use_kahan: bool = True,
                           sum_method: SummationMethod = SummationMethod.SORTED,
                           grid_strategy: GridStrategy = GridStrategy.FIXED_POINTS,
                           target_bin_width: Optional[float] = None) -> DiscreteDist:
    """Convolve two PMFs with configurable numerical precision and grid strategy."""
    if X.kind != DistKind.PMF or Y.kind != DistKind.PMF:
        raise ValueError('convolve_pmf_pmf_to_pmf expects PMF inputs')
    
    result = convolve_pmf_pmf_to_pmf_core(X, Y, mode, spacing, beta,
                                         use_kahan=use_kahan, sum_method=sum_method,
                                         grid_strategy=grid_strategy,
                                         target_bin_width=target_bin_width)
    result.name = 'pmf⊕pmf'
    return result