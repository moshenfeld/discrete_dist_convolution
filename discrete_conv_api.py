
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np

Kind = Literal["pmf", "cdf", "ccdf"]
Mode = Literal["DOMINATES", "IS_DOMINATED"]

from implementation import utils as _impl_utils
from implementation import steps as _impl_steps
from implementation import kernels as _impl_kernels
from implementation import grids as _impl_grids
from implementation import selfconv as _impl_selfconv

# Export grid generation function and enums
from implementation.grids import discretize_continuous_to_pmf, build_grid_from_support_bounds, Mode as GridMode, Spacing

@dataclass
class DiscreteDist:
    x: np.ndarray
    kind: Kind
    vals: np.ndarray
    p_neg_inf: float = 0.0
    p_pos_inf: float = 0.0
    name: Optional[str] = None
    debug_check: bool = False
    tol: float = 1e-12
    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.vals = np.ascontiguousarray(self.vals, dtype=np.float64)
        if self.x.ndim != 1 or self.vals.ndim != 1 or self.x.shape != self.vals.shape:
            raise ValueError("x and vals must be 1-D arrays of equal length")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x must be strictly increasing")
        if self.p_neg_inf < -self.tol or self.p_pos_inf < -self.tol:
            raise ValueError("p_neg_inf and p_pos_inf must be nonnegative")
        if self.debug_check:
            if self.kind == "pmf":
                if np.any(self.vals < -self.tol):
                    raise ValueError("PMF must be nonnegative")
            elif self.kind == "cdf":
                if np.any(np.diff(self.vals) < -1e-12):
                    raise ValueError("CDF must be nondecreasing")
            elif self.kind == "ccdf":
                if np.any(np.diff(self.vals) > 1e-12):
                    raise ValueError("CCDF must be nonincreasing")

def cdf_to_pmf(x: np.ndarray, F: np.ndarray, p_neg_inf: float, p_pos_inf: float, *, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.float64)
    pmf = np.diff(np.concatenate(([float(p_neg_inf)], F))).astype(np.float64, copy=False)
    pmf[pmf < 0.0] = 0.0
    # Create temporary dist for budget correction
    temp_dist = DiscreteDist(x=x, kind='pmf', vals=pmf, p_neg_inf=float(p_neg_inf), p_pos_inf=float(p_pos_inf))
    _impl_utils.budget_correction_last_bin(temp_dist, expected_total=1.0, tol=tol)
    return x, temp_dist.vals, float(temp_dist.p_neg_inf), float(temp_dist.p_pos_inf)

def ccdf_to_pmf(x: np.ndarray, S: np.ndarray, p_neg_inf: float, p_pos_inf: float, *, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    S = np.ascontiguousarray(S, dtype=np.float64)
    pmf = np.empty_like(S)
    if S.size == 0:
        pmf = S.copy()
    else:
        pmf[0] = max(0.0, 1.0 - float(p_neg_inf) - float(S[0]))
        if S.size > 1:
            pmf[1:] = S[:-1] - S[1:]
            pmf[1:][pmf[1:] < 0.0] = 0.0
    # Create temporary dist for budget correction
    temp_dist = DiscreteDist(x=x, kind='pmf', vals=pmf, p_neg_inf=float(p_neg_inf), p_pos_inf=float(p_pos_inf))
    _impl_utils.budget_correction_last_bin(temp_dist, expected_total=1.0, tol=tol)
    return x, temp_dist.vals, float(temp_dist.p_neg_inf), float(temp_dist.p_pos_inf)

def step_cdf_left(dist: DiscreteDist, q: float):  return _impl_steps.step_cdf_left(dist, q)
def step_cdf_right(dist: DiscreteDist, q: float): return _impl_steps.step_cdf_right(dist, q)
def step_ccdf_left(dist: DiscreteDist, q: float): return _impl_steps.step_ccdf_left(dist, q)
def step_ccdf_right(dist: DiscreteDist, q: float):return _impl_steps.step_ccdf_right(dist, q)

def convolve_pmf_pmf_to_pmf(X: DiscreteDist, Y: DiscreteDist, mode: Mode = 'DOMINATES', spacing: Spacing = Spacing.LINEAR, t: Optional[np.ndarray] = None) -> DiscreteDist:
    """
    Convolve two PMFs.
    
    Parameters:
    -----------
    X, Y : DiscreteDist
        Input distributions (must be PMFs)
    mode : Mode
        Tie-breaking mode (default: 'DOMINATES')
    spacing : Spacing
        Grid spacing strategy - LINEAR or GEOMETRIC (default: LINEAR)
    t : Optional[np.ndarray]
        Explicit output grid (optional, for backwards compatibility).
        If provided, spacing is ignored and uses legacy grid-based convolution.
        
    Returns:
    --------
    DiscreteDist
        Result distribution as PMF
        
    Usage:
    ------
    # Automatic grid generation (recommended)
    Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES', spacing=Spacing.LINEAR)
    
    # Explicit grid (legacy)
    t = np.linspace(x_min + y_min, x_max + y_max, 1000)
    Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES', t=t)
    """
    if X.kind != 'pmf' or Y.kind != 'pmf':
        raise ValueError('convolve_pmf_pmf_to_pmf expects PMF inputs')
    
    if t is not None:
        # Legacy interface: explicit grid
        t = np.ascontiguousarray(t, dtype=np.float64)
        pmf_out, pneg, ppos = _impl_kernels._convolve_pmf_pmf_on_grid(X, Y, t, mode)
        return DiscreteDist(x=t, kind='pmf', vals=pmf_out, p_neg_inf=pneg, p_pos_inf=ppos, name='pmf⊕pmf')
    else:
        # New interface: automatic grid generation
        result = _impl_kernels.convolve_pmf_pmf_to_pmf_core(X, Y, mode, spacing)
        result.name = 'pmf⊕pmf'
        return result

def convolve_pmf_cdf_to_cdf(X: DiscreteDist, Y: DiscreteDist, t: Optional[np.ndarray] = None, mode: Mode = 'DOMINATES', spacing: Optional[Spacing] = None) -> DiscreteDist:
    """
    Convolve PMF with CDF.
    
    Parameters:
    -----------
    X : DiscreteDist
        PMF distribution
    Y : DiscreteDist
        CDF distribution
    t : Optional[np.ndarray]
        Output grid. If None, will be auto-generated based on spacing parameter
    mode : Mode
        Tie-breaking mode
    spacing : Optional[Spacing]
        If provided, uses build_grid_from_support_bounds with specified spacing.
        If None and t is None, uses default trim-log strategy.
        Ignored if t is provided.
        
    Returns:
    --------
    DiscreteDist
        Result distribution as CDF
    """
    if X.kind != 'pmf' or Y.kind != 'cdf':
        raise ValueError('convolve_pmf_cdf_to_cdf expects X:PMF, Y:CDF')
    if t is None:
        # Use new support-bounds grid generation
        t = _impl_grids.build_grid_from_support_bounds(X, Y, spacing or Spacing.LINEAR, beta=1e-6)
    t = np.ascontiguousarray(t, dtype=np.float64)
    F, pneg, ppos = _impl_kernels.convolve_pmf_cdf_to_cdf_core(X, Y, t, mode)
    return DiscreteDist(x=t, kind='cdf', vals=F, p_neg_inf=pneg, p_pos_inf=ppos, name='pmf⊕cdf')

def convolve_pmf_ccdf_to_ccdf(X: DiscreteDist, Y: DiscreteDist, t: Optional[np.ndarray] = None, mode: Mode = 'DOMINATES', spacing: Optional[Spacing] = None) -> DiscreteDist:
    """
    Convolve PMF with CCDF.
    
    Parameters:
    -----------
    X : DiscreteDist
        PMF distribution
    Y : DiscreteDist
        CCDF distribution
    t : Optional[np.ndarray]
        Output grid. If None, will be auto-generated based on spacing parameter
    mode : Mode
        Tie-breaking mode
    spacing : Optional[Spacing]
        If provided, uses build_grid_from_support_bounds with specified spacing.
        If None and t is None, uses default trim-log strategy.
        Ignored if t is provided.
        
    Returns:
    --------
    DiscreteDist
        Result distribution as CCDF
    """
    if X.kind != 'pmf' or Y.kind != 'ccdf':
        raise ValueError('convolve_pmf_ccdf_to_ccdf expects X:PMF, Y:CCDF')
    if t is None:
        # Use new support-bounds grid generation
        t = _impl_grids.build_grid_from_support_bounds(X, Y, spacing or Spacing.LINEAR, beta=1e-6)
    t = np.ascontiguousarray(t, dtype=np.float64)
    S, pneg, ppos = _impl_kernels.convolve_pmf_ccdf_to_ccdf_core(X, Y, t, mode)
    return DiscreteDist(x=t, kind='ccdf', vals=S, p_neg_inf=pneg, p_pos_inf=ppos, name='pmf⊕ccdf')

def self_convolve_pmf(base: DiscreteDist, T: int, mode: Mode = 'DOMINATES', spacing: Spacing = Spacing.LINEAR) -> DiscreteDist:
    """
    Self-convolve a PMF T times: compute X + X + ... + X (T times).
    
    Uses exponentiation-by-squaring for efficiency: O(log T) convolutions instead of O(T).
    Grids evolve naturally at each step based on support bounds.
    
    Parameters:
    -----------
    base: Base distribution (must be PMF)
    T: Number of times to convolve (must be >= 1)
    mode: Tie-breaking mode (default: 'DOMINATES')
    spacing: Grid spacing strategy - LINEAR or GEOMETRIC (default: LINEAR)
    
    Returns:
    --------
    Result distribution as PMF with T-fold self-convolution
    
    Usage:
    ------
    # With linear spacing (default)
    Z = self_convolve_pmf(base, T=10, mode='DOMINATES')
    
    # With geometric spacing (for positive distributions)
    Z = self_convolve_pmf(base, T=10, mode='DOMINATES', spacing=Spacing.GEOMETRIC)
    
    Note:
    -----
    Grids adapt automatically at each convolution step. No need to specify output grid.
    """
    if base.kind != 'pmf':
        raise ValueError('self_convolve_pmf expects base as PMF')
    if T < 1:
        raise ValueError(f'T must be >= 1, got {T}')
    
    result = _impl_selfconv.self_convolve_pmf_core(base, int(T), mode, spacing)
    result.name = f'{base.name or "X"}⊕^{T}'
    return result

def self_convolve_envelope(base: DiscreteDist, T: int, mode: Mode = 'DOMINATES', spacing: Spacing = Spacing.LINEAR, backend: Literal["cdf","ccdf"] = 'cdf') -> DiscreteDist:
    """
    Self-convolve envelope T times using exponentiation-by-squaring.
    
    Computes envelope bounds for X + X + ... + X (T times) where X ~ base.
    Grids evolve naturally at each step based on support bounds.
    
    Parameters:
    -----------
    base: Base distribution (must be PMF)
    T: Number of times to convolve (must be >= 1)
    mode: Tie-breaking mode (default: 'DOMINATES')
    spacing: Grid spacing strategy - LINEAR or GEOMETRIC (default: LINEAR)
    backend: "cdf" or "ccdf" for envelope representation (default: 'cdf')
    
    Returns:
    --------
    Result distribution as CDF or CCDF envelope with T-fold self-convolution
    
    Usage:
    ------
    # CDF envelope with linear spacing
    Z = self_convolve_envelope(base, T=10, mode='DOMINATES', backend='cdf')
    
    # CCDF envelope with geometric spacing
    Z = self_convolve_envelope(base, T=10, mode='DOMINATES', spacing=Spacing.GEOMETRIC, backend='ccdf')
    """
    if base.kind != 'pmf':
        raise ValueError('self_convolve_envelope currently expects base as PMF')
    
    result = _impl_selfconv.self_convolve_envelope_core(base, int(T), mode, spacing, backend)
    result.name = f'{base.name or "X"}⊕^{T}_{backend}'
    return result
