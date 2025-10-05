
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import numpy as np

from implementation import kernels as _impl_kernels
from implementation import grids as _impl_grids
from implementation import selfconv as _impl_selfconv
from implementation.types import Mode, Spacing, DistKind, DiscreteDist

def cdf_to_pmf(x: np.ndarray, F: np.ndarray, p_neg_inf: float, p_pos_inf: float, *, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.float64)
    pmf = np.diff(np.concatenate(([float(p_neg_inf)], F))).astype(np.float64, copy=False)
    pmf[pmf < 0.0] = 0.0
    return x, pmf, float(p_neg_inf), float(p_pos_inf)

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
    return x, pmf, float(p_neg_inf), float(p_pos_inf)

def convolve_pmf_pmf_to_pmf(X: DiscreteDist, Y: DiscreteDist, mode: Mode = Mode.DOMINATES, spacing: Spacing = Spacing.LINEAR, beta: float = 1e-12) -> DiscreteDist:
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
    beta : float
        Probability mass threshold for grid generation
        
    Returns:
    --------
    DiscreteDist
        Result distribution as PMF
        
    Usage:
    ------
    # Automatic grid generation
    Z = convolve_pmf_pmf_to_pmf(X, Y, mode=Mode.DOMINATES, spacing=Spacing.LINEAR, beta=1e-12)
    """
    if X.kind != DistKind.PMF or Y.kind != DistKind.PMF:
        raise ValueError('convolve_pmf_pmf_to_pmf expects PMF inputs')
    
    # Use automatic grid generation
    result = _impl_kernels.convolve_pmf_pmf_to_pmf_core(X, Y, mode, spacing, beta)
    result.name = 'pmf⊕pmf'
    return result

def convolve_pmf_cdf_to_cdf(X: DiscreteDist, Y: DiscreteDist, t: Optional[np.ndarray] = None, mode: Mode = Mode.DOMINATES, spacing: Optional[Spacing] = None) -> DiscreteDist:
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
    if X.kind != DistKind.PMF or Y.kind != DistKind.CDF:
        raise ValueError('convolve_pmf_cdf_to_cdf expects X:PMF, Y:CDF')
    if t is None:
        # Use new support-bounds grid generation
        t = _impl_grids.build_grid_from_support_bounds(X, Y, spacing or Spacing.LINEAR, beta=1e-6)
    t = np.ascontiguousarray(t, dtype=np.float64)
    F, pneg, ppos = _impl_kernels.convolve_pmf_cdf_to_cdf_core(X, Y, t, mode)
    return DiscreteDist(x=t, kind=DistKind.CDF, vals=F, p_neg_inf=pneg, p_pos_inf=ppos, name='pmf⊕cdf')

def convolve_pmf_ccdf_to_ccdf(X: DiscreteDist, Y: DiscreteDist, t: Optional[np.ndarray] = None, mode: Mode = Mode.DOMINATES, spacing: Optional[Spacing] = None) -> DiscreteDist:
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
    if X.kind != DistKind.PMF or Y.kind != DistKind.CCDF:
        raise ValueError('convolve_pmf_ccdf_to_ccdf expects X:PMF, Y:CCDF')
    if t is None:
        # Use new support-bounds grid generation
        t = _impl_grids.build_grid_from_support_bounds(X, Y, spacing or Spacing.LINEAR, beta=1e-6)
    t = np.ascontiguousarray(t, dtype=np.float64)
    S, pneg, ppos = _impl_kernels.convolve_pmf_ccdf_to_ccdf_core(X, Y, t, mode)
    return DiscreteDist(x=t, kind=DistKind.CCDF, vals=S, p_neg_inf=pneg, p_pos_inf=ppos, name='pmf⊕ccdf')

def self_convolve_pmf(base: DiscreteDist, T: int, mode: Mode = Mode.DOMINATES, spacing: Spacing = Spacing.LINEAR, beta: float = 1e-12) -> DiscreteDist:
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
    beta: Probability mass threshold for grid generation (default: 1e-12)
    
    Returns:
    --------
    Result distribution as PMF with T-fold self-convolution
    
    Usage:
    ------
    # With linear spacing (default)
    Z = self_convolve_pmf(base, T=10, mode='DOMINATES', beta=1e-12)
    
    # With geometric spacing (for positive distributions)
    Z = self_convolve_pmf(base, T=10, mode='DOMINATES', spacing=Spacing.GEOMETRIC, beta=1e-12)
    
    Note:
    -----
    Grids adapt automatically at each convolution step. No need to specify output grid.
    """
    if base.kind != DistKind.PMF:
        raise ValueError('self_convolve_pmf expects base as PMF')
    if T < 1:
        raise ValueError(f'T must be >= 1, got {T}')
    
    result = _impl_selfconv.self_convolve_pmf_core(base, int(T), mode, spacing, beta)
    result.name = f'{base.name or "X"}⊕^{T}'
    return result

