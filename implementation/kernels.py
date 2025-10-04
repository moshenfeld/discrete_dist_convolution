import numpy as np
from numba import njit
from .ledger import infinity_ledger_from_pmfs
from .types import Mode, Spacing, DistKind, DiscreteDist
from .grids import build_grid_from_support_bounds

@njit(cache=True)
def _pmf_pmf_kernel_numba(x1: np.ndarray, p1: np.ndarray, 
                          x2: np.ndarray, p2: np.ndarray,
                          x_out: np.ndarray, mode_is_dominates: bool):
    """
    Numba kernel for PMF×PMF convolution with proper tie-breaking.
    
    mode_is_dominates: True for DOMINATES mode, False for IS_DOMINATED mode
    Returns: (pmf_out, pneg_extra, ppos_extra)
    """
    pmf_out = np.zeros(x_out.size, dtype=np.float64)
    pneg_extra = 0.0
    ppos_extra = 0.0
    
    # Double loop over all (i,j) pairs
    for i in range(x1.size):
        for j in range(x2.size):
            z = x1[i] + x2[j]
            mass = p1[i] * p2[j]
            
            if mode_is_dominates:
                idx = np.searchsorted(x_out, z, side='left')
                if idx >= x_out.size:
                    ppos_extra += mass
                else:
                    pmf_out[idx] += mass
            else:
                idx = np.searchsorted(x_out, z, side='right') - 1
                if idx < 0:
                    pneg_extra += mass
                else:
                    pmf_out[idx] += mass
    
    return pmf_out, pneg_extra, ppos_extra

def convolve_pmf_pmf_to_pmf_core(X: DiscreteDist, Y: DiscreteDist, mode: Mode, spacing: Spacing, beta: float) -> DiscreteDist:
    """
    PMF×PMF → PMF convolution with proper tie-breaking and infinity handling.
    
    Computes the convolution Z = X + Y where X and Y are discrete PMFs.
    Grid is generated automatically based on support bounds and spacing strategy.
    
    Parameters:
    -----------
    X, Y: DiscreteDist objects (must have kind='pmf')
    mode: "DOMINATES" (exact hits up) or "IS_DOMINATED" (exact hits down)
    spacing: Grid spacing strategy (LINEAR or GEOMETRIC)
    beta: Probability mass threshold for grid generation
    
    Returns:
    --------
    DiscreteDist: Result distribution as PMF
    """
    if X.kind != DistKind.PMF or Y.kind != DistKind.PMF:
        raise ValueError(f'convolve_pmf_pmf_to_pmf_core expects PMF inputs, got {X.kind}, {Y.kind}')
    
    # Generate output grid
    t = build_grid_from_support_bounds(X, Y, spacing, beta)
    
    # Compute finite-finite convolution with tie-breaking
    mode_is_dominates = (mode == "DOMINATES")
    pmf_out, pneg_extra, ppos_extra = _pmf_pmf_kernel_numba(
        X.x, X.vals, Y.x, Y.vals, t, mode_is_dominates
    )
    
    # Add infinity ledger contributions
    add_neg, add_pos = infinity_ledger_from_pmfs(X, Y, mode)
    
    pnegZ = pneg_extra + add_neg
    pposZ = ppos_extra + add_pos
    
    return DiscreteDist(x=t, kind=DistKind.PMF, vals=pmf_out, p_neg_inf=pnegZ, p_pos_inf=pposZ)

def convolve_pmf_cdf_to_cdf_core(X: DiscreteDist, Y: DiscreteDist, t: np.ndarray, mode: Mode):
    """
    PMF×CDF → CDF envelope core (pseudocode in docstring). Returns (F, pnegZ_add, pposZ_add).
    """
    if X.kind != DistKind.PMF or Y.kind != DistKind.CDF:
        raise ValueError(f'convolve_pmf_cdf_to_cdf_core expects X:PMF, Y:CDF, got {X.kind}, {Y.kind}')
    
    F = np.zeros_like(t, dtype=np.float64)
    add_neg, add_pos = infinity_ledger_from_pmfs(X, Y, mode)
    return F, add_neg, add_pos

def convolve_pmf_ccdf_to_ccdf_core(X: DiscreteDist, Y: DiscreteDist, t: np.ndarray, mode: Mode):
    """
    PMF×CCDF → CCDF envelope core (pseudocode in docstring). Returns (S, pnegZ_add, pposZ_add).
    """
    if X.kind != DistKind.PMF or Y.kind != DistKind.CCDF:
        raise ValueError(f'convolve_pmf_ccdf_to_ccdf_core expects X:PMF, Y:CCDF, got {X.kind}, {Y.kind}')
    
    S = np.zeros_like(t, dtype=np.float64)
    add_neg, add_pos = infinity_ledger_from_pmfs(X, Y, mode)
    return S, add_neg, add_pos