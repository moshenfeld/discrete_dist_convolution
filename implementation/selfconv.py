
from typing import Literal, TYPE_CHECKING, Optional
import numpy as np
from .kernels import convolve_pmf_pmf_to_pmf_core, convolve_pmf_cdf_to_cdf_core, convolve_pmf_ccdf_to_ccdf_core

if TYPE_CHECKING:
    from discrete_conv_api import DiscreteDist
    from .grids import Spacing

Mode = Literal["DOMINATES", "IS_DOMINATED"]

def self_convolve_pmf_core(base: "DiscreteDist", T: int, mode: Mode, spacing: "Spacing") -> "DiscreteDist":
    """
    Self-convolve a PMF T times using exponentiation-by-squaring with evolving grids.
    
    Computes the distribution of X + X + ... + X (T times) where X ~ base.
    
    Algorithm: Exponentiation by squaring with flexible grids
    - Each convolution generates its own output grid based on support bounds
    - Grids adapt to the distribution support at each step
    - No need for user to specify output grid
    
    Parameters:
    -----------
    base: Base distribution (must be PMF)
    T: Number of times to convolve (must be >= 1)
    mode: Tie-breaking mode
    spacing: Grid spacing strategy (LINEAR or GEOMETRIC)
    
    Returns:
    --------
    result: DiscreteDist with T-fold self-convolution
    """
    if base.kind != 'pmf':
        raise ValueError(f'self_convolve_pmf_core expects PMF, got {base.kind}')
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if T == 1:
        return base
    
    # Binary exponentiation with evolving grids
    base_dist = base
    acc_dist = None
    
    T_remaining = T
    while T_remaining > 0:
        if T_remaining & 1:  # If bit is set
            if acc_dist is None:
                acc_dist = base_dist
            else:
                # Convolve acc with base_dist - grid computed automatically inside kernel
                acc_dist = convolve_pmf_pmf_to_pmf_core(acc_dist, base_dist, mode, spacing)
        
        T_remaining >>= 1  # Shift right (divide by 2)
        
        if T_remaining > 0:
            # Square base_dist - grid computed automatically inside kernel
            base_dist = convolve_pmf_pmf_to_pmf_core(base_dist, base_dist, mode, spacing)
    
    return acc_dist

def self_convolve_envelope_core(base: "DiscreteDist", T: int, mode: Mode, spacing: "Spacing", backend: Literal["cdf","ccdf"] = "cdf") -> "DiscreteDist":
    """
    Self-convolve envelope using exponentiation-by-squaring with evolving grids.
    
    Computes envelope bounds for X + X + ... + X (T times) where X ~ base.
    
    Parameters:
    -----------
    base: Base distribution (must be PMF)
    T: Number of times to convolve (must be >= 1)
    mode: Tie-breaking mode
    spacing: Grid spacing strategy (LINEAR or GEOMETRIC)
    backend: "cdf" or "ccdf" for envelope representation
    
    Returns:
    --------
    result: DiscreteDist with T-fold self-convolution envelope
    
    NOTE: Currently stub implementation.
    TODO: Implement using PMF×CDF or PMF×CCDF kernels when available
    """
    if base.kind != 'pmf':
        raise ValueError(f'self_convolve_envelope_core expects PMF, got {base.kind}')
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    # Import here to avoid circular dependency
    from discrete_conv_api import DiscreteDist
    from .grids import build_grid_from_support_bounds
    
    # For now, use PMF self-convolution and convert to envelope
    # TODO: Replace with proper PMF×CDF/CCDF convolution when implemented
    pmf_result = self_convolve_pmf_core(base, T, mode, spacing)
    
    # Convert PMF to requested envelope type
    if backend == "cdf":
        # Convert to CDF
        cdf_vals = np.cumsum(pmf_result.vals, dtype=np.float64)
        cdf_vals = np.minimum(cdf_vals, 1.0 - pmf_result.p_pos_inf)
        return DiscreteDist(x=pmf_result.x, kind='cdf', vals=cdf_vals,
                           p_neg_inf=pmf_result.p_neg_inf, p_pos_inf=pmf_result.p_pos_inf,
                           name=f'{base.name or "X"}⊕^{T}_cdf')
    else:
        # Convert to CCDF
        ccdf_vals = 1.0 - np.cumsum(pmf_result.vals, dtype=np.float64)
        ccdf_vals[0] = 1.0 - pmf_result.p_neg_inf - pmf_result.vals[0]
        ccdf_vals = np.maximum(ccdf_vals, pmf_result.p_pos_inf)
        return DiscreteDist(x=pmf_result.x, kind='ccdf', vals=ccdf_vals,
                           p_neg_inf=pmf_result.p_neg_inf, p_pos_inf=pmf_result.p_pos_inf,
                           name=f'{base.name or "X"}⊕^{T}_ccdf')
