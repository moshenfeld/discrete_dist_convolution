from typing import Literal
import numpy as np
from .kernels import convolve_pmf_pmf_to_pmf_core, check_mass_conservation
from .types import Mode, Spacing, DistKind, DiscreteDist

def self_convolve_pmf_core(base: DiscreteDist, T: int, mode: Mode, spacing: Spacing, beta: float) -> DiscreteDist:
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
    beta: Probability mass threshold for grid generation
    
    Returns:
    --------
    result: DiscreteDist with T-fold self-convolution
    """
    if base.kind != DistKind.PMF:
        raise ValueError(f'self_convolve_pmf_core expects PMF, got {base.kind}')
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if T == 1:
        return base
    
    # Check mass conservation at the beginning
    check_mass_conservation(base)
    
    # Binary exponentiation with evolving grids
    base_dist = base
    acc_dist = None
    while T > 0:
        if T & 1:  # If bit is set
            if acc_dist is None:
                acc_dist = base_dist
            else:
                # Convolve acc with base_dist - grid computed automatically inside kernel
                acc_dist = convolve_pmf_pmf_to_pmf_core(acc_dist, base_dist, mode, spacing, beta)
        T >>= 1  # Shift right (divide by 2)
        if T > 0:
            # Square base_dist - grid computed automatically inside kernel
            base_dist = convolve_pmf_pmf_to_pmf_core(base_dist, base_dist, mode, spacing, beta)
    
    # Check mass conservation at the end
    check_mass_conservation(acc_dist)
    
    return acc_dist

