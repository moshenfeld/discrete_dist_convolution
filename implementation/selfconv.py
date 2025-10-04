
from typing import Literal
import numpy as np
from .kernels import convolve_pmf_pmf_to_pmf_core, convolve_pmf_cdf_to_cdf_core, convolve_pmf_ccdf_to_ccdf_core
from .utils import identity_index_for_grid

Mode = Literal["DOMINATES", "IS_DOMINATED"]

def self_convolve_pmf_core(x_base: np.ndarray, pmf_base: np.ndarray, pneg_base: float, ppos_base: float,
                           T: int, t: np.ndarray, mode: Mode):
    """
    Self-convolve a PMF T times using exponentiation-by-squaring.
    
    Computes the distribution of X + X + ... + X (T times) where X ~ pmf_base.
    
    Algorithm: Exponentiation by squaring
    - Maintains cur_pmf = X^(2^k) via repeated self-convolution
    - Accumulates result when bit k is set in binary representation of T
    
    Parameters:
    -----------
    x_base: Grid points for base distribution
    pmf_base: PMF values for base distribution
    pneg_base, ppos_base: Mass at ±∞ for base
    T: Number of times to convolve (must be >= 1)
    t: Output grid
    mode: Tie-breaking mode
    
    Returns:
    --------
    pmf_result: PMF on output grid t
    pneg_result: Mass at -∞
    ppos_result: Mass at +∞
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    
    if T == 1:
        # Special case: just place base distribution on output grid
        # For simplicity, if grids match, return directly; otherwise use convolution with identity
        if len(x_base) == len(t) and np.allclose(x_base, t):
            return pmf_base.copy(), float(pneg_base), float(ppos_base)
    
    # Initialize accumulator with identity (delta at 0)
    # Find index closest to 0 in output grid
    i0 = identity_index_for_grid(t)
    acc_pmf = np.zeros_like(t, dtype=np.float64)
    acc_pmf[i0] = 1.0
    acc_pneg = 0.0
    acc_ppos = 0.0
    
    # Initialize cur with base distribution on output grid
    # We need to place base on output grid first
    cur_pmf, cur_pneg, cur_ppos = convolve_pmf_pmf_to_pmf_core(
        x_base, pmf_base, float(pneg_base), float(ppos_base),
        np.array([0.0]), np.array([1.0]), 0.0, 0.0,  # identity at 0
        t, mode
    )
    
    # Exponentiation by squaring
    T_remaining = T
    
    while T_remaining > 0:
        if T_remaining % 2 == 1:
            # Accumulate: acc = acc * cur
            acc_pmf, acc_pneg, acc_ppos = convolve_pmf_pmf_to_pmf_core(
                t, acc_pmf, acc_pneg, acc_ppos,
                t, cur_pmf, cur_pneg, cur_ppos,
                t, mode
            )
        
        T_remaining //= 2
        
        if T_remaining > 0:
            # Square: cur = cur * cur
            cur_pmf, cur_pneg, cur_ppos = convolve_pmf_pmf_to_pmf_core(
                t, cur_pmf, cur_pneg, cur_ppos,
                t, cur_pmf, cur_pneg, cur_ppos,
                t, mode
            )
    
    return acc_pmf, acc_pneg, acc_ppos

def self_convolve_envelope_core(x_base: np.ndarray, pmf_base: np.ndarray, pneg_base: float, ppos_base: float,
                                T: int, t: np.ndarray, mode: Mode, backend: Literal["cdf","ccdf"]):
    """
    Self-convolve envelope using exponentiation-by-squaring.
    
    NOTE: Currently only envelope backend is not fully implemented.
    The PMF self-convolution works via self_convolve_pmf_core.
    """
    # For now, return stub for envelope backend
    # TODO: Implement using PMF×CDF or PMF×CCDF kernels when available
    if backend == "cdf":
        env = np.zeros_like(t, dtype=np.float64)
    else:
        env = np.ones_like(t, dtype=np.float64)
    pneg = 0.0; ppos = 0.0
    return env, pneg, ppos
