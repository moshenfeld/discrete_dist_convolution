import numpy as np
from numba import njit
from .types import Mode, Spacing, DistKind, DiscreteDist
from .grids import build_grid_from_support_bounds

def check_mass_conservation(dist: DiscreteDist, tolerance: float = 1e-11) -> None:
    """
    Check if a distribution conserves probability mass within tolerance.
    Raises ValueError if mass conservation fails.
    
    Parameters:
    -----------
    dist : DiscreteDist
        Distribution to check
    tolerance : float
        Maximum allowed error in mass conservation
    context : str
        Context string for error messages
    """
    total_mass = dist.vals.sum() + dist.p_neg_inf + dist.p_pos_inf
    mass_error = abs(total_mass - 1.0)
    
    if mass_error > tolerance:
        error_msg = f"MASS CONSERVATION ERROR"
        error_msg += f": Error={mass_error:.2e} (tolerance={tolerance:.0e})"
        error_msg += f", PMF sum={dist.vals.sum():.15f}"
        error_msg += f", p_neg_inf={dist.p_neg_inf:.2e}"
        error_msg += f", p_pos_inf={dist.p_pos_inf:.2e}"
        error_msg += f", Total mass={total_mass:.15f}"
        raise ValueError(error_msg)


@njit(cache=True)
def _pmf_pmf_kernel_numba(x1: np.ndarray, p1: np.ndarray, 
                          x2: np.ndarray, p2: np.ndarray,
                          x_out: np.ndarray, mode: Mode):
    """
    Numba kernel for PMF×PMF convolution.
    
    Returns: (pmf_out, pneg_extra, ppos_extra)
    """
    pmf_out = np.zeros(x_out.size, dtype=np.float64)
    pneg_extra = 0.0
    ppos_extra = 0.0
    
    for i in range(x1.size):
        # Reset position for each outer loop iteration
        last_idx = 0
        
        for j in range(x2.size):
            z = x1[i] + x2[j]
            mass = p1[i] * p2[j]
            
            if mode == Mode.DOMINATES:
                # Walk forward from last_idx until we find the right position
                # Since z is monotonic as j increases, we only need to walk forward
                while last_idx < x_out.size and x_out[last_idx] < z:
                    last_idx += 1
                
                # last_idx is now the correct position (left insertion point)
                if last_idx >= x_out.size:
                    ppos_extra += mass
                else:
                    pmf_out[last_idx] += mass
            else:
                # IS_DOMINATED mode - walk forward to find right insertion point
                while last_idx < x_out.size and x_out[last_idx] <= z:
                    last_idx += 1
                
                # last_idx is now the right insertion point, so idx = last_idx - 1
                idx = last_idx - 1
                
                if idx < 0:
                    pneg_extra += mass
                else:
                    pmf_out[idx] += mass
    
    return pmf_out, pneg_extra, ppos_extra

def convolve_pmf_pmf_to_pmf_core(dist1: DiscreteDist, dist2: DiscreteDist, mode: Mode, spacing: Spacing, beta: float) -> DiscreteDist:
    """
    PMF×PMF → PMF convolution with proper tie-breaking and infinity handling.
    
    Computes the convolution Z = dist1 + dist2 where dist1 and dist2 are discrete PMFs.
    Grid is generated automatically based on support bounds and spacing strategy.
    
    Parameters:
    -----------
    dist1, dist2: DiscreteDist objects (must have kind='pmf')
    mode: "DOMINATES" (exact hits up) or "IS_DOMINATED" (exact hits down)
    spacing: Grid spacing strategy (LINEAR or GEOMETRIC)
    beta: Probability mass threshold for grid generation
    
    Returns:
    --------
    DiscreteDist: Result distribution as PMF
    """
    if dist1.kind != DistKind.PMF or dist2.kind != DistKind.PMF:
        raise ValueError(f'convolve_pmf_pmf_to_pmf_core expects PMF inputs, got {dist1.kind}, {dist2.kind}')    
    # Check mode-specific infinity mass constraints
    if mode == Mode.DOMINATES and (dist1.p_neg_inf > 0 or dist2.p_neg_inf > 0):
        raise ValueError(f'DOMINATES mode requires p_neg_inf=0 for both inputs, got dist1.p_neg_inf={dist1.p_neg_inf}, dist2.p_neg_inf={dist2.p_neg_inf}')
    if mode == Mode.IS_DOMINATED and (dist1.p_pos_inf > 0 or dist2.p_pos_inf > 0):
        raise ValueError(f'IS_DOMINATED mode requires p_pos_inf=0 for both inputs, got dist1.p_pos_inf={dist1.p_pos_inf}, dist2.p_pos_inf={dist2.p_pos_inf}')
    # Check input mass conservation
    check_mass_conservation(dist1)
    check_mass_conservation(dist2)

    # Generate output grid
    x_out = build_grid_from_support_bounds(dist1, dist2, spacing, beta)
    # Compute finite-finite convolution with tie-breaking
    pmf_out, pneg_extra, ppos_extra = _pmf_pmf_kernel_numba(
        dist1.x, dist1.vals, dist2.x, dist2.vals, x_out, mode
    )
        
    # Total infinity mass: p1 + p2 - p1*p2
    pnegZ = pneg_extra + dist1.p_neg_inf + dist2.p_neg_inf - dist1.p_neg_inf*dist2.p_neg_inf
    pposZ = ppos_extra + dist1.p_pos_inf + dist2.p_pos_inf - dist1.p_pos_inf*dist2.p_pos_inf
    
    # Check mode-specific infinity mass constraints on output
    if mode == Mode.DOMINATES and pnegZ > 0:
        raise ValueError(f'DOMINATES mode requires output p_neg_inf=0, got {pnegZ}')
    if mode == Mode.IS_DOMINATED and pposZ > 0:
        raise ValueError(f'IS_DOMINATED mode requires output p_pos_inf=0, got {pposZ}')
    
    # Create result distribution
    result = DiscreteDist(x=x_out, kind=DistKind.PMF, vals=pmf_out, p_neg_inf=pnegZ, p_pos_inf=pposZ)
    check_mass_conservation(result)
    return result

def convolve_pmf_cdf_to_cdf_core(dist1: DiscreteDist, dist2: DiscreteDist, t: np.ndarray, mode: Mode):
    """
    PMF×CDF → CDF envelope core (pseudocode in docstring). Returns (F, pnegZ_add, pposZ_add).
    """
    raise NotImplementedError("convolve_pmf_cdf_to_cdf_core is not implemented")

def convolve_pmf_ccdf_to_ccdf_core(dist1: DiscreteDist, dist2: DiscreteDist, t: np.ndarray, mode: Mode):
    """
    PMF×CCDF → CCDF envelope core (pseudocode in docstring). Returns (S, pnegZ_add, pposZ_add).
    """
    raise NotImplementedError("convolve_pmf_ccdf_to_ccdf_core is not implemented")