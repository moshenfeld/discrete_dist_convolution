from typing import Literal
import numpy as np
from numba import njit
from .ledger import infinity_ledger_from_pmfs
from .steps import step_cdf_right, step_cdf_left, step_ccdf_right, step_ccdf_left
from .utils import clip_to_feasible_cdf, clip_to_feasible_ccdf, running_max_inplace, running_min_reverse_inplace

Mode = Literal["DOMINATES", "IS_DOMINATED"]

@njit(cache=True)
def _pmf_pmf_kernel_numba(xX: np.ndarray, pX: np.ndarray, 
                          xY: np.ndarray, pY: np.ndarray,
                          t: np.ndarray, mode_is_dominates: bool):
    """
    Numba kernel for PMF×PMF convolution with proper tie-breaking.
    
    mode_is_dominates: True for DOMINATES mode, False for IS_DOMINATED mode
    Returns: (pmf_out, pneg_extra, ppos_extra)
    """
    pmf_out = np.zeros(t.size, dtype=np.float64)
    pneg_extra = 0.0
    ppos_extra = 0.0
    
    # Double loop over all (i,j) pairs
    for i in range(xX.size):
        for j in range(xY.size):
            z = xX[i] + xY[j]
            mass = pX[i] * pY[j]
            
            if mode_is_dominates:
                # DOMINATES: exact hits go up, use 'right'
                idx = np.searchsorted(t, z, side='right')
                
                if idx >= t.size:
                    # z > t[-1] or z == t[-1], mass goes to +∞
                    ppos_extra += mass
                else:
                    pmf_out[idx] += mass
            else:
                # IS_DOMINATED: exact hits go down, use 'left' - 1
                idx_raw = np.searchsorted(t, z, side='left')
                idx = idx_raw - 1
                
                if idx < 0:
                    # z < t[0] or z == t[0], mass goes to -∞
                    pneg_extra += mass
                elif idx_raw >= t.size:
                    # z > t[-1], mass goes to +∞
                    ppos_extra += mass
                else:
                    pmf_out[idx] += mass
    
    return pmf_out, pneg_extra, ppos_extra

def convolve_pmf_pmf_to_pmf_core(xX: np.ndarray, pX: np.ndarray, pnegX: float, pposX: float,
                                 xY: np.ndarray, pY: np.ndarray, pnegY: float, pposY: float,
                                 t: np.ndarray, mode: Mode):
    """
    PMF×PMF → PMF core with proper tie-breaking and infinity handling.
    
    Parameters:
    -----------
    xX, pX: Grid and PMF values for X
    pnegX, pposX: Mass at -∞ and +∞ for X
    xY, pY: Grid and PMF values for Y
    pnegY, pposY: Mass at -∞ and +∞ for Y
    t: Output grid
    mode: "DOMINATES" (exact hits up) or "IS_DOMINATED" (exact hits down)
    
    Returns:
    --------
    pmf_out: PMF on grid t
    pnegZ: Total mass at -∞
    pposZ: Total mass at +∞
    """
    # Compute finite-finite convolution with tie-breaking
    mode_is_dominates = (mode == "DOMINATES")
    pmf_out, pneg_extra, ppos_extra = _pmf_pmf_kernel_numba(
        xX, pX, xY, pY, t, mode_is_dominates
    )
    
    # Add infinity ledger contributions
    mX = float(pX.sum())
    mY = float(pY.sum())
    add_neg, add_pos = infinity_ledger_from_pmfs(mX, pnegX, pposX, mY, pnegY, pposY, mode)
    
    pnegZ = pneg_extra + add_neg
    pposZ = ppos_extra + add_pos
    
    return pmf_out, pnegZ, pposZ

def convolve_pmf_cdf_to_cdf_core(xX: np.ndarray, pX: np.ndarray, pnegX: float, pposX: float,
                                 xY: np.ndarray, FY: np.ndarray, pnegY: float, pposY: float,
                                 t: np.ndarray, mode: Mode):
    """
    PMF×CDF → CDF envelope core (pseudocode in docstring). Returns (F, pnegZ_add, pposZ_add).
    """
    F = np.zeros_like(t, dtype=np.float64)
    add_neg, add_pos = infinity_ledger_from_pmfs(float(pX.sum()), pnegX, pposX, float(FY[-1]-pnegY), pnegY, pposY, mode)
    return F, add_neg, add_pos

def convolve_pmf_ccdf_to_ccdf_core(xX: np.ndarray, pX: np.ndarray, pnegX: float, pposX: float,
                                   xY: np.ndarray, SY: np.ndarray, pnegY: float, pposY: float,
                                   t: np.ndarray, mode: Mode):
    """
    PMF×CCDF → CCDF envelope core (pseudocode in docstring). Returns (S, pnegZ_add, pposZ_add).
    """
    S = np.zeros_like(t, dtype=np.float64)
    add_neg, add_pos = infinity_ledger_from_pmfs(float(pX.sum()), pnegX, pposX, float(1.0 - SY[0] - pnegY), pnegY, pposY, mode)
    return S, add_neg, add_pos