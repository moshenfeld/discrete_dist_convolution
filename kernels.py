
from typing import Literal
import numpy as np
from .ledger import infinity_ledger_from_pmfs
from .steps import step_cdf_right, step_cdf_left, step_ccdf_right, step_ccdf_left
from .utils import clip_to_feasible_cdf, clip_to_feasible_ccdf, running_max_inplace, running_min_reverse_inplace

Mode = Literal["DOMINATES", "IS_DOMINATED"]

def convolve_pmf_pmf_to_pmf_core(xX: np.ndarray, pX: np.ndarray, pnegX: float, pposX: float,
                                 xY: np.ndarray, pY: np.ndarray, pnegY: float, pposY: float,
                                 t: np.ndarray, mode: Mode):
    """
    PMF×PMF → PMF core (pseudocode in docstring). Returns (pmf_out, pnegZ, pposZ).
    """
    pmf_out = np.zeros_like(t, dtype=np.float64)
    pneg_extra = 0.0
    ppos_extra = 0.0
    add_neg, add_pos = infinity_ledger_from_pmfs(float(pX.sum()), pnegX, pposX, float(pY.sum()), pnegY, pposY, mode)
    return pmf_out, pneg_extra + add_neg, ppos_extra + add_pos

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
