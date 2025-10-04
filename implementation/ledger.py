
from typing import Tuple, Literal, TYPE_CHECKING
import numpy as np
from .types import Mode, DistKind

if TYPE_CHECKING:
    from .types import DiscreteDist

def infinity_ledger_from_pmfs(X: "DiscreteDist", Y: "DiscreteDist", mode: Mode) -> Tuple[float, float]:
    """
    Compute infinity ledger contributions for convolution.
    
    Parameters:
    -----------
    X: DiscreteDist object (must have kind='pmf')
    Y: DiscreteDist object (can be 'pmf', 'cdf', or 'ccdf')
    mode: "DOMINATES" or "IS_DOMINATED"
    
    Returns:
    --------
    (add_neg, add_pos): Additional mass at -∞ and +∞
    """
    if X.kind != DistKind.PMF:
        raise ValueError(f"infinity_ledger expects X to be PMF, got {X.kind}")
    
    mX = float(X.vals.sum())
    pnegX = float(X.p_neg_inf)
    pposX = float(X.p_pos_inf)
    pnegY = float(Y.p_neg_inf)
    pposY = float(Y.p_pos_inf)
    
    # Compute finite mass in Y based on its kind
    if Y.kind == DistKind.PMF:
        mY = float(Y.vals.sum())
    elif Y.kind == DistKind.CDF:
        # For CDF, finite mass is F[-1] - p_neg_inf
        mY = float(Y.vals[-1] - pnegY) if len(Y.vals) > 0 else 0.0
    elif Y.kind == DistKind.CCDF:
        # For CCDF, finite mass is (1 - S[0]) - p_neg_inf
        mY = float(1.0 - Y.vals[0] - pnegY) if len(Y.vals) > 0 else 0.0
    else:
        raise ValueError(f"Unknown kind for Y: {Y.kind}")
    
    add_neg = pnegX * (mY + pnegY) + pnegY * mX
    add_pos = pposX * (mY + pposY) + pposY * mX
    m_cross = pnegX * pposY + pposX * pnegY
    
    # Handle both string and enum modes
    if isinstance(mode, str):
        mode_str = mode
    else:
        mode_str = mode.value
    
    if mode_str == "DOMINATES":
        add_pos += m_cross
    else:
        add_neg += m_cross
    return float(add_neg), float(add_pos)
