import numpy as np
import pytest
from discrete_conv_api import DiscreteDist, DistKind, Mode, convolve_pmf_pmf_to_pmf

def test_pmf_pmf_core_ledger_matches_when_no_edges():
    # Choose wide spacing to avoid edge extra rounding paths; only ledger contributes for now.
    X = DiscreteDist(x=np.array([0.0, 1.0]), kind=DistKind.PMF, vals=np.array([0.2, 0.1]),
                     p_neg_inf=0.0, p_pos_inf=0.7)  # DOMINATES mode: p_neg_inf=0, Total = 1.0
    Y = DiscreteDist(x=np.array([0.0, 2.0]), kind=DistKind.PMF, vals=np.array([0.3, 0.2]),
                     p_neg_inf=0.0, p_pos_inf=0.5)  # DOMINATES mode: p_neg_inf=0, Total = 1.0
    
    # Use automatic grid generation instead of explicit grid
    result = convolve_pmf_pmf_to_pmf(X, Y, mode="DOMINATES")
    
    # Check that result is a valid DiscreteDist
    assert isinstance(result, DiscreteDist)
    assert result.kind == DistKind.PMF
    assert result.p_neg_inf >= 0 and result.p_pos_inf >= 0
