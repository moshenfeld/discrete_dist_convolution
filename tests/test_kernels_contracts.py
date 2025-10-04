import numpy as np
import pytest
from implementation.kernels import (
    _convolve_pmf_pmf_on_grid,
    convolve_pmf_cdf_to_cdf_core,
    convolve_pmf_ccdf_to_ccdf_core,
)
from discrete_conv_api import DiscreteDist

def test_pmf_cdf_core_ledger_only_until_implemented():
    # Minimal inputs where FY[-1]-pnegY = mY (finite mass); check ledger scalars.
    X = DiscreteDist(x=np.array([0.0, 1.0]), kind='pmf', vals=np.array([0.3, 0.2]),
                     p_neg_inf=0.1, p_pos_inf=0.4)
    # Build FY: starts at pnegY, ends at 1-pposY
    pnegY, pposY = 0.05, 0.15
    Y = DiscreteDist(x=np.array([0.0, 2.0]), kind='cdf', 
                     vals=np.array([pnegY, 1.0 - pposY]),
                     p_neg_inf=pnegY, p_pos_inf=pposY)
    t = np.array([-1.0, 0.0, 1.0, 3.0])
    F, add_neg, add_pos = convolve_pmf_cdf_to_cdf_core(X, Y, t, "DOMINATES")
    # Expect non-negative scalars; values of F are zeros until kernel is implemented
    assert add_neg >= 0 and add_pos >= 0
    assert F.shape == t.shape

def test_pmf_pmf_core_ledger_matches_when_no_edges():
    # Choose t wide to avoid edge extra rounding paths; only ledger contributes for now.
    X = DiscreteDist(x=np.array([0.0, 1.0]), kind='pmf', vals=np.array([0.2, 0.1]),
                     p_neg_inf=0.05, p_pos_inf=0.15)
    Y = DiscreteDist(x=np.array([0.0, 2.0]), kind='pmf', vals=np.array([0.3, 0.2]),
                     p_neg_inf=0.0, p_pos_inf=0.35)
    t = np.linspace(-100, 100, 11)  # very wide
    pmf_out, pnegZ, pposZ = _convolve_pmf_pmf_on_grid(X, Y, t, "IS_DOMINATED")
    # pmf_out currently zeros; assert shapes and nonnegativity of infinities
    assert pmf_out.shape == t.shape
    assert pnegZ >= 0 and pposZ >= 0

@pytest.mark.xfail(reason="CCDF core kernel envelope not yet implemented")
def test_pmf_ccdf_core_future_envelope_behavior():
    X = DiscreteDist(x=np.array([0.0, 1.0]), kind='pmf', vals=np.array([0.2, 0.1]),
                     p_neg_inf=0.0, p_pos_inf=0.0)
    Y = DiscreteDist(x=np.array([0.0, 2.0]), kind='ccdf', vals=np.array([1.0, 0.0]),
                     p_neg_inf=0.0, p_pos_inf=0.0)
    t = np.array([0.0, 1.0, 2.0])
    S, add_neg, add_pos = convolve_pmf_ccdf_to_ccdf_core(X, Y, t, "DOMINATES")
    # Once implemented, S should be clipped to [p_pos_X,1] and monotone nonincreasing
    assert S.shape == t.shape
