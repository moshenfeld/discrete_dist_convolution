import numpy as np
import pytest
from implementation.kernels import (
    convolve_pmf_pmf_to_pmf_core,
    convolve_pmf_cdf_to_cdf_core,
    convolve_pmf_ccdf_to_ccdf_core,
)

def test_pmf_cdf_core_ledger_only_until_implemented():
    # Minimal inputs where FY[-1]-pnegY = mY (finite mass); check ledger scalars.
    xX = np.array([0.0, 1.0])
    pX = np.array([0.3, 0.2])
    pnegX, pposX = 0.1, 0.4
    xY = np.array([0.0, 2.0])
    # Build FY: starts at pnegY, ends at 1-pposY
    pnegY, pposY = 0.05, 0.15
    FY = np.array([pnegY, 1.0 - pposY])
    t = np.array([-1.0, 0.0, 1.0, 3.0])
    F, add_neg, add_pos = convolve_pmf_cdf_to_cdf_core(xX,pX,pnegX,pposX,xY,FY,pnegY,pposY,t,"DOMINATES")
    # Expect non-negative scalars; values of F are zeros until kernel is implemented
    assert add_neg >= 0 and add_pos >= 0
    assert F.shape == t.shape

def test_pmf_pmf_core_ledger_matches_when_no_edges():
    # Choose t wide to avoid edge extra rounding paths; only ledger contributes for now.
    xX = np.array([0.0, 1.0])
    pX = np.array([0.2, 0.1])
    pnegX, pposX = 0.05, 0.15
    xY = np.array([0.0, 2.0])
    pY = np.array([0.3, 0.2])
    pnegY, pposY = 0.0, 0.35
    t = np.linspace(-100, 100, 11)  # very wide
    pmf_out, pnegZ, pposZ = convolve_pmf_pmf_to_pmf_core(xX,pX,pnegX,pposX,xY,pY,pnegY,pposY,t,"IS_DOMINATED")
    # pmf_out currently zeros; assert shapes and nonnegativity of infinities
    assert pmf_out.shape == t.shape
    assert pnegZ >= 0 and pposZ >= 0

@pytest.mark.xfail(reason="CCDF core kernel envelope not yet implemented")
def test_pmf_ccdf_core_future_envelope_behavior():
    xX = np.array([0.0, 1.0])
    pX = np.array([0.2, 0.1])
    pnegX, pposX = 0.0, 0.0
    xY = np.array([0.0, 2.0])
    SY = np.array([1.0, 0.0])  # no infinities
    pnegY, pposY = 0.0, 0.0
    t = np.array([0.0, 1.0, 2.0])
    S, add_neg, add_pos = convolve_pmf_ccdf_to_ccdf_core(xX,pX,pnegX,pposX,xY,SY,pnegY,pposY,t,"DOMINATES")
    # Once implemented, S should be clipped to [p_pos_X,1] and monotone nonincreasing
    assert S.shape == t.shape
