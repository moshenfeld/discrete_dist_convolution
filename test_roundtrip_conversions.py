import numpy as np
from discrete_conv_api import cdf_to_pmf, ccdf_to_pmf
from tests.conftest import TOL_MICRO, assert_monotone_nondecreasing, assert_monotone_nonincreasing, assert_budget_close, pmf_to_cdf, pmf_to_ccdf

def test_cdf_to_pmf_roundtrip():
    x = np.array([0.0, 0.5, 1.5, 2.5], dtype=np.float64)
    pneg, ppos = 0.1, 0.2
    # Build a valid CDF
    F = np.array([0.1, 0.3, 0.6, 0.8], dtype=np.float64)
    assert_monotone_nondecreasing(F)
    # Convert and reconstruct
    _, pmf, pneg2, ppos2 = cdf_to_pmf(x, F, pneg, ppos, tol=1e-12)
    assert abs(pneg2 - pneg) < TOL_MICRO and abs(ppos2 - ppos) < TOL_MICRO
    assert_budget_close(pneg2, pmf, ppos2, expected=1.0, tol=1e-12)
    F2 = pmf_to_cdf(x, pmf, pneg2, ppos2)
    # Roundtrip within small tolerance
    assert np.allclose(F2, F, atol=1e-12, rtol=0.0)

def test_ccdf_to_pmf_roundtrip():
    x = np.array([0.0, 0.5, 1.5, 2.5], dtype=np.float64)
    pneg, ppos = 0.1, 0.2
    # Build a valid CCDF
    S = np.array([0.9, 0.7, 0.4, 0.2], dtype=np.float64)
    assert_monotone_nonincreasing(S)
    # Convert and reconstruct
    _, pmf, pneg2, ppos2 = ccdf_to_pmf(x, S, pneg, ppos, tol=1e-12)
    assert abs(pneg2 - pneg) < TOL_MICRO and abs(ppos2 - ppos) < TOL_MICRO
    assert_budget_close(pneg2, pmf, ppos2, expected=1.0, tol=1e-12)
    S2 = pmf_to_ccdf(x, pmf, pneg2, ppos2)
    assert np.allclose(S2, S, atol=1e-12, rtol=0.0)
