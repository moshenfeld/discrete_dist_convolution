import numpy as np
import pytest

# from discrete_conv_api import (
#     DiscreteDist, convolve_pmf_pmf_to_pmf, convolve_pmf_cdf_to_cdf,
#     convolve_pmf_ccdf_to_ccdf
# )

TOL_MICRO = 1e-12
TOL_STRESS = 1e-10

def assert_domination(lower: np.ndarray, upper: np.ndarray, tol: float = 0.0):
    """Return (ok, count, max_gap, idx) and assert if not ok."""
    gaps = lower - upper
    ok = np.all(gaps <= tol)
    count = int(np.sum(gaps > tol))
    max_gap = float(np.max(gaps)) if count else 0.0
    idx = int(np.argmax(gaps)) if count else -1
    if not ok:
        raise AssertionError(f"Domination violated {count} times; max overshoot {max_gap:.3e} at index {idx}")
    return ok, count, max_gap, idx

def test_pmf_rounding_exact_hits_and_edges():
    """PMF×PMF tie-breaking: ceil-to-+∞ (DOMINATES) and floor-to-−∞ (IS_DOMINATED)."""
    # PSEUDOCODE
    # - build tiny X,Y with deterministic sums z==t[0], z==t[-1], and exact interior hits
    # - call convolve_pmf_pmf_to_pmf in both modes; check edge mass goes to ±∞
    pass

def test_step_eval_boundaries_cdf_ccdf():
    """Boundary clamps for step evaluators on both sides."""
    # PSEUDOCODE
    # - construct simple monotone arrays; query far below/above grid; check clamps
    pass

def test_infinity_cross_terms_mode_dependent_routing():
    """Ambiguous cross mass routed to +∞ (DOMINATES) or −∞ (IS_DOMINATED)."""
    # PSEUDOCODE
    # - set X with p_neg_inf>0, Y with p_pos_inf>0 (and vice versa); zero finite mass
    # - verify routing per mode
    pass

def test_selfconv_identity_index_when_no_nonpos_anchor():
    """Identity δ0 placed at index 0 if all anchors are >0."""
    # PSEUDOCODE
    # - choose t_out strictly positive; build identity; check location
    pass

def test_reconcile_lower_upper_cdf_ccdf():
    """Optional lower/upper reconciliation keeps order."""
    # PSEUDOCODE
    # - craft two sequences with slight crossing; apply min/max reconciliation; assert order
    pass
