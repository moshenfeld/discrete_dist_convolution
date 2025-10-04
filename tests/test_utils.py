import numpy as np
from implementation import utils as U

def test_running_max_inplace_and_reverse_min():
    a = np.array([0.0, 0.5, 0.4, 0.6, 0.55], dtype=np.float64)
    U.running_max_inplace(a)
    assert np.allclose(a, [0.0, 0.5, 0.5, 0.6, 0.6])

    b = np.array([1.0, 0.9, 0.95, 0.7, 0.8], dtype=np.float64)
    U.running_min_reverse_inplace(b)
    assert np.allclose(b, [0.9, 0.9, 0.8, 0.7, 0.8][::-1], atol=0, rtol=0) is False  # sanity: not this pattern
    # Correct expected:
    # traverse from right: cur=inf -> 0.8 -> min(0.7,0.8)=0.7 -> min(0.95,0.7)=0.7 -> min(0.9,0.7)=0.7 -> min(1.0,0.7)=0.7
    assert np.allclose(b, [0.7, 0.7, 0.7, 0.7, 0.8])

def test_clip_feasible_cdf_and_ccdf():
    F = np.array([-1.0, 0.2, 0.8, 1.2], dtype=np.float64)
    U.clip_to_feasible_cdf(F, p_neg_x=0.1, mX=0.6, p_pos_y=0.25)
    # hi = 0.1 + 0.6*(1-0.25) = 0.55
    assert np.allclose(F, np.array([0.1, 0.2, 0.55, 0.55]))

    S = np.array([-0.1, 0.2, 1.2], dtype=np.float64)
    U.clip_to_feasible_ccdf(S, p_pos_x=0.3)
    assert np.allclose(S, np.array([0.3, 0.3, 1.0]))

def test_reconcile_helpers():
    lower = np.array([0.1, 0.5, 0.7], dtype=np.float64)
    upper = np.array([0.2, 0.4, 0.9], dtype=np.float64)
    U.reconcile_cdf(lower, upper)
    assert np.all(lower <= upper + 1e-15)
    # CCDF reconciliation
    lower2 = np.array([0.8, 0.6, 0.4], dtype=np.float64)
    upper2 = np.array([0.9, 0.5, 0.3], dtype=np.float64)
    U.reconcile_ccdf(lower2, upper2)
    assert np.all(upper2 >= lower2 - 1e-15)

def test_budget_correction_last_bin():
    from discrete_conv_api import DiscreteDist
    
    pmf = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    pneg, ppos = 0.0, 0.11
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    
    dist = DiscreteDist(x=x, kind='pmf', vals=pmf, p_neg_inf=pneg, p_pos_inf=ppos)
    # Sum = 1.01 -> eps = 0.01, within tol=0.02 -> last bin -0.01
    U.budget_correction_last_bin(dist, expected_total=1.0, tol=0.02)
    assert abs(dist.p_neg_inf + dist.vals.sum() + dist.p_pos_inf - 1.0) < 1e-15
    assert np.isclose(dist.vals[-1], 0.39)

def test_union_grid_and_identity_index():
    xs = [np.array([-1.0, 0.0, 1.0]), np.array([0.5, 2.0])]
    u = U.union_grid(xs)
    assert np.allclose(u, [-1.0, 0.0, 0.5, 1.0, 2.0])

    t = np.array([0.1, 0.2, 0.3])
    assert U.identity_index_for_grid(t) == 0
    t2 = np.array([-0.5, -0.1, 0.0, 0.4])
    assert U.identity_index_for_grid(t2) == 2  # last <= 0 -> index 2
