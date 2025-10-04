import numpy as np
from implementation import steps as S
from discrete_conv_api import DiscreteDist, DistKind

def test_cdf_steps_boundaries_and_interior():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    F = np.array([0.1, 0.5, 0.8], dtype=np.float64)
    pneg, ppos = 0.1, 0.2
    
    dist = DiscreteDist(x=x, kind=DistKind.CDF, vals=F, p_neg_inf=pneg, p_pos_inf=ppos)

    # Below grid
    assert S.step_cdf_left(dist, -5.0) == pneg
    assert S.step_cdf_right(dist, -5.0) == pneg
    # Above grid
    assert S.step_cdf_left(dist, 5.0) == 1.0 - ppos
    assert S.step_cdf_right(dist, 5.0) == 1.0 - ppos
    # Interior, exact hits
    assert S.step_cdf_left(dist, 1.0) == F[0]  # left of 1.0 is index 0
    assert S.step_cdf_right(dist, 1.0) == F[1] # right at 1.0 is index 1

def test_ccdf_steps_boundaries_and_interior():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    Scc = np.array([0.9, 0.6, 0.2], dtype=np.float64)  # last equals ppos
    pneg, ppos = 0.1, 0.2
    
    dist = DiscreteDist(x=x, kind=DistKind.CCDF, vals=Scc, p_neg_inf=pneg, p_pos_inf=ppos)

    # Below grid
    assert S.step_ccdf_left(dist, -5.0) == 1.0 - pneg
    assert S.step_ccdf_right(dist, -5.0) == 1.0 - pneg
    # Above grid
    assert S.step_ccdf_left(dist, 5.0) == ppos
    assert S.step_ccdf_right(dist, 5.0) == ppos
    # Interior, exact hits
    assert S.step_ccdf_left(dist, 1.0) == Scc[0]  # left of 1.0 is index 0
    assert S.step_ccdf_right(dist, 1.0) == Scc[1] # right at 1.0 is index 1
