import numpy as np
from implementation import steps as S

def test_cdf_steps_boundaries_and_interior():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    F = np.array([0.1, 0.5, 0.8], dtype=np.float64)
    pneg, ppos = 0.1, 0.2

    # Below grid
    assert S.step_cdf_left(x, F, pneg, ppos, -5.0) == pneg
    assert S.step_cdf_right(x, F, pneg, ppos, -5.0) == pneg
    # Above grid
    assert S.step_cdf_left(x, F, pneg, ppos, 5.0) == 1.0 - ppos
    assert S.step_cdf_right(x, F, pneg, ppos, 5.0) == 1.0 - ppos
    # Interior, exact hits
    assert S.step_cdf_left(x, F, pneg, ppos, 1.0) == F[0]  # left of 1.0 is index 0
    assert S.step_cdf_right(x, F, pneg, ppos, 1.0) == F[1] # right at 1.0 is index 1

def test_ccdf_steps_boundaries_and_interior():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    Scc = np.array([0.9, 0.6, 0.2], dtype=np.float64)  # last equals ppos
    pneg, ppos = 0.1, 0.2

    # Below grid
    assert S.step_ccdf_left(x, Scc, pneg, ppos, -5.0) == 1.0 - pneg
    assert S.step_ccdf_right(x, Scc, pneg, ppos, -5.0) == 1.0 - pneg
    # Above grid
    assert S.step_ccdf_left(x, Scc, pneg, ppos, 5.0) == ppos
    assert S.step_ccdf_right(x, Scc, pneg, ppos, 5.0) == ppos
    # Interior, exact hits
    assert S.step_ccdf_left(x, Scc, pneg, ppos, 1.0) == Scc[0]  # left of 1.0 is index 0
    assert S.step_ccdf_right(x, Scc, pneg, ppos, 1.0) == Scc[1] # right at 1.0 is index 1
