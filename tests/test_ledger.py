import numpy as np
from implementation.ledger import infinity_ledger_from_pmfs
from discrete_conv_api import DiscreteDist, DistKind

def test_ledger_no_ambiguous_cross():
    mX, pnegX, pposX = 0.6, 0.1, 0.0
    mY, pnegY, pposY = 0.7, 0.2, 0.0
    # add_neg = pnegX*(mY+pnegY)+pnegY*mX = 0.1*(0.7+0.2)+0.2*0.6 = 0.09+0.12=0.21
    # add_pos = pposX*(mY+pposY)+pposY*mX = 0
    
    # Create distributions with correct masses (need at least 2 points for grid generation)
    X = DiscreteDist(x=np.array([0.0, 0.1]), kind=DistKind.PMF, vals=np.array([mX, 0.0]), 
                     p_neg_inf=pnegX, p_pos_inf=pposX)
    Y = DiscreteDist(x=np.array([0.0, 0.1]), kind=DistKind.PMF, vals=np.array([mY, 0.0]), 
                     p_neg_inf=pnegY, p_pos_inf=pposY)
    
    for mode in ("DOMINATES","IS_DOMINATED"):
        add_neg, add_pos = infinity_ledger_from_pmfs(X, Y, mode)
        assert abs(add_neg - 0.21) < 1e-15
        assert abs(add_pos - 0.0) < 1e-15

def test_ledger_with_ambiguous_cross_routing():
    mX, pnegX, pposX = 0.5, 0.3, 0.0
    mY, pnegY, pposY = 0.4, 0.0, 0.4
    # base adds:
    # add_neg_base = pnegX*(mY+pnegY)+pnegY*mX = 0.3*(0.4+0.0)+0 = 0.12
    # add_pos_base = pposX*(mY+pposY)+pposY*mX = 0 + 0.4*0.5 = 0.2
    # ambiguous cross m_cross = pnegX*pposY + pposX*pnegY = 0.3*0.4 + 0 = 0.12
    
    # Create distributions (need at least 2 points for grid generation)
    X = DiscreteDist(x=np.array([0.0, 0.1]), kind=DistKind.PMF, vals=np.array([mX, 0.0]), 
                     p_neg_inf=pnegX, p_pos_inf=pposX)
    Y = DiscreteDist(x=np.array([0.0, 0.1]), kind=DistKind.PMF, vals=np.array([mY, 0.0]), 
                     p_neg_inf=pnegY, p_pos_inf=pposY)
    
    add_neg, add_pos = infinity_ledger_from_pmfs(X, Y, "DOMINATES")
    assert abs(add_neg - 0.12) < 1e-15     # unchanged
    assert abs(add_pos - (0.2 + 0.12)) < 1e-15  # cross to +∞
    add_neg2, add_pos2 = infinity_ledger_from_pmfs(X, Y, "IS_DOMINATED")
    assert abs(add_neg2 - (0.12 + 0.12)) < 1e-15  # cross to −∞
    assert abs(add_pos2 - 0.2) < 1e-15
