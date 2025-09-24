"""Tests for PMF×PMF kernel implementation."""
import numpy as np
import pytest
from implementations.kernels import convolve_pmf_pmf_to_pmf_core

def test_pmf_pmf_simple_no_infinity():
    """Test basic PMF×PMF convolution without infinity masses."""
    # X: delta at 0
    xX = np.array([0.0])
    pX = np.array([1.0])
    pnegX, pposX = 0.0, 0.0
    
    # Y: delta at 1
    xY = np.array([1.0])
    pY = np.array([1.0])
    pnegY, pposY = 0.0, 0.0
    
    # Output grid includes the sum
    t = np.array([-1.0, 0.0, 1.0, 2.0])
    
    # DOMINATES mode
    pmf_out, pnegZ, pposZ = convolve_pmf_pmf_to_pmf_core(
        xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, "DOMINATES"
    )
    
    # Sum is 0+1=1, which is exactly t[2]
    # In DOMINATES mode, exact hits go up, so searchsorted(..., 'right') gives index 3
    assert np.allclose(pmf_out, [0.0, 0.0, 0.0, 1.0])
    assert pnegZ == 0.0
    assert pposZ == 0.0
    
    # IS_DOMINATED mode
    pmf_out2, pnegZ2, pposZ2 = convolve_pmf_pmf_to_pmf_core(
        xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, "IS_DOMINATED"
    )
    
    # In IS_DOMINATED mode, exact hits go down, so searchsorted(..., 'left')-1 gives index 1
    assert np.allclose(pmf_out2, [0.0, 1.0, 0.0, 0.0])
    assert pnegZ2 == 0.0
    assert pposZ2 == 0.0

def test_pmf_pmf_edge_routing():
    """Test that edge masses route correctly to ±∞."""
    # X: delta at 0
    xX = np.array([0.0])
    pX = np.array([1.0])
    pnegX, pposX = 0.0, 0.0
    
    # Y: two masses
    xY = np.array([0.0, 2.0])
    pY = np.array([0.3, 0.7])
    pnegY, pposY = 0.0, 0.0
    
    # Output grid that captures only the first sum
    t = np.array([0.0, 1.0])
    
    # DOMINATES mode: z=0 is exact hit on t[0], goes up to index 1
    # z=2 > t[-1], goes to +∞
    pmf_out, pnegZ, pposZ = convolve_pmf_pmf_to_pmf_core(
        xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, "DOMINATES"
    )
    
    assert np.allclose(pmf_out, [0.0, 0.3])  # z=0 goes to index 1
    assert pnegZ == 0.0
    assert np.isclose(pposZ, 0.7)  # z=2 goes to +∞
    
    # IS_DOMINATED mode: z=0 exact hit on t[0], goes down to -∞
    # z=2 > t[-1], also goes to +∞ (not affected by mode for upper edge)
    pmf_out2, pnegZ2, pposZ2 = convolve_pmf_pmf_to_pmf_core(
        xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, "IS_DOMINATED"
    )
    
    assert np.allclose(pmf_out2, [0.0, 0.0])  # both masses go to infinities
    assert np.isclose(pnegZ2, 0.3)  # z=0 goes to -∞
    assert np.isclose(pposZ2, 0.7)  # z=2 goes to +∞

def test_pmf_pmf_with_infinity_masses():
    """Test PMF×PMF with existing infinity masses."""
    # X with mass at -∞
    xX = np.array([0.0])
    pX = np.array([0.5])
    pnegX, pposX = 0.3, 0.2
    
    # Y with mass at +∞
    xY = np.array([0.0])
    pY = np.array([0.4])
    pnegY, pposY = 0.1, 0.5
    
    t = np.array([-1.0, 0.0, 1.0])
    
    # Test DOMINATES mode
    pmf_out, pnegZ, pposZ = convolve_pmf_pmf_to_pmf_core(
        xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, "DOMINATES"
    )
    
    # Check ledger calculations:
    # To -∞: pnegX*(mY + pnegY) + pnegY*mX = 0.3*(0.4+0.1) + 0.1*0.5 = 0.15 + 0.05 = 0.2
    # To +∞: pposX*(mY + pposY) + pposY*mX = 0.2*(0.4+0.5) + 0.5*0.5 = 0.18 + 0.25 = 0.43
    # Cross: pnegX*pposY + pposX*pnegY = 0.3*0.5 + 0.2*0.1 = 0.15 + 0.02 = 0.17
    # DOMINATES routes cross to +∞: add_pos += 0.17
    
    # Finite convolution: 0.5 * 0.4 = 0.2 at z=0, which goes to index 2 (DOMINATES: exact hits go up)
    assert np.allclose(pmf_out, [0.0, 0.0, 0.2])
    assert np.isclose(pnegZ, 0.2)
    assert np.isclose(pposZ, 0.43 + 0.17)  # 0.6

def test_pmf_pmf_budget_conservation():
    """Test that total probability mass is conserved."""
    # Random PMFs
    np.random.seed(42)
    xX = np.sort(np.random.randn(5))
    pX = np.random.rand(5)
    pX /= pX.sum() * 1.2  # Make sum < 1
    pnegX = 0.1
    pposX = 1.0 - pX.sum() - pnegX
    
    xY = np.sort(np.random.randn(4))
    pY = np.random.rand(4)
    pY /= pY.sum() * 1.1
    pnegY = 0.05
    pposY = 1.0 - pY.sum() - pnegY
    
    # Wide output grid to capture everything
    t = np.linspace(xX[0] + xY[0] - 1, xX[-1] + xY[-1] + 1, 50)
    
    for mode in ["DOMINATES", "IS_DOMINATED"]:
        pmf_out, pnegZ, pposZ = convolve_pmf_pmf_to_pmf_core(
            xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, mode
        )
        
        total = pmf_out.sum() + pnegZ + pposZ
        assert np.isclose(total, 1.0, atol=1e-12), f"Budget not conserved in {mode}: {total}"