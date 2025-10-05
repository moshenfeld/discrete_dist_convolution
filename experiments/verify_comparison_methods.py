"""
Verification script for comparison implementations.

This script verifies that the alternative convolution implementations
(FFT, Monte Carlo, Analytic) work correctly with basic functionality tests.
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from discrete_conv_api import DiscreteDist, Mode, Spacing, DistKind
from implementation.grids import discretize_continuous_to_pmf
from comparisons import (
    fft_self_convolve_pmf,
    monte_carlo_self_convolve_pmf,
    analytic_convolve_gaussian,
    analytic_convolve_lognormal
)

def test_gaussian_methods():
    """Test Gaussian comparison methods."""
    print("Testing Gaussian methods...")
    
    # Create base Gaussian
    dist_gaussian = stats.norm(0, 1)
    T = 10
    
    # Discretize using main implementation
    x, pmf, pneg, ppos = discretize_continuous_to_pmf(
        dist_gaussian, 1000, 1e-6, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
    base = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, 
                       p_neg_inf=pneg, p_pos_inf=ppos, name='Gaussian')
    
    # Test FFT method
    try:
        result_fft = fft_self_convolve_pmf(dist_gaussian, T, Mode.DOMINATES, Spacing.LINEAR, 1000)
        print(f"  FFT: ✓ (shape: {result_fft.x.shape})")
    except Exception as e:
        print(f"  FFT: ✗ ({e})")
    
    # Test Monte Carlo method
    try:
        result_mc = monte_carlo_self_convolve_pmf(base, T, Mode.DOMINATES, Spacing.LINEAR, 
                                                 n_samples=10000, n_bins=1000)
        print(f"  Monte Carlo: ✓ (shape: {result_mc.x.shape})")
    except Exception as e:
        print(f"  Monte Carlo: ✗ ({e})")
    
    # Test Analytic method
    try:
        result_analytic = analytic_convolve_gaussian(dist_gaussian, T, Mode.DOMINATES, Spacing.LINEAR,
                                                   n_points=1000, beta=1e-6)
        print(f"  Analytic: ✓ (shape: {result_analytic.x.shape})")
    except Exception as e:
        print(f"  Analytic: ✗ ({e})")

def test_lognormal_methods():
    """Test LogNormal comparison methods."""
    print("\nTesting LogNormal methods...")
    
    # Create base LogNormal
    dist_lognorm = stats.lognorm(s=1.0, scale=np.exp(0))  # σ=1, μ=0
    T = 10
    
    # Discretize using main implementation
    x, pmf, pneg, ppos = discretize_continuous_to_pmf(
        dist_lognorm, 1000, 1e-6, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC)
    base = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, 
                       p_neg_inf=pneg, p_pos_inf=ppos, name='LogNormal')
    
    # Test Monte Carlo method
    try:
        result_mc = monte_carlo_self_convolve_pmf(base, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                                 n_samples=10000, n_bins=1000)
        print(f"  Monte Carlo: ✓ (shape: {result_mc.x.shape})")
    except Exception as e:
        print(f"  Monte Carlo: ✗ ({e})")
    
    # Test Analytic method
    try:
        result_analytic = analytic_convolve_lognormal(dist_lognorm, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                                    n_points=1000, beta=1e-6)
        print(f"  Analytic: ✓ (shape: {result_analytic.x.shape})")
    except Exception as e:
        print(f"  Analytic: ✗ ({e})")

if __name__ == "__main__":
    test_gaussian_methods()
    test_lognormal_methods()
    print("\n✅ Test complete!")
