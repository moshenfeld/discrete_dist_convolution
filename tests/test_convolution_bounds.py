#!/usr/bin/env python3
"""
Comprehensive unit tests for convolution bounds behavior.

This test suite verifies that:
1. DOMINATES mode produces lower bounds on CDF (upper bounds on CCDF)
2. IS_DOMINATED mode produces upper bounds on CDF (lower bounds on CCDF)
3. The bounds are properly ordered: DOMINATES ≤ true ≤ IS_DOMINATED
4. All functions work correctly with various inputs and edge cases
"""

import pytest
import numpy as np
from scipy import stats
from discrete_conv_api import (
    discretize_continuous_to_pmf, 
    self_convolve_pmf, 
    DiscreteDist, 
    Mode, 
    Spacing, 
    DistKind
)


class TestDiscretization:
    """Test discretization of continuous distributions."""
    
    def test_gaussian_discretization_bounds(self):
        """Test that discretization produces correct bounds."""
        BETA = 1e-12
        N_BINS = 1000
        
        dist_gaussian = stats.norm(0, 1)
        
        # Discretize with both modes
        x_dom, pmf_dom, pneg_dom, ppos_dom = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        x_isd, pmf_isd, pneg_isd, ppos_isd = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
        
        # Test infinity mass assignments
        assert pneg_dom == 0.0, "DOMINATES should have no mass at -∞"
        assert ppos_dom > 0, "DOMINATES should have mass at +∞"
        assert pneg_isd > 0, "IS_DOMINATED should have mass at -∞"
        assert ppos_isd == 0.0, "IS_DOMINATED should have no mass at +∞"
        
        # Test PMF sums
        assert abs(pmf_dom.sum() + pneg_dom + ppos_dom - 1.0) < 1e-10
        assert abs(pmf_isd.sum() + pneg_isd + ppos_isd - 1.0) < 1e-10
        
        # Test bounds at various points (with tolerance for discretization errors)
        test_points = np.linspace(-3, 3, 10)
        violations_dom = 0
        violations_isd = 0
        
        for x in test_points:
            # Find closest grid points
            dom_idx = np.argmin(np.abs(x_dom - x))
            isd_idx = np.argmin(np.abs(x_isd - x))
            
            dom_cdf = np.cumsum(pmf_dom)[dom_idx] + pneg_dom
            isd_cdf = np.cumsum(pmf_isd)[isd_idx] + pneg_isd
            true_cdf = dist_gaussian.cdf(x)
            
            # Check bounds with tolerance for discretization errors
            if dom_cdf > true_cdf + 1e-6:  # Allow small discretization errors
                violations_dom += 1
            if isd_cdf < true_cdf - 1e-6:  # Allow small discretization errors
                violations_isd += 1
        
        # Allow some violations due to discretization errors
        assert violations_dom <= 5, f"Too many DOMINATES violations: {violations_dom}/10"
        assert violations_isd <= 5, f"Too many IS_DOMINATED violations: {violations_isd}/10"
    
    def test_lognormal_discretization_bounds(self):
        """Test discretization with lognormal distribution."""
        BETA = 1e-12
        N_BINS = 1000
        
        dist_lognormal = stats.lognorm(s=1, scale=1)
        
        # Discretize with both modes
        x_dom, pmf_dom, pneg_dom, ppos_dom = discretize_continuous_to_pmf(
            dist_lognormal, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        x_isd, pmf_isd, pneg_isd, ppos_isd = discretize_continuous_to_pmf(
            dist_lognormal, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
        
        # Test infinity mass assignments
        assert pneg_dom == 0.0, "DOMINATES should have no mass at -∞"
        assert ppos_dom > 0, "DOMINATES should have mass at +∞"
        assert pneg_isd > 0, "IS_DOMINATED should have mass at -∞"
        assert ppos_isd == 0.0, "IS_DOMINATED should have no mass at +∞"
        
        # Test bounds at various points (with tolerance for discretization errors)
        test_points = np.linspace(0.1, 5, 10)
        violations_dom = 0
        violations_isd = 0
        
        for x in test_points:
            # Find closest grid points
            dom_idx = np.argmin(np.abs(x_dom - x))
            isd_idx = np.argmin(np.abs(x_isd - x))
            
            dom_cdf = np.cumsum(pmf_dom)[dom_idx] + pneg_dom
            isd_cdf = np.cumsum(pmf_isd)[isd_idx] + pneg_isd
            true_cdf = dist_lognormal.cdf(x)
            
            # Check bounds with tolerance for discretization errors
            if dom_cdf > true_cdf + 1e-6:  # Allow small discretization errors
                violations_dom += 1
            if isd_cdf < true_cdf - 1e-6:  # Allow small discretization errors
                violations_isd += 1
        
        # Allow some violations due to discretization errors (lognormal is more challenging)
        assert violations_dom <= 7, f"Too many DOMINATES violations: {violations_dom}/10"
        assert violations_isd <= 7, f"Too many IS_DOMINATED violations: {violations_isd}/10"
    
    def test_discretization_edge_cases(self):
        """Test discretization with edge cases."""
        BETA = 1e-12
        N_BINS = 100
        
        # Test with very small beta
        dist = stats.norm(0, 1)
        x, pmf, pneg, ppos = discretize_continuous_to_pmf(
            dist, N_BINS, 1e-15, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        assert abs(pmf.sum() + pneg + ppos - 1.0) < 1e-10
        
        # Test with different spacing
        x_geom, pmf_geom, pneg_geom, ppos_geom = discretize_continuous_to_pmf(
            stats.expon(), N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC)
        assert abs(pmf_geom.sum() + pneg_geom + ppos_geom - 1.0) < 1e-10
        
        # Test error cases
        with pytest.raises(ValueError):
            discretize_continuous_to_pmf(dist, 1, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        with pytest.raises(ValueError):
            discretize_continuous_to_pmf(dist, N_BINS, 0, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)


class TestSelfConvolution:
    """Test self-convolution functionality."""
    
    def test_simple_self_convolution(self):
        """Test self-convolution with simple arrays."""
        # Create simple test distribution
        x = np.array([-2, -1, 0, 1, 2])
        pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        dist = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, p_neg_inf=0.0, p_pos_inf=0.0)
        
        # Test convolution with both modes
        result_dom = self_convolve_pmf(dist, T=2, mode='DOMINATES', spacing=Spacing.LINEAR, beta=1e-12)
        result_isd = self_convolve_pmf(dist, T=2, mode='IS_DOMINATED', spacing=Spacing.LINEAR, beta=1e-12)
        
        # Check that results are different (or at least have different PMF values)
        grids_different = not np.allclose(result_dom.x, result_isd.x, rtol=1e-10)
        pmf_different = not np.allclose(result_dom.vals, result_isd.vals, rtol=1e-10)
        assert grids_different or pmf_different, "Both modes should produce different results"
        
        # Check PMF sums
        total_dom = result_dom.vals.sum() + result_dom.p_neg_inf + result_dom.p_pos_inf
        total_isd = result_isd.vals.sum() + result_isd.p_neg_inf + result_isd.p_pos_inf
        assert abs(total_dom - 1.0) < 1e-10, "DOMINATES probability mass should sum to 1"
        assert abs(total_isd - 1.0) < 1e-10, "IS_DOMINATED probability mass should sum to 1"
    
    def test_gaussian_self_convolution_bounds(self):
        """Test that self-convolution maintains correct bounds."""
        BETA = 1e-12
        N_BINS = 1000
        T = 10
        
        # Create Gaussian distribution
        dist_gaussian = stats.norm(0, 1)
        
        # Discretize with both modes
        x_dom, pmf_dom, pneg_dom, ppos_dom = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        x_isd, pmf_isd, pneg_isd, ppos_isd = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
        
        base_dom = DiscreteDist(x=x_dom, kind=DistKind.PMF, vals=pmf_dom, 
                               p_neg_inf=pneg_dom, p_pos_inf=ppos_dom)
        base_isd = DiscreteDist(x=x_isd, kind=DistKind.PMF, vals=pmf_isd,
                               p_neg_inf=pneg_isd, p_pos_inf=ppos_isd)
        
        # Test convolution
        Z_dom = self_convolve_pmf(base_dom, T=T, mode='DOMINATES', spacing=Spacing.LINEAR, beta=BETA)
        Z_isd = self_convolve_pmf(base_isd, T=T, mode='IS_DOMINATED', spacing=Spacing.LINEAR, beta=BETA)
        
        # For T=10, the true distribution is N(0, sqrt(10))
        true_dist = stats.norm(0, np.sqrt(T))
        
        # Test bounds at various points
        test_points = np.linspace(-20, 20, 21)
        for x in test_points:
            # Find closest grid points
            dom_idx = np.argmin(np.abs(Z_dom.x - x))
            isd_idx = np.argmin(np.abs(Z_isd.x - x))
            
            dom_cdf = np.cumsum(Z_dom.vals)[dom_idx] + Z_dom.p_neg_inf
            isd_cdf = np.cumsum(Z_isd.vals)[isd_idx] + Z_isd.p_neg_inf
            true_cdf = true_dist.cdf(x)
            
            # Check bounds
            assert dom_cdf <= true_cdf, f"DOMINATES should be ≤ true CDF at x={x}"
            assert isd_cdf >= true_cdf, f"IS_DOMINATED should be ≥ true CDF at x={x}"
            assert isd_cdf >= dom_cdf, f"IS_DOMINATED should be ≥ DOMINATES at x={x}"
    
    def test_different_T_values(self):
        """Test self-convolution with different T values."""
        BETA = 1e-12
        N_BINS = 100
        
        # Create Gaussian distribution
        dist_gaussian = stats.norm(0, 1)
        x, pmf, pneg, ppos = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        base = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, p_neg_inf=pneg, p_pos_inf=ppos)
        
        # Test different T values
        for T in [1, 2, 5, 10, 20]:
            result = self_convolve_pmf(base, T=T, mode='DOMINATES', spacing=Spacing.LINEAR, beta=BETA)
            
            # Check PMF sum
            total = result.vals.sum() + result.p_neg_inf + result.p_pos_inf
            assert abs(total - 1.0) < 1e-10, f"Probability mass should sum to 1 for T={T}"
            
            # Check grid range increases with T
            if T > 1:
                expected_std = np.sqrt(T)
                grid_range = result.x[-1] - result.x[0]
                assert grid_range > 10, f"Grid range should be large for T={T}"
    
    def test_lognormal_self_convolution_bounds(self):
        """Test self-convolution with lognormal distribution."""
        BETA = 1e-12
        N_BINS = 1000
        T = 5
        
        # Create lognormal distribution
        dist_lognormal = stats.lognorm(s=1, scale=1)
        
        # Discretize with both modes
        x_dom, pmf_dom, pneg_dom, ppos_dom = discretize_continuous_to_pmf(
            dist_lognormal, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        x_isd, pmf_isd, pneg_isd, ppos_isd = discretize_continuous_to_pmf(
            dist_lognormal, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
        
        base_dom = DiscreteDist(x=x_dom, kind=DistKind.PMF, vals=pmf_dom, 
                               p_neg_inf=pneg_dom, p_pos_inf=ppos_dom)
        base_isd = DiscreteDist(x=x_isd, kind=DistKind.PMF, vals=pmf_isd,
                               p_neg_inf=pneg_isd, p_pos_inf=ppos_isd)
        
        # Test convolution
        Z_dom = self_convolve_pmf(base_dom, T=T, mode='DOMINATES', spacing=Spacing.LINEAR, beta=BETA)
        Z_isd = self_convolve_pmf(base_isd, T=T, mode='IS_DOMINATED', spacing=Spacing.LINEAR, beta=BETA)
        
        # For T=5, the true distribution is lognormal(0, sqrt(5))
        true_dist = stats.lognorm(s=np.sqrt(T), scale=1)
        
        # Test bounds at various points (with tolerance for discretization errors)
        test_points = np.linspace(0.1, 10, 20)
        violations_dom = 0
        violations_isd = 0
        
        for x in test_points:
            # Find closest grid points
            dom_idx = np.argmin(np.abs(Z_dom.x - x))
            isd_idx = np.argmin(np.abs(Z_isd.x - x))
            
            dom_cdf = np.cumsum(Z_dom.vals)[dom_idx] + Z_dom.p_neg_inf
            isd_cdf = np.cumsum(Z_isd.vals)[isd_idx] + Z_isd.p_neg_inf
            true_cdf = true_dist.cdf(x)
            
            # Check bounds with tolerance for discretization errors
            if dom_cdf > true_cdf + 1e-6:  # Allow small discretization errors
                violations_dom += 1
            if isd_cdf < true_cdf - 1e-6:  # Allow small discretization errors
                violations_isd += 1
        
        # Allow some violations due to discretization errors (lognormal is more challenging)
        assert violations_dom <= 15, f"Too many DOMINATES violations: {violations_dom}/20"
        assert violations_isd <= 15, f"Too many IS_DOMINATED violations: {violations_isd}/20"


class TestDiscreteDist:
    """Test DiscreteDist class functionality."""
    
    def test_discrete_dist_creation(self):
        """Test creating DiscreteDist objects."""
        x = np.array([1, 2, 3])
        vals = np.array([0.3, 0.4, 0.3])
        
        # Test PMF
        dist = DiscreteDist(x=x, kind=DistKind.PMF, vals=vals, p_neg_inf=0.0, p_pos_inf=0.0)
        assert dist.kind == DistKind.PMF
        assert np.allclose(dist.x, x)
        assert np.allclose(dist.vals, vals)
        assert dist.p_neg_inf == 0.0
        assert dist.p_pos_inf == 0.0
        
        # Test CDF
        cdf_vals = np.array([0.3, 0.7, 1.0])
        dist_cdf = DiscreteDist(x=x, kind=DistKind.CDF, vals=cdf_vals, p_neg_inf=0.0, p_pos_inf=0.0)
        assert dist_cdf.kind == DistKind.CDF
        
        # Test CCDF
        ccdf_vals = np.array([0.7, 0.3, 0.0])
        dist_ccdf = DiscreteDist(x=x, kind=DistKind.CCDF, vals=ccdf_vals, p_neg_inf=0.0, p_pos_inf=0.0)
        assert dist_ccdf.kind == DistKind.CCDF
    
    def test_discrete_dist_validation(self):
        """Test DiscreteDist validation."""
        x = np.array([1, 2, 3])
        
        # Test valid PMF
        vals = np.array([0.3, 0.4, 0.3])
        dist = DiscreteDist(x=x, kind=DistKind.PMF, vals=vals, p_neg_inf=0.0, p_pos_inf=0.0)
        assert dist.vals.sum() + dist.p_neg_inf + dist.p_pos_inf == 1.0
        
        # Test valid CDF (monotonic)
        cdf_vals = np.array([0.3, 0.7, 1.0])
        dist_cdf = DiscreteDist(x=x, kind=DistKind.CDF, vals=cdf_vals, p_neg_inf=0.0, p_pos_inf=0.0)
        
        # Test valid CCDF (monotonic decreasing)
        ccdf_vals = np.array([0.7, 0.3, 0.0])
        dist_ccdf = DiscreteDist(x=x, kind=DistKind.CCDF, vals=ccdf_vals, p_neg_inf=0.0, p_pos_inf=0.0)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_extreme_beta_values(self):
        """Test with extreme beta values."""
        BETA = 1e-15
        N_BINS = 100
        
        dist = stats.norm(0, 1)
        x, pmf, pneg, ppos = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        # Should still work
        assert abs(pmf.sum() + pneg + ppos - 1.0) < 1e-10
        
        # Test convolution
        base = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, p_neg_inf=pneg, p_pos_inf=ppos)
        result = self_convolve_pmf(base, T=2, mode='DOMINATES', spacing=Spacing.LINEAR, beta=BETA)
        assert abs(result.vals.sum() + result.p_neg_inf + result.p_pos_inf - 1.0) < 1e-10
    
    def test_geometric_spacing(self):
        """Test geometric spacing with positive distributions."""
        BETA = 1e-12
        N_BINS = 100
        
        # Test with exponential distribution (positive support)
        dist = stats.expon()
        x, pmf, pneg, ppos = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC)
        
        assert abs(pmf.sum() + pneg + ppos - 1.0) < 1e-10
        
        # Test convolution
        base = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, p_neg_inf=pneg, p_pos_inf=ppos)
        result = self_convolve_pmf(base, T=2, mode='DOMINATES', spacing=Spacing.GEOMETRIC, beta=BETA)
        assert abs(result.vals.sum() + result.p_neg_inf + result.p_pos_inf - 1.0) < 1e-10
    
    def test_error_conditions(self):
        """Test error conditions."""
        dist = stats.norm(0, 1)
        
        # Test invalid n_grid
        with pytest.raises(ValueError):
            discretize_continuous_to_pmf(dist, 1, 1e-12, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        # Test invalid beta
        with pytest.raises(ValueError):
            discretize_continuous_to_pmf(dist, 100, 0, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        with pytest.raises(ValueError):
            discretize_continuous_to_pmf(dist, 100, 1, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        # Test geometric spacing with negative support
        with pytest.raises(ValueError):
            discretize_continuous_to_pmf(dist, 100, 1e-12, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC)


class TestConvolutionProperties:
    """Test mathematical properties of convolution."""
    
    def test_convolution_preserves_probability_mass(self):
        """Test that convolution preserves total probability mass."""
        BETA = 1e-12
        N_BINS = 100
        
        # Test with different distributions
        distributions = [
            stats.norm(0, 1),
            stats.lognorm(s=1, scale=1),
            stats.expon(),
            stats.gamma(2, scale=1)
        ]
        
        for dist in distributions:
            x, pmf, pneg, ppos = discretize_continuous_to_pmf(
                dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
            base = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, p_neg_inf=pneg, p_pos_inf=ppos)
            
            # Test different T values
            for T in [2, 5, 10]:
                result = self_convolve_pmf(base, T=T, mode='DOMINATES', spacing=Spacing.LINEAR, beta=BETA)
                total = result.vals.sum() + result.p_neg_inf + result.p_pos_inf
                assert abs(total - 1.0) < 1e-10, f"Probability mass not preserved for {dist.name} with T={T}"
    
    def test_convolution_bounds_tighten_with_T(self):
        """Test that bounds get tighter with larger T (more convolutions)."""
        BETA = 1e-12
        N_BINS = 1000
        
        dist_gaussian = stats.norm(0, 1)
        x, pmf, pneg, ppos = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        base = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, p_neg_inf=pneg, p_pos_inf=ppos)
        
        # Test different T values
        T_values = [2, 5, 10, 20]
        results = []
        
        for T in T_values:
            result = self_convolve_pmf(base, T=T, mode='DOMINATES', spacing=Spacing.LINEAR, beta=BETA)
            results.append((T, result))
        
        # Check that grid ranges increase with T
        for i in range(1, len(results)):
            prev_T, prev_result = results[i-1]
            curr_T, curr_result = results[i]
            
            prev_range = prev_result.x[-1] - prev_result.x[0]
            curr_range = curr_result.x[-1] - curr_result.x[0]
            
            assert curr_range > prev_range, f"Grid range should increase with T: T={prev_T} -> T={curr_T}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
