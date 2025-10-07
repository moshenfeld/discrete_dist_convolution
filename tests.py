"""
Enhanced unit tests for discrete distribution convolution v2.

New tests in v2:
- CDF domination property tests
- Comprehensive property validation
- Grid strategy tests (fixed points vs fixed width)
- More edge cases
"""

import pytest
import numpy as np
from scipy import stats

from implementation_enhanced_v2 import (
    DiscreteDist, Mode, Spacing, DistKind, SummationMethod, GridStrategy,
    self_convolve_pmf, convolve_pmf_pmf_to_pmf,
    discretize_continuous_to_pmf, check_mass_conservation,
    kahan_sum, sorted_sum, compensated_sum,
    pmf_to_cdf, cdf_to_pmf, ccdf_to_pmf
)

# Test constants
TOL_MICRO = 1e-12
TOL_STRESS = 1e-10

# =============================================================================
# CDF DOMINATION TESTS
# =============================================================================

class TestCDFDomination:
    """Test that upper bound CDF dominates lower bound CDF."""
    
    def test_cdf_domination_gaussian(self):
        """Test CDF domination for Gaussian distribution.
        
        CRITICAL: For CCDF, S_upper >= S_lower (upper bound is conservative)
        This means for CDF: F_lower >= F_upper (lower bound CDF is higher)
        """
        BETA = 1e-12
        N_BINS = 1000
        
        dist_gaussian = stats.norm(0, 1)
        
        # Get upper and lower bounds
        dist_upper = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR,
            name='Gaussian-upper')
        dist_lower = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR,
            name='Gaussian-lower')
        
        # Convert to CDFs
        cdf_upper = pmf_to_cdf(dist_upper)
        cdf_lower = pmf_to_cdf(dist_lower)
        
        # CORRECTED: F_lower(x) >= F_upper(x) for all x
        for x_val in dist_upper.x:
            # Find CDF values at this point
            idx_upper = np.searchsorted(cdf_upper.x, x_val, side='right') - 1
            idx_lower = np.searchsorted(cdf_lower.x, x_val, side='right') - 1
            
            if idx_upper >= 0 and idx_lower >= 0:
                F_upper = cdf_upper.vals[idx_upper]
                F_lower = cdf_lower.vals[idx_lower]
                
                # Correct inequality: F_lower >= F_upper
                assert F_lower >= F_upper - 1e-10, \
                    f"CDF domination violated at x={x_val}: F_lower={F_lower} < F_upper={F_upper}"
    
    def test_cdf_domination_after_convolution(self):
        """Test that CDF domination is preserved after convolution.
        
        CRITICAL: For CCDF, S_upper >= S_lower (upper bound is conservative)
        This means for CDF: F_lower >= F_upper (lower bound CDF is higher)
        """
        BETA = 1e-12
        N_BINS = 500
        T = 10
        
        dist_gaussian = stats.norm(0, 1)
        
        # Get upper and lower bounds
        base_upper = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR,
            name='Gaussian-upper')
        base_lower = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR,
            name='Gaussian-lower')
        
        # Convolve T times
        result_upper = self_convolve_pmf(base_upper, T=T, mode=Mode.DOMINATES, 
                                        spacing=Spacing.LINEAR, beta=BETA)
        result_lower = self_convolve_pmf(base_lower, T=T, mode=Mode.IS_DOMINATED, 
                                        spacing=Spacing.LINEAR, beta=BETA)
        
        # Convert to CDFs
        cdf_upper = pmf_to_cdf(result_upper)
        cdf_lower = pmf_to_cdf(result_lower)
        
        # CORRECTED: Check F_lower >= F_upper at all points
        for i, x_val in enumerate(cdf_lower.x):
            F_lower = cdf_lower.vals[i]
            
            # Interpolate upper bound CDF at this point
            if x_val < cdf_upper.x[0]:
                F_upper = cdf_upper.p_neg_inf
            elif x_val > cdf_upper.x[-1]:
                F_upper = 1.0 - cdf_upper.p_pos_inf
            else:
                idx = np.searchsorted(cdf_upper.x, x_val, side='right') - 1
                if idx >= 0:
                    F_upper = cdf_upper.vals[idx]
                else:
                    F_upper = cdf_upper.p_neg_inf
            
            # Correct inequality: F_lower >= F_upper
            assert F_lower >= F_upper - 1e-9, \
                f"CDF domination violated after convolution at x={x_val}: " \
                f"F_lower={F_lower} < F_upper={F_upper}"
    
    def test_expected_value_bounds(self):
        """Test that E[upper] >= E[lower] for convolutions.
        
        Since S_upper(x) >= S_lower(x) (CCDF), by first-order stochastic dominance,
        the upper bound assigns more probability to higher values.
        Therefore: E[upper] >= E[lower].
        
        Note: This computes E[X] over the finite grid only, not including
        infinity masses (which can't be multiplied by ±∞).
        """
        BETA = 1e-12
        N_BINS = 500
        
        dist_gaussian = stats.norm(0, 1)
        
        base_upper = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        base_lower = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
        
        for T in [1, 5, 10, 20]:
            result_upper = self_convolve_pmf(base_upper, T=T, mode=Mode.DOMINATES, 
                                            spacing=Spacing.LINEAR, beta=BETA)
            result_lower = self_convolve_pmf(base_lower, T=T, mode=Mode.IS_DOMINATED, 
                                            spacing=Spacing.LINEAR, beta=BETA)
            
            E_upper = np.sum(result_upper.x * result_upper.vals)
            E_lower = np.sum(result_lower.x * result_lower.vals)
            
            # CORRECTED: Upper bound should have higher expectation (stochastic dominance)
            assert E_upper >= E_lower - 1e-6, \
                f"Expected value bound violated for T={T}: E_upper={E_upper} < E_lower={E_lower}"

# =============================================================================
# PROPERTY TESTS
# =============================================================================

class TestDistributionProperties:
    """Test fundamental properties of distributions."""
    
    def test_pmf_nonnegativity(self):
        """Test that PMF values are non-negative."""
        BETA = 1e-12
        N_BINS = 1000
        
        dist = stats.norm(0, 1)
        result = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        assert np.all(result.vals >= -1e-12), "PMF contains negative values"
    
    def test_cdf_monotonicity(self):
        """Test that CDF is monotonically non-decreasing."""
        BETA = 1e-12
        N_BINS = 1000
        
        dist = stats.norm(0, 1)
        pmf_dist = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        cdf_dist = pmf_to_cdf(pmf_dist)
        
        # Check CDF is non-decreasing
        diffs = np.diff(cdf_dist.vals)
        assert np.all(diffs >= -1e-12), \
            f"CDF not monotonic: min diff = {np.min(diffs)}"
    
    def test_cdf_bounds(self):
        """Test that CDF values are in [0, 1]."""
        BETA = 1e-12
        N_BINS = 1000
        
        dist = stats.norm(0, 1)
        pmf_dist = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        cdf_dist = pmf_to_cdf(pmf_dist)
        
        assert np.all(cdf_dist.vals >= -1e-12), "CDF contains values < 0"
        assert np.all(cdf_dist.vals <= 1 + 1e-12), "CDF contains values > 1"
    
    def test_grid_monotonicity(self):
        """Test that grid points are strictly increasing."""
        BETA = 1e-12
        N_BINS = 1000
        
        dist = stats.norm(0, 1)
        result = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        diffs = np.diff(result.x)
        assert np.all(diffs > 0), "Grid not strictly increasing"
    
    def test_infinity_mass_non_negative(self):
        """Test that infinity masses are non-negative."""
        BETA = 1e-12
        N_BINS = 1000
        
        dist = stats.norm(0, 1)
        
        result_dom = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        result_isd = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
        
        assert result_dom.p_neg_inf >= 0, "Negative p_neg_inf in DOMINATES mode"
        assert result_dom.p_pos_inf >= 0, "Negative p_pos_inf in DOMINATES mode"
        assert result_isd.p_neg_inf >= 0, "Negative p_neg_inf in IS_DOMINATED mode"
        assert result_isd.p_pos_inf >= 0, "Negative p_pos_inf in IS_DOMINATED mode"
    
    def test_mode_constraints(self):
        """Test that modes enforce correct infinity mass constraints."""
        BETA = 1e-12
        N_BINS = 1000
        
        dist = stats.norm(0, 1)
        
        result_dom = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        result_isd = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
        
        # DOMINATES should have p_neg_inf = 0
        assert abs(result_dom.p_neg_inf) < 1e-15, \
            f"DOMINATES mode should have p_neg_inf=0, got {result_dom.p_neg_inf}"
        
        # IS_DOMINATED should have p_pos_inf = 0
        assert abs(result_isd.p_pos_inf) < 1e-15, \
            f"IS_DOMINATED mode should have p_pos_inf=0, got {result_isd.p_pos_inf}"

# =============================================================================
# GRID STRATEGY TESTS
# =============================================================================

class TestGridStrategies:
    """Test fixed points vs fixed width grid strategies."""
    
    def test_fixed_points_strategy(self):
        """Test that FIXED_POINTS maintains number of grid points."""
        BETA = 1e-12
        N_BINS = 100
        
        dist = stats.norm(0, 1)
        base = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        # Self-convolve with fixed points strategy
        result = self_convolve_pmf(
            base, T=2, mode=Mode.DOMINATES, spacing=Spacing.LINEAR,
            grid_strategy=GridStrategy.FIXED_POINTS, beta=BETA)
        
        # Result should have approximately N_BINS points
        assert result.x.size >= N_BINS * 0.9, \
            f"FIXED_POINTS strategy lost too many points: {result.x.size} < {N_BINS}"
        assert result.x.size <= N_BINS * 1.5, \
            f"FIXED_POINTS strategy gained too many points: {result.x.size} > {N_BINS}"
    
    def test_fixed_width_strategy(self):
        """Test that FIXED_WIDTH maintains approximate bin width."""
        BETA = 1e-12
        N_BINS = 100
        
        dist = stats.norm(0, 1)
        base = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        # Compute initial bin width
        initial_width = np.median(np.diff(base.x))
        
        # Self-convolve with fixed width strategy
        result = self_convolve_pmf(
            base, T=2, mode=Mode.DOMINATES, spacing=Spacing.LINEAR,
            grid_strategy=GridStrategy.FIXED_WIDTH, beta=BETA)
        
        # Result should have approximately same bin width
        result_width = np.median(np.diff(result.x))
        
        # Width should be within 50% of original (allows some adaptation)
        assert result_width >= initial_width * 0.5, \
            f"FIXED_WIDTH strategy changed width too much: {result_width} < {initial_width * 0.5}"
        assert result_width <= initial_width * 1.5, \
            f"FIXED_WIDTH strategy changed width too much: {result_width} > {initial_width * 1.5}"
    
    def test_fixed_width_increases_points_with_T(self):
        """Test that FIXED_WIDTH increases number of points as T increases."""
        BETA = 1e-12
        N_BINS = 100
        
        dist = stats.norm(0, 1)
        base = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        sizes = []
        for T in [1, 2, 5, 10]:
            if T == 1:
                result = base
            else:
                result = self_convolve_pmf(
                    base, T=T, mode=Mode.DOMINATES, spacing=Spacing.LINEAR,
                    grid_strategy=GridStrategy.FIXED_WIDTH, beta=BETA)
            sizes.append(result.x.size)
        
        # Sizes should generally increase with T (allowing some variability)
        assert sizes[-1] > sizes[0], \
            f"FIXED_WIDTH should increase grid size with T: {sizes[-1]} <= {sizes[0]}"

# =============================================================================
# CONVERSION TESTS
# =============================================================================

class TestConversions:
    """Test conversions between PMF, CDF, and CCDF."""
    
    def test_pmf_to_cdf_to_pmf(self):
        """Test PMF -> CDF -> PMF roundtrip."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        
        dist_pmf = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, 
                               p_neg_inf=0.0, p_pos_inf=0.0)
        
        # Convert to CDF and back
        dist_cdf = pmf_to_cdf(dist_pmf)
        dist_pmf_back = cdf_to_pmf(dist_cdf)
        
        # Should recover original PMF (within tolerance)
        assert np.allclose(dist_pmf.vals, dist_pmf_back.vals, atol=1e-12)
    
    def test_cdf_properties_after_conversion(self):
        """Test that CDF has correct properties after conversion from PMF."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        
        dist_pmf = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, 
                               p_neg_inf=0.0, p_pos_inf=0.0)
        dist_cdf = pmf_to_cdf(dist_pmf)
        
        # CDF should be non-decreasing
        assert np.all(np.diff(dist_cdf.vals) >= -1e-12)
        
        # CDF should start near p_neg_inf and end near 1-p_pos_inf
        assert abs(dist_cdf.vals[0] - (pmf[0] + dist_cdf.p_neg_inf)) < 1e-12
        assert abs(dist_cdf.vals[-1] - (1.0 - dist_cdf.p_pos_inf)) < 1e-12

# =============================================================================
# NUMERICAL UTILITIES TESTS
# =============================================================================

class TestNumericalUtilities:
    """Test numerical utility functions."""
    
    def test_kahan_sum_accuracy(self):
        """Test that Kahan summation improves accuracy."""
        arr = np.array([1.0, 1e-15, 1e-15, 1e-15, -1.0])
        
        std_sum = np.sum(arr)
        kahan_result = kahan_sum(arr)
        
        true_value = 3e-15
        
        kahan_error = abs(kahan_result - true_value)
        std_error = abs(std_sum - true_value)
        
        assert kahan_error <= std_error or kahan_error < 1e-30
    
    def test_sorted_sum_accuracy(self):
        """Test that sorted summation improves accuracy."""
        arr = np.array([1e10, 1e10, -1e10, -1e10, 1.0, 1.0, 1.0])
        
        sorted_result = sorted_sum(arr)
        true_value = 3.0
        
        assert abs(sorted_result - true_value) < 1e-10
    
    def test_compensated_sum_methods(self):
        """Test different summation methods."""
        arr = np.array([1.0, 1e-15, 1e-15])
        
        result_std = compensated_sum(arr, SummationMethod.STANDARD)
        result_kahan = compensated_sum(arr, SummationMethod.KAHAN)
        result_sorted = compensated_sum(arr, SummationMethod.SORTED)
        
        assert abs(result_std - 1.0) < 1e-10
        assert abs(result_kahan - 1.0) < 1e-10
        assert abs(result_sorted - 1.0) < 1e-10

# =============================================================================
# MASS CONSERVATION TESTS
# =============================================================================

class TestMassConservation:
    """Test mass conservation with different methods."""
    
    def test_mass_conservation_after_discretization(self):
        """Test that discretization conserves mass."""
        BETA = 1e-12
        N_BINS = 1000
        
        for dist in [stats.norm(0, 1), stats.expon(), stats.uniform(-1, 2)]:
            result = discretize_continuous_to_pmf(
                dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
            
            total = result.vals.sum() + result.p_neg_inf + result.p_pos_inf
            assert abs(total - 1.0) < 1e-10, \
                f"Mass not conserved for {dist.__class__.__name__}: {total}"
    
    def test_mass_conservation_after_convolution(self):
        """Test that convolution conserves mass."""
        BETA = 1e-12
        N_BINS = 500
        
        dist = stats.norm(0, 1)
        base = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        for T in [2, 5, 10, 20]:
            result = self_convolve_pmf(base, T=T, mode=Mode.DOMINATES, 
                                      spacing=Spacing.LINEAR, beta=BETA)
            
            total = result.vals.sum() + result.p_neg_inf + result.p_pos_inf
            assert abs(total - 1.0) < 1e-9, \
                f"Mass not conserved after T={T} convolutions: {total}"

# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_T_equals_1(self):
        """Test that self-convolution with T=1 returns original distribution."""
        BETA = 1e-12
        N_BINS = 100
        
        dist = stats.norm(0, 1)
        base = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        result = self_convolve_pmf(base, T=1, mode=Mode.DOMINATES, 
                                  spacing=Spacing.LINEAR, beta=BETA)
        
        # Should be identical (same object)
        assert result is base
    
    def test_extreme_beta_values(self):
        """Test with extreme beta values."""
        N_BINS = 100
        
        dist = stats.norm(0, 1)
        
        # Very small beta
        result1 = discretize_continuous_to_pmf(
            dist, N_BINS, 1e-15, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        assert abs(result1.vals.sum() + result1.p_neg_inf + result1.p_pos_inf - 1.0) < 1e-10
        
        # Larger beta
        result2 = discretize_continuous_to_pmf(
            dist, N_BINS, 1e-3, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        assert abs(result2.vals.sum() + result2.p_neg_inf + result2.p_pos_inf - 1.0) < 1e-10
    
    def test_geometric_spacing_positive_distributions(self):
        """Test geometric spacing with positive distributions."""
        BETA = 1e-12
        N_BINS = 100
        
        # Exponential distribution (positive support)
        dist = stats.expon()
        result = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC)
        
        assert abs(result.vals.sum() + result.p_neg_inf + result.p_pos_inf - 1.0) < 1e-10
        
        # Check geometric spacing
        if len(result.x) > 1:
            ratios = result.x[1:] / result.x[:-1]
            # Ratios should be approximately constant for geometric spacing
            assert np.std(ratios) / np.mean(ratios) < 0.1
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        BETA = 1e-12
        N_BINS = 100
        
        dist = stats.norm(0, 1)
        
        # n_grid < 2
        with pytest.raises(ValueError):
            discretize_continuous_to_pmf(
                dist, 1, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        # beta outside (0, 1)
        with pytest.raises(ValueError):
            discretize_continuous_to_pmf(
                dist, N_BINS, 0.0, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        with pytest.raises(ValueError):
            discretize_continuous_to_pmf(
                dist, N_BINS, 1.0, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        # T < 1
        base = discretize_continuous_to_pmf(
            dist, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        
        with pytest.raises(ValueError):
            self_convolve_pmf(base, T=0, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)

# =============================================================================
# KAHAN CONVOLUTION TESTS
# =============================================================================

class TestKahanConvolution:
    """Test convolution with Kahan summation."""
    
    def test_kahan_vs_standard_simple(self):
        """Test that Kahan summation produces same or better results."""
        x = np.array([-2, -1, 0, 1, 2])
        pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        dist = DiscreteDist(x=x, kind=DistKind.PMF, vals=pmf, 
                           p_neg_inf=0.0, p_pos_inf=0.0)
        
        result_std = self_convolve_pmf(dist, T=2, mode=Mode.DOMINATES, 
                                      spacing=Spacing.LINEAR, use_kahan=False)
        result_kahan = self_convolve_pmf(dist, T=2, mode=Mode.DOMINATES, 
                                        spacing=Spacing.LINEAR, use_kahan=True)
        
        total_std = result_std.vals.sum() + result_std.p_neg_inf + result_std.p_pos_inf
        total_kahan = result_kahan.vals.sum() + result_kahan.p_neg_inf + result_kahan.p_pos_inf
        
        assert abs(total_std - 1.0) < 1e-10
        assert abs(total_kahan - 1.0) < 1e-10

# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])