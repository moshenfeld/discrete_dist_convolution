"""
Comparison test between main implementation and alternative convolution methods.

This script compares the main implementation (upper and lower bounds) with:
1. Monte Carlo sampling convolution
2. Analytic convolution (exact for Gaussian, approximate for lognormal)

For Gaussian: compares all 3 methods (main upper/lower + Monte Carlo + Analytic)
For Lognormal: compares 3 methods (main upper/lower + Monte Carlo + Analytic)
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from discrete_conv_api import self_convolve_pmf, DiscreteDist, Mode, Spacing, DistKind
from implementation.grids import discretize_continuous_to_pmf

# Import comparison methods
from comparisons import (
    fft_self_convolve_pmf,
    monte_carlo_self_convolve_pmf,
    analytic_convolve_gaussian,
    analytic_convolve_lognormal
)

# Import data utilities
import sys
from pathlib import Path
experiments_dir = Path(__file__).parent
sys.path.insert(0, str(experiments_dir))
from data_utils import save_results, load_latest_results, save_plot

# Import plotting functions
from visualization.method_comparison_plots import plot_gaussian_results, plot_lognormal_results

# Standard parameters
N_BINS = 20000
BETA = 1e-15
T_VALUES = [10, 100, 1000]  # Reduced for investigation
MC_SAMPLES = 1_000_000  # Monte Carlo samples

# Control variables
RERUN_TESTS = True  # Set to False to only run visualization from saved data

def run_gaussian_comparison(T_values):
    """
    Run comparison experiment for Gaussian distribution.
    
    Compares:
    - Main implementation (upper/lower bounds)
    - Monte Carlo convolution
    - Analytic convolution (exact)
    """
    print(f"\n{'='*80}")
    print(f"GAUSSIAN COMPARISON: N(0,1) with linear spacing")
    print(f"{'='*80}")
    
    # Create base Gaussian distribution
    dist_gaussian = stats.norm(0, 1)
    
    # Discretize base distribution using main implementation
    try:
        x_upper, pmf_upper, pneg_upper, ppos_upper = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        x_lower, pmf_lower, pneg_lower, ppos_lower = discretize_continuous_to_pmf(
            dist_gaussian, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
        
        base_upper = DiscreteDist(x=x_upper, kind=DistKind.PMF, vals=pmf_upper, 
                                  p_neg_inf=pneg_upper, p_pos_inf=ppos_upper, name='Gaussian-upper')
        base_lower = DiscreteDist(x=x_lower, kind=DistKind.PMF, vals=pmf_lower,
                                  p_neg_inf=pneg_lower, p_pos_inf=ppos_lower, name='Gaussian-lower')
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"Base: {base_upper.name}, {len(base_upper.x):,} bins")
    print(f"  Range: [{base_upper.x[0]:.6f}, {base_upper.x[-1]:.6f}]")
    
    # Compute theoretical expectation
    E_base = dist_gaussian.mean()
    print(f"  Theoretical E[X] = {E_base:.6f}")
    
    results = {}
    
    for T in T_values:
        print(f"\n  T={T} copies...")
        
        # Warmup on first run
        if T == T_values[0]:
            _ = self_convolve_pmf(base_upper, T, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
            print(f"    [JIT warmup complete]")
        
        # Method 1: Main implementation (upper/lower bounds)
        start = time.perf_counter()
        Z_upper = self_convolve_pmf(base_upper, T, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
        Z_lower = self_convolve_pmf(base_lower, T, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
        main_time = time.perf_counter() - start
        
        # Method 2: FFT convolution
        start = time.perf_counter()
        Z_fft = fft_self_convolve_pmf(dist_gaussian, T, Mode.DOMINATES, Spacing.LINEAR, N_BINS, BETA)
        fft_time = time.perf_counter() - start
        
        # Method 3: Monte Carlo convolution
        start = time.perf_counter()
        Z_mc = monte_carlo_self_convolve_pmf(dist_gaussian, T, Mode.DOMINATES, Spacing.LINEAR, 
                                            MC_SAMPLES, N_BINS)
        mc_time = time.perf_counter() - start
        
        # Method 4: Analytic convolution (exact for Gaussian)
        start = time.perf_counter()
        Z_analytic = analytic_convolve_gaussian(dist_gaussian, T, Mode.DOMINATES, Spacing.LINEAR,
                                               n_points=N_BINS, beta=BETA)
        analytic_time = time.perf_counter() - start
        
        # Compute statistics
        E_theoretical = T * E_base
        
        E_upper = np.sum(Z_upper.x * Z_upper.vals)
        E_lower = np.sum(Z_lower.x * Z_lower.vals)
        E_fft = np.sum(Z_fft.x * Z_fft.vals)
        
        # Handle Monte Carlo expectation calculation (filter out infinite values)
        finite_mask_mc = np.isfinite(Z_mc.x)
        if np.any(finite_mask_mc):
            E_mc = np.sum(Z_mc.x[finite_mask_mc] * Z_mc.vals[finite_mask_mc])
        else:
            E_mc = np.nan
            
        E_analytic = np.sum(Z_analytic.x * Z_analytic.vals)
        
        bias_main = E_upper - E_lower
        
        print(f"    Times: Main={main_time:.3f}s, FFT={fft_time:.3f}s, MC={mc_time:.3f}s, Analytic={analytic_time:.3f}s")
        print(f"    E[theoretical] = {E_theoretical:.6f}")
        print(f"    E[upper] = {E_upper:.6f}  (error: {E_upper - E_theoretical:+.6f})")
        print(f"    E[lower] = {E_lower:.6f}  (error: {E_lower - E_theoretical:+.6f})")
        print(f"    E[FFT] = {E_fft:.6f}  (error: {E_fft - E_theoretical:+.6f})")
        print(f"    E[MC] = {E_mc:.6f}  (error: {E_mc - E_theoretical:+.6f})")
        print(f"    E[Analytic] = {E_analytic:.6f}  (error: {E_analytic - E_theoretical:+.6f})")
        print(f"    Bias (upper-lower) = {bias_main:.6f}")
        
        results[T] = {
            'dist_upper': Z_upper,
            'dist_lower': Z_lower,
            'dist_fft': Z_fft,
            'dist_mc': Z_mc,
            'dist_analytic': Z_analytic,
            'E_upper': E_upper,
            'E_lower': E_lower,
            'E_fft': E_fft,
            'E_mc': E_mc,
            'E_analytic': E_analytic,
            'E_theoretical': E_theoretical,
            'bias_main': bias_main,
            'times': {'main': main_time, 'fft': fft_time, 'mc': mc_time, 'analytic': analytic_time}
        }
    
    return results

def run_lognormal_comparison(T_values):
    """
    Run comparison experiment for LogNormal distribution.
    
    Compares:
    - Main implementation (upper/lower bounds)
    - Monte Carlo convolution
    - Analytic convolution (approximate)
    """
    print(f"\n{'='*80}")
    print(f"LOGNORMAL COMPARISON: LogNormal(μ=0, σ=1) with geometric spacing")
    print(f"{'='*80}")
    
    # Create base LogNormal distribution
    dist_lognorm = stats.lognorm(s=1, scale=1)
    
    # Discretize base distribution using main implementation
    try:
        x_upper, pmf_upper, pneg_upper, ppos_upper = discretize_continuous_to_pmf(
            dist_lognorm, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC)
        x_lower, pmf_lower, pneg_lower, ppos_lower = discretize_continuous_to_pmf(
            dist_lognorm, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.GEOMETRIC)
        
        base_upper = DiscreteDist(x=x_upper, kind=DistKind.PMF, vals=pmf_upper, 
                                  p_neg_inf=pneg_upper, p_pos_inf=ppos_upper, name='LogNormal-upper')
        base_lower = DiscreteDist(x=x_lower, kind=DistKind.PMF, vals=pmf_lower,
                                  p_neg_inf=pneg_lower, p_pos_inf=ppos_lower, name='LogNormal-lower')
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"Base: {base_upper.name}, {len(base_upper.x):,} bins")
    print(f"  Range: [{base_upper.x[0]:.6f}, {base_upper.x[-1]:.6f}]")
    
    # Compute theoretical expectation
    E_base = dist_lognorm.mean()
    print(f"  Theoretical E[X] = {E_base:.6f}")
    
    results = {}
    
    for T in T_values:
        print(f"\n  T={T} copies...")
        
        # Warmup on first run
        if T == T_values[0]:
            _ = self_convolve_pmf(base_upper, T, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC)
            print(f"    [JIT warmup complete]")
        
        # Method 1: Main implementation (upper/lower bounds)
        start = time.perf_counter()
        Z_upper = self_convolve_pmf(base_upper, T, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC)
        Z_lower = self_convolve_pmf(base_lower, T, mode=Mode.IS_DOMINATED, spacing=Spacing.GEOMETRIC)
        main_time = time.perf_counter() - start
        
        # Method 2: FFT convolution
        start = time.perf_counter()
        Z_fft = fft_self_convolve_pmf(dist_lognorm, T, Mode.DOMINATES, Spacing.LINEAR, N_BINS, BETA)
        fft_time = time.perf_counter() - start
        
        # Method 3: Monte Carlo convolution
        start = time.perf_counter()
        Z_mc = monte_carlo_self_convolve_pmf(dist_lognorm, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                            MC_SAMPLES, N_BINS)
        mc_time = time.perf_counter() - start
        
        # Method 4: Analytic convolution (approximate for LogNormal)
        start = time.perf_counter()
        Z_analytic = analytic_convolve_lognormal(dist_lognorm, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                                n_points=N_BINS, beta=BETA)
        analytic_time = time.perf_counter() - start
        
        # Compute statistics
        E_theoretical = T * E_base
        
        E_upper = np.sum(Z_upper.x * Z_upper.vals)
        E_lower = np.sum(Z_lower.x * Z_lower.vals)
        E_fft = np.sum(Z_fft.x * Z_fft.vals)
        
        # Handle Monte Carlo expectation calculation (filter out infinite values)
        finite_mask_mc = np.isfinite(Z_mc.x)
        if np.any(finite_mask_mc):
            E_mc = np.sum(Z_mc.x[finite_mask_mc] * Z_mc.vals[finite_mask_mc])
        else:
            E_mc = np.nan
            
        E_analytic = np.sum(Z_analytic.x * Z_analytic.vals)
        
        bias_main = E_upper - E_lower
        
        print(f"    Times: Main={main_time:.3f}s, FFT={fft_time:.3f}s, MC={mc_time:.3f}s, Analytic={analytic_time:.3f}s")
        print(f"    E[theoretical] = {E_theoretical:.6f}")
        print(f"    E[upper] = {E_upper:.6f}  (error: {E_upper - E_theoretical:+.6f})")
        print(f"    E[lower] = {E_lower:.6f}  (error: {E_lower - E_theoretical:+.6f})")
        print(f"    E[FFT] = {E_fft:.6f}  (error: {E_fft - E_theoretical:+.6f})")
        print(f"    E[MC] = {E_mc:.6f}  (error: {E_mc - E_theoretical:+.6f})")
        print(f"    E[Analytic] = {E_analytic:.6f}  (error: {E_analytic - E_theoretical:+.6f})")
        print(f"    Bias (upper-lower) = {bias_main:.6f}")
        
        results[T] = {
            'dist_upper': Z_upper,
            'dist_lower': Z_lower,
            'dist_fft': Z_fft,
            'dist_mc': Z_mc,
            'dist_analytic': Z_analytic,
            'E_upper': E_upper,
            'E_lower': E_lower,
            'E_fft': E_fft,
            'E_mc': E_mc,
            'E_analytic': E_analytic,
            'E_theoretical': E_theoretical,
            'bias_main': bias_main,
            'times': {'main': main_time, 'fft': fft_time, 'mc': mc_time, 'analytic': analytic_time}
        }
    
    return results

def main():
    """Run comparison experiments."""
    print("="*80)
    print("METHOD COMPARISON: Main vs Alternative Implementations")
    print("="*80)
    print(f"Configuration: {N_BINS:,} bins, β={BETA}")
    print(f"Testing T = {T_VALUES}")
    print(f"Monte Carlo samples: {MC_SAMPLES:,}")
    print(f"RERUN_TESTS = {RERUN_TESTS}")
    
    # Run or load Gaussian comparison
    if RERUN_TESTS:
        print("\n" + "="*80)
        print("GAUSSIAN COMPARISON: N(0,1) with linear spacing")
        print("="*80)
        gaussian_results = run_gaussian_comparison(T_VALUES)
        save_results(gaussian_results, "gaussian_method_comparison", "results")
    else:
        gaussian_results = load_latest_results("gaussian_method_comparison", "results")
        if gaussian_results is None:
            print("No saved Gaussian results found. Set RERUN_TESTS=True to run experiments.")
            return
    
    # Run or load LogNormal comparison
    if RERUN_TESTS:
        print("\n" + "="*80)
        print("LOGNORMAL COMPARISON: LogNormal(μ=0, σ=1) with geometric spacing")
        print("="*80)
        lognormal_results = run_lognormal_comparison(T_VALUES)
        save_results(lognormal_results, "lognormal_method_comparison", "results")
    else:
        lognormal_results = load_latest_results("lognormal_method_comparison", "results")
        if lognormal_results is None:
            print("No saved LogNormal results found. Set RERUN_TESTS=True to run experiments.")
            return
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if gaussian_results:
        print(f"\nGaussian N(0,1):")
        for T in T_VALUES:
            if T not in gaussian_results:
                continue
            r = gaussian_results[T]
            print(f"  T={T:4d}: Main bias={r['bias_main']:8.3f}, "
                  f"FFT error={abs(r['E_fft']-r['E_theoretical']):7.4f}, "
                  f"MC error={abs(r['E_mc']-r['E_theoretical']):7.4f}, "
                  f"Analytic error={abs(r['E_analytic']-r['E_theoretical']):7.4f}")
    
    if lognormal_results:
        print(f"\nLogNormal(μ=0, σ=1):")
        for T in T_VALUES:
            if T not in lognormal_results:
                continue
            r = lognormal_results[T]
            print(f"  T={T:4d}: Main bias={r['bias_main']:8.3f}, "
                  f"MC error={abs(r['E_mc']-r['E_theoretical']):7.4f}, "
                  f"Analytic error={abs(r['E_analytic']-r['E_theoretical']):7.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_gaussian_results(gaussian_results, T_VALUES, save_plot_func=save_plot)
    plot_lognormal_results(lognormal_results, T_VALUES, save_plot_func=save_plot)
    
    print("\n" + "="*80)
    print("✅ All comparison experiments complete!")
    print("="*80)
    print("\nPlots saved:")
    print("  • plots/method_comparison_gaussian.png")
    print("  • plots/method_comparison_lognormal.png")
    print("="*80)

if __name__ == "__main__":
    main()
