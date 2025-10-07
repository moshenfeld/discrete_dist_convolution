"""
Enhanced experiment and visualization module v2.

Updates in v2:
- Works with new DiscreteDist API
- No tuple unpacking for discretization
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from implementation import (
    DiscreteDist, DistKind, Mode, Spacing, SummationMethod, GridStrategy,
    self_convolve_pmf, discretize_continuous_to_pmf, check_mass_conservation,
    pmf_to_cdf
)

from alternative_methods import (
    fft_self_convolve_continuous,
    monte_carlo_self_convolve_pmf,
    analytic_convolve_gaussian,
    analytic_convolve_lognormal
)

# =============================================================================
# EXPERIMENTAL CONFIGURATION
# =============================================================================

class ExperimentConfig:
    """Centralized configuration for all experiments."""
    
    # Method comparison experiments
    T_VALUES = [10, 100, 1000]
    
    # Gaussian-specific parameters
    GAUSSIAN_N_BINS = 20000
    GAUSSIAN_BETA = 1e-12  # Note: 1e-18 causes infinite quantiles
    GAUSSIAN_FFT_SIZE = 5000
    
    # Lognormal-specific parameters
    LOGNORMAL_N_BINS = 20000
    LOGNORMAL_BETA = 1e-18
    LOGNORMAL_FFT_SIZE = 5000
    
    # Monte Carlo parameters
    MC_SAMPLES = 1_000_000
    MC_BLOCK_SIZE = 100_000
    
    # Grid strategy comparison
    GRID_T_VALUES = [10, 50, 100, 500, 1000]
    GRID_N_BINS = 1000
    GRID_BETA = 1e-12
    
    # Diagnostics
    DIAGNOSTICS_T = 1000
    DIAGNOSTICS_N_BINS = 20000
    DIAGNOSTICS_BETA_DISCRETIZE = 1e-15
    DIAGNOSTICS_BETA_CONVOLVE = 1e-12
    
    # Numerical precision settings
    USE_KAHAN = True
    SUM_METHOD = SummationMethod.SORTED
    
    # Experiment control flags
    RUN_DIAGNOSTICS = True
    RUN_GRID_COMPARISON = True
    RUN_METHOD_COMPARISON = True
    
    # Monte Carlo settings (deprecated importance sampling)
    USE_MC_IMPORTANCE = False  # Always False - importance sampling was buggy

# =============================================================================
# DATA UTILITIES
# =============================================================================

def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_results(results: Dict[str, Any], experiment_type: str, filename_prefix: str = "results") -> str:
    """Save results to JSON file with timestamp."""
    exp_dir = Path("data") / experiment_type
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp()
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = exp_dir / filename
    
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, DiscreteDist):
                    serializable_results[key][subkey] = {
                        'x': subvalue.x.tolist(),
                        'vals': subvalue.vals.tolist(),
                        'kind': subvalue.kind.value,
                        'p_neg_inf': float(subvalue.p_neg_inf),
                        'p_pos_inf': float(subvalue.p_pos_inf)
                    }
                elif isinstance(subvalue, (np.ndarray, np.floating, np.integer)):
                    serializable_results[key][subkey] = subvalue.tolist() if hasattr(subvalue, 'tolist') else float(subvalue)
                else:
                    serializable_results[key][subkey] = subvalue
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"  Saved results to: {filepath}")
    return str(filepath)

def save_plot(fig, filename_prefix: str) -> str:
    """Save a matplotlib figure."""
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{filename_prefix}.png"
    filepath = plots_dir / filename
    
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {filepath}")
    
    return str(filepath)

# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def pmf_to_cdf_arrays(dist: DiscreteDist) -> tuple:
    """Convert PMF to CDF arrays."""
    cdf_dist = pmf_to_cdf(dist)
    return cdf_dist.x, cdf_dist.vals

def pmf_to_ccdf_arrays(dist: DiscreteDist) -> tuple:
    """Convert PMF to CCDF arrays."""
    cdf_dist = pmf_to_cdf(dist)
    ccdf_vals = 1.0 - cdf_dist.vals
    return cdf_dist.x, ccdf_vals

# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def diagnose_error_sources(base_dist: DiscreteDist, 
                          T: Optional[int] = None,
                          mode: Mode = Mode.DOMINATES,
                          spacing: Spacing = Spacing.LINEAR, 
                          beta: Optional[float] = None):
    """
    Diagnose sources of numerical error accumulation.
    
    Tests:
    1. Single convolution error (isolates kernel precision)
    2. T convolutions error (total accumulation)
    3. Comparison of Kahan vs standard summation
    4. Comparison of sort-and-sum vs standard summation
    """
    # Use config defaults if not specified
    if T is None:
        T = ExperimentConfig.DIAGNOSTICS_T
    if beta is None:
        beta = ExperimentConfig.DIAGNOSTICS_BETA_CONVOLVE
    
    print("\n" + "="*80)
    print(f"ERROR DIAGNOSTICS FOR T={T}")
    print("="*80)
    
    # Test 1: Single convolution with standard method
    print("\n1. Single Convolution (Standard):")
    try:
        result1_std = self_convolve_pmf(base_dist, T=2, mode=mode, spacing=spacing, beta=beta,
                                       use_kahan=False, sum_method=SummationMethod.STANDARD)
        error1_std = abs(result1_std.vals.sum() + result1_std.p_neg_inf + result1_std.p_pos_inf - 1.0)
        print(f"   Error: {error1_std:.2e}")
    except ValueError as e:
        error1_std = float('inf')
        print(f"   Error: FAILED - {e}")
    
    # Test 2: Single convolution with Kahan
    print("\n2. Single Convolution (Kahan):")
    try:
        result1_kahan = self_convolve_pmf(base_dist, T=2, mode=mode, spacing=spacing, beta=beta,
                                         use_kahan=True, sum_method=SummationMethod.STANDARD)
        error1_kahan = abs(result1_kahan.vals.sum() + result1_kahan.p_neg_inf + result1_kahan.p_pos_inf - 1.0)
        print(f"   Error: {error1_kahan:.2e}")
        print(f"   Improvement: {error1_std/error1_kahan:.1f}x")
    except ValueError as e:
        error1_kahan = float('inf')
        print(f"   Error: FAILED - {e}")
    
    # Test 3: T convolutions with standard method
    print(f"\n3. T={T} Convolutions (Standard + Standard):")
    try:
        resultT_std = self_convolve_pmf(base_dist, T=T, mode=mode, spacing=spacing, beta=beta,
                                       use_kahan=False, sum_method=SummationMethod.STANDARD)
        errorT_std = abs(resultT_std.vals.sum() + resultT_std.p_neg_inf + resultT_std.p_pos_inf - 1.0)
        print(f"   Error: {errorT_std:.2e}")
        if error1_std < float('inf'):
            print(f"   Growth rate: {errorT_std/error1_std:.1f}x")
            print(f"   Expected from log2(T): {np.log2(T):.1f}x")
    except ValueError as e:
        errorT_std = float('inf')
        print(f"   Error: FAILED - {e}")
    
    # Test 4: T convolutions with Kahan
    print(f"\n4. T={T} Convolutions (Kahan + Standard):")
    try:
        resultT_kahan = self_convolve_pmf(base_dist, T=T, mode=mode, spacing=spacing, beta=beta,
                                         use_kahan=True, sum_method=SummationMethod.STANDARD)
        errorT_kahan = abs(resultT_kahan.vals.sum() + resultT_kahan.p_neg_inf + resultT_kahan.p_pos_inf - 1.0)
        print(f"   Error: {errorT_kahan:.2e}")
        if error1_kahan < float('inf'):
            print(f"   Growth rate: {errorT_kahan/error1_kahan:.1f}x")
        if errorT_std < float('inf'):
            print(f"   Improvement over standard: {errorT_std/errorT_kahan:.1f}x")
    except ValueError as e:
        errorT_kahan = float('inf')
        print(f"   Error: FAILED - {e}")
    
    # Test 5: T convolutions with Kahan + Sorted summation
    print(f"\n5. T={T} Convolutions (Kahan + Sorted):")
    try:
        resultT_kahan_sorted = self_convolve_pmf(base_dist, T=T, mode=mode, spacing=spacing, beta=beta,
                                                use_kahan=True, sum_method=SummationMethod.SORTED)
        errorT_kahan_sorted = abs(np.sum(np.sort(np.abs(resultT_kahan_sorted.vals)) * 
                                         np.sign(resultT_kahan_sorted.vals[np.argsort(np.abs(resultT_kahan_sorted.vals))])) + 
                                 resultT_kahan_sorted.p_neg_inf + resultT_kahan_sorted.p_pos_inf - 1.0)
        print(f"   Error: {errorT_kahan_sorted:.2e}")
        if errorT_kahan < float('inf'):
            print(f"   Improvement over Kahan+Standard: {errorT_kahan/errorT_kahan_sorted:.1f}x")
    except ValueError as e:
        errorT_kahan_sorted = float('inf')
        print(f"   Error: FAILED - {e}")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"  Best single convolution: {min(error1_std, error1_kahan):.2e}")
    print(f"  Best T={T} convolution: {min(errorT_std, errorT_kahan, errorT_kahan_sorted):.2e}")
    print("="*80)

# =============================================================================
# GRID STRATEGY COMPARISON
# =============================================================================

def run_grid_strategy_comparison(T_values: Optional[List[int]] = None,
                                n_bins: Optional[int] = None,
                                beta: Optional[float] = None):
    """Compare FIXED_POINTS vs FIXED_WIDTH grid strategies."""
    # Use config defaults if not specified
    if T_values is None:
        T_values = ExperimentConfig.GRID_T_VALUES
    if n_bins is None:
        n_bins = ExperimentConfig.GRID_N_BINS
    if beta is None:
        beta = ExperimentConfig.GRID_BETA
    
    print(f"\n{'='*80}")
    print(f"GRID STRATEGY COMPARISON: LogNormal(μ=0, σ=1)")
    print(f"{'='*80}")
    
    dist_lognorm = stats.lognorm(s=1, scale=1)
    
    base = discretize_continuous_to_pmf(
        dist_lognorm, n_bins, beta, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC,
        name='LogNormal-base')
    
    print(f"Base distribution: {len(base.x)} bins")
    print(f"  Median bin width (relative): {np.median(np.diff(np.log(base.x))):.6f}")
    
    results_fixed_points = {}
    results_fixed_width = {}
    
    for T in T_values:
        print(f"\n  T={T}:")
        
        # FIXED_POINTS strategy
        start = time.perf_counter()
        result_fp = self_convolve_pmf(
            base, T=T, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC,
            grid_strategy=GridStrategy.FIXED_POINTS,
            use_kahan=ExperimentConfig.USE_KAHAN, 
            sum_method=ExperimentConfig.SUM_METHOD, 
            beta=beta
        )
        time_fp = time.perf_counter() - start
        
        # FIXED_WIDTH strategy
        start = time.perf_counter()
        result_fw = self_convolve_pmf(
            base, T=T, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC,
            grid_strategy=GridStrategy.FIXED_WIDTH,
            use_kahan=ExperimentConfig.USE_KAHAN, 
            sum_method=ExperimentConfig.SUM_METHOD, 
            beta=beta
        )
        time_fw = time.perf_counter() - start
        
        # Compute statistics
        E_theoretical = T * dist_lognorm.mean()
        
        E_fp = np.sum(result_fp.x * result_fp.vals)
        E_fw = np.sum(result_fw.x * result_fw.vals)
        
        # For geometric spacing, compute relative width (ratio-1)
        median_width_fp = np.median(result_fp.x[1:] / result_fp.x[:-1]) - 1.0
        median_width_fw = np.median(result_fw.x[1:] / result_fw.x[:-1]) - 1.0
        
        base_width = np.median(base.x[1:] / base.x[:-1]) - 1.0
        
        print(f"    FIXED_POINTS:")
        print(f"      Grid size: {len(result_fp.x)} (vs {len(base.x)} base)")
        print(f"      Median relative width: {median_width_fp:.6f} ({median_width_fp/base_width:.2f}x base)")
        print(f"      E[X]: {E_fp:.6f} (error: {E_fp - E_theoretical:+.6f})")
        print(f"      Time: {time_fp:.3f}s")
        
        print(f"    FIXED_WIDTH:")
        print(f"      Grid size: {len(result_fw.x)} (vs {len(base.x)} base)")
        print(f"      Median relative width: {median_width_fw:.6f} ({median_width_fw/base_width:.2f}x base)")
        print(f"      E[X]: {E_fw:.6f} (error: {E_fw - E_theoretical:+.6f})")
        print(f"      Time: {time_fw:.3f}s")
        
        results_fixed_points[T] = {
            'dist': result_fp,
            'n_bins': len(result_fp.x),
            'median_width': median_width_fp,
            'E_X': E_fp,
            'E_error': E_fp - E_theoretical,
            'time': time_fp
        }
        
        results_fixed_width[T] = {
            'dist': result_fw,
            'n_bins': len(result_fw.x),
            'median_width': median_width_fw,
            'E_X': E_fw,
            'E_error': E_fw - E_theoretical,
            'time': time_fw
        }
    
    return results_fixed_points, results_fixed_width, base_width

def plot_grid_strategy_comparison(results_fp, results_fw, base_width, T_values, save_plot_func=None):
    """Plot comparison of FIXED_POINTS vs FIXED_WIDTH strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fig.suptitle('Grid Strategy Comparison: FIXED_POINTS vs FIXED_WIDTH\nLogNormal(μ=0, σ=1) with Geometric Spacing', 
                 fontsize=14, fontweight='bold')
    
    T_list = sorted(T_values)
    
    # Plot 1: Grid size vs T
    ax = axes[0, 0]
    n_bins_fp = [results_fp[T]['n_bins'] for T in T_list]
    n_bins_fw = [results_fw[T]['n_bins'] for T in T_list]
    
    ax.plot(T_list, n_bins_fp, 'o-', label='FIXED_POINTS', linewidth=2, markersize=8)
    ax.plot(T_list, n_bins_fw, 's-', label='FIXED_WIDTH', linewidth=2, markersize=8)
    ax.set_xlabel('T (number of convolutions)')
    ax.set_ylabel('Grid size (number of bins)')
    ax.set_title('Grid Size vs T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 2: Median bin width vs T
    ax = axes[0, 1]
    width_fp = [results_fp[T]['median_width'] / base_width for T in T_list]
    width_fw = [results_fw[T]['median_width'] / base_width for T in T_list]
    
    ax.plot(T_list, width_fp, 'o-', label='FIXED_POINTS', linewidth=2, markersize=8)
    ax.plot(T_list, width_fw, 's-', label='FIXED_WIDTH', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Base width')
    ax.set_xlabel('T (number of convolutions)')
    ax.set_ylabel('Median bin width (relative to base)')
    ax.set_title('Bin Width vs T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 3: Expectation error vs T
    ax = axes[1, 0]
    error_fp = [abs(results_fp[T]['E_error']) for T in T_list]
    error_fw = [abs(results_fw[T]['E_error']) for T in T_list]
    
    ax.plot(T_list, error_fp, 'o-', label='FIXED_POINTS', linewidth=2, markersize=8)
    ax.plot(T_list, error_fw, 's-', label='FIXED_WIDTH', linewidth=2, markersize=8)
    ax.set_xlabel('T (number of convolutions)')
    ax.set_ylabel('|E[X] - E[X]_theoretical|')
    ax.set_title('Expectation Error vs T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 4: Computation time vs T
    ax = axes[1, 1]
    time_fp = [results_fp[T]['time'] for T in T_list]
    time_fw = [results_fw[T]['time'] for T in T_list]
    
    ax.plot(T_list, time_fp, 'o-', label='FIXED_POINTS', linewidth=2, markersize=8)
    ax.plot(T_list, time_fw, 's-', label='FIXED_WIDTH', linewidth=2, markersize=8)
    ax.set_xlabel('T (number of convolutions)')
    ax.set_ylabel('Computation time (s)')
    ax.set_title('Computation Time vs T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_plot_func:
        filename = save_plot_func(plt.gcf(), "grid_strategy_comparison")
        print(f"Plot saved: {filename}")
    
    plt.close()

# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def run_gaussian_comparison(T_values: Optional[List[int]] = None,
                           n_bins: Optional[int] = None,
                           beta: Optional[float] = None,
                           fft_size: Optional[int] = None,
                           mc_samples: Optional[int] = None):
    """Run comparison experiment for Gaussian distribution."""
    # Use config defaults if not specified
    if T_values is None:
        T_values = ExperimentConfig.T_VALUES
    if n_bins is None:
        n_bins = ExperimentConfig.GAUSSIAN_N_BINS
    if beta is None:
        beta = ExperimentConfig.GAUSSIAN_BETA
    if fft_size is None:
        fft_size = ExperimentConfig.GAUSSIAN_FFT_SIZE
    if mc_samples is None:
        mc_samples = ExperimentConfig.MC_SAMPLES
    
    print(f"\n{'='*80}")
    print(f"GAUSSIAN COMPARISON: N(0,1) with linear spacing")
    print(f"  Kahan summation: {ExperimentConfig.USE_KAHAN}")
    print(f"  Sum method: {ExperimentConfig.SUM_METHOD.value}")
    print(f"  N_BINS: {n_bins}, BETA: {beta:.2e}")
    print(f"{'='*80}")
    
    dist_gaussian = stats.norm(0, 1)
    
    try:
        base_upper = discretize_continuous_to_pmf(
            dist_gaussian, n_bins, beta, mode=Mode.DOMINATES, spacing=Spacing.LINEAR,
            name='Gaussian-upper')
        base_lower = discretize_continuous_to_pmf(
            dist_gaussian, n_bins, beta, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR,
            name='Gaussian-lower')
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"Base: {base_upper.name}, {len(base_upper.x):,} bins")
    print(f"  Range: [{base_upper.x[0]:.6f}, {base_upper.x[-1]:.6f}]")
    
    E_base = dist_gaussian.mean()
    print(f"  Theoretical E[X] = {E_base:.6f}")
    
    results = {}
    
    for T in T_values:
        print(f"\n  T={T} copies...")
        
        # Method 1: Main implementation
        start = time.perf_counter()
        Z_upper = self_convolve_pmf(base_upper, T, mode=Mode.DOMINATES, spacing=Spacing.LINEAR,
                                   use_kahan=ExperimentConfig.USE_KAHAN, 
                                   sum_method=ExperimentConfig.SUM_METHOD)
        Z_lower = self_convolve_pmf(base_lower, T, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR,
                                   use_kahan=ExperimentConfig.USE_KAHAN, 
                                   sum_method=ExperimentConfig.SUM_METHOD)
        main_time = time.perf_counter() - start
        
        # Method 2: FFT convolution
        start = time.perf_counter()
        Z_fft = fft_self_convolve_continuous(dist_gaussian, T, Mode.DOMINATES, Spacing.LINEAR, fft_size, beta)
        fft_time = time.perf_counter() - start
        
        # Method 3: Monte Carlo convolution
        start = time.perf_counter()
        Z_mc = monte_carlo_self_convolve_pmf(dist_gaussian, T, Mode.DOMINATES, Spacing.LINEAR, 
                                            mc_samples, n_bins, 
                                            use_importance_sampling=ExperimentConfig.USE_MC_IMPORTANCE)
        mc_time = time.perf_counter() - start
        
        # Method 4: Analytic convolution
        start = time.perf_counter()
        Z_analytic = analytic_convolve_gaussian(dist_gaussian, T, Mode.DOMINATES, Spacing.LINEAR,
                                               n_points=n_bins, beta=beta)
        analytic_time = time.perf_counter() - start
        
        # Compute statistics
        E_theoretical = T * E_base
        
        E_upper = np.sum(Z_upper.x * Z_upper.vals)
        E_lower = np.sum(Z_lower.x * Z_lower.vals)
        E_fft = np.sum(Z_fft.x * Z_fft.vals)
        
        finite_mask_mc = np.isfinite(Z_mc.x)
        E_mc = np.sum(Z_mc.x[finite_mask_mc] * Z_mc.vals[finite_mask_mc]) if np.any(finite_mask_mc) else np.nan
        
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

def run_lognormal_comparison(T_values: Optional[List[int]] = None,
                            n_bins: Optional[int] = None,
                            beta: Optional[float] = None,
                            fft_size: Optional[int] = None,
                            mc_samples: Optional[int] = None):
    """Run comparison experiment for LogNormal distribution."""
    # Use config defaults if not specified
    if T_values is None:
        T_values = ExperimentConfig.T_VALUES
    if n_bins is None:
        n_bins = ExperimentConfig.LOGNORMAL_N_BINS
    if beta is None:
        beta = ExperimentConfig.LOGNORMAL_BETA
    if fft_size is None:
        fft_size = ExperimentConfig.LOGNORMAL_FFT_SIZE
    if mc_samples is None:
        mc_samples = ExperimentConfig.MC_SAMPLES
    
    print(f"\n{'='*80}")
    print(f"LOGNORMAL COMPARISON: LogNormal(μ=0, σ=1) with geometric spacing")
    print(f"  Kahan summation: {ExperimentConfig.USE_KAHAN}")
    print(f"  Sum method: {ExperimentConfig.SUM_METHOD.value}")
    print(f"  N_BINS: {n_bins}, BETA: {beta:.2e}")
    print(f"{'='*80}")
    
    dist_lognorm = stats.lognorm(s=1, scale=1)
    
    try:
        base_upper = discretize_continuous_to_pmf(
            dist_lognorm, n_bins, beta, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC,
            name='LogNormal-upper')
        base_lower = discretize_continuous_to_pmf(
            dist_lognorm, n_bins, beta, mode=Mode.IS_DOMINATED, spacing=Spacing.GEOMETRIC,
            name='LogNormal-lower')
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"Base: {base_upper.name}, {len(base_upper.x):,} bins")
    print(f"  Range: [{base_upper.x[0]:.6f}, {base_upper.x[-1]:.6f}]")
    
    E_base = dist_lognorm.mean()
    print(f"  Theoretical E[X] = {E_base:.6f}")
    
    results = {}
    
    for T in T_values:
        print(f"\n  T={T} copies...")
        
        # Method 1: Main implementation
        start = time.perf_counter()
        Z_upper = self_convolve_pmf(base_upper, T, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC,
                                   use_kahan=ExperimentConfig.USE_KAHAN, 
                                   sum_method=ExperimentConfig.SUM_METHOD)
        Z_lower = self_convolve_pmf(base_lower, T, mode=Mode.IS_DOMINATED, spacing=Spacing.GEOMETRIC,
                                   use_kahan=ExperimentConfig.USE_KAHAN, 
                                   sum_method=ExperimentConfig.SUM_METHOD)
        main_time = time.perf_counter() - start
        
        # Method 2: FFT convolution
        start = time.perf_counter()
        Z_fft = fft_self_convolve_continuous(dist_lognorm, T, Mode.DOMINATES, Spacing.LINEAR, fft_size, beta)
        fft_time = time.perf_counter() - start
        
        # Method 3: Monte Carlo convolution
        start = time.perf_counter()
        Z_mc = monte_carlo_self_convolve_pmf(dist_lognorm, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                            mc_samples, n_bins, 
                                            use_importance_sampling=ExperimentConfig.USE_MC_IMPORTANCE)
        mc_time = time.perf_counter() - start
        
        # Method 4: Analytic convolution
        start = time.perf_counter()
        Z_analytic = analytic_convolve_lognormal(dist_lognorm, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                                n_points=n_bins, beta=beta)
        analytic_time = time.perf_counter() - start
        
        # Compute statistics
        E_theoretical = T * E_base
        
        E_upper = np.sum(Z_upper.x * Z_upper.vals)
        E_lower = np.sum(Z_lower.x * Z_lower.vals)
        E_fft = np.sum(Z_fft.x * Z_fft.vals)
        
        finite_mask_mc = np.isfinite(Z_mc.x)
        E_mc = np.sum(Z_mc.x[finite_mask_mc] * Z_mc.vals[finite_mask_mc]) if np.any(finite_mask_mc) else np.nan
        
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

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_gaussian_results(results: Dict[int, Dict[str, Any]], T_values: List[int], 
                         save_plot_func=None) -> None:
    """Generate plots for Gaussian comparison."""
    if results is None or len(results) == 0:
        print("  No Gaussian results to plot")
        return
    
    fig, axes = plt.subplots(2, len(T_values), figsize=(6*len(T_values), 10))
    if len(T_values) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Gaussian N(0,1) - Method Comparison (CDFs and CCDFs)', fontsize=14, fontweight='bold')
    
    for idx, T in enumerate(T_values):
        ax_cdf = axes[0, idx]
        ax_ccdf = axes[1, idx]
        
        T_key = str(T) if str(T) in results else T
        if T_key not in results:
            ax_cdf.text(0.5, 0.5, f'T={T}\nNo data', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nNo data', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        r = results[T_key]
        
        if r['dist_upper'].vals.sum() < 0.01:
            ax_cdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        x_upper, cdf_upper = pmf_to_cdf_arrays(r['dist_upper'])
        x_lower, cdf_lower = pmf_to_cdf_arrays(r['dist_lower'])
        x_fft, cdf_fft = pmf_to_cdf_arrays(r['dist_fft'])
        x_mc, cdf_mc = pmf_to_cdf_arrays(r['dist_mc'])
        x_analytic, cdf_analytic = pmf_to_cdf_arrays(r['dist_analytic'])
        
        _, ccdf_upper = pmf_to_ccdf_arrays(r['dist_upper'])
        _, ccdf_lower = pmf_to_ccdf_arrays(r['dist_lower'])
        _, ccdf_fft = pmf_to_ccdf_arrays(r['dist_fft'])
        _, ccdf_mc = pmf_to_ccdf_arrays(r['dist_mc'])
        _, ccdf_analytic = pmf_to_ccdf_arrays(r['dist_analytic'])
        
        x_min = min(x_upper[0], x_lower[0])
        x_max = max(x_upper[-1], x_lower[-1])
        
        # Plot CDFs
        ax_cdf.plot(x_upper, cdf_upper, 'C0-', linewidth=2, label='Main Upper', alpha=0.6)
        ax_cdf.plot(x_lower, cdf_lower, 'C1--', linewidth=2, label='Main Lower', alpha=0.6)
        ax_cdf.plot(x_fft, cdf_fft, 'C2:', linewidth=2, label='FFT', alpha=0.6)
        ax_cdf.plot(x_mc, cdf_mc, 'C3-.', linewidth=2, label='Monte Carlo', alpha=0.6)
        ax_cdf.plot(x_analytic, cdf_analytic, 'C4-', linewidth=2, label='Analytic (exact)', alpha=0.6)
        ax_cdf.fill_between(x_upper, cdf_lower, cdf_upper, alpha=0.1, color='C0')
        
        # Plot CCDFs
        ax_ccdf.plot(x_upper, ccdf_upper, 'C0-', linewidth=2, label='Main Upper', alpha=0.6)
        ax_ccdf.plot(x_lower, ccdf_lower, 'C1--', linewidth=2, label='Main Lower', alpha=0.6)
        ax_ccdf.plot(x_fft, ccdf_fft, 'C2:', linewidth=2, label='FFT', alpha=0.6)
        ax_ccdf.plot(x_mc, ccdf_mc, 'C3-.', linewidth=2, label='Monte Carlo', alpha=0.6)
        ax_ccdf.plot(x_analytic, ccdf_analytic, 'C4-', linewidth=2, label='Analytic (exact)', alpha=0.6)
        ax_ccdf.fill_between(x_upper, ccdf_lower, ccdf_upper, alpha=0.1, color='C0')
        
        ax_cdf.set_xlim(x_min, x_max)
        ax_ccdf.set_xlim(x_min, x_max)
        
        ax_cdf.set_xlabel('x')
        ax_cdf.set_ylabel('CDF')
        ax_cdf.set_title(f'T={T} - CDF\nBias={r["bias_main"]:.3f}')
        ax_cdf.legend(fontsize=8, loc='best')
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.set_yscale('log')
        
        ax_ccdf.set_xlabel('x')
        ax_ccdf.set_ylabel('CCDF')
        ax_ccdf.set_title(f'T={T} - CCDF\nBias={r["bias_main"]:.3f}')
        ax_ccdf.legend(fontsize=8, loc='best')
        ax_ccdf.grid(True, alpha=0.3)
        ax_ccdf.set_yscale('log')
    
    plt.tight_layout()
    
    if save_plot_func:
        filename = save_plot_func(plt.gcf(), "method_comparison_gaussian_enhanced_v2")
        print(f"Plot saved: {filename}")
    
    plt.close()

def plot_lognormal_results(results: Dict[int, Dict[str, Any]], T_values: List[int],
                          save_plot_func=None) -> None:
    """Generate plots for LogNormal comparison."""
    if results is None or len(results) == 0:
        print("  No LogNormal results to plot")
        return
    
    fig, axes = plt.subplots(2, len(T_values), figsize=(6*len(T_values), 10))
    if len(T_values) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('LogNormal(μ=0, σ=1) - Method Comparison (CDFs and CCDFs)', fontsize=14, fontweight='bold')
    
    for idx, T in enumerate(T_values):
        ax_cdf = axes[0, idx]
        ax_ccdf = axes[1, idx]
        
        T_key = str(T) if str(T) in results else T
        if T_key not in results:
            ax_cdf.text(0.5, 0.5, f'T={T}\nNo data', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nNo data', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        r = results[T_key]
        
        if r['dist_upper'].vals.sum() < 0.01:
            ax_cdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        x_upper, cdf_upper = pmf_to_cdf_arrays(r['dist_upper'])
        x_lower, cdf_lower = pmf_to_cdf_arrays(r['dist_lower'])
        x_fft, cdf_fft = pmf_to_cdf_arrays(r['dist_fft'])
        x_mc, cdf_mc = pmf_to_cdf_arrays(r['dist_mc'])
        x_analytic, cdf_analytic = pmf_to_cdf_arrays(r['dist_analytic'])
        
        _, ccdf_upper = pmf_to_ccdf_arrays(r['dist_upper'])
        _, ccdf_lower = pmf_to_ccdf_arrays(r['dist_lower'])
        _, ccdf_fft = pmf_to_ccdf_arrays(r['dist_fft'])
        _, ccdf_mc = pmf_to_ccdf_arrays(r['dist_mc'])
        _, ccdf_analytic = pmf_to_ccdf_arrays(r['dist_analytic'])
        
        def safe_log_transform(x_vals, cdf_vals):
            positive_mask = (x_vals > 0) & np.isfinite(x_vals)
            if np.any(positive_mask):
                log_x = np.log(x_vals[positive_mask])
                log_x_full = np.full_like(x_vals, -np.inf)
                log_x_full[positive_mask] = log_x
                return log_x_full, cdf_vals
            else:
                return np.full_like(x_vals, -np.inf), cdf_vals
        
        log_x_upper, cdf_upper = safe_log_transform(x_upper, cdf_upper)
        log_x_lower, cdf_lower = safe_log_transform(x_lower, cdf_lower)
        log_x_fft, cdf_fft = safe_log_transform(x_fft, cdf_fft)
        log_x_mc, cdf_mc = safe_log_transform(x_mc, cdf_mc)
        log_x_analytic, cdf_analytic = safe_log_transform(x_analytic, cdf_analytic)
        
        finite_log_values = []
        for log_x in [log_x_upper, log_x_lower]:
            finite_mask = np.isfinite(log_x)
            if np.any(finite_mask):
                finite_log_values.extend(log_x[finite_mask])
        
        if finite_log_values:
            log_x_min = min(finite_log_values)
            log_x_max = max(finite_log_values)
        else:
            log_x_min, log_x_max = -10, 10
        
        ax_cdf.plot(log_x_upper, cdf_upper, 'C0-', linewidth=2, label='Main Upper', alpha=0.6)
        ax_cdf.plot(log_x_lower, cdf_lower, 'C1--', linewidth=2, label='Main Lower', alpha=0.6)
        ax_cdf.plot(log_x_fft, cdf_fft, 'C2:', linewidth=2, label='FFT', alpha=0.6)
        ax_cdf.plot(log_x_mc, cdf_mc, 'C3-.', linewidth=2, label='Monte Carlo', alpha=0.6)
        ax_cdf.plot(log_x_analytic, cdf_analytic, 'C4-', linewidth=2, label='Analytic (approx)', alpha=0.6)
        ax_cdf.fill_between(log_x_upper, cdf_lower, cdf_upper, alpha=0.1, color='C0')
        
        ax_ccdf.plot(log_x_upper, ccdf_upper, 'C0-', linewidth=2, label='Main Upper', alpha=0.6)
        ax_ccdf.plot(log_x_lower, ccdf_lower, 'C1--', linewidth=2, label='Main Lower', alpha=0.6)
        ax_ccdf.plot(log_x_fft, ccdf_fft, 'C2:', linewidth=2, label='FFT', alpha=0.6)
        ax_ccdf.plot(log_x_mc, ccdf_mc, 'C3-.', linewidth=2, label='Monte Carlo', alpha=0.6)
        ax_ccdf.plot(log_x_analytic, ccdf_analytic, 'C4-', linewidth=2, label='Analytic (approx)', alpha=0.6)
        ax_ccdf.fill_between(log_x_upper, ccdf_lower, ccdf_upper, alpha=0.1, color='C0')
        
        E_theoretical = r['E_theoretical']
        log_E_theoretical = np.log(E_theoretical) if E_theoretical > 0 else log_x_min
        
        ax_cdf.set_xlim(log_x_min, log_E_theoretical)
        ax_ccdf.set_xlim(log_E_theoretical, log_x_max)
        
        ax_cdf.set_xlabel('log(x)')
        ax_cdf.set_ylabel('CDF')
        ax_cdf.set_title(f'T={T} - CDF (log space)\nBias={r["bias_main"]:.3f}')
        ax_cdf.legend(fontsize=8, loc='best')
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.set_yscale('log')
        
        ax_ccdf.set_xlabel('log(x)')
        ax_ccdf.set_ylabel('CCDF')
        ax_ccdf.set_title(f'T={T} - CCDF (log space)\nBias={r["bias_main"]:.3f}')
        ax_ccdf.legend(fontsize=8, loc='best')
        ax_ccdf.grid(True, alpha=0.3)
        ax_ccdf.set_yscale('log')
    
    plt.tight_layout()
    
    if save_plot_func:
        filename = save_plot_func(plt.gcf(), "method_comparison_lognormal_enhanced_v2")
        print(f"Plot saved: {filename}")
    
    plt.close()

# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def main():
    """Run comparison experiments with enhanced numerical stability."""
    print("="*80)
    print("ENHANCED METHOD COMPARISON V2: Configurable Precision")
    print("="*80)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  T values: {ExperimentConfig.T_VALUES}")
    print(f"  Grid comparison T values: {ExperimentConfig.GRID_T_VALUES}")
    print(f"  Use Kahan summation: {ExperimentConfig.USE_KAHAN}")
    print(f"  Sum method: {ExperimentConfig.SUM_METHOD.value}")
    print(f"  Gaussian: N_BINS={ExperimentConfig.GAUSSIAN_N_BINS}, BETA={ExperimentConfig.GAUSSIAN_BETA:.2e}")
    print(f"  Lognormal: N_BINS={ExperimentConfig.LOGNORMAL_N_BINS}, BETA={ExperimentConfig.LOGNORMAL_BETA:.2e}")
    print(f"  Monte Carlo: {ExperimentConfig.MC_SAMPLES:,} samples")
    print(f"  Run diagnostics: {ExperimentConfig.RUN_DIAGNOSTICS}")
    print(f"  Run grid comparison: {ExperimentConfig.RUN_GRID_COMPARISON}")
    print(f"  Run method comparison: {ExperimentConfig.RUN_METHOD_COMPARISON}")
    
    if ExperimentConfig.RUN_DIAGNOSTICS:
        # Run diagnostics for Gaussian
        print("\n" + "="*80)
        print("DIAGNOSTICS: Gaussian N(0,1)")
        print("="*80)
        dist_gaussian = stats.norm(0, 1)
        base_gaussian = discretize_continuous_to_pmf(
            dist_gaussian, 
            ExperimentConfig.DIAGNOSTICS_N_BINS, 
            ExperimentConfig.DIAGNOSTICS_BETA_DISCRETIZE, 
            mode=Mode.DOMINATES, 
            spacing=Spacing.LINEAR,
            name='Gaussian')
        diagnose_error_sources(base_gaussian)  # Uses config defaults for T and beta
    
    if ExperimentConfig.RUN_GRID_COMPARISON:
        print("\n" + "="*80)
        print("GRID STRATEGY COMPARISON")
        print("="*80)
        results_fp, results_fw, base_width = run_grid_strategy_comparison()  # Uses config defaults
        plot_grid_strategy_comparison(results_fp, results_fw, base_width, 
                                     ExperimentConfig.GRID_T_VALUES, save_plot_func=save_plot)
    
    if ExperimentConfig.RUN_METHOD_COMPARISON:
        print("\n" + "="*80)
        print("GAUSSIAN COMPARISON")
        print("="*80)
        gaussian_results = run_gaussian_comparison()  # Uses config defaults
        if gaussian_results:
            save_results(gaussian_results, "gaussian_method_comparison_enhanced_v2", "results")
        
        print("\n" + "="*80)
        print("LOGNORMAL COMPARISON")
        print("="*80)
        lognormal_results = run_lognormal_comparison()  # Uses config defaults
        if lognormal_results:
            save_results(lognormal_results, "lognormal_method_comparison_enhanced_v2", "results")
    
        # Generate plots
        print("\nGenerating plots...")
        if gaussian_results:
            plot_gaussian_results(gaussian_results, ExperimentConfig.T_VALUES, save_plot_func=save_plot)
        if lognormal_results:
            plot_lognormal_results(lognormal_results, ExperimentConfig.T_VALUES, save_plot_func=save_plot)
    
    print("\n" + "="*80)
    print("✅ All enhanced comparison experiments complete!")
    print("="*80)

if __name__ == "__main__":
    main()