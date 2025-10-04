"""
Comparison test between main implementation and alternative convolution methods.

This script compares the main implementation (upper and lower bounds) with:
1. FFT-based convolution (for Gaussian only, requires linear grids)
2. Monte Carlo sampling convolution
3. Analytic convolution (exact for Gaussian, approximate for lognormal)

For Gaussian: compares all 4 methods (main upper/lower + 3 alternatives)
For Lognormal: compares 3 methods (main upper/lower + Monte Carlo + Analytic, no FFT)
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from discrete_conv_api import discretize_continuous_to_pmf, self_convolve_pmf, DiscreteDist, Mode, Spacing, DistKind

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

# Standard parameters
N_BINS = 5000
BETA = 1e-12
T_VALUES = [10, 100]  # Reduced for investigation
MC_SAMPLES = 100000  # Monte Carlo samples

# Control variables
RERUN_TESTS = True  # Set to False to only run visualization from saved data

def run_gaussian_comparison(T_values):
    """
    Run comparison experiment for Gaussian distribution.
    
    Compares:
    - Main implementation (upper/lower bounds)
    - FFT convolution
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
            _ = self_convolve_pmf(base_upper, T, mode='DOMINATES', spacing=Spacing.LINEAR)
            print(f"    [JIT warmup complete]")
        
        # Method 1: Main implementation (upper/lower bounds)
        start = time.perf_counter()
        Z_upper = self_convolve_pmf(base_upper, T, mode='DOMINATES', spacing=Spacing.LINEAR)
        Z_lower = self_convolve_pmf(base_lower, T, mode='IS_DOMINATED', spacing=Spacing.LINEAR)
        main_time = time.perf_counter() - start
        
        # Method 2: FFT convolution
        start = time.perf_counter()
        Z_fft = fft_self_convolve_pmf(base_upper, T, Mode.DOMINATES, Spacing.LINEAR)
        fft_time = time.perf_counter() - start
        
        # Method 3: Monte Carlo convolution
        start = time.perf_counter()
        Z_mc = monte_carlo_self_convolve_pmf(base_upper, T, Mode.DOMINATES, Spacing.LINEAR, 
                                            n_samples=MC_SAMPLES, n_bins=N_BINS)
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
    dist_lognorm = stats.lognorm(s=1.0, scale=np.exp(0))  # σ=1, μ=0
    
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
            _ = self_convolve_pmf(base_upper, T, mode='DOMINATES', spacing=Spacing.GEOMETRIC)
            print(f"    [JIT warmup complete]")
        
        # Method 1: Main implementation (upper/lower bounds)
        start = time.perf_counter()
        Z_upper = self_convolve_pmf(base_upper, T, mode='DOMINATES', spacing=Spacing.GEOMETRIC)
        Z_lower = self_convolve_pmf(base_lower, T, mode='IS_DOMINATED', spacing=Spacing.GEOMETRIC)
        main_time = time.perf_counter() - start
        
        # Method 2: Monte Carlo convolution
        start = time.perf_counter()
        Z_mc = monte_carlo_self_convolve_pmf(base_upper, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                            n_samples=MC_SAMPLES, n_bins=N_BINS)
        mc_time = time.perf_counter() - start
        
        # Method 3: Analytic convolution (approximate for LogNormal)
        start = time.perf_counter()
        Z_analytic = analytic_convolve_lognormal(dist_lognorm, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                                n_points=N_BINS, beta=BETA)
        analytic_time = time.perf_counter() - start
        
        # Compute statistics
        E_theoretical = T * E_base
        
        E_upper = np.sum(Z_upper.x * Z_upper.vals)
        E_lower = np.sum(Z_lower.x * Z_lower.vals)
        
        # Handle Monte Carlo expectation calculation (filter out infinite values)
        finite_mask_mc = np.isfinite(Z_mc.x)
        if np.any(finite_mask_mc):
            E_mc = np.sum(Z_mc.x[finite_mask_mc] * Z_mc.vals[finite_mask_mc])
        else:
            E_mc = np.nan
            
        E_analytic = np.sum(Z_analytic.x * Z_analytic.vals)
        
        bias_main = E_upper - E_lower
        
        print(f"    Times: Main={main_time:.3f}s, MC={mc_time:.3f}s, Analytic={analytic_time:.3f}s")
        print(f"    E[theoretical] = {E_theoretical:.6f}")
        print(f"    E[upper] = {E_upper:.6f}  (error: {E_upper - E_theoretical:+.6f})")
        print(f"    E[lower] = {E_lower:.6f}  (error: {E_lower - E_theoretical:+.6f})")
        print(f"    E[MC] = {E_mc:.6f}  (error: {E_mc - E_theoretical:+.6f})")
        print(f"    E[Analytic] = {E_analytic:.6f}  (error: {E_analytic - E_theoretical:+.6f})")
        print(f"    Bias (upper-lower) = {bias_main:.6f}")
        
        results[T] = {
            'dist_upper': Z_upper,
            'dist_lower': Z_lower,
            'dist_mc': Z_mc,
            'dist_analytic': Z_analytic,
            'E_upper': E_upper,
            'E_lower': E_lower,
            'E_mc': E_mc,
            'E_analytic': E_analytic,
            'E_theoretical': E_theoretical,
            'bias_main': bias_main,
            'times': {'main': main_time, 'mc': mc_time, 'analytic': analytic_time}
        }
    
    return results

def pmf_to_cdf(dist: DiscreteDist) -> tuple:
    """Convert PMF to CDF."""
    cdf_vals = np.cumsum(dist.vals)
    # Add infinity masses
    cdf_vals = cdf_vals + dist.p_neg_inf
    return dist.x, cdf_vals

def pmf_to_ccdf(dist: DiscreteDist) -> tuple:
    """Convert PMF to CCDF (Complementary CDF)."""
    cdf_vals = np.cumsum(dist.vals)
    # Add infinity masses
    cdf_vals = cdf_vals + dist.p_neg_inf
    ccdf_vals = 1.0 - cdf_vals
    return dist.x, ccdf_vals

def plot_gaussian_results(results, T_values):
    """Generate plots for Gaussian comparison."""
    if results is None or len(results) == 0:
        print("  No Gaussian results to plot")
        return
    
    # Create figure with subplots for each T value (2 rows: CDF and CCDF)
    fig, axes = plt.subplots(2, len(T_values), figsize=(6*len(T_values), 10))
    if len(T_values) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Gaussian N(0,1) - Method Comparison (CDFs and CCDFs)', fontsize=14, fontweight='bold')
    
    for idx, T in enumerate(T_values):
        ax_cdf = axes[0, idx]  # CDF subplot
        ax_ccdf = axes[1, idx]  # CCDF subplot
        
        if T not in results:
            ax_cdf.text(0.5, 0.5, f'T={T}\nNo data', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nNo data', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        r = results[T]
        
        # Check if distributions are valid
        if r['dist_upper'].vals.sum() < 0.01:
            ax_cdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        # Convert PMFs to CDFs and CCDFs
        x_upper, cdf_upper = pmf_to_cdf(r['dist_upper'])
        x_lower, cdf_lower = pmf_to_cdf(r['dist_lower'])
        x_fft, cdf_fft = pmf_to_cdf(r['dist_fft'])
        x_mc, cdf_mc = pmf_to_cdf(r['dist_mc'])
        x_analytic, cdf_analytic = pmf_to_cdf(r['dist_analytic'])
        
        _, ccdf_upper = pmf_to_ccdf(r['dist_upper'])
        _, ccdf_lower = pmf_to_ccdf(r['dist_lower'])
        _, ccdf_fft = pmf_to_ccdf(r['dist_fft'])
        _, ccdf_mc = pmf_to_ccdf(r['dist_mc'])
        _, ccdf_analytic = pmf_to_ccdf(r['dist_analytic'])
        
        # Determine x-axis range based on main upper and lower bounds
        x_min = min(x_upper[0], x_lower[0])
        x_max = max(x_upper[-1], x_lower[-1])
        
        # Plot CDFs
        ax_cdf.plot(x_upper, cdf_upper, 'C0-', linewidth=1.5, 
                   label='Main Upper', alpha=0.8)
        ax_cdf.plot(x_lower, cdf_lower, 'C1--', linewidth=1.5,
                   label='Main Lower', alpha=0.8)
        ax_cdf.plot(x_fft, cdf_fft, 'C2:', linewidth=2,
                   label='FFT', alpha=0.9)
        ax_cdf.plot(x_mc, cdf_mc, 'C3-.', linewidth=1.5,
                   label='Monte Carlo', alpha=0.8)
        ax_cdf.plot(x_analytic, cdf_analytic, 'C4-', linewidth=2,
                   label='Analytic (exact)', alpha=0.9)
        
        # Fill between upper and lower bounds
        ax_cdf.fill_between(x_upper, cdf_lower, cdf_upper, 
                           alpha=0.1, color='C0')
        
        # Plot CCDFs
        ax_ccdf.plot(x_upper, ccdf_upper, 'C0-', linewidth=1.5, 
                    label='Main Upper', alpha=0.8)
        ax_ccdf.plot(x_lower, ccdf_lower, 'C1--', linewidth=1.5,
                    label='Main Lower', alpha=0.8)
        ax_ccdf.plot(x_fft, ccdf_fft, 'C2:', linewidth=2,
                    label='FFT', alpha=0.9)
        ax_ccdf.plot(x_mc, ccdf_mc, 'C3-.', linewidth=1.5,
                    label='Monte Carlo', alpha=0.8)
        ax_ccdf.plot(x_analytic, ccdf_analytic, 'C4-', linewidth=2,
                    label='Analytic (exact)', alpha=0.9)
        
        # Fill between upper and lower bounds for CCDF
        ax_ccdf.fill_between(x_upper, ccdf_lower, ccdf_upper, 
                            alpha=0.1, color='C0')
        
        # Add vertical line at theoretical expectation
        E_theoretical = r['E_theoretical']
        ax_cdf.axvline(E_theoretical, color='red', linestyle='--', 
                      alpha=0.5, linewidth=2, label=f'E[theory]={E_theoretical:.1f}')
        ax_ccdf.axvline(E_theoretical, color='red', linestyle='--', 
                      alpha=0.5, linewidth=2, label=f'E[theory]={E_theoretical:.1f}')
        
        # Set x-axis range based on main bounds
        ax_cdf.set_xlim(x_min, x_max)
        ax_ccdf.set_xlim(x_min, x_max)
        
        # Labels and titles
        ax_cdf.set_xlabel('x')
        ax_cdf.set_ylabel('CDF')
        ax_cdf.set_title(f'T={T} - CDF\nBias={r["bias_main"]:.3f}')
        ax_cdf.legend(fontsize=8, loc='best')
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.set_yscale('log')
        ax_cdf.set_ylim(1e-6, 1)
        
        ax_ccdf.set_xlabel('x')
        ax_ccdf.set_ylabel('CCDF')
        ax_ccdf.set_title(f'T={T} - CCDF\nBias={r["bias_main"]:.3f}')
        ax_ccdf.legend(fontsize=8, loc='best')
        ax_ccdf.grid(True, alpha=0.3)
        ax_ccdf.set_yscale('log')
        ax_ccdf.set_ylim(1e-6, 1)
    
    plt.tight_layout()
    filename = save_plot(plt.gcf(), "method_comparison_gaussian")
    plt.close()

def plot_lognormal_results(results, T_values):
    """Generate plots for LogNormal comparison."""
    if results is None or len(results) == 0:
        print("  No LogNormal results to plot")
        return
    
    # Create figure with subplots for each T value (2 rows: CDF and CCDF)
    fig, axes = plt.subplots(2, len(T_values), figsize=(6*len(T_values), 10))
    if len(T_values) == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('LogNormal(μ=0, σ=1) - Method Comparison (CDFs and CCDFs)', fontsize=14, fontweight='bold')
    
    for idx, T in enumerate(T_values):
        ax_cdf = axes[0, idx]  # CDF subplot
        ax_ccdf = axes[1, idx]  # CCDF subplot
        
        if T not in results:
            ax_cdf.text(0.5, 0.5, f'T={T}\nNo data', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nNo data', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        r = results[T]
        
        # Check if distributions are valid
        if r['dist_upper'].vals.sum() < 0.01:
            ax_cdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        # Convert PMFs to CDFs and CCDFs
        x_upper, cdf_upper = pmf_to_cdf(r['dist_upper'])
        x_lower, cdf_lower = pmf_to_cdf(r['dist_lower'])
        x_mc, cdf_mc = pmf_to_cdf(r['dist_mc'])
        x_analytic, cdf_analytic = pmf_to_cdf(r['dist_analytic'])
        
        _, ccdf_upper = pmf_to_ccdf(r['dist_upper'])
        _, ccdf_lower = pmf_to_ccdf(r['dist_lower'])
        _, ccdf_mc = pmf_to_ccdf(r['dist_mc'])
        _, ccdf_analytic = pmf_to_ccdf(r['dist_analytic'])
        
        # Determine x-axis range based on main upper and lower bounds
        x_min = min(x_upper[0], x_lower[0])
        x_max = max(x_upper[-1], x_lower[-1])
        
        # Plot CDFs
        ax_cdf.plot(x_upper, cdf_upper, 'C0-', linewidth=1.5, 
                   label='Main Upper', alpha=0.8)
        ax_cdf.plot(x_lower, cdf_lower, 'C1--', linewidth=1.5,
                   label='Main Lower', alpha=0.8)
        ax_cdf.plot(x_mc, cdf_mc, 'C3-.', linewidth=1.5,
                   label='Monte Carlo', alpha=0.8)
        ax_cdf.plot(x_analytic, cdf_analytic, 'C4-', linewidth=2,
                   label='Analytic (approx)', alpha=0.9)
        
        # Fill between upper and lower bounds
        ax_cdf.fill_between(x_upper, cdf_lower, cdf_upper, 
                           alpha=0.1, color='C0')
        
        # Plot CCDFs
        ax_ccdf.plot(x_upper, ccdf_upper, 'C0-', linewidth=1.5, 
                    label='Main Upper', alpha=0.8)
        ax_ccdf.plot(x_lower, ccdf_lower, 'C1--', linewidth=1.5,
                    label='Main Lower', alpha=0.8)
        ax_ccdf.plot(x_mc, ccdf_mc, 'C3-.', linewidth=1.5,
                    label='Monte Carlo', alpha=0.8)
        ax_ccdf.plot(x_analytic, ccdf_analytic, 'C4-', linewidth=2,
                    label='Analytic (approx)', alpha=0.9)
        
        # Fill between upper and lower bounds for CCDF
        ax_ccdf.fill_between(x_upper, ccdf_lower, ccdf_upper, 
                            alpha=0.1, color='C0')
        
        # Add vertical line at theoretical expectation
        E_theoretical = r['E_theoretical']
        ax_cdf.axvline(E_theoretical, color='red', linestyle='--', 
                      alpha=0.5, linewidth=2, label=f'E[theory]={E_theoretical:.1f}')
        ax_ccdf.axvline(E_theoretical, color='red', linestyle='--', 
                      alpha=0.5, linewidth=2, label=f'E[theory]={E_theoretical:.1f}')
        
        # Set x-axis range based on main bounds
        ax_cdf.set_xlim(x_min, x_max)
        ax_ccdf.set_xlim(x_min, x_max)
        
        # Labels and titles
        ax_cdf.set_xlabel('x')
        ax_cdf.set_ylabel('CDF')
        ax_cdf.set_title(f'T={T} - CDF\nBias={r["bias_main"]:.3f}')
        ax_cdf.legend(fontsize=8, loc='best')
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.set_yscale('log')
        ax_cdf.set_ylim(1e-6, 1)
        
        ax_ccdf.set_xlabel('x')
        ax_ccdf.set_ylabel('CCDF')
        ax_ccdf.set_title(f'T={T} - CCDF\nBias={r["bias_main"]:.3f}')
        ax_ccdf.legend(fontsize=8, loc='best')
        ax_ccdf.grid(True, alpha=0.3)
        ax_ccdf.set_yscale('log')
        ax_ccdf.set_ylim(1e-6, 1)
    
    plt.tight_layout()
    filename = save_plot(plt.gcf(), "method_comparison_lognormal")
    plt.close()

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
    plot_gaussian_results(gaussian_results, T_VALUES)
    plot_lognormal_results(lognormal_results, T_VALUES)
    
    print("\n" + "="*80)
    print("✅ All comparison experiments complete!")
    print("="*80)
    print("\nPlots saved:")
    print("  • plots/method_comparison_gaussian.png")
    print("  • plots/method_comparison_lognormal.png")
    print("="*80)

if __name__ == "__main__":
    main()
