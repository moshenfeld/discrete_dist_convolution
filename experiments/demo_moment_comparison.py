"""
Alpha moment comparison demo for LogNormal convolution methods.

This script compares different convolution methods by computing alpha moments
for T=1000 convolution of LogNormal distributions, plotting moment curves
as a function of alpha.
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from discrete_conv_api import self_convolve_pmf, DiscreteDist, Mode, Spacing, DistKind
from implementation.grids import discretize_continuous_to_pmf
from comparisons import (
    fft_self_convolve_continuous,
    monte_carlo_self_convolve_pmf,
    analytic_convolve_lognormal
)
from evaluation.moments import compute_moment_sequence, compute_log_moment_sequence, moments_of_t_convolution

# Parameters
N_BINS = 20000
FFT_SIZE = 5000
BETA = 1e-15
T_VALUES = [10, 100, 1000]  # Number of convolutions to test
MC_SAMPLES = 1_000_000  # Monte Carlo samples
ALPHA_RANGE = np.arange(1, 21, dtype=int)  # Alpha values from 1 to 20 (integers)


def save_results_to_file(results, filename_prefix):
    """
    Save results to a JSON file with timestamp.
    
    Parameters:
    -----------
    results : dict
        Results dictionary containing all computed data
    filename_prefix : str
        Prefix for the filename
    """
    # Create data directory for lognormal moment comparison
    data_dir = project_root / "experiments" / "data" / "lognormal_moment_comparison"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = data_dir / filename
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    json_results[key][subkey] = subvalue.tolist()
                else:
                    json_results[key][subkey] = subvalue
        else:
            json_results[key] = value
    
    # Add metadata
    json_results['metadata'] = {
        'timestamp': timestamp,
        'alpha_range': ALPHA_RANGE.tolist(),
        'T_values': T_VALUES,
        'N_BINS': N_BINS,
        'FFT_SIZE': FFT_SIZE,
        'BETA': BETA,
        'MC_SAMPLES': MC_SAMPLES
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {filepath}")
    return filepath


def run_lognormal_moment_comparison():
    """
    Run alpha moment comparison for LogNormal convolution methods across multiple T values.
    
    Compares:
    - Main implementation (upper/lower bounds)
    - FFT convolution
    - Monte Carlo convolution
    - Analytic convolution
    
    Creates 2x3 subplot layout: 3 columns for T values, 2 rows for moment types
    """
    print("="*80)
    print("ALPHA MOMENT COMPARISON: LogNormal Convolution (T=10,100,1000)")
    print("="*80)
    print(f"Configuration: {N_BINS:,} bins, β={BETA}")
    print(f"T values: {T_VALUES}")
    print(f"Monte Carlo samples: {MC_SAMPLES:,}")
    print(f"Alpha range: [{ALPHA_RANGE[0]}, {ALPHA_RANGE[-1]}] ({len(ALPHA_RANGE)} points)")
    
    # Create base LogNormal distribution
    dist_lognorm = stats.lognorm(s=1, scale=1)
    
    print(f"\nBase distribution: LogNormal(μ=0, σ=1)")
    print(f"  Theoretical E[X] = {dist_lognorm.mean():.6f}")
    print(f"  Theoretical E[X²] = {dist_lognorm.moment(2):.6f}")
    
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
    
    print(f"Base discretization: {len(base_upper.x):,} bins")
    print(f"  Range: [{base_upper.x[0]:.6f}, {base_upper.x[-1]:.6f}]")
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LogNormal Convolution - Alpha Moments Comparison', fontsize=16, fontweight='bold')
    
    all_results = {}
    
    for col_idx, T in enumerate(T_VALUES):
        print(f"\nComputing T={T} convolutions...")
        
        # Method 1: Main implementation (upper/lower bounds)
        print("  Method 1: Main implementation...")
        start = time.perf_counter()
        Z_upper = self_convolve_pmf(base_upper, T, mode=Mode.DOMINATES, spacing=Spacing.GEOMETRIC)
        Z_lower = self_convolve_pmf(base_lower, T, mode=Mode.IS_DOMINATED, spacing=Spacing.GEOMETRIC)
        main_time = time.perf_counter() - start
        print(f"    Time: {main_time:.3f}s")
        
        # Method 2: FFT convolution
        print("  Method 2: FFT convolution...")
        start = time.perf_counter()
        Z_fft = fft_self_convolve_continuous(dist_lognorm, T, Mode.DOMINATES, Spacing.LINEAR, FFT_SIZE, BETA)
        fft_time = time.perf_counter() - start
        print(f"    Time: {fft_time:.3f}s")
        
        # Method 3: Monte Carlo convolution
        print("  Method 3: Monte Carlo convolution...")
        start = time.perf_counter()
        Z_mc = monte_carlo_self_convolve_pmf(dist_lognorm, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                            MC_SAMPLES, N_BINS)
        mc_time = time.perf_counter() - start
        print(f"    Time: {mc_time:.3f}s")
        
        # Method 4: Analytic convolution
        print("  Method 4: Analytic convolution...")
        start = time.perf_counter()
        Z_analytic = analytic_convolve_lognormal(dist_lognorm, T, Mode.DOMINATES, Spacing.GEOMETRIC,
                                                n_points=N_BINS, beta=BETA)
        analytic_time = time.perf_counter() - start
        print(f"    Time: {analytic_time:.3f}s")
        
        # Compute theoretical moments for comparison
        print(f"  Computing theoretical moments...")
        try:
            # Round alpha values to integers since the function requires integer moments
            alpha_ints = np.round(ALPHA_RANGE).astype(int)
            theoretical_moments = moments_of_t_convolution(dist_lognorm, T, alpha_ints)
        except (OverflowError, ValueError) as e:
            print(f"    Warning: Theoretical moment computation failed: {e}")
            theoretical_moments = np.full_like(ALPHA_RANGE, np.inf, dtype=float)
        
        print(f"  Computing alpha moments for all methods...")
        
        # Compute regular moments
        moments_upper = compute_moment_sequence(Z_upper, ALPHA_RANGE)
        moments_lower = compute_moment_sequence(Z_lower, ALPHA_RANGE)
        moments_fft = compute_moment_sequence(Z_fft, ALPHA_RANGE)
        moments_mc = compute_moment_sequence(Z_mc, ALPHA_RANGE)
        moments_analytic = compute_moment_sequence(Z_analytic, ALPHA_RANGE)
        
        # Compute log moments
        log_moments_upper = compute_log_moment_sequence(Z_upper, ALPHA_RANGE)
        log_moments_lower = compute_log_moment_sequence(Z_lower, ALPHA_RANGE)
        log_moments_fft = compute_log_moment_sequence(Z_fft, ALPHA_RANGE)
        log_moments_mc = compute_log_moment_sequence(Z_mc, ALPHA_RANGE)
        log_moments_analytic = compute_log_moment_sequence(Z_analytic, ALPHA_RANGE)
        
        # Plotting helper function
        def plot_log_safe(ax, x, y, *args, **kwargs):
            """Plot data safely on log scale by filtering out non-positive values."""
            mask = (y > 0) & np.isfinite(y)
            if np.any(mask):
                ax.plot(x[mask], y[mask], *args, **kwargs)
                print(f"    Plotted {np.sum(mask)}/{len(y)} points for {kwargs.get('label', 'unknown')}")
            else:
                print(f"    No valid points to plot for {kwargs.get('label', 'unknown')}")
        
        # Plot regular moments (top row)
        ax_top = axes[0, col_idx]
        print(f"  Plotting raw moments for T={T}...")
        plot_log_safe(ax_top, ALPHA_RANGE, theoretical_moments, 'k-', linewidth=2, 
                      label='Theoretical', alpha=0.8)
        plot_log_safe(ax_top, ALPHA_RANGE, moments_upper, 'C0-', linewidth=2, 
                      label='Main Upper', alpha=0.7)
        plot_log_safe(ax_top, ALPHA_RANGE, moments_lower, 'C1--', linewidth=2,
                      label='Main Lower', alpha=0.7)
        plot_log_safe(ax_top, ALPHA_RANGE, moments_fft, 'C2:', linewidth=2,
                      label='FFT', alpha=0.7)
        plot_log_safe(ax_top, ALPHA_RANGE, moments_mc, 'C3-.', linewidth=2,
                      label='Monte Carlo', alpha=0.7)
        plot_log_safe(ax_top, ALPHA_RANGE, moments_analytic, 'C4-', linewidth=2,
                      label='Analytic', alpha=0.7)
        
        ax_top.set_xlabel('α (moment order)')
        ax_top.set_ylabel('E[X^α]')
        ax_top.set_title(f'T={T} - Raw Moments')
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)
        ax_top.set_yscale('log')
        print(f"    Set log scale on raw moments plot")
        
        # Plot log moments (bottom row)
        ax_bottom = axes[1, col_idx]
        print(f"  Plotting log moments for T={T}...")
        plot_log_safe(ax_bottom, ALPHA_RANGE, log_moments_upper, 'C0-', linewidth=2, 
                      label='Main Upper', alpha=0.7)
        plot_log_safe(ax_bottom, ALPHA_RANGE, log_moments_lower, 'C1--', linewidth=2,
                      label='Main Lower', alpha=0.7)
        plot_log_safe(ax_bottom, ALPHA_RANGE, log_moments_fft, 'C2:', linewidth=2,
                      label='FFT', alpha=0.7)
        plot_log_safe(ax_bottom, ALPHA_RANGE, log_moments_mc, 'C3-.', linewidth=2,
                      label='Monte Carlo', alpha=0.7)
        plot_log_safe(ax_bottom, ALPHA_RANGE, log_moments_analytic, 'C4-', linewidth=2,
                      label='Analytic', alpha=0.7)
        
        ax_bottom.set_xlabel('α (moment order)')
        ax_bottom.set_ylabel('E[(log X)^α]')
        ax_bottom.set_title(f'T={T} - Log Moments')
        ax_bottom.legend(fontsize=8)
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.set_yscale('log')
        print(f"    Set log scale on log moments plot")
        
        # Store results for this T value
        all_results[T] = {
            'alpha_values': ALPHA_RANGE,
            'theoretical_moments': theoretical_moments,
            'moments_upper': moments_upper,
            'moments_lower': moments_lower,
            'moments_fft': moments_fft,
            'moments_mc': moments_mc,
            'moments_analytic': moments_analytic,
            'log_moments_upper': log_moments_upper,
            'log_moments_lower': log_moments_lower,
            'log_moments_fft': log_moments_fft,
            'log_moments_mc': log_moments_mc,
            'log_moments_analytic': log_moments_analytic,
            'times': {
                'main': main_time,
                'fft': fft_time,
                'mc': mc_time,
                'analytic': analytic_time
            }
        }
    
    plt.tight_layout()
    
    # Save plot
    plot_path = project_root / "experiments" / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_path / f"lognormal_moment_comparison_multiT.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {plot_filename}")
    
    
    # Summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for T in T_VALUES:
        print(f"\nT={T}:")
        times = all_results[T]['times']
        print(f"  Times: Main={times['main']:.3f}s, FFT={times['fft']:.3f}s, MC={times['mc']:.3f}s, Analytic={times['analytic']:.3f}s")
    
    # Save results to file
    save_results_to_file(all_results, "lognormal_moment_comparison_multiT")
    
    return all_results


def main():
    """Run the alpha moment comparison demo."""
    results = run_lognormal_moment_comparison()
    
    if results is not None:
        print(f"\n" + "="*80)
        print("✅ Alpha moment comparison complete!")
        print("="*80)
    else:
        print(f"\n" + "="*80)
        print("❌ Alpha moment comparison failed!")
        print("="*80)


if __name__ == "__main__":
    main()
