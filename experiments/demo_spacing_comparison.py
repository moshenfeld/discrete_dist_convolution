"""
Demonstration comparing different distributions and spacing strategies.

Tests 4 configurations:
1. Gaussian N(0,1) - linear spacing (crosses zero)
2. Uniform [-0.5, 0.5] - linear spacing (crosses zero)
3. Exponential(λ=1) - geometric spacing (positive support)
4. LogNormal(μ=0, σ=1) - geometric spacing (positive support)

For each, we perform self-convolution T=10, 100, 1000 times.
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

# Standard parameters
N_BINS = 5000
BETA = 1e-6
T_VALUES = [10, 100, 1000]

def run_experiment(dist_name, dist_base, spacing, T_values):
    """
    Run self-convolution experiment for a given distribution.
    
    Parameters:
    -----------
    dist_name : str
        Name of the distribution
    dist_base : scipy.stats distribution
        Base distribution
    spacing : str
        "linear" or "geometric"
    T_values : list
        List of T values to test
        
    Returns:
    --------
    dict with results for each T
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {dist_name} with {spacing} spacing")
    print(f"{'='*80}")
    
    # Convert spacing string to enum
    spacing_enum = Spacing.GEOMETRIC if spacing == "geometric" else Spacing.LINEAR
    
    # Discretize base distribution
    try:
        x_upper, pmf_upper, pneg_upper, ppos_upper = discretize_continuous_to_pmf(
            dist_base, N_BINS, BETA, mode=Mode.DOMINATES, spacing=spacing_enum)
        x_lower, pmf_lower, pneg_lower, ppos_lower = discretize_continuous_to_pmf(
            dist_base, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=spacing_enum)
        
        base_upper = DiscreteDist(x=x_upper, kind=DistKind.PMF, vals=pmf_upper, 
                                  p_neg_inf=pneg_upper, p_pos_inf=ppos_upper, name=f'{dist_name}-upper')
        base_lower = DiscreteDist(x=x_lower, kind=DistKind.PMF, vals=pmf_lower,
                                  p_neg_inf=pneg_lower, p_pos_inf=ppos_lower, name=f'{dist_name}-lower')
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"Base: {base_upper.name}, {len(base_upper.x):,} bins")
    print(f"  Range: [{base_upper.x[0]:.6f}, {base_upper.x[-1]:.6f}]")
    bin_sizes = np.diff(base_upper.x)
    print(f"  Bin size: min={bin_sizes.min():.6e}, max={bin_sizes.max():.6e}, mean={bin_sizes.mean():.6e}")
    
    # Compute theoretical expectation
    E_base = dist_base.mean()
    print(f"  Theoretical E[X] = {E_base:.6f}")
    
    results = {}
    
    for T in T_values:
        print(f"\n  T={T} copies...")
        
        # Warmup on first run
        if T == T_values[0]:
            _ = self_convolve_pmf(base_upper, T, mode='DOMINATES', spacing=spacing_enum)
            print(f"    [JIT warmup complete]")
        
        # Compute bounds using automatic grid generation
        start = time.perf_counter()
        Z_upper = self_convolve_pmf(base_upper, T, mode='DOMINATES', spacing=spacing_enum)
        Z_lower = self_convolve_pmf(base_lower, T, mode='IS_DOMINATED', spacing=spacing_enum)
        elapsed = time.perf_counter() - start
        
        # Compute statistics
        E_upper = np.sum(Z_upper.x * Z_upper.vals)
        E_lower = np.sum(Z_lower.x * Z_lower.vals)
        E_theoretical = T * E_base
        bias = E_upper - E_lower
        
        # Output grid bin sizes
        out_bin_sizes = np.diff(Z_upper.x)
        
        print(f"    Time: {elapsed:.3f}s")
        print(f"    Output grid: [{Z_upper.x[0]:.3f}, {Z_upper.x[-1]:.3f}], {len(Z_upper.x)} bins")
        print(f"    Output bin size: mean={out_bin_sizes.mean():.6e}")
        print(f"    E[theoretical] = {E_theoretical:.6f}")
        print(f"    E[upper] = {E_upper:.6f}  (error: {E_upper - E_theoretical:+.6f})")
        print(f"    E[lower] = {E_lower:.6f}  (error: {E_lower - E_theoretical:+.6f})")
        print(f"    Bias (upper-lower) = {bias:.6f}")
        print(f"    Bias / T = {bias/T:.6f}")
        print(f"    Bias / mean_bin_size = {bias/out_bin_sizes.mean():.2f} bins")
        
        results[T] = {
            'dist_upper': Z_upper,
            'dist_lower': Z_lower,
            'E_upper': E_upper,
            'E_lower': E_lower,
            'E_theoretical': E_theoretical,
            'bias': bias,
            'time': elapsed,
            'output_grid': Z_upper.x,
            'bin_size': out_bin_sizes.mean()
        }
    
    return results

def plot_all_results(all_results, T_values):
    """Generate plots for all experiments."""
    
    experiment_configs = [
        ('gaussian_linear', 'Gaussian N(0,1) - Linear Spacing'),
        ('uniform_linear', 'Uniform[-0.5,0.5] - Linear Spacing'),
        ('exp_geometric', 'Exponential(λ=1) - Geometric Spacing'),
        ('lognorm_geometric', 'LogNormal(μ=0,σ=1) - Geometric Spacing')
    ]
    
    for exp_key, exp_title in experiment_configs:
        results = all_results.get(exp_key)
        if results is None or len(results) == 0:
            print(f"  Skipping {exp_key}: no results")
            continue
        
        # Create figure with subplots for each T value
        fig, axes = plt.subplots(1, len(T_values), figsize=(5*len(T_values), 4))
        if len(T_values) == 1:
            axes = [axes]
        
        fig.suptitle(exp_title, fontsize=14, fontweight='bold')
        
        for idx, T in enumerate(T_values):
            ax = axes[idx]
            
            if T not in results:
                ax.text(0.5, 0.5, f'T={T}\nNo data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'T={T}')
                continue
            
            r = results[T]
            Z_upper = r['dist_upper']
            Z_lower = r['dist_lower']
            
            # Check if distributions are valid (not all zeros)
            if Z_upper.vals.sum() < 0.01:
                ax.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'T={T}')
                continue
            
            # Plot upper and lower bounds
            ax.plot(Z_upper.x, Z_upper.vals, 'C0-', linewidth=1.5, 
                   label='DOMINATES (upper)', alpha=0.8)
            ax.plot(Z_lower.x, Z_lower.vals, 'C1--', linewidth=1.5,
                   label='IS_DOMINATED (lower)', alpha=0.8)
            ax.fill_between(Z_upper.x, Z_lower.vals, Z_upper.vals, alpha=0.2, color='C2')
            
            # Add vertical line at theoretical expectation
            E_theoretical = r['E_theoretical']
            ax.axvline(E_theoretical, color='red', linestyle='--', 
                      alpha=0.5, linewidth=2, label=f'E[theory]={E_theoretical:.1f}')
            
            # Labels and title
            ax.set_xlabel('x')
            ax.set_ylabel('PMF')
            ax.set_title(f'T={T}\nBias={r["bias"]:.3f}, E_up={r["E_upper"]:.2f}, E_low={r["E_lower"]:.2f}')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'plots/spacing_comparison_{exp_key}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

def main():
    """Run all 4 experiments."""
    print("="*80)
    print("Spacing Strategy Comparison")
    print("="*80)
    print(f"Configuration: {N_BINS:,} bins, β={BETA}")
    print(f"Testing T = {T_VALUES}")
    
    Path('plots').mkdir(exist_ok=True)
    
    all_results = {}
    
    # Experiment 1: Gaussian with linear spacing
    dist_gaussian = stats.norm(0, 1)
    results_gaussian = run_experiment(
        "Gaussian N(0,1)",
        dist_gaussian,
        "linear",
        T_VALUES
    )
    all_results['gaussian_linear'] = results_gaussian
    
    # Experiment 2: Uniform with linear spacing
    dist_uniform = stats.uniform(-0.5, 1.0)  # loc=-0.5, scale=1.0 gives [-0.5, 0.5]
    results_uniform = run_experiment(
        "Uniform[-0.5, 0.5]",
        dist_uniform,
        "linear",
        T_VALUES
    )
    all_results['uniform_linear'] = results_uniform
    
    # Experiment 3: Exponential with geometric spacing
    dist_exp = stats.expon(scale=1.0)  # rate=1, mean=1
    results_exp = run_experiment(
        "Exponential(λ=1)",
        dist_exp,
        "geometric",
        T_VALUES
    )
    all_results['exp_geometric'] = results_exp
    
    # Experiment 4: LogNormal with geometric spacing
    dist_lognorm = stats.lognorm(s=1.0, scale=np.exp(0))  # σ=1, μ=0
    results_lognorm = run_experiment(
        "LogNormal(μ=0, σ=1)",
        dist_lognorm,
        "geometric",
        T_VALUES
    )
    all_results['lognorm_geometric'] = results_lognorm
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for exp_name, results in all_results.items():
        if results is None:
            print(f"\n{exp_name}: FAILED")
            continue
        
        print(f"\n{exp_name}:")
        for T in T_VALUES:
            if T not in results:
                continue
            r = results[T]
            print(f"  T={T:4d}: bias={r['bias']:8.3f}, bias/T={r['bias']/T:7.4f}, "
                  f"bias/bin={r['bias']/r['bin_size']:6.1f} bins")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_all_results(all_results, T_VALUES)
    
    print("\n" + "="*80)
    print("✅ All experiments complete!")
    print("="*80)
    print("\nPlots saved:")
    print("  • plots/spacing_comparison_gaussian_linear.png")
    print("  • plots/spacing_comparison_uniform_linear.png")
    print("  • plots/spacing_comparison_exp_geometric.png")
    print("  • plots/spacing_comparison_lognorm_geometric.png")
    print("="*80)

if __name__ == "__main__":
    main()

