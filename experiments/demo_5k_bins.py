"""
Standard demonstration using 5,000 bins (optimal balance of accuracy vs speed).

Demonstrates:
1. PMF×PMF convolution (pairwise)
2. Self-convolution (X + X + ... + X) for T=10, 100, 1000

All tests use proper quantile-based grid generation per IMPLEMENTATION_GUIDE_NUMBA.md
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from discrete_conv_api import (
    DistKind,
    discretize_continuous_to_pmf, 
    convolve_pmf_pmf_to_pmf, 
    self_convolve_pmf, 
    DiscreteDist,
    Mode,
    Spacing
)

# Standard grid resolution
N_BINS = 5000
BETA = 1e-6  # Tail probability to trim

def demo_pairwise_convolution():
    """Demonstrate pairwise PMF×PMF convolution."""
    print("\n" + "="*80)
    print("1. Pairwise PMF×PMF Convolution")
    print("="*80)
    
    # Create two distributions using proper discretization
    dist_X = stats.norm(0, 1)
    dist_Y = stats.norm(1, 0.8)
    
    # Discretize using the refactored API
    x_x, pmf_x_upper, pneg_x_upper, ppos_x_upper = discretize_continuous_to_pmf(
        dist_X, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
    X_upper = DiscreteDist(x=x_x, kind=DistKind.PMF, vals=pmf_x_upper, 
                          p_neg_inf=pneg_x_upper, p_pos_inf=ppos_x_upper, name='N(0,1)-upper')
    
    x_x_lower, pmf_x_lower, pneg_x_lower, ppos_x_lower = discretize_continuous_to_pmf(
        dist_X, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
    X_lower = DiscreteDist(x=x_x_lower, kind=DistKind.PMF, vals=pmf_x_lower, 
                          p_neg_inf=pneg_x_lower, p_pos_inf=ppos_x_lower, name='N(0,1)-lower')
    
    x_y, pmf_y_upper, pneg_y_upper, ppos_y_upper = discretize_continuous_to_pmf(
        dist_Y, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
    Y_upper = DiscreteDist(x=x_y, kind=DistKind.PMF, vals=pmf_y_upper, 
                          p_neg_inf=pneg_y_upper, p_pos_inf=ppos_y_upper, name='N(1,0.8)-upper')
    
    print(f"Input X (upper): {X_upper.name}, {len(X_upper.x):,} bins, mass={X_upper.vals.sum():.10f}")
    print(f"Input Y (upper): {Y_upper.name}, {len(Y_upper.x):,} bins, mass={Y_upper.vals.sum():.10f}")
    print("Computing X + Y...")
    
    start = time.perf_counter()
    Z = convolve_pmf_pmf_to_pmf(X_upper, Y_upper, mode='DOMINATES')
    elapsed = time.perf_counter() - start
    
    total_mass = Z.vals.sum() + Z.p_neg_inf + Z.p_pos_inf
    ops = len(X_upper.x) * len(Y_upper.x)
    throughput = ops / elapsed / 1e6
    
    print(f"✓ Complete in {elapsed:.3f}s ({throughput:.1f} M ops/sec)")
    print(f"  Output: {len(Z.x):,} bins, mass={total_mass:.10f}")
    print(f"  Mass at ±∞: p(-∞)={Z.p_neg_inf:.2e}, p(+∞)={Z.p_pos_inf:.2e}")
    
    # Save plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(X_upper.x, X_upper.vals, 'C0-', linewidth=1)
    axes[0].fill_between(X_upper.x, 0, X_upper.vals, alpha=0.3)
    axes[0].set_title(f'Input X: {X_upper.name}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('PMF')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(Y_upper.x, Y_upper.vals, 'C1-', linewidth=1)
    axes[1].fill_between(Y_upper.x, 0, Y_upper.vals, alpha=0.3)
    axes[1].set_title(f'Input Y: {Y_upper.name}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('PMF')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(Z.x, Z.vals, 'C2-', linewidth=1)
    axes[2].fill_between(Z.x, 0, Z.vals, alpha=0.3)
    axes[2].set_title(f'Output X+Y\n{elapsed:.3f}s, mass={total_mass:.8f}')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('PMF')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/pairwise_convolution_5k.png', dpi=150, bbox_inches='tight')
    print("  → Saved plot: plots/pairwise_convolution_5k.png")

def demo_self_convolution():
    """Demonstrate self-convolution for multiple T values."""
    print("\n" + "="*80)
    print("2. Self-Convolution: X + X + ... + X")
    print("="*80)
    
    # Create base distribution using proper discretization
    dist_base = stats.norm(0, 1)
    x_base, pmf_base_upper, pneg_base_upper, ppos_base_upper = discretize_continuous_to_pmf(
        dist_base, N_BINS, BETA, mode=Mode.DOMINATES, spacing=Spacing.LINEAR)
    base_upper = DiscreteDist(x=x_base, kind=DistKind.PMF, vals=pmf_base_upper,
                             p_neg_inf=pneg_base_upper, p_pos_inf=ppos_base_upper, name='N(0,1)')
    
    x_base_lower, pmf_base_lower, pneg_base_lower, ppos_base_lower = discretize_continuous_to_pmf(
        dist_base, N_BINS, BETA, mode=Mode.IS_DOMINATED, spacing=Spacing.LINEAR)
    base_lower = DiscreteDist(x=x_base_lower, kind=DistKind.PMF, vals=pmf_base_lower,
                             p_neg_inf=pneg_base_lower, p_pos_inf=ppos_base_lower, name='N(0,1)')
    
    print(f"Base: {base_upper.name}, {len(base_upper.x):,} bins, mass={base_upper.vals.sum():.10f}")
    
    T_values = [10, 100, 1000]
    results = {}
    
    for T in T_values:
        print(f"\nComputing sum of {T} copies...")
        
        # Warmup on first run
        if T == 10:
            _ = self_convolve_pmf(base_upper, T, mode='DOMINATES', spacing=Spacing.LINEAR)
            print("  [JIT warmup complete]")
        
        # Compute DOMINATES (upper bound) with automatic grid generation
        start = time.perf_counter()
        Z_upper = self_convolve_pmf(base_upper, T, mode='DOMINATES', spacing=Spacing.LINEAR)
        elapsed = time.perf_counter() - start
        
        # Compute IS_DOMINATED (lower bound) with automatic grid generation
        Z_lower = self_convolve_pmf(base_lower, T, mode='IS_DOMINATED', spacing=Spacing.LINEAR)
        
        total_mass = Z_upper.vals.sum() + Z_upper.p_neg_inf + Z_upper.p_pos_inf
        n_convs = int(np.ceil(np.log2(T))) + bin(T).count('1') - 1
        
        print(f"✓ Complete in {elapsed:.3f}s")
        print(f"  Convolutions: {n_convs} (vs {T-1} naive) - {100*(1-n_convs/(T-1)):.1f}% reduction")
        print(f"  Output: {len(Z_upper.x):,} bins, mass={total_mass:.10f}")
        
        # Compute expectations
        E_upper = np.sum(Z_upper.x * Z_upper.vals)
        E_lower = np.sum(Z_lower.x * Z_lower.vals)
        print(f"  Expectation (DOMINATES): {E_upper:.6f}")
        print(f"  Expectation (IS_DOMINATED): {E_lower:.6f}")
        print(f"  Bias: {E_upper - E_lower:.6f}")
        
        results[T] = {'dist_upper': Z_upper, 'dist_lower': Z_lower, 'time': elapsed, 'n_convs': n_convs}
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Base distribution
    ax = axes[0, 0]
    ax.plot(base_upper.x, base_upper.vals, 'C0-', linewidth=1.5)
    ax.fill_between(base_upper.x, 0, base_upper.vals, alpha=0.3)
    ax.set_title(f'Base: {base_upper.name} ({N_BINS:,} bins)')
    ax.set_xlabel('x')
    ax.set_ylabel('PMF')
    ax.grid(True, alpha=0.3)
    
    # Plot each result
    colors = ['C1', 'C2', 'C3']
    for idx, (T, color) in enumerate(zip(T_values, colors)):
        row = 0 if idx == 0 else 1
        col = 1 if idx == 0 else idx - 1
        ax = axes[row, col]
        
        Z_upper = results[T]['dist_upper']
        Z_lower = results[T]['dist_lower']
        elapsed = results[T]['time']
        n_convs = results[T]['n_convs']
        
        # Plot both upper (DOMINATES) and lower (IS_DOMINATED) bounds
        ax.plot(Z_upper.x, Z_upper.vals, color=color, linewidth=1.5, label='DOMINATES (upper)', alpha=0.8)
        ax.plot(Z_lower.x, Z_lower.vals, color=color, linewidth=1.5, linestyle='--', label='IS_DOMINATED (lower)', alpha=0.8)
        ax.fill_between(Z_upper.x, Z_lower.vals, Z_upper.vals, alpha=0.2, color=color)
        
        # Compute expectations
        E_upper = np.sum(Z_upper.x * Z_upper.vals)
        E_lower = np.sum(Z_lower.x * Z_lower.vals)
        bias = E_upper - E_lower
        
        expected_std = np.sqrt(T)
        ax.set_title(f'Sum of {T} copies ({n_convs} convolutions)\n'
                    f'{elapsed:.3f}s | E_upper={E_upper:.2f}, E_lower={E_lower:.2f}, bias={bias:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('PMF')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Add expected mean and std markers
        ax.axvline(0, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(-expected_std, color='orange', linestyle=':', alpha=0.3)
        ax.axvline(expected_std, color='orange', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/self_convolution_5k.png', dpi=150, bbox_inches='tight')
    print("\n  → Saved plot: plots/self_convolution_5k.png")

def main():
    """Run all demonstrations."""
    print("="*80)
    print("Standard Demonstrations (5,000-bin grids)")
    print("="*80)
    print(f"Grid resolution: {N_BINS:,} bins")
    print(f"Tail probability β: {BETA}")
    print("Using automatic grid generation with support bounds")
    
    Path('plots').mkdir(exist_ok=True)
    
    # Run demonstrations
    demo_pairwise_convolution()
    demo_self_convolution()
    
    # Summary
    print("\n" + "="*80)
    print("✅ All demonstrations complete!")
    print("="*80)
    print(f"Configuration: {N_BINS:,} bins, β={BETA}")
    print("\nPlots saved in: plots/")
    print("  • plots/pairwise_convolution_5k.png")
    print("  • plots/self_convolution_5k.png")
    print("="*80)

if __name__ == "__main__":
    main()
