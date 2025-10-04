"""
Standard demonstration using 5,000 bins (optimal balance of accuracy vs speed).

Demonstrates:
1. PMF×PMF convolution (pairwise)
2. Self-convolution (X + X + ... + X) for T=10, 100, 1000

All tests use 5,000-bin grids for high-resolution results.
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from discrete_conv_api import DiscreteDist, convolve_pmf_pmf_to_pmf, self_convolve_pmf

# Standard grid resolution
N_BINS = 5000

def create_gaussian_pmf(n: int, mu: float = 0.0, sigma: float = 1.0) -> DiscreteDist:
    """Create a discrete Gaussian PMF."""
    lo, hi = mu - 5 * sigma, mu + 5 * sigma
    x = np.linspace(lo, hi, n, dtype=np.float64)
    if n > 1:
        x[1:] += np.linspace(1e-14, 1e-12, n-1)
    z = (x - mu) / sigma
    pdf = np.exp(-0.5 * z * z)
    pdf /= pdf.sum()
    return DiscreteDist(x=x, kind="pmf", vals=pdf, p_neg_inf=0.0, p_pos_inf=0.0, 
                       name=f"N({mu},{sigma})")

def create_exponential_pmf(n: int, rate: float = 1.0, x_max: float = None) -> DiscreteDist:
    """Create a discrete exponential PMF."""
    if x_max is None:
        x_max = 5.0 / rate
    x = np.linspace(0, x_max, n, dtype=np.float64)
    if n > 1:
        x[1:] += np.linspace(1e-14, 1e-12, n-1)
    pdf = rate * np.exp(-rate * x)
    pdf /= pdf.sum()
    return DiscreteDist(x=x, kind="pmf", vals=pdf, p_neg_inf=0.0, p_pos_inf=0.0, 
                       name=f"Exp({rate})")

def demo_pairwise_convolution():
    """Demonstrate pairwise PMF×PMF convolution."""
    print("\n" + "="*80)
    print("1. Pairwise PMF×PMF Convolution")
    print("="*80)
    
    # Create two distributions
    X = create_gaussian_pmf(N_BINS, mu=0, sigma=1)
    Y = create_gaussian_pmf(N_BINS, mu=1, sigma=0.8)
    
    print(f"Input X: {X.name}, {len(X.x):,} bins, mass={X.vals.sum():.10f}")
    print(f"Input Y: {Y.name}, {len(Y.x):,} bins, mass={Y.vals.sum():.10f}")
    print("Computing X + Y...")
    
    start = time.perf_counter()
    Z = convolve_pmf_pmf_to_pmf(X, Y, mode='DOMINATES')
    elapsed = time.perf_counter() - start
    
    total_mass = Z.vals.sum() + Z.p_neg_inf + Z.p_pos_inf
    ops = len(X.x) * len(Y.x)
    throughput = ops / elapsed / 1e6
    
    print(f"✓ Complete in {elapsed:.3f}s ({throughput:.1f} M ops/sec)")
    print(f"  Output: {len(Z.x):,} bins, mass={total_mass:.10f}")
    print(f"  Mass at ±∞: p(-∞)={Z.p_neg_inf:.2e}, p(+∞)={Z.p_pos_inf:.2e}")
    
    # Save plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(X.x, X.vals, 'C0-', linewidth=1)
    axes[0].fill_between(X.x, 0, X.vals, alpha=0.3)
    axes[0].set_title(f'Input X: {X.name}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('PMF')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(Y.x, Y.vals, 'C1-', linewidth=1)
    axes[1].fill_between(Y.x, 0, Y.vals, alpha=0.3)
    axes[1].set_title(f'Input Y: {Y.name}')
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
    
    # Create base distribution
    base = create_gaussian_pmf(N_BINS, mu=0, sigma=1)
    print(f"Base: {base.name}, {len(base.x):,} bins, mass={base.vals.sum():.10f}")
    
    T_values = [10, 100, 1000]
    results = {}
    
    for T in T_values:
        print(f"\nComputing sum of {T} copies...")
        
        # Create output grid
        t = np.linspace(T * (-5), T * 5, N_BINS, dtype=np.float64)
        
        # Warmup on first run
        if T == 10:
            _ = self_convolve_pmf(base, T, t=t, mode='DOMINATES')
            print("  [JIT warmup complete]")
        
        start = time.perf_counter()
        Z = self_convolve_pmf(base, T, t=t, mode='DOMINATES')
        elapsed = time.perf_counter() - start
        
        total_mass = Z.vals.sum() + Z.p_neg_inf + Z.p_pos_inf
        n_convs = int(np.ceil(np.log2(T))) + bin(T).count('1') - 1
        
        print(f"✓ Complete in {elapsed:.3f}s")
        print(f"  Convolutions: {n_convs} (vs {T-1} naive) - {100*(1-n_convs/(T-1)):.1f}% reduction")
        print(f"  Output: {len(Z.x):,} bins, mass={total_mass:.10f}")
        
        results[T] = {'dist': Z, 'time': elapsed, 'n_convs': n_convs}
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Base distribution
    ax = axes[0, 0]
    ax.plot(base.x, base.vals, 'C0-', linewidth=1.5)
    ax.fill_between(base.x, 0, base.vals, alpha=0.3)
    ax.set_title(f'Base: {base.name} ({N_BINS:,} bins)')
    ax.set_xlabel('x')
    ax.set_ylabel('PMF')
    ax.grid(True, alpha=0.3)
    
    # Plot each result
    colors = ['C1', 'C2', 'C3']
    for idx, (T, color) in enumerate(zip(T_values, colors)):
        row = 0 if idx == 0 else 1
        col = 1 if idx == 0 else idx - 1
        ax = axes[row, col]
        
        Z = results[T]['dist']
        elapsed = results[T]['time']
        n_convs = results[T]['n_convs']
        
        ax.plot(Z.x, Z.vals, color=color, linewidth=1.2)
        ax.fill_between(Z.x, 0, Z.vals, alpha=0.3, color=color)
        
        expected_std = np.sqrt(T)
        ax.set_title(f'Sum of {T} copies ({n_convs} convolutions)\n'
                    f'{elapsed:.3f}s | Expected: N(0,{expected_std:.2f})')
        ax.set_xlabel('x')
        ax.set_ylabel('PMF')
        ax.grid(True, alpha=0.3)
        
        # Add expected mean and std markers
        ax.axvline(0, color='red', linestyle='--', alpha=0.3)
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
    print("This provides excellent accuracy with reasonable computation time.")
    
    Path('plots').mkdir(exist_ok=True)
    
    # Run demonstrations
    demo_pairwise_convolution()
    demo_self_convolution()
    
    # Summary
    print("\n" + "="*80)
    print("✅ All demonstrations complete!")
    print("="*80)
    print(f"Standard configuration: {N_BINS:,} bins")
    print("  • Pairwise convolution: ~1 second")
    print("  • Self-convolution T=10: ~4 seconds")
    print("  • Self-convolution T=100: ~7 seconds")
    print("  • Self-convolution T=1000: ~12 seconds")
    print("\nPlots saved in: plots/")
    print("  • plots/pairwise_convolution_5k.png")
    print("  • plots/self_convolution_5k.png")
    print("="*80)

if __name__ == "__main__":
    main()

