"""
Visualize successful self-convolution results from 1k and 5k bin tests.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from discrete_conv_api import DiscreteDist, self_convolve_pmf

def create_gaussian_pmf(n: int, mu: float = 0.0, sigma: float = 1.0) -> DiscreteDist:
    """Create a discrete Gaussian PMF."""
    lo, hi = mu - 5 * sigma, mu + 5 * sigma
    x = np.linspace(lo, hi, n, dtype=np.float64)
    if n > 1:
        x[1:] += np.linspace(1e-14, 1e-12, n-1)
    z = (x - mu) / sigma
    pdf = np.exp(-0.5 * z * z)
    pdf /= pdf.sum()
    return DiscreteDist(x=x, kind="pmf", vals=pdf, p_neg_inf=0.0, p_pos_inf=0.0, name=f"N(0,1)")

def plot_self_convolutions(n_bins: int, T_values: list, save_path: str):
    """Generate and plot self-convolutions."""
    print(f"\nGenerating visualizations for {n_bins:,} bins...")
    
    # Create base distribution
    base = create_gaussian_pmf(n=n_bins, mu=0, sigma=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Base distribution
    ax = axes[0, 0]
    ax.plot(base.x, base.vals, 'C0-', linewidth=1.5)
    ax.fill_between(base.x, 0, base.vals, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('PMF')
    ax.set_title(f'Base: {base.name} ({n_bins:,} bins)')
    ax.grid(True, alpha=0.3)
    
    # Compute and plot each T
    colors = ['C1', 'C2', 'C3']
    for idx, (T, color) in enumerate(zip(T_values, colors)):
        print(f"  Computing T={T}...")
        
        # Create output grid
        t = np.linspace(T * (-5), T * 5, n_bins, dtype=np.float64)
        Z = self_convolve_pmf(base, T, t=t, mode='DOMINATES')
        
        row = 0 if idx == 0 else 1
        col = 1 if idx == 0 else idx - 1
        ax = axes[row, col]
        
        ax.plot(Z.x, Z.vals, color=color, linewidth=1.2)
        ax.fill_between(Z.x, 0, Z.vals, alpha=0.3, color=color)
        ax.set_xlabel('x')
        ax.set_ylabel('PMF')
        
        total_mass = Z.vals.sum() + Z.p_neg_inf + Z.p_pos_inf
        n_convs = int(np.ceil(np.log2(T))) + bin(T).count('1') - 1
        
        # Expected mean and std for sum of T gaussians
        expected_mean = 0.0  # T * 0
        expected_std = np.sqrt(T)  # sqrt(T * 1^2)
        
        ax.set_title(f'Sum of {T} copies ({n_convs} convolutions)\n'
                    f'Mass={total_mass:.8f} | Expected: N(0,{expected_std:.2f})')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at expected mean
        ax.axvline(0, color='red', linestyle='--', alpha=0.3, label=f'Expected mean=0')
        
        # Add ±1 std markers
        ax.axvline(-expected_std, color='orange', linestyle=':', alpha=0.3)
        ax.axvline(expected_std, color='orange', linestyle=':', alpha=0.3, 
                  label=f'±1 std = ±{expected_std:.2f}')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {save_path}")

def main():
    print("="*80)
    print("Self-Convolution Visualization")
    print("="*80)
    
    Path("plots").mkdir(exist_ok=True)
    
    # Visualize 1k bins (fast)
    plot_self_convolutions(1000, [10, 100, 1000], "plots/self_conv_1k_bins.png")
    
    # Visualize 5k bins (reasonable)
    plot_self_convolutions(5000, [10, 100, 1000], "plots/self_conv_5k_bins.png")
    
    print("\n" + "="*80)
    print("✅ Visualizations complete!")
    print("   plots/self_conv_1k_bins.png")
    print("   plots/self_conv_5k_bins.png")
    print("="*80)

if __name__ == "__main__":
    main()

