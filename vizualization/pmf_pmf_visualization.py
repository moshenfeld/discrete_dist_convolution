"""
Visualization for PMF×PMF convolution method.

This script demonstrates how the PMF×PMF kernel works by showing:
1. Input distributions X and Y
2. The convolution process (all pairwise sums)
3. Output distribution Z with tie-breaking behavior
4. Comparison between DOMINATES and IS_DOMINATED modes
"""

import numpy as np
import matplotlib.pyplot as plt
from implementations.kernels import convolve_pmf_pmf_to_pmf_core

def create_pmf_distribution(x, p, pneg=0.0, ppos=0.0):
    """Create a simple PMF distribution for visualization."""
    return {
        'x': np.array(x),
        'p': np.array(p),
        'pneg': pneg,
        'ppos': ppos,
        'total_mass': np.sum(p) + pneg + ppos
    }

def visualize_pmf_pmf_convolution(X, Y, output_grid=None, title="PMF×PMF Convolution"):
    """
    Visualize the PMF×PMF convolution process.
    
    Parameters:
    -----------
    X, Y : dict
        PMF distributions with keys 'x', 'p', 'pneg', 'ppos'
    output_grid : array, optional
        Output grid for convolution. If None, creates a reasonable grid.
    title : str
        Title for the plot
    """
    if output_grid is None:
        # Create a reasonable output grid
        x_min = X['x'][0] + Y['x'][0] - 1
        x_max = X['x'][-1] + Y['x'][-1] + 1
        output_grid = np.linspace(x_min, x_max, 20)
    
    # Compute convolution for both modes
    pmf_dom, pneg_dom, ppos_dom = convolve_pmf_pmf_to_pmf_core(
        X['x'], X['p'], X['pneg'], X['ppos'],
        Y['x'], Y['p'], Y['pneg'], Y['ppos'],
        output_grid, "DOMINATES"
    )
    
    pmf_idom, pneg_idom, ppos_idom = convolve_pmf_pmf_to_pmf_core(
        X['x'], X['p'], X['pneg'], X['ppos'],
        Y['x'], Y['p'], Y['pneg'], Y['ppos'],
        output_grid, "IS_DOMINATED"
    )
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Input distributions
    ax1 = axes[0, 0]
    ax1.stem(X['x'], X['p'], linefmt='b-', markerfmt='bo', basefmt='b-', label='X')
    ax1.stem(Y['x'], Y['p'], linefmt='r-', markerfmt='ro', basefmt='r-', label='Y')
    if X['pneg'] > 0:
        ax1.axvline(x=X['x'][0]-1, ymin=0, ymax=X['pneg'], color='b', linestyle='--', alpha=0.7, label=f'X at -∞ ({X["pneg"]:.2f})')
    if X['ppos'] > 0:
        ax1.axvline(x=X['x'][-1]+1, ymin=0, ymax=X['ppos'], color='b', linestyle='--', alpha=0.7, label=f'X at +∞ ({X["ppos"]:.2f})')
    if Y['pneg'] > 0:
        ax1.axvline(x=Y['x'][0]-1, ymin=0, ymax=Y['pneg'], color='r', linestyle='--', alpha=0.7, label=f'Y at -∞ ({Y["pneg"]:.2f})')
    if Y['ppos'] > 0:
        ax1.axvline(x=Y['x'][-1]+1, ymin=0, ymax=Y['ppos'], color='r', linestyle='--', alpha=0.7, label=f'Y at +∞ ({Y["ppos"]:.2f})')
    ax1.set_title('Input Distributions X and Y')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability Mass')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All pairwise sums (convolution process)
    ax2 = axes[0, 1]
    pairwise_sums = []
    pairwise_masses = []
    for i, x_val in enumerate(X['x']):
        for j, y_val in enumerate(Y['x']):
            pairwise_sums.append(x_val + y_val)
            pairwise_masses.append(X['p'][i] * Y['p'][j])
    
    # Sort for better visualization
    sorted_indices = np.argsort(pairwise_sums)
    pairwise_sums = np.array(pairwise_sums)[sorted_indices]
    pairwise_masses = np.array(pairwise_masses)[sorted_indices]
    
    ax2.stem(pairwise_sums, pairwise_masses, linefmt='g-', markerfmt='go', basefmt='g-')
    ax2.set_title('All Pairwise Sums (X[i] + Y[j])')
    ax2.set_xlabel('Sum Value')
    ax2.set_ylabel('Joint Probability Mass')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: DOMINATES mode result
    ax3 = axes[1, 0]
    ax3.stem(output_grid, pmf_dom, linefmt='purple', markerfmt='o', basefmt='purple', label='Finite PMF')
    if pneg_dom > 0:
        ax3.axvline(x=output_grid[0]-1, ymin=0, ymax=pneg_dom, color='purple', linestyle='--', alpha=0.7, label=f'Mass at -∞ ({pneg_dom:.2f})')
    if ppos_dom > 0:
        ax3.axvline(x=output_grid[-1]+1, ymin=0, ymax=ppos_dom, color='purple', linestyle='--', alpha=0.7, label=f'Mass at +∞ ({ppos_dom:.2f})')
    ax3.set_title('DOMINATES Mode (Exact Hits Go Up)')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Probability Mass')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: IS_DOMINATED mode result
    ax4 = axes[1, 1]
    ax4.stem(output_grid, pmf_idom, linefmt='orange', markerfmt='o', basefmt='orange', label='Finite PMF')
    if pneg_idom > 0:
        ax4.axvline(x=output_grid[0]-1, ymin=0, ymax=pneg_idom, color='orange', linestyle='--', alpha=0.7, label=f'Mass at -∞ ({pneg_idom:.2f})')
    if ppos_idom > 0:
        ax4.axvline(x=output_grid[-1]+1, ymin=0, ymax=ppos_idom, color='orange', linestyle='--', alpha=0.7, label=f'Mass at +∞ ({ppos_idom:.2f})')
    ax4.set_title('IS_DOMINATED Mode (Exact Hits Go Down)')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Probability Mass')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print summary information
    print(f"\n=== {title} ===")
    print(f"Input X: {len(X['x'])} points, total mass = {X['total_mass']:.3f}")
    print(f"Input Y: {len(Y['x'])} points, total mass = {Y['total_mass']:.3f}")
    print(f"Output grid: {len(output_grid)} points from {output_grid[0]:.2f} to {output_grid[-1]:.2f}")
    print(f"\nDOMINATES mode:")
    print(f"  Finite PMF mass: {np.sum(pmf_dom):.3f}")
    print(f"  Mass at -∞: {pneg_dom:.3f}")
    print(f"  Mass at +∞: {ppos_dom:.3f}")
    print(f"  Total: {np.sum(pmf_dom) + pneg_dom + ppos_dom:.3f}")
    print(f"\nIS_DOMINATED mode:")
    print(f"  Finite PMF mass: {np.sum(pmf_idom):.3f}")
    print(f"  Mass at -∞: {pneg_idom:.3f}")
    print(f"  Mass at +∞: {ppos_idom:.3f}")
    print(f"  Total: {np.sum(pmf_idom) + pneg_idom + ppos_idom:.3f}")
    
    return fig, axes

def demonstrate_tie_breaking():
    """Demonstrate tie-breaking behavior with a simple example."""
    print("=== Tie-Breaking Demonstration ===")
    
    # Simple example: X = δ(0), Y = δ(1), output grid includes exact sum
    X = create_pmf_distribution([0.0], [1.0])
    Y = create_pmf_distribution([1.0], [1.0])
    output_grid = np.array([-1.0, 0.0, 1.0, 2.0])
    
    print(f"X: delta at {X['x'][0]}")
    print(f"Y: delta at {Y['x'][0]}")
    print(f"Sum: {X['x'][0]} + {Y['x'][0]} = {X['x'][0] + Y['x'][0]}")
    print(f"Output grid: {output_grid}")
    print()
    
    # Test both modes
    for mode in ["DOMINATES", "IS_DOMINATED"]:
        pmf_out, pneg, ppos = convolve_pmf_pmf_to_pmf_core(
            X['x'], X['p'], X['pneg'], X['ppos'],
            Y['x'], Y['p'], Y['pneg'], Y['ppos'],
            output_grid, mode
        )
        
        print(f"{mode} mode:")
        print(f"  PMF: {pmf_out}")
        print(f"  Mass at -∞: {pneg}")
        print(f"  Mass at +∞: {ppos}")
        
        # Find where the mass went
        mass_idx = np.where(pmf_out > 0)[0]
        if len(mass_idx) > 0:
            print(f"  Mass goes to grid point {mass_idx[0]} (value {output_grid[mass_idx[0]]})")
        print()

if __name__ == "__main__":
    # Example 1: Simple case with no infinity masses
    print("Creating visualization for PMF×PMF convolution...")
    
    X1 = create_pmf_distribution([0.0, 2.0], [0.6, 0.4])
    Y1 = create_pmf_distribution([1.0, 3.0], [0.7, 0.3])
    
    fig1, axes1 = visualize_pmf_pmf_convolution(X1, Y1, title="Example 1: Simple PMF×PMF")
    
    # Example 2: Case with infinity masses
    X2 = create_pmf_distribution([0.0], [0.5], pneg=0.3, ppos=0.2)
    Y2 = create_pmf_distribution([0.0], [0.4], pneg=0.1, ppos=0.5)
    
    fig2, axes2 = visualize_pmf_pmf_convolution(X2, Y2, title="Example 2: PMF×PMF with Infinity Masses")
    
    # Demonstrate tie-breaking
    demonstrate_tie_breaking()
    
    plt.show()
