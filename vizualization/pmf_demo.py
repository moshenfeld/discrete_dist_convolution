"""
Interactive PMF×PMF Convolution Visualization

This script provides an easy way to visualize how the PMF×PMF convolution works.
You can modify the input distributions and see how the convolution behaves.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from implementations.kernels import convolve_pmf_pmf_to_pmf_core

def quick_pmf_convolution_demo():
    """Quick demonstration of PMF×PMF convolution with visual output."""
    
    print("🎯 PMF×PMF Convolution Visualization")
    print("=" * 50)
    
    # Example 1: Simple case - two discrete distributions
    print("\n📊 Example 1: Simple Discrete Distributions")
    print("-" * 40)
    
    # X: Two-point distribution
    xX = np.array([0.0, 2.0])
    pX = np.array([0.6, 0.4])
    pnegX, pposX = 0.0, 0.0
    
    # Y: Two-point distribution  
    xY = np.array([1.0, 3.0])
    pY = np.array([0.7, 0.3])
    pnegY, pposY = 0.0, 0.0
    
    # Output grid
    t = np.linspace(0.0, 6.0, 15)
    
    print(f"X: points at {xX} with masses {pX}")
    print(f"Y: points at {xY} with masses {pY}")
    print(f"Output grid: {len(t)} points from {t[0]:.1f} to {t[-1]:.1f}")
    
    # Show all pairwise sums
    print("\n🔄 Convolution Process:")
    print("All pairwise sums X[i] + Y[j]:")
    for i, x_val in enumerate(xX):
        for j, y_val in enumerate(xY):
            z = x_val + y_val
            mass = pX[i] * pY[j]
            print(f"  {x_val} + {y_val} = {z} (mass: {mass:.3f})")
    
    # Compute convolution for both modes
    pmf_dom, pneg_dom, ppos_dom = convolve_pmf_pmf_to_pmf_core(
        xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, "DOMINATES"
    )
    
    pmf_idom, pneg_idom, ppos_idom = convolve_pmf_pmf_to_pmf_core(
        xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, "IS_DOMINATED"
    )
    
    print(f"\n📈 Results:")
    print(f"DOMINATES mode: {np.sum(pmf_dom):.3f} finite + {pneg_dom:.3f} at -∞ + {ppos_dom:.3f} at +∞")
    print(f"IS_DOMINATED mode: {np.sum(pmf_idom):.3f} finite + {pneg_idom:.3f} at -∞ + {ppos_idom:.3f} at +∞")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('PMF×PMF Convolution Visualization', fontsize=14)
    
    # Plot 1: Input distributions
    ax1 = axes[0, 0]
    ax1.stem(xX, pX, linefmt='b-', markerfmt='bo', basefmt='b-', label='X')
    ax1.stem(xY, pY, linefmt='r-', markerfmt='ro', basefmt='r-', label='Y')
    ax1.set_title('Input Distributions')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability Mass')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Pairwise sums
    ax2 = axes[0, 1]
    pairwise_sums = []
    pairwise_masses = []
    for i, x_val in enumerate(xX):
        for j, y_val in enumerate(xY):
            pairwise_sums.append(x_val + y_val)
            pairwise_masses.append(pX[i] * pY[j])
    
    ax2.stem(pairwise_sums, pairwise_masses, linefmt='g-', markerfmt='go', basefmt='g-')
    ax2.set_title('All Pairwise Sums')
    ax2.set_xlabel('Sum Value')
    ax2.set_ylabel('Joint Probability Mass')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: DOMINATES mode
    ax3 = axes[1, 0]
    ax3.stem(t, pmf_dom, linefmt='purple', markerfmt='o', basefmt='purple')
    ax3.set_title('DOMINATES Mode (Exact Hits Go Up)')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Probability Mass')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: IS_DOMINATED mode
    ax4 = axes[1, 1]
    ax4.stem(t, pmf_idom, linefmt='orange', markerfmt='o', basefmt='orange')
    ax4.set_title('IS_DOMINATED Mode (Exact Hits Go Down)')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Probability Mass')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Example 2: Tie-breaking demonstration
    print("\n🎯 Example 2: Tie-Breaking Demonstration")
    print("-" * 40)
    
    # Simple case where sum exactly hits a grid point
    xX2 = np.array([0.0])
    pX2 = np.array([1.0])
    xY2 = np.array([1.0])
    pY2 = np.array([1.0])
    t2 = np.array([-1.0, 0.0, 1.0, 2.0])
    
    print(f"X: delta at {xX2[0]}")
    print(f"Y: delta at {xY2[0]}")
    print(f"Sum: {xX2[0]} + {xY2[0]} = {xX2[0] + xY2[0]}")
    print(f"Grid: {t2}")
    print(f"Sum {xX2[0] + xY2[0]} exactly equals grid point t[2] = {t2[2]}")
    
    for mode in ["DOMINATES", "IS_DOMINATED"]:
        pmf_out, pneg, ppos = convolve_pmf_pmf_to_pmf_core(
            xX2, pX2, 0.0, 0.0, xY2, pY2, 0.0, 0.0, t2, mode
        )
        
        mass_idx = np.where(pmf_out > 0)[0]
        print(f"\n{mode} mode:")
        print(f"  PMF: {pmf_out}")
        if len(mass_idx) > 0:
            print(f"  Mass goes to grid point {mass_idx[0]} (value {t2[mass_idx[0]]})")
        print(f"  Explanation: In {mode} mode, exact hits go {'up' if mode == 'DOMINATES' else 'down'}")

if __name__ == "__main__":
    quick_pmf_convolution_demo()
