#!/usr/bin/env python3
"""
Interactive PMF×PMF Convolution Explorer

Usage: python pmf_explorer.py

This tool lets you experiment with different input distributions
and see how the PMF×PMF convolution behaves.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from implementations.kernels import convolve_pmf_pmf_to_pmf_core

def explore_pmf_convolution():
    """Interactive exploration of PMF×PMF convolution."""
    
    print("🔬 PMF×PMF Convolution Explorer")
    print("=" * 50)
    print("This tool helps you understand how PMF×PMF convolution works.")
    print("You can modify the input distributions and see the results.\n")
    
    # Interactive input
    print("📝 Define Input Distribution X:")
    x_input = input("Enter X values (comma-separated, e.g., 0,2): ").strip()
    p_input = input("Enter X probabilities (comma-separated, e.g., 0.6,0.4): ").strip()
    
    try:
        xX = np.array([float(x.strip()) for x in x_input.split(',')])
        pX = np.array([float(p.strip()) for p in p_input.split(',')])
        
        # Normalize probabilities
        pX = pX / np.sum(pX)
        
        print(f"\n📝 Define Input Distribution Y:")
        y_input = input("Enter Y values (comma-separated, e.g., 1,3): ").strip()
        py_input = input("Enter Y probabilities (comma-separated, e.g., 0.7,0.3): ").strip()
        
        xY = np.array([float(y.strip()) for y in y_input.split(',')])
        pY = np.array([float(p.strip()) for p in py_input.split(',')])
        
        # Normalize probabilities
        pY = pY / np.sum(pY)
        
    except ValueError as e:
        print(f"❌ Error parsing input: {e}")
        print("Using default values...")
        xX = np.array([0.0, 2.0])
        pX = np.array([0.6, 0.4])
        xY = np.array([1.0, 3.0])
        pY = np.array([0.7, 0.3])
    
    # Ask about infinity masses
    print(f"\n♾️  Infinity Masses (optional):")
    pnegX = float(input("X mass at -∞ (default 0.0): ") or "0.0")
    pposX = float(input("X mass at +∞ (default 0.0): ") or "0.0")
    pnegY = float(input("Y mass at -∞ (default 0.0): ") or "0.0")
    pposY = float(input("Y mass at +∞ (default 0.0): ") or "0.0")
    
    # Create output grid
    x_min = xX[0] + xY[0] - 1
    x_max = xX[-1] + xY[-1] + 1
    grid_size = int(input(f"\n📊 Output grid size (default 20): ") or "20")
    t = np.linspace(x_min, x_max, grid_size)
    
    print(f"\n🎯 Computing PMF×PMF Convolution...")
    print(f"X: {len(xX)} points, total mass = {np.sum(pX) + pnegX + pposX:.3f}")
    print(f"Y: {len(xY)} points, total mass = {np.sum(pY) + pnegY + pposY:.3f}")
    print(f"Output grid: {len(t)} points from {t[0]:.2f} to {t[-1]:.2f}")
    
    # Show all pairwise sums
    print(f"\n🔄 Convolution Process:")
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
    print(f"DOMINATES mode:")
    print(f"  Finite PMF mass: {np.sum(pmf_dom):.3f}")
    print(f"  Mass at -∞: {pneg_dom:.3f}")
    print(f"  Mass at +∞: {ppos_dom:.3f}")
    print(f"  Total: {np.sum(pmf_dom) + pneg_dom + ppos_dom:.3f}")
    
    print(f"\nIS_DOMINATED mode:")
    print(f"  Finite PMF mass: {np.sum(pmf_idom):.3f}")
    print(f"  Mass at -∞: {pneg_idom:.3f}")
    print(f"  Mass at +∞: {ppos_idom:.3f}")
    print(f"  Total: {np.sum(pmf_idom) + pneg_idom + ppos_idom:.3f}")
    
    # Show where masses went
    print(f"\n📍 Mass Distribution:")
    print("DOMINATES mode:")
    for i, mass in enumerate(pmf_dom):
        if mass > 0:
            print(f"  Grid point {i} (value {t[i]:.2f}): mass {mass:.3f}")
    
    print("IS_DOMINATED mode:")
    for i, mass in enumerate(pmf_idom):
        if mass > 0:
            print(f"  Grid point {i} (value {t[i]:.2f}): mass {mass:.3f}")
    
    # Create visualization
    show_plot = input(f"\n📊 Show visualization plot? (y/n): ").strip().lower()
    if show_plot in ['y', 'yes']:
        create_visualization(xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, pmf_dom, pneg_dom, ppos_dom, pmf_idom, pneg_idom, ppos_idom)

def create_visualization(xX, pX, pnegX, pposX, xY, pY, pnegY, pposY, t, pmf_dom, pneg_dom, ppos_dom, pmf_idom, pneg_idom, ppos_idom):
    """Create a visualization of the PMF×PMF convolution."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('PMF×PMF Convolution Results', fontsize=14)
    
    # Plot 1: Input distributions
    ax1 = axes[0, 0]
    ax1.stem(xX, pX, linefmt='b-', markerfmt='bo', basefmt='b-', label='X')
    ax1.stem(xY, pY, linefmt='r-', markerfmt='ro', basefmt='r-', label='Y')
    
    # Add infinity masses if present
    if pnegX > 0 or pposX > 0:
        ax1.axvline(x=xX[0]-1, ymin=0, ymax=max(pnegX, pposX), color='b', linestyle='--', alpha=0.7, label=f'X at ±∞')
    if pnegY > 0 or pposY > 0:
        ax1.axvline(x=xY[0]-1, ymin=0, ymax=max(pnegY, pposY), color='r', linestyle='--', alpha=0.7, label=f'Y at ±∞')
    
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
    ax3.stem(t, pmf_dom, linefmt='purple', markerfmt='o', basefmt='purple', label='Finite PMF')
    if pneg_dom > 0:
        ax3.axvline(x=t[0]-1, ymin=0, ymax=pneg_dom, color='purple', linestyle='--', alpha=0.7, label=f'Mass at -∞')
    if ppos_dom > 0:
        ax3.axvline(x=t[-1]+1, ymin=0, ymax=ppos_dom, color='purple', linestyle='--', alpha=0.7, label=f'Mass at +∞')
    ax3.set_title('DOMINATES Mode (Exact Hits Go Up)')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Probability Mass')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: IS_DOMINATED mode
    ax4 = axes[1, 1]
    ax4.stem(t, pmf_idom, linefmt='orange', markerfmt='o', basefmt='orange', label='Finite PMF')
    if pneg_idom > 0:
        ax4.axvline(x=t[0]-1, ymin=0, ymax=pneg_idom, color='orange', linestyle='--', alpha=0.7, label=f'Mass at -∞')
    if ppos_idom > 0:
        ax4.axvline(x=t[-1]+1, ymin=0, ymax=ppos_idom, color='orange', linestyle='--', alpha=0.7, label=f'Mass at +∞')
    ax4.set_title('IS_DOMINATED Mode (Exact Hits Go Down)')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Probability Mass')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore_pmf_convolution()
