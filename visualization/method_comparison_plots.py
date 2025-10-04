"""
Method comparison plotting functions.

This module contains plotting functions for comparing different convolution methods
(Gaussian and LogNormal distributions).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from discrete_conv_api import DiscreteDist


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


def plot_gaussian_results(results: Dict[int, Dict[str, Any]], T_values: List[int], 
                         save_plot_func=None) -> None:
    """
    Generate plots for Gaussian comparison.
    
    Parameters:
    -----------
    results : Dict[int, Dict[str, Any]]
        Results dictionary with T values as keys
    T_values : List[int]
        List of T values to plot
    save_plot_func : callable, optional
        Function to save plots (e.g., from data_utils)
    """
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
        ax_cdf.plot(x_upper, cdf_upper, 'C0-', linewidth=2, 
                   label='Main Upper', alpha=0.6)
        ax_cdf.plot(x_lower, cdf_lower, 'C1--', linewidth=2,
                   label='Main Lower', alpha=0.6)
        ax_cdf.plot(x_fft, cdf_fft, 'C2:', linewidth=2,
                   label='FFT', alpha=0.6)
        ax_cdf.plot(x_mc, cdf_mc, 'C3-.', linewidth=2,
                   label='Monte Carlo', alpha=0.6)
        ax_cdf.plot(x_analytic, cdf_analytic, 'C4-', linewidth=2,
                   label='Analytic (exact)', alpha=0.6)
        
        # Fill between upper and lower bounds
        ax_cdf.fill_between(x_upper, cdf_lower, cdf_upper, 
                           alpha=0.1, color='C0')
        
        # Plot CCDFs
        ax_ccdf.plot(x_upper, ccdf_upper, 'C0-', linewidth=2, 
                    label='Main Upper', alpha=0.6)
        ax_ccdf.plot(x_lower, ccdf_lower, 'C1--', linewidth=2,
                    label='Main Lower', alpha=0.6)
        ax_ccdf.plot(x_fft, ccdf_fft, 'C2:', linewidth=2,
                    label='FFT', alpha=0.6)
        ax_ccdf.plot(x_mc, ccdf_mc, 'C3-.', linewidth=2,
                    label='Monte Carlo', alpha=0.6)
        ax_ccdf.plot(x_analytic, ccdf_analytic, 'C4-', linewidth=2,
                    label='Analytic (exact)', alpha=0.6)
        
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
        
        ax_ccdf.set_xlabel('x')
        ax_ccdf.set_ylabel('CCDF')
        ax_ccdf.set_title(f'T={T} - CCDF\nBias={r["bias_main"]:.3f}')
        ax_ccdf.legend(fontsize=8, loc='best')
        ax_ccdf.grid(True, alpha=0.3)
        ax_ccdf.set_yscale('log')
    
    plt.tight_layout()
    
    if save_plot_func:
        filename = save_plot_func(plt.gcf(), "method_comparison_gaussian")
        print(f"Plot saved: {filename}")
    
    plt.close()


def plot_lognormal_results(results: Dict[int, Dict[str, Any]], T_values: List[int],
                          save_plot_func=None) -> None:
    """
    Generate plots for LogNormal comparison.
    
    Parameters:
    -----------
    results : Dict[int, Dict[str, Any]]
        Results dictionary with T values as keys
    T_values : List[int]
        List of T values to plot
    save_plot_func : callable, optional
        Function to save plots (e.g., from data_utils)
    """
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
        ax_cdf.plot(x_upper, cdf_upper, 'C0-', linewidth=2, 
                   label='Main Upper', alpha=0.6)
        ax_cdf.plot(x_lower, cdf_lower, 'C1--', linewidth=2,
                   label='Main Lower', alpha=0.6)
        ax_cdf.plot(x_mc, cdf_mc, 'C3-.', linewidth=2,
                   label='Monte Carlo', alpha=0.6)
        ax_cdf.plot(x_analytic, cdf_analytic, 'C4-', linewidth=2,
                   label='Analytic (approx)', alpha=0.6)
        
        # Fill between upper and lower bounds
        ax_cdf.fill_between(x_upper, cdf_lower, cdf_upper, 
                           alpha=0.1, color='C0')
        
        # Plot CCDFs
        ax_ccdf.plot(x_upper, ccdf_upper, 'C0-', linewidth=2, 
                    label='Main Upper', alpha=0.6)
        ax_ccdf.plot(x_lower, ccdf_lower, 'C1--', linewidth=2,
                    label='Main Lower', alpha=0.6)
        ax_ccdf.plot(x_mc, ccdf_mc, 'C3-.', linewidth=2,
                    label='Monte Carlo', alpha=0.6)
        ax_ccdf.plot(x_analytic, ccdf_analytic, 'C4-', linewidth=2,
                    label='Analytic (approx)', alpha=0.6)
        
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
        
        ax_ccdf.set_xlabel('x')
        ax_ccdf.set_ylabel('CCDF')
        ax_ccdf.set_title(f'T={T} - CCDF\nBias={r["bias_main"]:.3f}')
        ax_ccdf.legend(fontsize=8, loc='best')
        ax_ccdf.grid(True, alpha=0.3)
        ax_ccdf.set_yscale('log')
    
    plt.tight_layout()
    
    if save_plot_func:
        filename = save_plot_func(plt.gcf(), "method_comparison_lognormal")
        print(f"Plot saved: {filename}")
    
    plt.close()
