"""
Simple experiment module using the simplified implementation.

This module runs the same experiments as experiments.py but uses the 
simplified_implementation.py API instead of the full implementation.py API.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d

from simplified_implementation import (
    DiscreteDist, BoundType, SpacingType, GridStrategy,
    self_convolve_discrete_distributions, discretize_continuous_to_pmf,
    convolve_discrete_distributions
)

from alternative_methods import (
    fft_self_convolve_continuous,
    clean_fft_self_convolve_continuous,
    monte_carlo_self_convolve_pmf,
    analytic_convolve_gaussian,
    analytic_convolve_lognormal
)

# Import plotting functions from the original experiments
from experiments import (
    plot_grid_strategy_comparison
)

# =============================================================================
# EXPERIMENTAL CONFIGURATION
# =============================================================================

class SimpleExperimentConfig:
    """Configuration for simple experiments using simplified implementation - matching original parameters."""
    
    # Method comparison experiments - matching original
    T_VALUES = [100, 1000, 10000]
    
    # Gaussian-specific parameters - matching original
    GAUSSIAN_N_BINS = 20000
    GAUSSIAN_BETA = 1e-12
    GAUSSIAN_FFT_SIZE = 5000
    
    # Lognormal-specific parameters - matching original
    LOGNORMAL_N_BINS = 20000
    LOGNORMAL_BETA = 1e-15
    LOGNORMAL_FFT_SIZE = 5000
    
    # Monte Carlo parameters - matching original
    MC_SAMPLES = 1_000_000
    MC_BLOCK_SIZE = 100_000
    
    # Grid strategy comparison - matching original
    GRID_T_VALUES = [10, 50, 100, 500, 1000]
    GRID_N_BINS = 1000
    GRID_BETA = 1e-12
    
    # Diagnostics - matching original
    DIAGNOSTICS_T = 1000
    DIAGNOSTICS_N_BINS = 20000
    DIAGNOSTICS_BETA_DISCRETIZE = 1e-15
    DIAGNOSTICS_BETA_CONVOLVE = 1e-12
    
    # Experiment control flags
    RUN_DIAGNOSTICS = True
    RUN_GRID_COMPARISON = True
    RUN_METHOD_COMPARISON = True
    
    # Method control flags
    RUN_MONTE_CARLO = False
    RUN_FFT = True
    RUN_ANALYTIC = True
    
    # Grid strategy configuration
    # Options: GridStrategy.FIXED_NUM_POINTS (maintains constant number of bins)
    #          GridStrategy.FIXED_WIDTH (maintains constant bin width)
    GRID_STRATEGY = GridStrategy.FIXED_WIDTH
    
    # Monte Carlo settings
    USE_MC_IMPORTANCE = False

# =============================================================================
# DATA UTILITIES
# =============================================================================

def get_timestamp() -> str:
    """Get current timestamp in format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_results(results: Dict[str, Any], experiment_type: str, filename_prefix: str = "results") -> str:
    """Save results to JSON file with timestamp."""
    exp_dir = Path("data") / experiment_type
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp()
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = exp_dir / filename
    
    def serialize_value(obj):
        """Recursively serialize objects for JSON."""
        if hasattr(obj, '__class__') and 'DiscreteDist' in obj.__class__.__name__:
            return {
                'x': obj.x.tolist(),
                'PMF': obj.PMF.tolist() if hasattr(obj, 'PMF') else obj.vals.tolist(),
                'p_neg_inf': float(obj.p_neg_inf),
                'p_pos_inf': float(obj.p_pos_inf)
            }
        elif isinstance(obj, dict):
            return {k: serialize_value(v) for k, v in obj.items()}
        elif isinstance(obj, (np.ndarray, np.floating, np.integer)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif isinstance(obj, (list, tuple)):
            return [serialize_value(item) for item in obj]
        else:
            return obj
    
    serializable_results = serialize_value(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"  Saved results to: {filepath}")
    return str(filepath)

def save_plot(fig, filename_prefix: str) -> str:
    """Save a matplotlib figure."""
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{filename_prefix}.png"
    filepath = plots_dir / filename
    
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {filepath}")
    
    return str(filepath)

# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def pmf_to_cdf_arrays(dist: DiscreteDist) -> tuple:
    """Convert PMF to CDF arrays."""
    # Handle both simplified (PMF) and full implementation (vals) DiscreteDist objects
    pmf_vals = dist.PMF if hasattr(dist, 'PMF') else dist.vals
    cdf_vals = np.cumsum(pmf_vals) + dist.p_neg_inf
    return dist.x, cdf_vals

def pmf_to_ccdf_arrays(dist: DiscreteDist) -> tuple:
    """Convert PMF to CCDF arrays."""
    _, cdf_vals = pmf_to_cdf_arrays(dist)
    ccdf_vals = 1.0 - cdf_vals
    return dist.x, ccdf_vals

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_gaussian_results(results: Dict[int, Dict[str, Any]], T_values: List[int], 
                         save_plot_func=None) -> None:
    """Generate plots for Gaussian comparison using simplified implementation."""
    if results is None or len(results) == 0:
        print("  No Gaussian results to plot")
        return
    
    fig, axes = plt.subplots(2, len(T_values), figsize=(6*len(T_values), 10))
    if len(T_values) == 1:
        axes = axes.reshape(2, 1)
    
    strategy_name = "Fixed Points" if SimpleExperimentConfig.GRID_STRATEGY == GridStrategy.FIXED_NUM_POINTS else "Constant Width"
    fig.suptitle(f'Gaussian N(0,1) - Method Comparison (Simplified Implementation, {strategy_name})', fontsize=14, fontweight='bold')
    
    for idx, T in enumerate(T_values):
        ax_cdf = axes[0, idx]
        ax_ccdf = axes[1, idx]
        
        if T not in results:
            ax_cdf.text(0.5, 0.5, f'T={T}\nNo data', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nNo data', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        r = results[T]
        
        # Check if distribution has enough mass
        pmf_vals = r['dist_upper'].PMF if hasattr(r['dist_upper'], 'PMF') else r['dist_upper'].vals
        if pmf_vals.sum() < 0.01:
            ax_cdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        x_upper, cdf_upper = pmf_to_cdf_arrays(r['dist_upper'])
        x_lower, cdf_lower = pmf_to_cdf_arrays(r['dist_lower'])
        
        # Handle optional methods that might be None
        if r['dist_fft'] is not None:
            x_fft, cdf_fft = pmf_to_cdf_arrays(r['dist_fft'])
        else:
            x_fft, cdf_fft = None, None
            
        if r['dist_mc'] is not None:
            x_mc, cdf_mc = pmf_to_cdf_arrays(r['dist_mc'])
        else:
            x_mc, cdf_mc = None, None
            
        if r['dist_analytic'] is not None:
            x_analytic, cdf_analytic = pmf_to_cdf_arrays(r['dist_analytic'])
        else:
            x_analytic, cdf_analytic = None, None
        
        _, ccdf_upper = pmf_to_ccdf_arrays(r['dist_upper'])
        _, ccdf_lower = pmf_to_ccdf_arrays(r['dist_lower'])
        
        # Handle optional methods for CCDF
        if r['dist_fft'] is not None:
            _, ccdf_fft = pmf_to_ccdf_arrays(r['dist_fft'])
        else:
            ccdf_fft = None
            
        if r['dist_mc'] is not None:
            _, ccdf_mc = pmf_to_ccdf_arrays(r['dist_mc'])
        else:
            ccdf_mc = None
            
        if r['dist_analytic'] is not None:
            _, ccdf_analytic = pmf_to_ccdf_arrays(r['dist_analytic'])
        else:
            ccdf_analytic = None
        
        x_min = min(x_upper[0], x_lower[0])
        x_max = max(x_upper[-1], x_lower[-1])
        
        # Create a common grid for plotting to avoid size mismatches
        x_common = np.linspace(x_min, x_max, 1000)
        
        # Interpolate all CDFs to the common grid
        cdf_upper_interp = interp1d(x_upper, cdf_upper, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common)
        cdf_lower_interp = interp1d(x_lower, cdf_lower, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common)
        
        # Handle optional methods for interpolation
        if x_fft is not None and cdf_fft is not None:
            cdf_fft_interp = interp1d(x_fft, cdf_fft, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common)
        else:
            cdf_fft_interp = None
            
        if x_mc is not None and cdf_mc is not None:
            cdf_mc_interp = interp1d(x_mc, cdf_mc, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common)
        else:
            cdf_mc_interp = None
            
        if x_analytic is not None and cdf_analytic is not None:
            cdf_analytic_interp = interp1d(x_analytic, cdf_analytic, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common)
        else:
            cdf_analytic_interp = None
        
        ccdf_upper_interp = 1.0 - cdf_upper_interp
        ccdf_lower_interp = 1.0 - cdf_lower_interp
        
        # Handle optional methods for CCDF
        if cdf_fft_interp is not None:
            ccdf_fft_interp = 1.0 - cdf_fft_interp
        else:
            ccdf_fft_interp = None
            
        if cdf_mc_interp is not None:
            ccdf_mc_interp = 1.0 - cdf_mc_interp
        else:
            ccdf_mc_interp = None
            
        if cdf_analytic_interp is not None:
            ccdf_analytic_interp = 1.0 - cdf_analytic_interp
        else:
            ccdf_analytic_interp = None
        
        # Plot CDFs
        ax_cdf.plot(x_common, cdf_upper_interp, 'C0-', linewidth=2, label='Main Upper', alpha=0.6)
        ax_cdf.plot(x_common, cdf_lower_interp, 'C1--', linewidth=2, label='Main Lower', alpha=0.6)
        
        if cdf_fft_interp is not None:
            ax_cdf.plot(x_common, cdf_fft_interp, 'C2:', linewidth=2, label='FFT', alpha=0.6)
        if cdf_mc_interp is not None:
            ax_cdf.plot(x_common, cdf_mc_interp, 'C3-.', linewidth=2, label='Monte Carlo', alpha=0.6)
        if cdf_analytic_interp is not None:
            ax_cdf.plot(x_common, cdf_analytic_interp, 'C4-', linewidth=2, label='Analytic (exact)', alpha=0.6)
            
        ax_cdf.fill_between(x_common, cdf_lower_interp, cdf_upper_interp, alpha=0.1, color='C0')
        
        # Plot CCDFs
        ax_ccdf.plot(x_common, ccdf_upper_interp, 'C0-', linewidth=2, label='Main Upper', alpha=0.6)
        ax_ccdf.plot(x_common, ccdf_lower_interp, 'C1--', linewidth=2, label='Main Lower', alpha=0.6)
        
        if ccdf_fft_interp is not None:
            ax_ccdf.plot(x_common, ccdf_fft_interp, 'C2:', linewidth=2, label='FFT', alpha=0.6)
        if ccdf_mc_interp is not None:
            ax_ccdf.plot(x_common, ccdf_mc_interp, 'C3-.', linewidth=2, label='Monte Carlo', alpha=0.6)
        if ccdf_analytic_interp is not None:
            ax_ccdf.plot(x_common, ccdf_analytic_interp, 'C4-', linewidth=2, label='Analytic (exact)', alpha=0.6)
        ax_ccdf.fill_between(x_common, ccdf_lower_interp, ccdf_upper_interp, alpha=0.1, color='C0')
        
        ax_cdf.set_xlim(x_min, x_max)
        ax_ccdf.set_xlim(x_min, x_max)
        
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
        strategy_suffix = "fixed_points" if SimpleExperimentConfig.GRID_STRATEGY == GridStrategy.FIXED_NUM_POINTS else "constant_width"
        filename = save_plot_func(plt.gcf(), f"method_comparison_gaussian_{strategy_suffix}")
        print(f"Plot saved: {filename}")
    
    plt.close()

def plot_lognormal_results(results: Dict[int, Dict[str, Any]], T_values: List[int],
                          save_plot_func=None) -> None:
    """Generate plots for LogNormal comparison using simplified implementation."""
    if results is None or len(results) == 0:
        print("  No LogNormal results to plot")
        return
    
    fig, axes = plt.subplots(2, len(T_values), figsize=(6*len(T_values), 10))
    if len(T_values) == 1:
        axes = axes.reshape(2, 1)
    
    strategy_name = "Fixed Points" if SimpleExperimentConfig.GRID_STRATEGY == GridStrategy.FIXED_NUM_POINTS else "Constant Width"
    fig.suptitle(f'LogNormal(μ=0, σ=1) - Method Comparison (Simplified Implementation, {strategy_name})', fontsize=14, fontweight='bold')
    
    for idx, T in enumerate(T_values):
        ax_cdf = axes[0, idx]
        ax_ccdf = axes[1, idx]
        
        if T not in results:
            ax_cdf.text(0.5, 0.5, f'T={T}\nNo data', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nNo data', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        r = results[T]
        
        # Check if distribution has enough mass
        pmf_vals = r['dist_upper'].PMF if hasattr(r['dist_upper'], 'PMF') else r['dist_upper'].vals
        if pmf_vals.sum() < 0.01:
            ax_cdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                       ha='center', va='center', transform=ax_cdf.transAxes)
            ax_cdf.set_title(f'T={T}')
            ax_ccdf.text(0.5, 0.5, f'T={T}\nMass outside grid', 
                        ha='center', va='center', transform=ax_ccdf.transAxes)
            continue
        
        x_upper, cdf_upper = pmf_to_cdf_arrays(r['dist_upper'])
        x_lower, cdf_lower = pmf_to_cdf_arrays(r['dist_lower'])
        
        # Handle optional methods that might be None
        if r['dist_fft'] is not None:
            x_fft, cdf_fft = pmf_to_cdf_arrays(r['dist_fft'])
        else:
            x_fft, cdf_fft = None, None
            
        if r['dist_mc'] is not None:
            x_mc, cdf_mc = pmf_to_cdf_arrays(r['dist_mc'])
        else:
            x_mc, cdf_mc = None, None
            
        if r['dist_analytic'] is not None:
            x_analytic, cdf_analytic = pmf_to_cdf_arrays(r['dist_analytic'])
        else:
            x_analytic, cdf_analytic = None, None
        
        _, ccdf_upper = pmf_to_ccdf_arrays(r['dist_upper'])
        _, ccdf_lower = pmf_to_ccdf_arrays(r['dist_lower'])
        
        # Handle optional methods for CCDF
        if r['dist_fft'] is not None:
            _, ccdf_fft = pmf_to_ccdf_arrays(r['dist_fft'])
        else:
            ccdf_fft = None
            
        if r['dist_mc'] is not None:
            _, ccdf_mc = pmf_to_ccdf_arrays(r['dist_mc'])
        else:
            ccdf_mc = None
            
        if r['dist_analytic'] is not None:
            _, ccdf_analytic = pmf_to_ccdf_arrays(r['dist_analytic'])
        else:
            ccdf_analytic = None
        
        # Create log-space common grid
        x_min = min(x_upper[0], x_lower[0])
        x_max = max(x_upper[-1], x_lower[-1])
        log_x_min = np.log(max(x_min, 1e-10))  # Ensure positive values for log
        log_x_max = np.log(x_max)
        log_x_common = np.linspace(log_x_min, log_x_max, 1000)
        x_common_log = np.exp(log_x_common)
        
        # Interpolate all CDFs to the log-space common grid
        cdf_upper_interp = interp1d(x_upper, cdf_upper, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common_log)
        cdf_lower_interp = interp1d(x_lower, cdf_lower, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common_log)
        
        # Handle optional methods for interpolation
        if x_fft is not None and cdf_fft is not None:
            cdf_fft_interp = interp1d(x_fft, cdf_fft, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common_log)
        else:
            cdf_fft_interp = None
            
        if x_mc is not None and cdf_mc is not None:
            cdf_mc_interp = interp1d(x_mc, cdf_mc, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common_log)
        else:
            cdf_mc_interp = None
            
        if x_analytic is not None and cdf_analytic is not None:
            cdf_analytic_interp = interp1d(x_analytic, cdf_analytic, kind='linear', bounds_error=False, fill_value=(0, 1))(x_common_log)
        else:
            cdf_analytic_interp = None
        
        ccdf_upper_interp = 1.0 - cdf_upper_interp
        ccdf_lower_interp = 1.0 - cdf_lower_interp
        
        # Handle optional methods for CCDF
        if cdf_fft_interp is not None:
            ccdf_fft_interp = 1.0 - cdf_fft_interp
        else:
            ccdf_fft_interp = None
            
        if cdf_mc_interp is not None:
            ccdf_mc_interp = 1.0 - cdf_mc_interp
        else:
            ccdf_mc_interp = None
            
        if cdf_analytic_interp is not None:
            ccdf_analytic_interp = 1.0 - cdf_analytic_interp
        else:
            ccdf_analytic_interp = None
        
        ax_cdf.plot(log_x_common, cdf_upper_interp, 'C0-', linewidth=2, label='Main Upper', alpha=0.6)
        ax_cdf.plot(log_x_common, cdf_lower_interp, 'C1--', linewidth=2, label='Main Lower', alpha=0.6)
        
        if cdf_fft_interp is not None:
            ax_cdf.plot(log_x_common, cdf_fft_interp, 'C2:', linewidth=2, label='FFT', alpha=0.6)
        if cdf_mc_interp is not None:
            ax_cdf.plot(log_x_common, cdf_mc_interp, 'C3-.', linewidth=2, label='Monte Carlo', alpha=0.6)
        if cdf_analytic_interp is not None:
            ax_cdf.plot(log_x_common, cdf_analytic_interp, 'C4-', linewidth=2, label='Analytic (approx)', alpha=0.6)
            
        ax_cdf.fill_between(log_x_common, cdf_lower_interp, cdf_upper_interp, alpha=0.1, color='C0')
        
        ax_ccdf.plot(log_x_common, ccdf_upper_interp, 'C0-', linewidth=2, label='Main Upper', alpha=0.6)
        ax_ccdf.plot(log_x_common, ccdf_lower_interp, 'C1--', linewidth=2, label='Main Lower', alpha=0.6)
        
        if ccdf_fft_interp is not None:
            ax_ccdf.plot(log_x_common, ccdf_fft_interp, 'C2:', linewidth=2, label='FFT', alpha=0.6)
        if ccdf_mc_interp is not None:
            ax_ccdf.plot(log_x_common, ccdf_mc_interp, 'C3-.', linewidth=2, label='Monte Carlo', alpha=0.6)
        if ccdf_analytic_interp is not None:
            ax_ccdf.plot(log_x_common, ccdf_analytic_interp, 'C4-', linewidth=2, label='Analytic (approx)', alpha=0.6)
        ax_ccdf.fill_between(log_x_common, ccdf_lower_interp, ccdf_upper_interp, alpha=0.1, color='C0')
        
        E_theoretical = r['E_theoretical']
        log_E_theoretical = np.log(E_theoretical) if E_theoretical > 0 else log_x_min
        
        ax_cdf.set_xlim(log_x_min, log_x_max)
        ax_ccdf.set_xlim(log_x_min, log_x_max)
        
        ax_cdf.set_xlabel('log(x)')
        ax_cdf.set_ylabel('CDF')
        ax_cdf.set_title(f'T={T} - CDF (log space)\nBias={r["bias_main"]:.3f}')
        ax_cdf.legend(fontsize=8, loc='best')
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.set_yscale('log')
        
        ax_ccdf.set_xlabel('log(x)')
        ax_ccdf.set_ylabel('CCDF')
        ax_ccdf.set_title(f'T={T} - CCDF (log space)\nBias={r["bias_main"]:.3f}')
        ax_ccdf.legend(fontsize=8, loc='best')
        ax_ccdf.grid(True, alpha=0.3)
        ax_ccdf.set_yscale('log')
    
    plt.tight_layout()
    
    if save_plot_func:
        strategy_suffix = "fixed_points" if SimpleExperimentConfig.GRID_STRATEGY == GridStrategy.FIXED_NUM_POINTS else "constant_width"
        filename = save_plot_func(plt.gcf(), f"method_comparison_lognormal_{strategy_suffix}")
        print(f"Plot saved: {filename}")
    
    plt.close()

# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def diagnose_error_sources(base_dist: DiscreteDist, 
                          T: Optional[int] = None,
                          bound_type: BoundType = BoundType.DOMINATES,
                          spacing_type: SpacingType = SpacingType.LINEAR, 
                          beta: Optional[float] = None):
    """
    Diagnose sources of numerical error accumulation using simplified implementation.
    
    Tests:
    1. Single convolution error (isolates kernel precision)
    2. T convolutions error (total accumulation)
    """
    # Use config defaults if not specified
    if T is None:
        T = SimpleExperimentConfig.DIAGNOSTICS_T
    if beta is None:
        beta = SimpleExperimentConfig.DIAGNOSTICS_BETA_CONVOLVE
    
    print("\n" + "="*80)
    print(f"ERROR DIAGNOSTICS FOR T={T} (Simplified Implementation)")
    print("="*80)
    
    # Test 1: Single convolution
    print("\n1. Single Convolution:")
    try:
        result1 = self_convolve_discrete_distributions(
            base_dist, T=2, beta=beta, bound_type=bound_type, 
            spacing_type=spacing_type, grid_strategy=GridStrategy.FIXED_WIDTH
        )
        error1 = abs(result1.PMF.sum() + result1.p_neg_inf + result1.p_pos_inf - 1.0)
        print(f"   Error: {error1:.2e}")
    except ValueError as e:
        error1 = float('inf')
        print(f"   Error: FAILED - {e}")
    
    # Test 2: T convolutions
    print(f"\n2. T={T} Convolutions:")
    print(f"   Computing T={T} convolutions (this may take a while for large T)...")
    try:
        resultT = self_convolve_discrete_distributions(
            base_dist, T=T, beta=beta, bound_type=bound_type, 
            spacing_type=spacing_type, grid_strategy=GridStrategy.FIXED_NUM_POINTS
        )
        errorT = abs(resultT.PMF.sum() + resultT.p_neg_inf + resultT.p_pos_inf - 1.0)
        print(f"   Error: {errorT:.2e}")
        if error1 < float('inf') and error1 > 0:
            print(f"   Growth rate: {errorT/error1:.1f}x")
            print(f"   Expected from log2(T): {np.log2(T):.1f}x")
        elif error1 == 0 and errorT == 0:
            print(f"   Growth rate: Perfect (both errors = 0)")
            print(f"   Expected from log2(T): {np.log2(T):.1f}x")
    except ValueError as e:
        errorT = float('inf')
        print(f"   Error: FAILED - {e}")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"  Single convolution error: {error1:.2e}")
    print(f"  T={T} convolution error: {errorT:.2e}")
    print("="*80)

# =============================================================================
# GRID STRATEGY COMPARISON
# =============================================================================

def run_grid_strategy_comparison(T_values: Optional[List[int]] = None,
                                n_bins: Optional[int] = None,
                                beta: Optional[float] = None):
    """Compare FIXED_NUM_POINTS vs FIXED_WIDTH grid strategies using simplified implementation."""
    # Use config defaults if not specified
    if T_values is None:
        T_values = SimpleExperimentConfig.GRID_T_VALUES
    if n_bins is None:
        n_bins = SimpleExperimentConfig.GRID_N_BINS
    if beta is None:
        beta = SimpleExperimentConfig.GRID_BETA
    
    print(f"\n{'='*80}")
    print(f"GRID STRATEGY COMPARISON: LogNormal(μ=0, σ=1) (Simplified Implementation)")
    print(f"{'='*80}")
    
    dist_lognorm = stats.lognorm(s=1, scale=1)
    
    base = discretize_continuous_to_pmf(
        dist_lognorm, n_bins, beta, bound_type=BoundType.DOMINATES, 
        spacing_type=SpacingType.GEOMETRIC
    )
    
    print(f"Base distribution: {len(base.x)} bins")
    print(f"  Median bin width (relative): {np.median(np.diff(np.log(base.x))):.6f}")
    
    results_fixed_points = {}
    results_fixed_width = {}
    
    for T in T_values:
        print(f"\n  T={T}:")
        
        # FIXED_NUM_POINTS strategy
        start = time.perf_counter()
        result_fp = self_convolve_discrete_distributions(
            base, T=T, beta=beta, bound_type=BoundType.DOMINATES, 
            spacing_type=SpacingType.GEOMETRIC, grid_strategy=GridStrategy.FIXED_NUM_POINTS
        )
        time_fp = time.perf_counter() - start
        
        # FIXED_WIDTH strategy
        start = time.perf_counter()
        result_fw = self_convolve_discrete_distributions(
            base, T=T, beta=beta, bound_type=BoundType.DOMINATES, 
            spacing_type=SpacingType.GEOMETRIC, grid_strategy=GridStrategy.FIXED_WIDTH
        )
        time_fw = time.perf_counter() - start
        
        # Compute statistics
        E_theoretical = T * dist_lognorm.mean()
        
        E_fp = np.sum(result_fp.x * result_fp.PMF)
        E_fw = np.sum(result_fw.x * result_fw.PMF)
        
        # For geometric spacing, compute relative width (ratio-1)
        median_width_fp = np.median(result_fp.x[1:] / result_fp.x[:-1]) - 1.0
        median_width_fw = np.median(result_fw.x[1:] / result_fw.x[:-1]) - 1.0
        
        base_width = np.median(base.x[1:] / base.x[:-1]) - 1.0
        
        print(f"    FIXED_NUM_POINTS:")
        print(f"      Grid size: {len(result_fp.x)} (vs {len(base.x)} base)")
        print(f"      Median relative width: {median_width_fp:.6f} ({median_width_fp/base_width:.2f}x base)")
        print(f"      E[X]: {E_fp:.6f} (error: {E_fp - E_theoretical:+.6f})")
        print(f"      Time: {time_fp:.3f}s")
        
        print(f"    FIXED_WIDTH:")
        print(f"      Grid size: {len(result_fw.x)} (vs {len(base.x)} base)")
        print(f"      Median relative width: {median_width_fw:.6f} ({median_width_fw/base_width:.2f}x base)")
        print(f"      E[X]: {E_fw:.6f} (error: {E_fw - E_theoretical:+.6f})")
        print(f"      Time: {time_fw:.3f}s")
        
        results_fixed_points[T] = {
            'dist': result_fp,
            'n_bins': len(result_fp.x),
            'median_width': median_width_fp,
            'E_X': E_fp,
            'E_error': E_fp - E_theoretical,
            'time': time_fp
        }
        
        results_fixed_width[T] = {
            'dist': result_fw,
            'n_bins': len(result_fw.x),
            'median_width': median_width_fw,
            'E_X': E_fw,
            'E_error': E_fw - E_theoretical,
            'time': time_fw
        }
    
    return results_fixed_points, results_fixed_width, base_width

# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def run_gaussian_comparison(T_values: Optional[List[int]] = None,
                           n_bins: Optional[int] = None,
                           beta: Optional[float] = None,
                           fft_size: Optional[int] = None,
                           mc_samples: Optional[int] = None):
    """Run comparison experiment for Gaussian distribution using simplified implementation."""
    # Use config defaults if not specified
    if T_values is None:
        T_values = SimpleExperimentConfig.T_VALUES
    if n_bins is None:
        n_bins = SimpleExperimentConfig.GAUSSIAN_N_BINS
    if beta is None:
        beta = SimpleExperimentConfig.GAUSSIAN_BETA
    if fft_size is None:
        fft_size = SimpleExperimentConfig.GAUSSIAN_FFT_SIZE
    if mc_samples is None:
        mc_samples = SimpleExperimentConfig.MC_SAMPLES
    
    print(f"\n{'='*80}")
    print(f"GAUSSIAN COMPARISON: N(0,1) with linear spacing (Simplified Implementation)")
    print(f"  N_BINS: {n_bins}, BETA: {beta:.2e}")
    print(f"{'='*80}")
    
    dist_gaussian = stats.norm(0, 1)
    
    try:
        base_upper = discretize_continuous_to_pmf(
            dist_gaussian, n_bins, beta, bound_type=BoundType.DOMINATES, 
            spacing_type=SpacingType.LINEAR
        )
        base_lower = discretize_continuous_to_pmf(
            dist_gaussian, n_bins, beta, bound_type=BoundType.IS_DOMINATED, 
            spacing_type=SpacingType.LINEAR
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"Base distribution: {len(base_upper.x):,} bins")
    print(f"  Range: [{base_upper.x[0]:.6f}, {base_upper.x[-1]:.6f}]")
    
    E_base = dist_gaussian.mean()
    print(f"  Theoretical E[X] = {E_base:.6f}")
    
    results = {}
    
    for T in T_values:
        print(f"\n  T={T} copies...")
        print(f"    Computing main implementation (T={T} convolutions)...")
        
        # Method 1: Simplified implementation
        start = time.perf_counter()
        Z_upper = self_convolve_discrete_distributions(
            base_upper, T, beta=beta, bound_type=BoundType.DOMINATES, 
            spacing_type=SpacingType.LINEAR, grid_strategy=SimpleExperimentConfig.GRID_STRATEGY
        )
        Z_lower = self_convolve_discrete_distributions(
            base_lower, T, beta=beta, bound_type=BoundType.IS_DOMINATED, 
            spacing_type=SpacingType.LINEAR, grid_strategy=SimpleExperimentConfig.GRID_STRATEGY
        )
        main_time = time.perf_counter() - start
        print(f"    Main implementation completed in {main_time:.3f}s")
        
        # Method 2: FFT convolution (using alternative methods)
        if SimpleExperimentConfig.RUN_FFT:
            start = time.perf_counter()
            Z_fft = clean_fft_self_convolve_continuous(dist_gaussian, T, BoundType.DOMINATES, fft_size, beta)
            fft_time = time.perf_counter() - start
        else:
            Z_fft = None
            fft_time = 0.0
        
        # Method 3: Monte Carlo convolution
        if SimpleExperimentConfig.RUN_MONTE_CARLO:
            start = time.perf_counter()
            Z_mc = monte_carlo_self_convolve_pmf(dist_gaussian, T, BoundType.DOMINATES, SpacingType.LINEAR,
                                                mc_samples, n_bins, 
                                                block_size=SimpleExperimentConfig.MC_BLOCK_SIZE,
                                                use_importance_sampling=SimpleExperimentConfig.USE_MC_IMPORTANCE)
            mc_time = time.perf_counter() - start
        else:
            Z_mc = None
            mc_time = 0.0
        
        # Method 4: Analytic convolution
        if SimpleExperimentConfig.RUN_ANALYTIC:
            start = time.perf_counter()
            Z_analytic = analytic_convolve_gaussian(dist_gaussian, T, BoundType.DOMINATES, SpacingType.LINEAR,
                                                   n_points=n_bins, beta=beta)
            analytic_time = time.perf_counter() - start
        else:
            Z_analytic = None
            analytic_time = 0.0
        
        # Compute statistics
        E_theoretical = T * E_base
        
        E_upper = np.sum(Z_upper.x * Z_upper.PMF)
        E_lower = np.sum(Z_lower.x * Z_lower.PMF)
        
        # Handle optional methods
        if Z_fft is not None:
            E_fft = np.sum(Z_fft.x * (Z_fft.PMF if hasattr(Z_fft, 'PMF') else Z_fft.vals))
        else:
            E_fft = np.nan
            
        if Z_mc is not None:
            finite_mask_mc = np.isfinite(Z_mc.x)
            E_mc = np.sum(Z_mc.x[finite_mask_mc] * (Z_mc.PMF[finite_mask_mc] if hasattr(Z_mc, 'PMF') else Z_mc.vals[finite_mask_mc])) if np.any(finite_mask_mc) else np.nan
        else:
            E_mc = np.nan
            
        if Z_analytic is not None:
            E_analytic = np.sum(Z_analytic.x * (Z_analytic.PMF if hasattr(Z_analytic, 'PMF') else Z_analytic.vals))
        else:
            E_analytic = np.nan
        
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

def run_lognormal_comparison(T_values: Optional[List[int]] = None,
                            n_bins: Optional[int] = None,
                            beta: Optional[float] = None,
                            fft_size: Optional[int] = None,
                            mc_samples: Optional[int] = None):
    """Run comparison experiment for LogNormal distribution using simplified implementation."""
    # Use config defaults if not specified
    if T_values is None:
        T_values = SimpleExperimentConfig.T_VALUES
    if n_bins is None:
        n_bins = SimpleExperimentConfig.LOGNORMAL_N_BINS
    if beta is None:
        beta = SimpleExperimentConfig.LOGNORMAL_BETA
    if fft_size is None:
        fft_size = SimpleExperimentConfig.LOGNORMAL_FFT_SIZE
    if mc_samples is None:
        mc_samples = SimpleExperimentConfig.MC_SAMPLES
    
    print(f"\n{'='*80}")
    print(f"LOGNORMAL COMPARISON: LogNormal(μ=0, σ=1) with geometric spacing (Simplified Implementation)")
    print(f"  N_BINS: {n_bins}, BETA: {beta:.2e}")
    print(f"{'='*80}")
    
    dist_lognorm = stats.lognorm(s=1, scale=1)
    
    try:
        base_upper = discretize_continuous_to_pmf(
            dist_lognorm, n_bins, beta, bound_type=BoundType.DOMINATES, 
            spacing_type=SpacingType.GEOMETRIC
        )
        base_lower = discretize_continuous_to_pmf(
            dist_lognorm, n_bins, beta, bound_type=BoundType.IS_DOMINATED, 
            spacing_type=SpacingType.GEOMETRIC
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
    
    print(f"Base distribution: {len(base_upper.x):,} bins")
    print(f"  Range: [{base_upper.x[0]:.6f}, {base_upper.x[-1]:.6f}]")
    
    E_base = dist_lognorm.mean()
    print(f"  Theoretical E[X] = {E_base:.6f}")
    
    results = {}
    
    for T in T_values:
        print(f"\n  T={T} copies...")
        
        # Method 1: Simplified implementation
        start = time.perf_counter()
        Z_upper = self_convolve_discrete_distributions(
            base_upper, T, beta=beta, bound_type=BoundType.DOMINATES, 
            spacing_type=SpacingType.GEOMETRIC, grid_strategy=SimpleExperimentConfig.GRID_STRATEGY
        )
        Z_lower = self_convolve_discrete_distributions(
            base_lower, T, beta=beta, bound_type=BoundType.IS_DOMINATED, 
            spacing_type=SpacingType.GEOMETRIC, grid_strategy=SimpleExperimentConfig.GRID_STRATEGY
        )
        main_time = time.perf_counter() - start
        
        # Method 2: FFT convolution (using alternative methods)
        if SimpleExperimentConfig.RUN_FFT:
            start = time.perf_counter()
            Z_fft = clean_fft_self_convolve_continuous(dist_lognorm, T, BoundType.DOMINATES, fft_size, beta)
            fft_time = time.perf_counter() - start
        else:
            Z_fft = None
            fft_time = 0.0
        
        # Method 3: Monte Carlo convolution
        if SimpleExperimentConfig.RUN_MONTE_CARLO:
            start = time.perf_counter()
            Z_mc = monte_carlo_self_convolve_pmf(dist_lognorm, T, BoundType.DOMINATES, SpacingType.GEOMETRIC,
                                                mc_samples, n_bins, 
                                                block_size=SimpleExperimentConfig.MC_BLOCK_SIZE,
                                                use_importance_sampling=SimpleExperimentConfig.USE_MC_IMPORTANCE)
            mc_time = time.perf_counter() - start
        else:
            Z_mc = None
            mc_time = 0.0
        
        # Method 4: Analytic convolution
        if SimpleExperimentConfig.RUN_ANALYTIC:
            start = time.perf_counter()
            Z_analytic = analytic_convolve_lognormal(dist_lognorm, T, BoundType.DOMINATES, SpacingType.GEOMETRIC,
                                                    n_points=n_bins, beta=beta)
            analytic_time = time.perf_counter() - start
        else:
            Z_analytic = None
            analytic_time = 0.0
        
        # Compute statistics
        E_theoretical = T * E_base
        
        E_upper = np.sum(Z_upper.x * Z_upper.PMF)
        E_lower = np.sum(Z_lower.x * Z_lower.PMF)
        
        # Handle optional methods
        if Z_fft is not None:
            E_fft = np.sum(Z_fft.x * (Z_fft.PMF if hasattr(Z_fft, 'PMF') else Z_fft.vals))
        else:
            E_fft = np.nan
            
        if Z_mc is not None:
            finite_mask_mc = np.isfinite(Z_mc.x)
            E_mc = np.sum(Z_mc.x[finite_mask_mc] * (Z_mc.PMF[finite_mask_mc] if hasattr(Z_mc, 'PMF') else Z_mc.vals[finite_mask_mc])) if np.any(finite_mask_mc) else np.nan
        else:
            E_mc = np.nan
            
        if Z_analytic is not None:
            E_analytic = np.sum(Z_analytic.x * (Z_analytic.PMF if hasattr(Z_analytic, 'PMF') else Z_analytic.vals))
        else:
            E_analytic = np.nan
        
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

# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def main():
    """Run comparison experiments using simplified implementation."""
    print("="*80)
    print("SIMPLE METHOD COMPARISON: Using Simplified Implementation (Original Parameters)")
    print("="*80)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  T values: {SimpleExperimentConfig.T_VALUES}")
    print(f"  Grid comparison T values: {SimpleExperimentConfig.GRID_T_VALUES}")
    print(f"  Gaussian: N_BINS={SimpleExperimentConfig.GAUSSIAN_N_BINS}, BETA={SimpleExperimentConfig.GAUSSIAN_BETA:.2e}")
    print(f"  Lognormal: N_BINS={SimpleExperimentConfig.LOGNORMAL_N_BINS}, BETA={SimpleExperimentConfig.LOGNORMAL_BETA:.2e}")
    print(f"  Monte Carlo: {SimpleExperimentConfig.MC_SAMPLES:,} samples")
    print(f"  Run diagnostics: {SimpleExperimentConfig.RUN_DIAGNOSTICS}")
    print(f"  Run grid comparison: {SimpleExperimentConfig.RUN_GRID_COMPARISON}")
    print(f"  Run method comparison: {SimpleExperimentConfig.RUN_METHOD_COMPARISON}")
    
    if SimpleExperimentConfig.RUN_DIAGNOSTICS:
        # Run diagnostics for Gaussian
        print("\n" + "="*80)
        print("DIAGNOSTICS: Gaussian N(0,1)")
        print("="*80)
        dist_gaussian = stats.norm(0, 1)
        base_gaussian = discretize_continuous_to_pmf(
            dist_gaussian, 
            SimpleExperimentConfig.DIAGNOSTICS_N_BINS, 
            SimpleExperimentConfig.DIAGNOSTICS_BETA_DISCRETIZE, 
            bound_type=BoundType.DOMINATES, 
            spacing_type=SpacingType.LINEAR
        )
        diagnose_error_sources(base_gaussian)  # Uses config defaults for T and beta
    
    if SimpleExperimentConfig.RUN_GRID_COMPARISON:
        print("\n" + "="*80)
        print("GRID STRATEGY COMPARISON")
        print("="*80)
        results_fp, results_fw, base_width = run_grid_strategy_comparison()  # Uses config defaults
        plot_grid_strategy_comparison(results_fp, results_fw, base_width, 
                                     SimpleExperimentConfig.GRID_T_VALUES, save_plot_func=save_plot)
    
    if SimpleExperimentConfig.RUN_METHOD_COMPARISON:
        print("\n" + "="*80)
        print("GAUSSIAN COMPARISON")
        print("="*80)
        gaussian_results = run_gaussian_comparison()  # Uses config defaults
        if gaussian_results:
            strategy_suffix = "fixed_points" if SimpleExperimentConfig.GRID_STRATEGY == GridStrategy.FIXED_NUM_POINTS else "constant_width"
            save_results(gaussian_results, f"gaussian_method_comparison_{strategy_suffix}", "results")
        
        print("\n" + "="*80)
        print("LOGNORMAL COMPARISON")
        print("="*80)
        lognormal_results = run_lognormal_comparison()  # Uses config defaults
        if lognormal_results:
            strategy_suffix = "fixed_points" if SimpleExperimentConfig.GRID_STRATEGY == GridStrategy.FIXED_NUM_POINTS else "constant_width"
            save_results(lognormal_results, f"lognormal_method_comparison_{strategy_suffix}", "results")
    
        # Generate plots
        print("\nGenerating plots...")
        if gaussian_results:
            plot_gaussian_results(gaussian_results, SimpleExperimentConfig.T_VALUES, save_plot_func=save_plot)
        if lognormal_results:
            plot_lognormal_results(lognormal_results, SimpleExperimentConfig.T_VALUES, save_plot_func=save_plot)
    
    print("\n" + "="*80)
    print("✅ All simple comparison experiments complete!")
    print("="*80)

if __name__ == "__main__":
    main()
