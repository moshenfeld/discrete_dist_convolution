"""
Progressive self-convolution testing with detailed timing.

Tests T=10, 100, 1000 with n=1000, 5000, 10000 bins.
Starts small and only continues if performance is reasonable.
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from discrete_conv_api import DiscreteDist, self_convolve_pmf

def create_gaussian_pmf(n: int, mu: float = 0.0, sigma: float = 1.0) -> DiscreteDist:
    """Create a discrete Gaussian PMF."""
    lo = mu - 5 * sigma
    hi = mu + 5 * sigma
    
    x = np.linspace(lo, hi, n, dtype=np.float64)
    if n > 1:
        x[1:] += np.linspace(1e-14, 1e-12, n-1)
    
    z = (x - mu) / sigma
    pdf = np.exp(-0.5 * z * z)
    pdf /= pdf.sum()
    return DiscreteDist(x=x, kind="pmf", vals=pdf, p_neg_inf=0.0, p_pos_inf=0.0, 
                       name=f"N({mu},{sigma})")

def test_self_convolution(base: DiscreteDist, T: int, n_output: int):
    """Test self-convolution with detailed timing."""
    # Create output grid
    x_min, x_max = np.min(base.x), np.max(base.x)
    t_min = T * x_min - abs(x_min) * 2
    t_max = T * x_max + abs(x_max) * 2
    t = np.linspace(t_min, t_max, n_output, dtype=np.float64)
    
    # Expected number of convolutions
    n_convolutions = int(np.ceil(np.log2(T))) + bin(T).count('1') - 1
    
    print(f"    T={T:4d}: ", end='', flush=True)
    
    # Warmup (JIT compilation)
    if T == 10:  # Only warmup on first run
        _ = self_convolve_pmf(base, T, t=t, mode='DOMINATES')
        print("[warmup] ", end='', flush=True)
    
    # Timed run
    start = time.perf_counter()
    Z = self_convolve_pmf(base, T, t=t, mode='DOMINATES')
    elapsed = time.perf_counter() - start
    
    total_mass = Z.vals.sum() + Z.p_neg_inf + Z.p_pos_inf
    
    # Calculate effective throughput
    ops_per_conv = n_output * n_output
    total_ops = n_convolutions * ops_per_conv
    throughput = total_ops / elapsed / 1e6  # M ops/sec
    
    print(f"{elapsed:7.3f}s | {n_convolutions:2d} convs | {throughput:5.1f} M ops/s | mass={total_mass:.8f}")
    
    return elapsed, n_convolutions

def run_progressive_tests():
    """Run progressive tests, stopping if performance is too slow."""
    print("="*80)
    print("Progressive Self-Convolution Testing")
    print("Testing T=10, 100, 1000 with different grid sizes")
    print("="*80)
    
    T_values = [10, 100, 1000]
    n_values = [1000, 5000, 10000]
    
    # Store results for comparison
    results = {}
    
    for n_bins in n_values:
        print(f"\n{'='*80}")
        print(f"Grid Size: {n_bins:,} bins")
        print(f"{'='*80}")
        
        # Create base distribution
        print(f"Creating Gaussian base distribution with {n_bins:,} bins...")
        base = create_gaussian_pmf(n=n_bins, mu=0, sigma=1)
        print(f"  Base mass: {base.vals.sum():.10f}")
        
        results[n_bins] = {}
        
        for T in T_values:
            try:
                elapsed, n_convs = test_self_convolution(base, T, n_bins)
                results[n_bins][T] = (elapsed, n_convs)
                
                # Check if we should continue
                if T == 10 and elapsed > 5.0:
                    print(f"\n  ⚠️  T=10 took {elapsed:.1f}s (too slow). Skipping larger T values for this grid size.")
                    break
                elif T == 100 and elapsed > 60.0:
                    print(f"\n  ⚠️  T=100 took {elapsed:.1f}s (too slow). Skipping T=1000 for this grid size.")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n  ⚠️  Interrupted by user.")
                break
        
        # Check if we should continue to larger grid
        if 10 in results[n_bins]:
            t10 = results[n_bins][10][0]
            # Estimate time for next grid size (scales as n^2)
            if n_bins == 1000:
                next_n = 5000
                scale = (next_n / n_bins) ** 2
                estimated = t10 * scale
                if estimated > 30.0:
                    print(f"\n  ⚠️  Estimated T=10 time for {next_n:,} bins: ~{estimated:.1f}s")
                    print(f"     This seems too slow. Consider stopping here.")
                    response = input(f"     Continue anyway? (y/n): ").strip().lower()
                    if response != 'y':
                        break
            elif n_bins == 5000:
                next_n = 10000
                scale = (next_n / n_bins) ** 2
                estimated = t10 * scale
                if estimated > 60.0:
                    print(f"\n  ⚠️  Estimated T=10 time for {next_n:,} bins: ~{estimated:.1f}s")
                    print(f"     This seems too slow. Consider stopping here.")
                    response = input(f"     Continue anyway? (y/n): ").strip().lower()
                    if response != 'y':
                        break
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: Self-Convolution Performance")
    print(f"{'='*80}")
    
    print(f"\n{'Bins':<10} {'T=10':<15} {'T=100':<15} {'T=1000':<15}")
    print("-" * 60)
    
    for n_bins in n_values:
        if n_bins not in results:
            break
        row = f"{n_bins:,}".ljust(10)
        
        for T in T_values:
            if T in results[n_bins]:
                elapsed, n_convs = results[n_bins][T]
                row += f"{elapsed:.3f}s ({n_convs:2d})".ljust(15)
            else:
                row += "-".ljust(15)
        
        print(row)
    
    print(f"\n{'='*80}")
    print("Algorithm Efficiency (number of convolutions):")
    print(f"  T=10:   ~7 convolutions  (vs 9 naive)")
    print(f"  T=100:  ~13 convolutions (vs 99 naive)")
    print(f"  T=1000: ~17 convolutions (vs 999 naive)")
    print(f"{'='*80}")
    
    # Analysis
    print("\nPerformance Analysis:")
    if 1000 in results and 10 in results[1000]:
        t1k = results[1000][10][0]
        ops_1k = 7 * (1000 ** 2)  # 7 convolutions, 1000x1000 each
        throughput_1k = ops_1k / t1k / 1e6
        print(f"  1k bins: {throughput_1k:.1f} M ops/sec")
        
        if 5000 in results and 10 in results[5000]:
            t5k = results[5000][10][0]
            ratio = t5k / t1k
            expected_ratio = (5000 / 1000) ** 2
            print(f"  5k bins: {ratio:.1f}x slower than 1k (expected {expected_ratio:.1f}x)")
            
        if 10000 in results and 10 in results[10000]:
            t10k = results[10000][10][0]
            ratio = t10k / t1k
            expected_ratio = (10000 / 1000) ** 2
            print(f"  10k bins: {ratio:.1f}x slower than 1k (expected {expected_ratio:.1f}x)")

if __name__ == "__main__":
    run_progressive_tests()

