"""
Empirical Computational Complexity Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from dgps import DGP
from methods import z_test, Bonferroni_correction, Hochberg_correction, FDR_control

def time_function(func, *args, repeats=100):
    """Time a function with multiple runs"""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return np.mean(times)

def analyze_complexity():
    """Empirical complexity analysis of key operations"""
    
    # Test parameters
    m_values = [4, 8, 16, 32, 64, 128, 256]
    rng = np.random.default_rng(42)
    
    print("Empirical Computational Complexity Analysis")
    print("=" * 50)
    
    # Results storage
    results = {
        'data_gen': [],
        'z_test': [],
        'bonferroni': [],
        'hochberg': [],
        'fdr': []
    }
    
    for m in m_values:
        print(f"Testing m = {m}...")
        
        # Setup
        dgp = DGP(m=m, m0=m//2, L=8, mode='E')
        mus = dgp.generate_mus()
        X = dgp.generate_data(rng=rng)
        p_values = z_test(X)
        
        # Time each operation
        results['data_gen'].append(time_function(dgp.generate_data, rng))
        results['z_test'].append(time_function(z_test, X))
        results['bonferroni'].append(time_function(Bonferroni_correction, p_values))
        results['hochberg'].append(time_function(Hochberg_correction, p_values))
        results['fdr'].append(time_function(FDR_control, p_values))
    
    # Analysis
    print("\nComplexity Analysis:")
    print("-" * 30)
    
    for name, times in results.items():
        # Linear fit: log(time) vs log(m)
        log_m = np.log(m_values)
        log_times = np.log(times)
        
        slope = np.polyfit(log_m, log_times, 1)[0]
        
        if slope < 0.5:
            complexity = "O(1)"
        elif slope < 1.5:
            complexity = "O(n)"  
        elif slope < 2.5:
            complexity = "O(n²)"
        else:
            complexity = f"O(n^{slope:.1f})"
            
        print(f"{name:<12}: {complexity:<8} (slope: {slope:.2f})")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    for i, (name, times) in enumerate(results.items(), 1):
        plt.subplot(2, 3, i)
        plt.loglog(m_values, times, 'o-', label=name)
        plt.xlabel('m (problem size)')
        plt.ylabel('Time (seconds)')
        plt.title(f'{name.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/complexity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: results/complexity_analysis.png")
    
    return results, m_values

if __name__ == "__main__":
    analyze_complexity()
