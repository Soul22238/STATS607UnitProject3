"""
Comprehensive Empirical Complexity Assessment
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from pathlib import Path

def empirical_complexity_study():
    """
    Comprehensive empirical analysis of computational complexity
    for multiple testing procedures
    """
    
    # Import functions locally to avoid circular imports
    from dgps import DGP
    from methods import z_test, Bonferroni_correction, Hochberg_correction, FDR_control
    
    print("EMPIRICAL COMPUTATIONAL COMPLEXITY ASSESSMENT")
    print("=" * 60)
    
    # Test range: powers of 2 for clear scaling patterns
    m_values = [2**i for i in range(2, 11)]  # 4 to 1024
    n_trials = 50  # Reduced for speed
    
    print(f"Testing problem sizes: {m_values}")
    print(f"Trials per size: {n_trials}")
    print("-" * 60)
    
    results = []
    
    for m in m_values:
        print(f"Testing m={m:4d}...", end=" ")
        
        # Setup problem instance
        dgp = DGP(m=m, m0=m//2, L=8.0, mode='E')
        rng = np.random.default_rng(42)
        
        # Time each operation
        times = {}
        
        # Data generation
        start = time.perf_counter()
        for _ in range(n_trials):
            dgp.generate_data(rng=rng)
        times['data_gen'] = (time.perf_counter() - start) / n_trials
        
        # Statistical testing
        X = dgp.generate_data(rng=rng)
        start = time.perf_counter()
        for _ in range(n_trials):
            z_test(X)
        times['z_test'] = (time.perf_counter() - start) / n_trials
        
        # Multiple testing corrections
        p_values = z_test(X)
        
        for method_name, method_func in [
            ('bonferroni', Bonferroni_correction),
            ('hochberg', Hochberg_correction), 
            ('fdr', FDR_control)
        ]:
            start = time.perf_counter()
            for _ in range(n_trials):
                method_func(p_values.copy())
            times[method_name] = (time.perf_counter() - start) / n_trials
        
        # Store results
        result = {'m': m}
        result.update(times)
        results.append(result)
        
        print(f"Completed in {sum(times.values())*1000:.2f}ms")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Complexity analysis
    print("\nCOMPLEXITY ANALYSIS")
    print("-" * 40)
    print(f"{'Method':<12} {'Complexity':<10} {'R²':<6} {'Theory':<10}")
    print("-" * 40)
    
    methods = ['data_gen', 'z_test', 'bonferroni', 'hochberg', 'fdr']
    theoretical = ['O(m)', 'O(m)', 'O(m)', 'O(m log m)', 'O(m log m)']
    
    complexity_results = {}
    
    for method, theory in zip(methods, theoretical):
        times = df[method].values
        
        # Log-log linear regression
        log_m = np.log2(df['m'].values)
        log_times = np.log2(times)
        
        # Remove any inf/-inf values
        valid = np.isfinite(log_times)
        log_m_clean = log_m[valid]
        log_times_clean = log_times[valid]
        
        if len(log_times_clean) > 1:
            slope, intercept = np.polyfit(log_m_clean, log_times_clean, 1)
            r_squared = np.corrcoef(log_m_clean, log_times_clean)[0,1]**2
        else:
            slope, r_squared = 0, 0
        
        # Classify empirical complexity
        if abs(slope) < 0.3:
            empirical = "O(1)"
        elif 0.3 <= slope < 0.8:
            empirical = "O(m^0.5)"
        elif 0.8 <= slope < 1.3:
            empirical = "O(m)"
        elif 1.3 <= slope < 1.8:
            empirical = "O(m log m)"
        elif 1.8 <= slope < 2.3:
            empirical = "O(m²)"
        else:
            empirical = f"O(m^{slope:.1f})"
        
        complexity_results[method] = {
            'empirical': empirical,
            'slope': slope,
            'r_squared': r_squared,
            'theoretical': theory
        }
        
        print(f"{method:<12} {empirical:<10} {r_squared:<6.3f} {theory:<10}")
    
    # Generate visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, method in enumerate(methods):
        ax = axes[i]
        times = df[method].values * 1000  # Convert to milliseconds
        
        # Plot empirical data
        ax.loglog(df['m'], times, 'bo-', label='Empirical', markersize=4)
        
        # Plot theoretical scaling
        slope = complexity_results[method]['slope']
        theoretical_times = times[0] * (df['m'] / df['m'].iloc[0])**slope
        ax.loglog(df['m'], theoretical_times, 'r--', label=f'Fitted: O(m^{slope:.2f})', alpha=0.7)
        
        ax.set_xlabel('Problem size (m)')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{method.replace("_", " ").title()}\n{complexity_results[method]["empirical"]}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/empirical_complexity.png', dpi=300, bbox_inches='tight')
    df.to_csv('results/complexity_data.csv', index=False)
    
    print(f"\nResults saved:")
    print(f"  Plot: results/empirical_complexity.png") 
    print(f"  Data: results/complexity_data.csv")
    
    # Summary
    print(f"\nSUMMARY")
    print("-" * 20)
    print(f"• Data generation: {complexity_results['data_gen']['empirical']}")
    print(f"• Statistical testing: {complexity_results['z_test']['empirical']}")  
    print(f"• Bonferroni: {complexity_results['bonferroni']['empirical']}")
    print(f"• Hochberg: {complexity_results['hochberg']['empirical']}")
    print(f"• FDR: {complexity_results['fdr']['empirical']}")
    
    return df, complexity_results

if __name__ == "__main__":
    empirical_complexity_study()
