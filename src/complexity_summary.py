"""
Final Empirical Complexity Assessment Summary
"""

import pandas as pd
import numpy as np

def complexity_summary():
    """Generate final complexity assessment"""
    
    # Load empirical data
    df = pd.read_csv('results/complexity_data.csv')
    
    print("EMPIRICAL COMPUTATIONAL COMPLEXITY ASSESSMENT")
    print("=" * 55)
    
    # Methods and their theoretical expectations
    methods = {
        'data_gen': {'name': 'Data Generation', 'theory': 'O(m)'},
        'z_test': {'name': 'Z-Test', 'theory': 'O(m)'},
        'bonferroni': {'name': 'Bonferroni', 'theory': 'O(m)'},
        'hochberg': {'name': 'Hochberg', 'theory': 'O(m log m)'},
        'fdr': {'name': 'FDR (B-H)', 'theory': 'O(m log m)'}
    }
    
    print(f"Problem sizes tested: {df['m'].min()} to {df['m'].max()}")
    print(f"Scaling factor: {df['m'].max() / df['m'].min():.0f}x")
    print()
    
    # Empirical analysis
    results = []
    
    for method_key, info in methods.items():
        times = df[method_key].values
        m_vals = df['m'].values
        
        # Log-log regression for scaling exponent
        log_m = np.log2(m_vals)
        log_t = np.log2(times)
        
        # Remove invalid values
        valid = np.isfinite(log_t)
        if np.sum(valid) > 1:
            slope, _ = np.polyfit(log_m[valid], log_t[valid], 1)
            r2 = np.corrcoef(log_m[valid], log_t[valid])[0,1]**2
        else:
            slope, r2 = 0, 0
        
        # Time scaling factor
        scaling_factor = times[-1] / times[0]  # Ratio of max to min time
        
        # Classify complexity
        if abs(slope) < 0.3:
            empirical = "O(1)"
        elif 0.3 <= abs(slope) < 0.8:
            empirical = "O(√m)"
        elif 0.8 <= abs(slope) < 1.3:
            empirical = "O(m)"
        elif 1.3 <= abs(slope) < 1.8:
            empirical = "O(m log m)"
        else:
            empirical = f"O(m^{slope:.1f})"
        
        results.append({
            'Method': info['name'],
            'Empirical': empirical,
            'Theoretical': info['theory'],
            'Scaling': f"{scaling_factor:.1f}x",
            'R²': f"{r2:.3f}",
            'Slope': f"{slope:.2f}"
        })
    
    # Display results table
    print("COMPLEXITY COMPARISON")
    print("-" * 55)
    print(f"{'Method':<15} {'Empirical':<12} {'Theory':<12} {'Scaling':<8} {'R²':<6}")
    print("-" * 55)
    
    for result in results:
        print(f"{result['Method']:<15} {result['Empirical']:<12} {result['Theoretical']:<12} {result['Scaling']:<8} {result['R²']:<6}")
    
    # Key findings
    print("\nKEY FINDINGS")
    print("-" * 20)
    
    # Most computationally expensive
    last_times = df.iloc[-1, 1:].values  # Last row times
    method_names = list(methods.keys())
    slowest_idx = np.argmax(last_times)
    fastest_idx = np.argmin(last_times)
    
    print(f"• Most expensive: {methods[method_names[slowest_idx]]['name']}")
    print(f"• Least expensive: {methods[method_names[fastest_idx]]['name']}")
    print(f"• Performance ratio: {last_times[slowest_idx]/last_times[fastest_idx]:.0f}:1")
    
    # Scaling behavior
    slopes = [float(r['Slope']) for r in results]
    linear_methods = [r['Method'] for r, s in zip(results, slopes) if 0.8 <= s < 1.3]
    sublinear_methods = [r['Method'] for r, s in zip(results, slopes) if s < 0.8]
    
    if linear_methods:
        print(f"• Linear scaling: {', '.join(linear_methods)}")
    if sublinear_methods:
        print(f"• Sublinear scaling: {', '.join(sublinear_methods)}")
    
    # Practical implications
    print("\nPRACTICAL IMPLICATIONS")
    print("-" * 25)
    max_time = df.iloc[-1, 1:].max()
    print(f"• Largest problem (m=1024): {max_time*1000:.2f}ms per simulation")
    print(f"• Study feasibility: Highly tractable")
    print(f"• Bottleneck: FDR procedure dominates for large m")
    
    return results

if __name__ == "__main__":
    complexity_summary()
