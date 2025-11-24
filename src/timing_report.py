"""
Generate a comprehensive timing analysis report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_timing_report():
    """Generate a comprehensive timing analysis report"""
    
    # Load the timing data
    df = pd.read_csv('results/timing/single_simulation_timing.csv')
    
    print("🕐 SINGLE SIMULATION TIMING ANALYSIS REPORT")
    print("=" * 80)
    
    # Overall statistics
    print("\n📊 OVERALL TIMING STATISTICS")
    print("-" * 50)
    print(f"Total configurations tested: {len(df)}")
    print(f"Average single simulation time: {df['avg_single_sim_time'].mean()*1000:.3f} ms")
    print(f"Std deviation: {df['avg_single_sim_time'].std()*1000:.3f} ms")
    print(f"Minimum time: {df['avg_single_sim_time'].min()*1000:.3f} ms")
    print(f"Maximum time: {df['avg_single_sim_time'].max()*1000:.3f} ms")
    
    # Scaling with m (number of hypotheses)
    print("\n📈 SCALING WITH m (Number of Hypotheses)")
    print("-" * 50)
    m_scaling = df.groupby('m')['avg_single_sim_time'].agg(['mean', 'std']).reset_index()
    m_scaling['mean_ms'] = m_scaling['mean'] * 1000
    m_scaling['std_ms'] = m_scaling['std'] * 1000
    
    print(f"{'m':<4} {'Mean (ms)':<12} {'Std (ms)':<12} {'Est. 20k (min)':<15}")
    print("-" * 50)
    for _, row in m_scaling.iterrows():
        est_20k = row['mean'] * 20000 / 60
        print(f"{row['m']:<4} {row['mean_ms']:<12.3f} {row['std_ms']:<12.3f} {est_20k:<15.1f}")
    
    # Scaling with L (signal strength)
    print("\n📈 SCALING WITH L (Signal Strength)")
    print("-" * 50)
    L_scaling = df.groupby('L')['avg_single_sim_time'].agg(['mean', 'std']).reset_index()
    L_scaling['mean_ms'] = L_scaling['mean'] * 1000
    L_scaling['std_ms'] = L_scaling['std'] * 1000
    
    print(f"{'L':<4} {'Mean (ms)':<12} {'Std (ms)':<12} {'Est. 20k (min)':<15}")
    print("-" * 50)
    for _, row in L_scaling.iterrows():
        est_20k = row['mean'] * 20000 / 60
        print(f"{row['L']:<4} {row['mean_ms']:<12.3f} {row['std_ms']:<12.3f} {est_20k:<15.1f}")
    
    # Mode comparison
    print("\n📈 SCALING WITH MODE (Distribution Pattern)")
    print("-" * 50)
    mode_scaling = df.groupby('mode')['avg_single_sim_time'].agg(['mean', 'std']).reset_index()
    mode_scaling['mean_ms'] = mode_scaling['mean'] * 1000
    mode_scaling['std_ms'] = mode_scaling['std'] * 1000
    
    print(f"{'Mode':<6} {'Mean (ms)':<12} {'Std (ms)':<12} {'Est. 20k (min)':<15}")
    print("-" * 50)
    for _, row in mode_scaling.iterrows():
        est_20k = row['mean'] * 20000 / 60
        print(f"{row['mode']:<6} {row['mean_ms']:<12.3f} {row['std_ms']:<12.3f} {est_20k:<15.1f}")
    
    # Component timing breakdown
    print("\n⚙️ TIMING BREAKDOWN BY COMPONENT")
    print("-" * 50)
    components = {
        'Data Generation': 'single_data_gen_time',
        'Z-test': 'z_test_time', 
        'Bonferroni': 'bonferroni_time',
        'Hochberg': 'hochberg_time',
        'FDR': 'fdr_time'
    }
    
    print(f"{'Component':<15} {'Mean (ms)':<12} {'% of Total':<12}")
    print("-" * 50)
    total_component_time = sum(df[comp].mean() for comp in components.values())
    
    for name, col in components.items():
        mean_time = df[col].mean() * 1000
        percentage = (df[col].mean() / total_component_time) * 100
        print(f"{name:<15} {mean_time:<12.3f} {percentage:<12.1f}%")
    
    # Full simulation estimates
    print("\n🚀 FULL SIMULATION STUDY ESTIMATES")
    print("-" * 50)
    
    # Parameters for full study
    m_values = [4, 8, 16, 32, 64]
    L_values = [5, 8, 10, 15] 
    modes = ['D', 'E', 'I']
    null_ratios = [0.75, 0.5, 0.25, 0.0]
    n_sim = 20000
    
    total_configs = len(m_values) * len(L_values) * len(modes) * len(null_ratios)
    total_simulations = total_configs * n_sim
    
    # Use average timing from our measurements
    avg_single_sim_time = df['avg_single_sim_time'].mean()
    estimated_total_time = total_simulations * avg_single_sim_time
    
    print(f"Total configurations: {total_configs}")
    print(f"Simulations per config: {n_sim:,}")
    print(f"Total simulations: {total_simulations:,}")
    print(f"Average single sim time: {avg_single_sim_time*1000:.3f} ms")
    print(f"")
    print(f"ESTIMATED TOTAL RUNTIME:")
    print(f"  Total time: {estimated_total_time:.1f} seconds")
    print(f"  Total time: {estimated_total_time/60:.1f} minutes")
    print(f"  Total time: {estimated_total_time/3600:.1f} hours")
    
    # Data generation vs analysis breakdown
    data_gen_time = total_simulations * df['single_data_gen_time'].mean()
    analysis_time = estimated_total_time - data_gen_time
    
    print(f"")
    print(f"BREAKDOWN:")
    print(f"  Data generation: {data_gen_time/60:.1f} minutes ({data_gen_time/estimated_total_time*100:.1f}%)")
    print(f"  Statistical analysis: {analysis_time/60:.1f} minutes ({analysis_time/estimated_total_time*100:.1f}%)")
    
    # Complexity analysis
    print("\n📊 COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("-" * 50)
    
    # Check if time scales linearly with m
    correlation_m = np.corrcoef(df['m'], df['avg_single_sim_time'])[0,1]
    print(f"Correlation with m: {correlation_m:.3f}")
    
    # Fit linear model to check scaling
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['m'], df['avg_single_sim_time'])
    print(f"Linear fit: time = {slope*1000:.4f}*m + {intercept*1000:.4f} (ms)")
    print(f"R² = {r_value**2:.3f}")
    
    if r_value**2 > 0.8:
        print("✅ Time scales approximately linearly with m")
    else:
        print("⚠️  Time scaling with m is not strongly linear")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 50)
    
    if estimated_total_time < 3600:  # Less than 1 hour
        print("✅ Full simulation study is computationally feasible")
        print(f"   Estimated runtime: {estimated_total_time/60:.1f} minutes")
    elif estimated_total_time < 3600 * 24:  # Less than 1 day
        print("⚠️  Full simulation study will take several hours")
        print(f"   Estimated runtime: {estimated_total_time/3600:.1f} hours")
        print("   Consider running overnight or in batches")
    else:
        print("❌ Full simulation study may be too computationally expensive")
        print("   Consider reducing n_sim or number of configurations")
    
    print(f"\n🎯 Most efficient approach:")
    print(f"   - Run data generation first (saves ~{data_gen_time/60:.1f} min per batch)")
    print(f"   - Process analysis in parallel if possible")
    print(f"   - Monitor progress with smaller test runs first")

if __name__ == "__main__":
    generate_timing_report()
