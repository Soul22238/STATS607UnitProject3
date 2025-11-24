"""
Estimate single simulation run time as a function of key parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyinstrument import Profiler
import time
from pathlib import Path
from dgps import DGP
from methods import z_test, Bonferroni_correction, Hochberg_correction, FDR_control
from metrics import get_avg_power

def time_single_simulation(m, L, mode, n_sim=1000):
    """
    Time a single simulation configuration
    
    Parameters
    ----------
    m : int
        Total number of hypotheses
    L : float
        Signal strength
    mode : str
        Mode ('D', 'E', 'I')
    n_sim : int
        Number of simulations to run for this configuration
    
    Returns
    -------
    dict with timing results
    """
    # Setup
    m0 = int(m * 0.5)  # 50% null hypotheses
    rng = np.random.default_rng(seed=607)
    dgp = DGP(m=m, m0=m0, L=L, mode=mode)
    
    # Time data generation
    start_time = time.time()
    mus = dgp.generate_mus()
    mus_time = time.time() - start_time
    
    # Time single data generation
    start_time = time.time()
    X = dgp.generate_data(rng=rng)
    single_data_gen_time = time.time() - start_time
    
    # Time statistical testing
    start_time = time.time()
    p_values = z_test(X)
    z_test_time = time.time() - start_time
    
    # Time corrections
    start_time = time.time()
    bonf_result = Bonferroni_correction(p_values)
    bonf_time = time.time() - start_time
    
    start_time = time.time()
    hoch_result = Hochberg_correction(p_values)
    hoch_time = time.time() - start_time
    
    start_time = time.time()
    fdr_result = FDR_control(p_values)
    fdr_time = time.time() - start_time
    
    # Time full simulation loop
    start_time = time.time()
    results = []
    for _ in range(n_sim):
        X = dgp.generate_data(rng=rng)
        p_values = z_test(X)
        bonf_result = Bonferroni_correction(p_values)
        results.append(bonf_result)
    
    full_loop_time = time.time() - start_time
    avg_single_sim_time = full_loop_time / n_sim
    
    return {
        'm': m,
        'L': L,
        'mode': mode,
        'm0': m0,
        'm1': m - m0,
        'n_sim': n_sim,
        'mus_generation_time': mus_time,
        'single_data_gen_time': single_data_gen_time,
        'z_test_time': z_test_time,
        'bonferroni_time': bonf_time,
        'hochberg_time': hoch_time,
        'fdr_time': fdr_time,
        'full_loop_time': full_loop_time,
        'avg_single_sim_time': avg_single_sim_time,
        'estimated_20k_time': avg_single_sim_time * 20000
    }

def analyze_parameter_scaling():
    """
    Analyze how simulation time scales with key parameters
    """
    print("🔍 Analyzing simulation time scaling with parameters...")
    print("=" * 70)
    
    # Key parameters to test
    m_values = [4, 8, 16, 32, 64]
    L_values = [5, 8, 10, 15]
    modes = ['D', 'E', 'I']
    
    results = []
    total_tests = len(m_values) * len(L_values) * len(modes)
    test_num = 0
    
    for m in m_values:
        for L in L_values:
            for mode in modes:
                test_num += 1
                print(f"\rTesting {test_num}/{total_tests}: m={m}, L={L}, mode={mode}", end='', flush=True)
                
                # Run timing test
                result = time_single_simulation(m, L, mode, n_sim=500)  # 500 simulations for good average
                results.append(result)
    
    print("\n" + "=" * 70)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Display results
    print("\n📊 Single Simulation Timing Results")
    print("-" * 70)
    print(f"{'m':<4} {'L':<4} {'Mode':<4} {'Avg Single (ms)':<15} {'Est. 20k (min)':<15}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        avg_ms = row['avg_single_sim_time'] * 1000
        est_20k_min = row['estimated_20k_time'] / 60
        print(f"{row['m']:<4} {row['L']:<4} {row['mode']:<4} {avg_ms:<15.3f} {est_20k_min:<15.1f}")
    
    # Save detailed results
    output_dir = Path('results/timing')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / 'single_simulation_timing.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Detailed results saved to: {csv_file}")
    
    return df

def plot_scaling_analysis(df):
    """
    Create plots showing how simulation time scales with parameters
    """
    output_dir = Path('results/timing')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Time vs m (number of hypotheses)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Single Simulation Time Scaling Analysis', fontsize=14)
    
    # Plot 1a: Average single simulation time vs m
    ax = axes[0, 0]
    for L in df['L'].unique():
        subset = df[df['L'] == L].groupby('m')['avg_single_sim_time'].mean()
        ax.plot(subset.index, subset.values * 1000, marker='o', label=f'L={L}')
    
    ax.set_xlabel('m (number of hypotheses)')
    ax.set_ylabel('Average time (ms)')
    ax.set_title('Single Simulation Time vs m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 1b: Estimated 20k simulation time vs m
    ax = axes[0, 1]
    for L in df['L'].unique():
        subset = df[df['L'] == L].groupby('m')['estimated_20k_time'].mean()
        ax.plot(subset.index, subset.values / 60, marker='o', label=f'L={L}')
    
    ax.set_xlabel('m (number of hypotheses)')
    ax.set_ylabel('Estimated time for 20k sims (minutes)')
    ax.set_title('Est. 20k Simulation Time vs m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2a: Time breakdown by component
    ax = axes[1, 0]
    components = ['single_data_gen_time', 'z_test_time', 'bonferroni_time', 'hochberg_time', 'fdr_time']
    component_labels = ['Data Gen', 'Z-test', 'Bonferroni', 'Hochberg', 'FDR']
    
    # Average across all configurations
    avg_times = [df[comp].mean() * 1000 for comp in components]  # Convert to ms
    
    bars = ax.bar(component_labels, avg_times)
    ax.set_ylabel('Average time (ms)')
    ax.set_title('Time Breakdown by Component')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{time:.3f}', ha='center', va='bottom')
    
    # Plot 2b: Time scaling by mode
    ax = axes[1, 1]
    for mode in df['mode'].unique():
        subset = df[df['mode'] == mode].groupby('m')['avg_single_sim_time'].mean()
        ax.plot(subset.index, subset.values * 1000, marker='o', label=f'Mode {mode}')
    
    ax.set_xlabel('m (number of hypotheses)')
    ax.set_ylabel('Average time (ms)')
    ax.set_title('Single Simulation Time by Mode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'simulation_timing_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"📈 Plots saved to: {plot_file}")
    plt.show()

def estimate_total_runtime():
    """
    Estimate total runtime for the full simulation study
    """
    print("\n🕐 Total Runtime Estimation")
    print("=" * 50)
    
    # Parameters for full study
    m_values = [4, 8, 16, 32, 64]
    L_values = [5, 8, 10, 15]
    modes = ['D', 'E', 'I']
    null_ratios = [0.75, 0.5, 0.25, 0.0]
    n_sim = 20000
    
    # Quick timing test with largest m
    print("Running quick timing test with m=64...")
    result = time_single_simulation(m=64, L=10, mode='E', n_sim=100)
    avg_single_time = result['avg_single_sim_time']
    
    # Calculate total configurations
    total_configs = len(m_values) * len(L_values) * len(modes) * len(null_ratios)
    
    # Estimate times
    total_simulations = total_configs * n_sim
    estimated_total_time = total_simulations * avg_single_time
    
    print(f"Configurations: {total_configs}")
    print(f"Total simulations: {total_simulations:,}")
    print(f"Average single simulation time: {avg_single_time*1000:.3f} ms")
    print(f"Estimated total time: {estimated_total_time/3600:.1f} hours")
    print(f"Estimated total time: {estimated_total_time/60:.1f} minutes")
    
    # Time for data generation vs analysis
    data_gen_time = total_configs * n_sim * result['single_data_gen_time']
    print(f"\nData generation time: {data_gen_time/60:.1f} minutes")
    print(f"Analysis time: {(estimated_total_time - data_gen_time)/60:.1f} minutes")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Quick test with just a few parameters
        print("🚀 Quick timing test...")
        result = time_single_simulation(m=32, L=8, mode='E', n_sim=1000)
        
        print("\n📊 Quick Results:")
        print(f"m={result['m']}, L={result['L']}, mode={result['mode']}")
        print(f"Single simulation time: {result['avg_single_sim_time']*1000:.3f} ms")
        print(f"Estimated 20k simulation time: {result['estimated_20k_time']/60:.1f} minutes")
        
    else:
        # Full analysis
        df = analyze_parameter_scaling()
        plot_scaling_analysis(df)
        estimate_total_runtime()
        
        print("\n✅ Timing analysis complete!")
        print("Check results/timing/ for detailed results and plots.")
