"""
Timing Analysis: Single Simulation Runtime vs Key Parameters
Measures runtime as function of m (hypotheses) and n_sim (simulations)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
from dgps import DGP
from methods import z_test, Bonferroni_correction, Hochberg_correction, FDR_control
from metrics import get_avg_power


def time_single_run(m, n_sim, L=10, mode='E', null_ratio=0.5):
    """Time one complete simulation run."""
    start = time.perf_counter()
    
    # Setup DGP
    m0 = int(m * null_ratio)
    rng = np.random.default_rng(seed=607)
    dgp = DGP(m=m, m0=m0, L=L, mode=mode)
    mus = dgp.generate_mus()
    
    # Run simulations with all correction methods
    corrections = {
        'bonferroni': Bonferroni_correction,
        'hochberg': Hochberg_correction,
        'fdr': FDR_control
    }
    
    results = {name: [] for name in corrections}
    
    for _ in range(n_sim):
        X = dgp.generate_data(rng=rng)
        p_values = z_test(X)
        for name, func in corrections.items():
            results[name].append(func(p_values))
    
    # Calculate metrics
    for name in corrections:
        power = get_avg_power(mus, np.array(results[name]))
    
    return time.perf_counter() - start


def measure_and_save():
    """Measure timing for different parameter values and save to CSV."""
    print("="*60)
    print("TIMING ANALYSIS: Single Simulation Runs")
    print("="*60)
    
    results = []
    
    # Test m scaling (n_sim fixed at 1000)
    print("\n Testing m scaling (n_sim=1000)...")
    for m in [4, 8, 16, 32, 64]:
        t = time_single_run(m=m, n_sim=1000)
        results.append({'param': 'm', 'value': m, 'time': t})
        print(f"  m={m:3d}: {t:.3f}s")
    
    # Test n_sim scaling (m fixed at 32)
    print("\n Testing n_sim scaling (m=32)...")
    for n_sim in [500, 1000, 2000, 5000, 10000, 20000]:
        t = time_single_run(m=32, n_sim=n_sim)
        results.append({'param': 'n_sim', 'value': n_sim, 'time': t})
        print(f"  n_sim={n_sim:5d}: {t:.3f}s")
    
    # Save data
    df = pd.DataFrame(results)
    Path('results/timing').mkdir(parents=True, exist_ok=True)
    df.to_csv('results/timing/single_simulation_timing.csv', index=False)
    print(f"\n Data saved to: results/timing/single_simulation_timing.csv")
    
    return df


def plot_results(df):
    """Create timing plots with linear and log-log scales."""
    from scipy import stats
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get data
    m_data = df[df['param'] == 'm']
    nsim_data = df[df['param'] == 'n_sim']
    m_vals = m_data['value'].values
    m_times = m_data['time'].values
    nsim_vals = nsim_data['value'].values
    nsim_times = nsim_data['time'].values
    
    # Top row: Linear scale plots
    ax1.plot(m_vals, m_times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Hypotheses (m)', fontsize=11)
    ax1.set_ylabel('Runtime (seconds)', fontsize=11)
    ax1.set_title('Runtime vs m (Linear Scale)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(nsim_vals, nsim_times, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Simulations (n_sim)', fontsize=11)
    ax2.set_ylabel('Runtime (seconds)', fontsize=11)
    ax2.set_title('Runtime vs n_sim (Linear Scale)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Bottom row: Log-log plots with complexity analysis
    # m log-log plot
    log_m = np.log2(m_vals)
    log_time_m = np.log2(m_times)
    slope_m, intercept_m, r_m, _, _ = stats.linregress(log_m, log_time_m)
    
    ax3.loglog(m_vals, m_times, 'bo-', linewidth=2, markersize=8, label='Observed')
    # Add fitted line
    m_fitted = 2**(slope_m * log_m + intercept_m)
    ax3.loglog(m_vals, m_fitted, 'b--', linewidth=2, alpha=0.7, 
               label=f'Fit: O(m^{slope_m:.2f}), R²={r_m**2:.3f}')
    ax3.set_xlabel('Number of Hypotheses (m)', fontsize=11)
    ax3.set_ylabel('Runtime (seconds)', fontsize=11)
    ax3.set_title('Log-Log: m Complexity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=9)
    
    # n_sim log-log plot
    log_nsim = np.log2(nsim_vals)
    log_time_nsim = np.log2(nsim_times)
    slope_nsim, intercept_nsim, r_nsim, _, _ = stats.linregress(log_nsim, log_time_nsim)
    
    ax4.loglog(nsim_vals, nsim_times, 'ro-', linewidth=2, markersize=8, label='Observed')
    # Add fitted line
    nsim_fitted = 2**(slope_nsim * log_nsim + intercept_nsim)
    ax4.loglog(nsim_vals, nsim_fitted, 'r--', linewidth=2, alpha=0.7,
               label=f'Fit: O(n^{slope_nsim:.2f}), R²={r_nsim**2:.3f}')
    ax4.set_xlabel('Number of Simulations (n_sim)', fontsize=11)
    ax4.set_ylabel('Runtime (seconds)', fontsize=11)
    ax4.set_title('Log-Log: n_sim Complexity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/timing/simulation_timing_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📈 Plot saved to: results/timing/simulation_timing_analysis.png")
    print(f"   Complexity: m ~ O(m^{slope_m:.2f}), n_sim ~ O(n^{slope_nsim:.2f})")
    plt.close()


def estimate_full_study(df):
    """Estimate total runtime for full study."""
    print("\n" + "="*60)
    print("FULL STUDY ESTIMATE")
    print("="*60)
    
    # Get time for n_sim=20000
    nsim_data = df[df['param'] == 'n_sim']
    time_20k = nsim_data[nsim_data['value'] == 20000]['time'].values[0]
    
    n_configs = 5 * 4 * 3 * 4  # 240 configurations
    total_time = n_configs * time_20k
    
    print(f"Configuration: 240 configs (5m × 4L × 3modes × 4ratios)")
    print(f"Per config (n_sim=20000): {time_20k:.2f}s")
    print(f"Total time: {total_time:.1f}s = {total_time/60:.1f} minutes")
    print(f"Total simulations: {n_configs * 20000:,}")


if __name__ == "__main__":
    # Measure timing and save data
    df = measure_and_save()
    
    # Create plots
    plot_results(df)
    
    # Estimate full study
    estimate_full_study(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
