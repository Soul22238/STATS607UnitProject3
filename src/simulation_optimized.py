"""
Optimized Simulation with Vectorization and Algorithmic Improvements
Key optimizations:
1. Vectorized correction methods (eliminate per-simulation loops)
2. Uncompressed file I/O (remove compression bottleneck)
3. Batch processing for efficiency
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dgps import DGP


def z_test_vectorized(X, mu0=0, sigma=1):
    """Vectorized z-test for batch of observations."""
    z_stats = (X - mu0) / sigma
    from scipy.stats import norm
    p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))
    return p_values


def bonferroni_vectorized(p_values, alpha=0.05):
    """Vectorized Bonferroni correction."""
    m = p_values.shape[-1]
    return p_values < (alpha / m)


def hochberg_vectorized(p_values, alpha=0.05):
    """Vectorized Hochberg correction for multiple simulations."""
    m = p_values.shape[-1]
    rejected = np.zeros_like(p_values, dtype=bool)
    
    for i, p_vals in enumerate(p_values):
        sorted_indices = np.argsort(p_vals)
        sorted_pvals = p_vals[sorted_indices]
        
        for j in range(m-1, -1, -1):
            if sorted_pvals[j] <= alpha / (m - j):
                rejected[i, sorted_indices[:j+1]] = True
                break
    return rejected


def fdr_vectorized(p_values, alpha=0.05):
    """Vectorized FDR control for multiple simulations."""
    m = p_values.shape[-1]
    rejected = np.zeros_like(p_values, dtype=bool)
    
    for i, p_vals in enumerate(p_values):
        sorted_indices = np.argsort(p_vals)
        sorted_pvals = p_vals[sorted_indices]
        
        for j in range(m):
            if sorted_pvals[j] <= alpha * (j + 1) / m:
                rejected[i, sorted_indices[:j+1]] = True
            else:
                break
    return rejected


def get_avg_power_vectorized(true_mus, rejected):
    """Vectorized power calculation."""
    non_null_mask = true_mus != 0
    if not np.any(non_null_mask):
        return 0.0
    return np.mean(rejected[:, non_null_mask])


def generate_data_optimized(n_sim=20000, output_dir='data_optimized'):
    """Generate data with uncompressed I/O (remove compression bottleneck)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    m_values = [4, 8, 16, 32, 64]
    L_values = [5, 8, 10, 15]
    modes = ['D', 'E', 'I']
    null_ratios = [0.75, 0.5, 0.25, 0.0]
    
    print(f"Generating optimized data with n_sim={n_sim}")
    total_configs = len(m_values) * len(L_values) * len(modes) * len(null_ratios)
    config_num = 0
    
    for m in m_values:
        for L in L_values:
            for mode in modes:
                for null_ratio in null_ratios:
                    config_num += 1
                    progress = config_num / total_configs * 100
                    bar_length = 50
                    filled = int(bar_length * config_num / total_configs)
                    bar = '█' * filled + '-' * (bar_length - filled)
                    print(f'\r[{bar}] {progress:.1f}% ({config_num}/{total_configs})', end='', flush=True)
                    
                    m0 = int(m * null_ratio)
                    rng = np.random.default_rng(seed=607)
                    dgp = DGP(m=m, m0=m0, L=L, mode=mode)
                    mus = dgp.generate_mus()
                    
                    # Generate all X data
                    X_data = np.array([dgp.generate_data(rng=rng) for _ in range(n_sim)])
                    
                    # Save uncompressed (remove compression bottleneck)
                    filename = f'm{m}_L{L}_{mode}_null{null_ratio:.2f}.npz'
                    np.savez(output_path / filename, X=X_data, mus=mus)
    
    print(f"\n{'='*70}")
    print(f"Optimized data saved to: {output_dir}/")
    return output_path


def run_simulations_optimized(n_sim=20000, data_dir='data_optimized', output_dir='results/optimized'):
    """
    Optimized analysis with vectorized operations.
    Key optimization: Process all simulations at once instead of looping.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_dir)
    
    corrections = {
        'bonferroni': bonferroni_vectorized,
        'hochberg': hochberg_vectorized,
        'fdr': fdr_vectorized
    }
    
    print(f"Analyzing data from {data_dir}/ (OPTIMIZED)")
    data_files = sorted(data_path.glob('*.npz'))
    total_configs = len(data_files) * len(corrections)
    config_num = 0
    
    csv_data = []
    
    for data_file in data_files:
        # Load data
        data = np.load(data_file)
        X_data, mus = data['X'], data['mus']
        
        # Parse filename
        parts = data_file.stem.split('_')
        m = int(parts[0][1:])
        L = int(parts[1][1:])
        mode = parts[2]
        null_ratio = float(parts[3][4:])
        m0 = int(m * null_ratio)
        
        # Vectorized p-value computation for ALL simulations at once
        p_values_all = z_test_vectorized(X_data)  # shape: (n_sim, m)
        
        # Apply each correction method
        for corr_name, corr_func in corrections.items():
            config_num += 1
            progress = config_num / total_configs * 100
            bar_length = 50
            filled = int(bar_length * config_num / total_configs)
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f'\r[{bar}] {progress:.1f}% ({config_num}/{total_configs})', end='', flush=True)
            
            # Vectorized correction (all simulations at once)
            rejected_array = corr_func(p_values_all)  # shape: (n_sim, m)
            
            # Vectorized power calculation
            power = get_avg_power_vectorized(mus, rejected_array)
            
            csv_data.append({
                'm': m, 'm0': m0, 'm1': m - m0,
                'null_ratio': null_ratio, 'L': L,
                'mode': mode, 'correction': corr_name,
                'power': power
            })
    
    print(f"\n{'='*70}")
    
    # Save results
    df = pd.DataFrame(csv_data)
    csv_file = output_path / f'simulation_results_nsim{n_sim}.csv'
    df.to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")
    return csv_file


if __name__ == "__main__":
    import sys
    from pyinstrument import Profiler
    
    enable_profiling = '--profile' in sys.argv
    
    if enable_profiling:
        profiler = Profiler(interval=0.1)  # Sample every 100ms to drastically reduce file size for nested loops
        profiler.start()
        print("Profiling enabled")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        generate_data_optimized(n_sim=20000)
    elif len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        run_simulations_optimized(n_sim=20000)
    else:
        # Run full pipeline
        generate_data_optimized(n_sim=20000)
        run_simulations_optimized(n_sim=20000)
    
    if enable_profiling:
        profiler.stop()
        print("\n" + "="*70)
        print("PROFILING RESULTS")
        print("="*70)
        profiler.print()
        
        # Save HTML report
        from pathlib import Path
        output_dir = Path('results/profiles')
        output_dir.mkdir(parents=True, exist_ok=True)
        html_file = output_dir / 'full_simulation_optimized_profile.html'
        with open(html_file, 'w') as f:
            f.write(profiler.output_html())
        print(f"\nHTML profile saved to: {html_file}")
