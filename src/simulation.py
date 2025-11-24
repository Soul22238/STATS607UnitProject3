import numpy as np
import pandas as pd
from pathlib import Path
from methods import z_test, Bonferroni_correction, Hochberg_correction, FDR_control
from metrics import get_avg_power
from dgps import *
from pyinstrument import Profiler

def generate_data(n_sim=20000, output_dir='data'):
    """Generate raw simulation data (X and true mus) and save to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    m_values = [4, 8, 16, 32, 64]
    L_values = [5, 8, 10, 15]
    modes = ['D', 'E', 'I']
    
    print(f"Generating data with n_sim={n_sim}")
    total_configs = len(m_values) * len(L_values) * len(modes) * 4  # 4 null_ratios
    config_num = 0
    
    for m in m_values:
        for L in L_values:
            for mode in modes:
                for null_ratio in [0.75, 0.5, 0.25, 0.0]:
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
                    
                    # Save to file
                    filename = f'm{m}_L{L}_{mode}_null{null_ratio:.2f}.npz'
                    np.savez_compressed(output_path / filename, X=X_data, mus=mus)
    
    print(f"\n{'='*70}")
    print(f"Data saved to: {output_dir}/")
    return output_path


def run_all_simulations(n_sim=20000, data_dir='data', output_dir='results/raw'):
    """Analyze raw data and compute performance metrics."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_dir)
    
    corrections = {'bonferroni': Bonferroni_correction, 
                   'hochberg': Hochberg_correction, 
                   'fdr': FDR_control}
    
    print(f"Analyzing data from {data_dir}/")
    data_files = sorted(data_path.glob('*.npz'))
    total_configs = len(data_files) * len(corrections)
    config_num = 0
    
    csv_data = []
    
    for data_file in data_files:
        # Load data
        data = np.load(data_file)
        X_data, mus = data['X'], data['mus']
        
        # Parse filename: m4_L5_D_null0.75.npz
        parts = data_file.stem.split('_')
        m = int(parts[0][1:])
        L = int(parts[1][1:])
        mode = parts[2]
        null_ratio = float(parts[3][4:])
        m0 = int(m * null_ratio)
        
        # Test each correction method
        for corr_name, corr_func in corrections.items():
            config_num += 1
            progress = config_num / total_configs * 100
            bar_length = 50
            filled = int(bar_length * config_num / total_configs)
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f'\r[{bar}] {progress:.1f}% ({config_num}/{total_configs})', end='', flush=True)
            
            # Apply correction to all simulations
            rejected_list = []
            for X in X_data:
                p_values = z_test(X)
                rejected = corr_func(p_values)
                rejected_list.append(rejected)
            
            rejected_array = np.array(rejected_list)
            power = get_avg_power(mus, rejected_array)
            
            csv_data.append({
                'm': m, 'm0': m0, 'm1': m - m0,
                'null_ratio': null_ratio, 'L': L,
                'mode': mode, 'correction': corr_name,
                'power': power
            })
    
    print(f"\n{'='*70}")
    
    # Save CSV
    df = pd.DataFrame(csv_data)
    csv_file = output_path / f'simulation_results_nsim{n_sim}.csv'
    df.to_csv(csv_file, index=False)
    print(f"Results saved to: {csv_file}")
    return csv_file


if __name__ == "__main__":
    import sys
    
    # Check if profiling is requested
    enable_profiling = '--profile' in sys.argv
    if '--profile' in sys.argv:
        sys.argv.remove('--profile')
    
    if enable_profiling:
        profiler = Profiler()
        profiler.start()
        print("🔍 Profiling enabled...")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        # Generate raw data
        generate_data(n_sim=20000)
    elif len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        # Analyze existing data
        run_all_simulations(n_sim=20000)
    else:
        # Run full pipeline
        generate_data(n_sim=20000)
        run_all_simulations(n_sim=20000)
    
    if enable_profiling:
        profiler.stop()
        print("\n" + "="*70)
        print("🔍 PROFILING RESULTS")
        print("="*70)
        profiler.print()
        
        # Save HTML report
        output_dir = Path('results/profiles')
        output_dir.mkdir(parents=True, exist_ok=True)
        html_file = output_dir / 'full_simulation_profile.html'
        with open(html_file, 'w') as f:
            f.write(profiler.output_html())
        print(f"\n📊 HTML profile saved to: {html_file}")