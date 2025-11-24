"""
Performance profiling for simulation code
"""

import numpy as np
from pyinstrument import Profiler
from pathlib import Path
import time
from simulation import generate_data, run_all_simulations

def profile_generate_data():
    """Profile the data generation process with a smaller sample size"""
    print("=== Profiling Data Generation ===")
    
    profiler = Profiler()
    profiler.start()
    
    # Use smaller n_sim for profiling to get results faster
    generate_data(n_sim=100, output_dir='data_profile')
    
    profiler.stop()
    
    print("\n--- Data Generation Profile ---")
    profiler.print()
    
    # Save HTML report
    output_dir = Path('results/profiles')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_report = output_dir / 'generate_data_profile.html'
    with open(html_report, 'w') as f:
        f.write(profiler.output_html())
    print(f"\nHTML profile saved to: {html_report}")

def profile_analysis():
    """Profile the analysis process"""
    print("\n=== Profiling Analysis ===")
    
    # Make sure we have some test data first
    data_dir = 'data_profile'
    if not Path(data_dir).exists():
        print("Generating small test dataset first...")
        generate_data(n_sim=100, output_dir=data_dir)
    
    profiler = Profiler()
    profiler.start()
    
    # Analyze the test data
    run_all_simulations(n_sim=100, data_dir=data_dir, output_dir='results/profile_raw')
    
    profiler.stop()
    
    print("\n--- Analysis Profile ---")
    profiler.print()
    
    # Save HTML report
    output_dir = Path('results/profiles')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_report = output_dir / 'analysis_profile.html'
    with open(html_report, 'w') as f:
        f.write(profiler.output_html())
    print(f"\nHTML profile saved to: {html_report}")

def profile_single_methods():
    """Profile individual statistical methods to see which is slowest"""
    print("\n=== Profiling Individual Methods ===")
    
    from methods import z_test, Bonferroni_correction, Hochberg_correction, FDR_control
    
    # Generate test data
    np.random.seed(607)
    m = 64  # Larger m to see differences
    X = np.random.randn(m)
    
    # Time each method
    methods = {
        'z_test': lambda: z_test(X),
        'Bonferroni': lambda: Bonferroni_correction(z_test(X)),
        'Hochberg': lambda: Hochberg_correction(z_test(X)),
        'FDR': lambda: FDR_control(z_test(X))
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        profiler = Profiler()
        
        # Run multiple times to get better timing
        profiler.start()
        for _ in range(1000):  # Run 1000 times
            method_func()
        profiler.stop()
        
        print(f"\n--- {method_name} Profile (1000 runs) ---")
        profiler.print()
        
        # Save individual HTML reports
        output_dir = Path('results/profiles')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        html_report = output_dir / f'{method_name}_profile.html'
        with open(html_report, 'w') as f:
            f.write(profiler.output_html())
        print(f"HTML profile saved to: {html_report}")

def profile_dgp_generation():
    """Profile DGP data generation specifically"""
    print("\n=== Profiling DGP Generation ===")
    
    from dgps import DGP
    
    # Test different configurations
    configs = [
        (16, 8, 5, 'D'),
        (32, 16, 8, 'E'), 
        (64, 32, 10, 'I')
    ]
    
    for m, m0, L, mode in configs:
        print(f"\n--- Config: m={m}, m0={m0}, L={L}, mode={mode} ---")
        
        profiler = Profiler()
        profiler.start()
        
        rng = np.random.default_rng(seed=607)
        dgp = DGP(m=m, m0=m0, L=L, mode=mode)
        
        # Generate mus and data multiple times
        for _ in range(100):
            mus = dgp.generate_mus()
            X = dgp.generate_data(rng=rng)
        
        profiler.stop()
        
        profiler.print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        task = sys.argv[1]
        
        if task == 'generate':
            profile_generate_data()
        elif task == 'analysis':
            profile_analysis()
        elif task == 'methods':
            profile_single_methods()
        elif task == 'dgp':
            profile_dgp_generation()
        else:
            print("Available tasks: generate, analysis, methods, dgp")
    else:
        # Run all profiling tasks
        print("Running all profiling tasks...")
        profile_dgp_generation()
        profile_single_methods()
        profile_generate_data()
        profile_analysis()
        
        print("\n" + "="*70)
        print("All profiling completed!")
        print("Check results/profiles/ for HTML reports")
