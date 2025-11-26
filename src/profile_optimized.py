"""
Performance profiling for optimized simulation code
"""

import numpy as np
from pyinstrument import Profiler
from pathlib import Path
from simulation_optimized import generate_data_optimized, run_simulations_optimized


def profile_generate_optimized():
    """Profile optimized data generation"""
    print("=== Profiling Optimized Data Generation ===")
    
    profiler = Profiler()
    profiler.start()
    
    generate_data_optimized(n_sim=100, output_dir='data_profile_opt')
    
    profiler.stop()
    
    print("\n--- Optimized Data Generation Profile ---")
    profiler.print()
    
    output_dir = Path('results/profiles')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_report = output_dir / 'generate_optimized_profile.html'
    with open(html_report, 'w') as f:
        f.write(profiler.output_html())
    print(f"\nHTML profile saved to: {html_report}")


def profile_analysis_optimized():
    """Profile optimized analysis"""
    print("\n=== Profiling Optimized Analysis ===")
    
    data_dir = 'data_profile_opt'
    if not Path(data_dir).exists():
        print("Generating test dataset...")
        generate_data_optimized(n_sim=100, output_dir=data_dir)
    
    profiler = Profiler()
    profiler.start()
    
    run_simulations_optimized(n_sim=100, data_dir=data_dir, output_dir='results/profile_opt')
    
    profiler.stop()
    
    print("\n--- Optimized Analysis Profile ---")
    profiler.print()
    
    output_dir = Path('results/profiles')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_report = output_dir / 'analysis_optimized_profile.html'
    with open(html_report, 'w') as f:
        f.write(profiler.output_html())
    print(f"\nHTML profile saved to: {html_report}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        task = sys.argv[1]
        
        if task == 'generate':
            profile_generate_optimized()
        elif task == 'analysis':
            profile_analysis_optimized()
        else:
            print("Available tasks: generate, analysis")
    else:
        profile_generate_optimized()
        profile_analysis_optimized()
        
        print("\n" + "="*70)
        print("Optimized profiling completed!")
        print("Check results/profiles/ for HTML reports")
