# STATS607 Unit Project 2: Multiple Testing Simulation

Simulation study comparing Bonferroni, Hochberg, and FDR control methods.

## Quick Start

```bash
make all      # Run full pipeline: generate data → analyze → create figures
make test     # Run test suite
make clean    # Remove all generated files
```

## Project Structure

```
├── src/                      # Source code
│   ├── dgps.py              # Data generation process
│   ├── methods.py           # Statistical testing methods
│   ├── metrics.py           # Performance metrics
│   ├── simulation.py        # Main simulation pipeline
│   └── figures.py           # Visualization
├── data/                     # Raw simulation data (*.npz files)
├── results/
│   ├── raw/                 # Analysis results (CSV)
│   └── figures/             # Generated figures (PNG)
└── tests/                    # Unit tests

```

## Pipeline Steps

1. **Generate Data** (`make simulate`)
   - Creates raw X and mus data
   - Saves to `data/` as `.npz` files
   
2. **Analyze** (`make analyze`)
   - Applies correction methods to raw data
   - Computes power metrics
   - Saves results to `results/raw/simulation_results_nsim20000.csv`
   
3. **Visualize** (`make figures`)
   - Generates plots for each L value
   - Saves to `results/figures/power_comparison_L*.png`

## Parameters

- **m**: {4, 8, 16, 32, 64} hypotheses
- **L**: {5, 8, 10, 15} signal strength
- **mode**: D (decreasing), E (equal), I (increasing)
- **null_ratio**: {0.75, 0.5, 0.25, 0.0}
- **n_sim**: 20000 simulations per configuration

## Requirements

```bash
pip install -r requirements.txt
```
