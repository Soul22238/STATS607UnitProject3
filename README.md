# STATS607 Unit Project 2: Multiple Testing Simulation

Simulation study comparing three multiple testing correction methods (Bonferroni, Hochberg, and Benjamini-Hochberg FDR control) across varying signal strengths, hypothesis counts, and allocation strategies.


## Key Findings

FDR control achieves 20-30% higher power than Bonferroni at large m, especially with strong signals. Signal allocation mode (D/E/I) has a larger impact on power than the proportion of null hypotheses.

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

Requirements: Python 3.11+, numpy, scipy, pandas, matplotlib, pytest

## Quick Start

**Run complete analysis:**
```bash
make all      # Generate data → analyze → create figures
```

**Or run steps individually:**
```bash
make simulate # Generate raw simulation data
make analyze  # Compute power metrics
make figures  # Create visualizations
make test     # Run test suite
make clean    # Remove all generated files
```

## Runtime

- `make simulate`: ~1-2 minutes (240 data files, 20,000 simulations each)
- `make analyze`: ~5-7 minutes (applies 3 correction methods)
- `make figures`: ~10-20 seconds (creates 4 PNG files)
- **Total:** ~10-15 minutes

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
