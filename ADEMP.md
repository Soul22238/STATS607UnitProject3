# ADEMP Framework: Multiple Testing Simulation Study

## Aims

**Research Question:** How do different multiple testing correction methods (Bonferroni, Hochberg, FDR) compare in terms of statistical power under various signal strength and allocation scenarios?

**Hypotheses:**
- H1: Hochberg's step-up procedure will demonstrate higher power than Bonferroni correction while maintaining family-wise error rate control
- H2: FDR control (Benjamini-Hochberg) will show the highest power among all three methods
- H3: Power will increase with signal strength (L) and vary systematically across allocation modes (D, E, I)

## Data-generating Mechanisms

**Distribution:** Each hypothesis test follows X_i ~ N(μ_i, 1) where:
- μ_i = 0 for null hypotheses (H0 true)
- μ_i ∈ {L/4, L/2, 3L/4, L} for non-null hypotheses (H0 false)

**Parameters varied across conditions:**
- **m** (total hypotheses): {4, 8, 16, 32, 64}
- **L** (signal strength): {5, 8, 10, 15}
- **null_ratio**: {0.75, 0.5, 0.25, 0.0} → determines m0 = m × null_ratio
- **mode** (allocation of non-null signals):
  - D (Decreasing): More weak signals (weights 4:3:2:1 for L/4, L/2, 3L/4, L)
  - E (Equal): Equal distribution across all signal levels
  - I (Increasing): More strong signals (weights 1:2:3:4 for L/4, L/2, 3L/4, L)

**Fixed parameters:**
- n_sim = 20,000 simulations per configuration
- α = 0.05 (significance level)
- Random seed = 607 (for reproducibility)

## Estimands/Targets

**Primary estimand:** Average Power
- Power = P(Reject H_i | μ_i ≠ 0)
- Measures the proportion of true non-null hypotheses correctly rejected across simulations

**Secondary estimands (for validation):**
- FDR (False Discovery Rate) = E[V/R] where V = false discoveries, R = total rejections
- FWER (Family-Wise Error Rate) = P(V ≥ 1)

## Methods

Three multiple testing correction methods are compared:

1. **Bonferroni Correction**
   - Reject H_i if p_i < α/m
   - Controls FWER at level α

2. **Hochberg Step-up Procedure**
   - Order p-values: p_(1) ≤ ... ≤ p_(m)
   - Find largest i where p_(i) ≤ α/(m-i+1)
   - Reject H_(1), ..., H_(i)
   - Controls FWER at level α (more powerful than Bonferroni)

3. **FDR Control (Benjamini-Hochberg)**
   - Order p-values: p_(1) ≤ ... ≤ p_(m)
   - Find largest i where p_(i) ≤ α·i/m
   - Reject H_(1), ..., H_(i)
   - Controls FDR at level α

## Performance Measures

**Primary metric:** Average Power
- Calculated as the mean proportion of correctly rejected non-null hypotheses across all simulations
- Higher power indicates better detection of true signals

**Evaluation criteria:**
- Power trends across m (should increase with more hypotheses in certain scenarios)
- Power trends across L (should increase with stronger signals)
- Power differences across modes (I > E > D expected)
- Power comparison across methods (FDR > Hochberg > Bonferroni expected)

## Simulation Design Matrix

| Factor | Levels | Count |
|--------|--------|-------|
| m | 4, 8, 16, 32, 64 | 5 |
| L | 5, 8, 10, 15 | 4 |
| mode | D, E, I | 3 |
| null_ratio | 0.75, 0.5, 0.25, 0.0 | 4 |
| correction | Bonferroni, Hochberg, FDR | 3 |

**Total configurations:** 5 × 4 × 3 × 4 × 3 = 720 condition combinations

**Total simulations:** 720 × 20,000 = 14,400,000 hypothesis tests

## Reproducibility Details

**Data generation:**
```bash
python src/simulation.py generate
```
- Generates 240 raw data files (60 configurations × 4 null_ratios)
- Saved as compressed `.npz` files in `data/` directory
- Each file contains: X (observations, shape: n_sim × m) and mus (true means, shape: m)

**Analysis:**
```bash
python src/simulation.py analyze
```
- Reads raw data from `data/`
- Applies three correction methods to each dataset
- Computes power for each configuration
- Saves results to `results/raw/simulation_results_nsim20000.csv`

**Visualization:**
```bash
python src/figures.py
```
- Creates 4×3 subplot grids for each L value
- Rows = null_ratio, Columns = mode
- Saves figures to `results/figures/power_comparison_L*.png`

**Complete pipeline:**
```bash
make all
```

**Computational environment:**
- Python 3.11+
- Key packages: numpy, scipy, pandas, matplotlib, pytest
- See `requirements.txt` for complete dependencies
