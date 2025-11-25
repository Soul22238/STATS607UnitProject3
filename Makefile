# Makefile for STATS607 Unit Project 2
# Simulation study on multiple testing procedures

.PHONY: all simulate analyze figures clean test profile profile-full complexity help optimize benchmark

# Default target: run complete pipeline
all: simulate analyze figures

# Display help information
help:
	@echo "Available targets:"
	@echo "  make all           - Run complete simulation pipeline (simulate + analyze + figures)"
	@echo "  make simulate      - Generate raw simulation data"
	@echo "  make analyze       - Analyze data and compute metrics"
	@echo "  make figures       - Generate visualization plots"
	@echo "  make complexity    - Run timing analysis and complexity study"
	@echo "  make optimize      - Run optimized simulation"
	@echo "  make benchmark     - Run timing comparison: baseline vs optimized"
	@echo "  make profile       - Run full simulation with profiling"
	@echo "  make profile-part  - Run quick profiling test"
	@echo "  make test          - Run test suite"
	@echo "  make clean         - Remove all generated files"

# Generate raw simulation data (X and mus)
simulate:
	@echo "Generating raw simulation data..."
	python3 src/simulation.py generate

# Analyze raw data and compute performance metrics
analyze:
	@echo "Analyzing data and computing metrics..."
	python3 src/simulation.py analyze

# Create all visualizations
figures:
	@echo "Generating figures..."
	python3 src/figures.py

# Run performance profiling
profile-part:
	@echo "Running performance profiling..."
	python3 src/profile_simulation.py

# Run full simulation with profiling
profile:
	@echo "Running full simulation with profiling..."
	python3 src/simulation.py generate --profile

# Run timing analysis and complexity study
complexity:
	@echo "Running timing analysis and complexity study..."
	python3 src/timing_analysis.py

# Run optimized simulation
optimize:
	@echo "Running optimized simulation..."
	python3 src/simulation_optimized.py

# Run timing comparison: baseline vs optimized
benchmark:
	@echo "Running timing comparison: baseline vs optimized..."
	python3 src/compare_performance.py

# Remove generated files
clean:
	@echo "Cleaning up generated files..."
	rm -rf data/*.npz
	rm -f results/raw/*.csv results/raw/*.json
	rm -f results/figures/*.png
	rm -rf src/__pycache__ tests/__pycache__
	@echo "Clean complete"

# Run test suite
test:
	@echo "Running tests..."
	pytest tests/ -v
