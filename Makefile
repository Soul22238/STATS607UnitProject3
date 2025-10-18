# Makefile for STATS607 Unit Project 2
# Simulation study on multiple testing procedures

.PHONY: all simulate analyze figures clean test

# Default target: run complete pipeline
all: simulate analyze figures

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
