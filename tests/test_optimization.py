"""
Test that optimized implementations produce equivalent results to baseline.
Run with: pytest tests/test_optimization.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from methods import z_test, Bonferroni_correction, Hochberg_correction, FDR_control
from simulation_optimized import (
    z_test_vectorized, 
    bonferroni_vectorized, 
    hochberg_vectorized, 
    fdr_vectorized,
    get_avg_power_vectorized
)
from metrics import get_avg_power


class TestZTest:
    """Test vectorized z-test produces same results as baseline."""
    
    def test_single_simulation(self):
        """Single simulation should give identical results."""
        np.random.seed(42)
        X = np.random.randn(10)
        
        # Baseline
        p_baseline = z_test(X)
        
        # Optimized (with batch dimension)
        X_batch = X.reshape(1, -1)
        p_optimized = z_test_vectorized(X_batch)[0]
        
        np.testing.assert_allclose(p_baseline, p_optimized, rtol=1e-10)
    
    def test_multiple_simulations(self):
        """Multiple simulations should match individual calls."""
        np.random.seed(42)
        n_sim = 100
        m = 8
        X_data = np.random.randn(n_sim, m)
        
        # Baseline: compute one at a time
        p_baseline = np.array([z_test(X) for X in X_data])
        
        # Optimized: compute all at once
        p_optimized = z_test_vectorized(X_data)
        
        np.testing.assert_allclose(p_baseline, p_optimized, rtol=1e-10)


class TestBonferroni:
    """Test vectorized Bonferroni produces same results as baseline."""
    
    def test_single_simulation(self):
        """Single simulation rejection decisions should match."""
        np.random.seed(42)
        p_values = np.random.uniform(0, 1, 10)
        
        # Baseline
        rejected_baseline = Bonferroni_correction(p_values)
        
        # Optimized
        p_batch = p_values.reshape(1, -1)
        rejected_optimized = bonferroni_vectorized(p_batch)[0]
        
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
    
    def test_multiple_simulations(self):
        """Multiple simulations should match individual calls."""
        np.random.seed(42)
        n_sim = 100
        m = 8
        p_values_all = np.random.uniform(0, 1, (n_sim, m))
        
        # Baseline: apply one at a time
        rejected_baseline = np.array([Bonferroni_correction(p) for p in p_values_all])
        
        # Optimized: apply all at once
        rejected_optimized = bonferroni_vectorized(p_values_all)
        
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
    
    def test_edge_cases(self):
        """Test edge cases: all null, all significant."""
        # All null (high p-values)
        p_null = np.ones((10, 5)) * 0.9
        rejected_baseline = np.array([Bonferroni_correction(p) for p in p_null])
        rejected_optimized = bonferroni_vectorized(p_null)
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
        assert not rejected_optimized.any()  # Should reject none
        
        # All significant (low p-values)
        p_sig = np.ones((10, 5)) * 0.001
        rejected_baseline = np.array([Bonferroni_correction(p) for p in p_sig])
        rejected_optimized = bonferroni_vectorized(p_sig)
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
        assert rejected_optimized.all()  # Should reject all


class TestHochberg:
    """Test vectorized Hochberg produces same results as baseline."""
    
    def test_single_simulation(self):
        """Single simulation rejection decisions should match."""
        np.random.seed(42)
        p_values = np.random.uniform(0, 1, 10)
        
        # Baseline
        rejected_baseline = Hochberg_correction(p_values)
        
        # Optimized
        p_batch = p_values.reshape(1, -1)
        rejected_optimized = hochberg_vectorized(p_batch)[0]
        
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
    
    def test_multiple_simulations(self):
        """Multiple simulations should match individual calls."""
        np.random.seed(42)
        n_sim = 50
        m = 8
        p_values_all = np.random.uniform(0, 1, (n_sim, m))
        
        # Baseline: apply one at a time
        rejected_baseline = np.array([Hochberg_correction(p) for p in p_values_all])
        
        # Optimized: apply all at once
        rejected_optimized = hochberg_vectorized(p_values_all)
        
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # All null
        p_null = np.ones((10, 5)) * 0.9
        rejected_baseline = np.array([Hochberg_correction(p) for p in p_null])
        rejected_optimized = hochberg_vectorized(p_null)
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
        
        # All significant
        p_sig = np.ones((10, 5)) * 0.001
        rejected_baseline = np.array([Hochberg_correction(p) for p in p_sig])
        rejected_optimized = hochberg_vectorized(p_sig)
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)


class TestFDR:
    """Test vectorized FDR produces same results as baseline."""
    
    def test_single_simulation(self):
        """Single simulation rejection decisions should match."""
        np.random.seed(42)
        p_values = np.random.uniform(0, 1, 10)
        
        # Baseline
        rejected_baseline = FDR_control(p_values)
        
        # Optimized
        p_batch = p_values.reshape(1, -1)
        rejected_optimized = fdr_vectorized(p_batch)[0]
        
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
    
    def test_multiple_simulations(self):
        """Multiple simulations should match individual calls."""
        np.random.seed(42)
        n_sim = 50
        m = 8
        p_values_all = np.random.uniform(0, 1, (n_sim, m))
        
        # Baseline: apply one at a time
        rejected_baseline = np.array([FDR_control(p) for p in p_values_all])
        
        # Optimized: apply all at once
        rejected_optimized = fdr_vectorized(p_values_all)
        
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # All null
        p_null = np.ones((10, 5)) * 0.9
        rejected_baseline = np.array([FDR_control(p) for p in p_null])
        rejected_optimized = fdr_vectorized(p_null)
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)
        
        # All significant
        p_sig = np.ones((10, 5)) * 0.001
        rejected_baseline = np.array([FDR_control(p) for p in p_sig])
        rejected_optimized = fdr_vectorized(p_sig)
        np.testing.assert_array_equal(rejected_baseline, rejected_optimized)


class TestPowerCalculation:
    """Test vectorized power calculation produces same results as baseline."""
    
    def test_power_computation(self):
        """Power estimates should match."""
        np.random.seed(42)
        n_sim = 100
        m = 8
        
        # Create true means (some null, some non-null)
        true_mus = np.array([0, 0, 0, 1, 1, 2, 2, 3])
        
        # Generate random rejection decisions
        rejected = np.random.rand(n_sim, m) > 0.5
        
        # Baseline: compute as list
        rejected_list = [rejected[i] for i in range(n_sim)]
        power_baseline = get_avg_power(true_mus, rejected_list)
        
        # Optimized: compute as array
        power_optimized = get_avg_power_vectorized(true_mus, rejected)
        
        np.testing.assert_allclose(power_baseline, power_optimized, rtol=1e-10)
    
    def test_power_all_null(self):
        """Power should be 0 when all hypotheses are null."""
        true_mus = np.zeros(8)
        rejected = np.random.rand(100, 8) > 0.5
        
        power_baseline = get_avg_power(true_mus, [rejected[i] for i in range(100)])
        power_optimized = get_avg_power_vectorized(true_mus, rejected)
        
        assert power_baseline == 0.0
        assert power_optimized == 0.0
    
    def test_power_all_non_null(self):
        """Power should match for all non-null case."""
        true_mus = np.ones(8)
        rejected = np.random.rand(100, 8) > 0.5
        
        power_baseline = get_avg_power(true_mus, [rejected[i] for i in range(100)])
        power_optimized = get_avg_power_vectorized(true_mus, rejected)
        
        np.testing.assert_allclose(power_baseline, power_optimized, rtol=1e-10)


class TestFullPipeline:
    """Integration test: full pipeline from data to power estimates."""
    
    def test_complete_workflow(self):
        """Test that complete workflow produces identical results."""
        np.random.seed(42)
        n_sim = 50
        m = 8
        
        # Generate test data
        true_mus = np.array([0, 0, 0, 1, 1, 2, 2, 3])
        X_data = np.random.randn(n_sim, m) + true_mus
        
        # Baseline workflow
        rejected_baseline = []
        for X in X_data:
            p_values = z_test(X)
            rejected = Hochberg_correction(p_values)
            rejected_baseline.append(rejected)
        power_baseline = get_avg_power(true_mus, rejected_baseline)
        
        # Optimized workflow
        p_values_all = z_test_vectorized(X_data)
        rejected_optimized = hochberg_vectorized(p_values_all)
        power_optimized = get_avg_power_vectorized(true_mus, rejected_optimized)
        
        # Results should match
        np.testing.assert_array_equal(np.array(rejected_baseline), rejected_optimized)
        np.testing.assert_allclose(power_baseline, power_optimized, rtol=1e-10)
    
    def test_all_correction_methods(self):
        """Test that all correction methods produce identical results."""
        np.random.seed(42)
        n_sim = 30
        m = 6
        
        true_mus = np.array([0, 0, 1, 1, 2, 3])
        X_data = np.random.randn(n_sim, m) + true_mus
        
        corrections = {
            'bonferroni': (Bonferroni_correction, bonferroni_vectorized),
            'hochberg': (Hochberg_correction, hochberg_vectorized),
            'fdr': (FDR_control, fdr_vectorized)
        }
        
        for name, (baseline_func, optimized_func) in corrections.items():
            # Baseline
            rejected_baseline = []
            for X in X_data:
                p_values = z_test(X)
                rejected = baseline_func(p_values)
                rejected_baseline.append(rejected)
            power_baseline = get_avg_power(true_mus, rejected_baseline)
            
            # Optimized
            p_values_all = z_test_vectorized(X_data)
            rejected_optimized = optimized_func(p_values_all)
            power_optimized = get_avg_power_vectorized(true_mus, rejected_optimized)
            
            # Should match
            np.testing.assert_array_equal(
                np.array(rejected_baseline), 
                rejected_optimized,
                err_msg=f"Mismatch in {name} correction"
            )
            np.testing.assert_allclose(
                power_baseline, 
                power_optimized, 
                rtol=1e-10,
                err_msg=f"Mismatch in {name} power estimate"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
