import pytest
import numpy as np
import sys
sys.path.append('src')
from methods import z_test, Bonferroni_correction, Hochberg_correction, FDR_control


def test_z_test_all_null():
    """Test z-test returns uniform p-values under null."""
    rng = np.random.default_rng(seed=42)
    data = rng.normal(0, 1, size=1000)  # All null
    p_values = z_test(data)
    
    assert len(p_values) == 1000, "Should return p-value for each test"
    assert np.all((p_values >= 0) & (p_values <= 1)), "P-values should be in [0,1]"


def test_z_test_detects_signal():
    """Test z-test has small p-value for strong signal."""
    data = np.array([10.0])  # Strong signal
    p_value = z_test(data)[0]
    
    assert p_value < 0.001, "Strong signal should have very small p-value"


def test_bonferroni_controls_fwer():
    """Test Bonferroni rejects nothing when all null at alpha=0.05."""
    rng = np.random.default_rng(seed=42)
    p_values = rng.uniform(0.1, 1.0, size=100)  # No small p-values
    rejected = Bonferroni_correction(p_values, alpha=0.05)
    
    assert np.sum(rejected) == 0, "Should reject nothing with large p-values"


def test_bonferroni_rejects_strong_signal():
    """Test Bonferroni rejects very small p-value."""
    p_values = np.array([0.001, 0.5, 0.8, 0.9])
    rejected = Bonferroni_correction(p_values, alpha=0.05)
    
    assert rejected[0] == True, "Should reject very small p-value"
    assert np.sum(rejected) >= 1, "Should reject at least one hypothesis"


def test_hochberg_more_powerful_than_bonferroni():
    """Test Hochberg rejects at least as many as Bonferroni."""
    p_values = np.array([0.001, 0.01, 0.02, 0.5, 0.8])
    bonf_rejected = Bonferroni_correction(p_values)
    hoch_rejected = Hochberg_correction(p_values)
    
    assert np.sum(hoch_rejected) >= np.sum(bonf_rejected), \
        "Hochberg should be at least as powerful as Bonferroni"


def test_fdr_control_basic():
    """Test FDR control rejects appropriately."""
    p_values = np.array([0.001, 0.01, 0.05, 0.5, 0.9])
    rejected = FDR_control(p_values, alpha=0.05)
    
    assert rejected[0] == True, "Should reject smallest p-value"
    assert rejected[-1] == False, "Should not reject large p-value"


def test_methods_return_boolean_array():
    """Test all methods return boolean arrays of correct length."""
    p_values = np.array([0.01, 0.05, 0.5, 0.8])
    
    for method in [Bonferroni_correction, Hochberg_correction, FDR_control]:
        rejected = method(p_values)
        assert len(rejected) == len(p_values), "Should return array of same length"
        assert rejected.dtype == bool, "Should return boolean array"
