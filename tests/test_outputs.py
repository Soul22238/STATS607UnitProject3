import pytest
import numpy as np
import sys
sys.path.append('src')
from dgps import DGP
from methods import z_test, Bonferroni_correction, Hochberg_correction, FDR_control
from metrics import get_avg_power, get_fdr, get_fwer


def test_power_in_valid_range():
    """Test that power is between 0 and 1."""
    mus = np.array([0, 0, 5, 10])
    rejected = np.array([[False, False, True, True],
                        [False, False, False, True],
                        [False, False, True, False]])
    
    power = get_avg_power(mus, rejected)
    assert 0 <= power <= 1, f"Power {power} should be in [0,1]"


def test_power_all_correct():
    """Test power is 1 when all non-nulls are rejected."""
    mus = np.array([0, 0, 5, 10])
    rejected = np.array([[False, False, True, True]])
    
    power = get_avg_power(mus, rejected)
    assert power == 1.0, "Power should be 1 when all non-nulls rejected"


def test_power_none_rejected():
    """Test power is 0 when no non-nulls are rejected."""
    mus = np.array([0, 0, 5, 10])
    rejected = np.array([[False, False, False, False]])
    
    power = get_avg_power(mus, rejected)
    assert power == 0.0, "Power should be 0 when no non-nulls rejected"


def test_fdr_in_valid_range():
    """Test that FDR is between 0 and 1."""
    mus = np.array([0, 0, 5, 10])
    rejected = np.array([[True, False, True, True]])
    
    fdr = get_fdr(mus, rejected)
    assert 0 <= fdr <= 1, f"FDR {fdr} should be in [0,1]"


def test_fwer_in_valid_range():
    """Test that FWER is between 0 and 1."""
    mus = np.array([0, 0, 5, 10])
    rejected = np.array([[True, False, True, True],
                        [False, False, True, True]])
    
    fwer = get_fwer(mus, rejected)
    assert 0 <= fwer <= 1, f"FWER {fwer} should be in [0,1]"


def test_high_power_with_strong_signals():
    """Test that strong signals lead to high power (reproducing paper pattern)."""
    rng = np.random.default_rng(seed=607)
    dgp = DGP(m=4, m0=1, L=10, mode='I')  # Strong signals, mode I
    mus = dgp.generate_mus()
    
    rejected_list = []
    for _ in range(1000):
        data = dgp.generate_data(rng=rng)
        p_values = z_test(data)
        rejected = Hochberg_correction(p_values)
        rejected_list.append(rejected)
    
    rejected_array = np.array(rejected_list)
    power = get_avg_power(mus, rejected_array)
    
    # With L=10, mode I, 75% null, power should be very high (close to 1)
    assert power > 0.95, f"Power {power} should be >0.95 for strong signals in mode I"


def test_low_power_with_weak_signals():
    """Test that weak signals lead to lower power (reproducing paper pattern)."""
    rng = np.random.default_rng(seed=607)
    dgp = DGP(m=4, m0=1, L=5, mode='D')  # Weak signals, mode D
    mus = dgp.generate_mus()
    
    rejected_list = []
    for _ in range(1000):
        data = dgp.generate_data(rng=rng)
        p_values = z_test(data)
        rejected = Bonferroni_correction(p_values)
        rejected_list.append(rejected)
    
    rejected_array = np.array(rejected_list)
    power = get_avg_power(mus, rejected_array)
    
    # With L=5, mode D, power should be lower
    assert power < 0.5, f"Power {power} should be <0.5 for weak signals in mode D"
