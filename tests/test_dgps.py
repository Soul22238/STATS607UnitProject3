import pytest
import numpy as np
import sys
sys.path.append('src')
from dgps import DGP, allocate_groups


def test_dgp_dimensions():
    """Test that DGP generates correct dimensions."""
    dgp = DGP(m=10, m0=6, L=5, mode='D')
    mus = dgp.generate_mus()
    rng = np.random.default_rng(seed=42)
    data = dgp.generate_data(rng=rng)
    
    assert len(mus) == 10, "mus should have length m"
    assert len(data) == 10, "data should have length m"


def test_dgp_null_hypotheses():
    """Test that null hypotheses have mean zero."""
    dgp = DGP(m=8, m0=6, L=5, mode='E')
    mus = dgp.generate_mus()
    
    null_count = np.sum(mus == 0)
    assert null_count == 6, f"Expected 6 null hypotheses, got {null_count}"


def test_dgp_signal_strengths():
    """Test that signal strengths come from expected levels."""
    L = 10
    dgp = DGP(m=8, m0=0, L=L, mode='E')  # All non-null for easier testing
    mus = dgp.generate_mus()
    
    expected_signals = {L/4, L/2, 3*L/4, L}
    # All mus should be from the expected signal set
    for mu in mus:
        assert mu in expected_signals, f"Signal {mu} not in expected set {expected_signals}"


def test_allocate_groups_mode_D():
    """Test decreasing mode prioritizes weak signals."""
    allocation = allocate_groups(4, 'D')
    
    # Mode D should have more weak signals (weights 4:3:2:1)
    assert allocation[0] >= allocation[3], "Mode D should prioritize weak signals"


def test_allocate_groups_mode_I():
    """Test increasing mode prioritizes strong signals."""
    allocation = allocate_groups(4, 'I')
    
    # Mode I should have more strong signals (weights 1:2:3:4)
    assert allocation[3] >= allocation[0], "Mode I should prioritize strong signals"


def test_dgp_data_mean():
    """Test that generated data has approximately correct mean."""
    dgp = DGP(m=1000, m0=0, L=5, mode='E')
    dgp.generate_mus()
    rng = np.random.default_rng(seed=42)
    
    # Generate many samples and check mean is close to expected
    all_data = []
    for _ in range(100):
        data = dgp.generate_data(rng=rng)
        all_data.extend(data)
    
    mean_data = np.mean(all_data)
    expected_mean = np.mean(dgp.mus)
    assert abs(mean_data - expected_mean) < 0.1, "Data mean should be close to expected mean"
