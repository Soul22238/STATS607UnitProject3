"""
Performance Metrics for Multiple Testing Procedures
"""

import numpy as np


def get_avg_power(true_mus, rejected):
    """
    Calculate average power across multiple simulations.
    
    Parameters
    ----------
    true_mus : array-like, shape (m,)
        True mean values for each hypothesis
    rejected : array-like, shape (n_sim, m) or (m,)
        Boolean array indicating which hypotheses are rejected
        Can be a matrix where each row is one simulation
    
    Returns
    -------
    avg_power : float
        Proportion of non-null hypotheses correctly rejected
    """
    true_mus = np.asarray(true_mus)
    rejected = np.asarray(rejected)
    
    # Identify non-null hypotheses (indices where true_mus != 0)
    non_null_indices = np.where(true_mus != 0)[0]
    
    if len(non_null_indices) == 0:
        return 0.0  # No non-null hypotheses to calculate power
    
    # If rejected is 2D (multiple simulations)
    if rejected.ndim == 2:
        # Extract columns corresponding to non-null hypotheses
        rejected_non_nulls = rejected[:, non_null_indices]  # shape: (n_sim, m1)
        
        # Calculate proportion of True values across all simulations
        avg_power = np.mean(rejected_non_nulls)
    
    # If rejected is 1D (single simulation)
    else:
        # Extract elements corresponding to non-null hypotheses
        rejected_non_nulls = rejected[non_null_indices]
        
        # Calculate proportion of True values
        avg_power = np.mean(rejected_non_nulls)
    
    return avg_power


def get_fdr(true_mus, rejected):
    """
    Calculate False Discovery Rate.
    
    FDR = E[V/R] where V is number of false discoveries, R is number of rejections
    
    Parameters
    ----------
    true_mus : array-like, shape (m,)
        True mean values for each hypothesis
    rejected : array-like, shape (n_sim, m) or (m,)
        Boolean array indicating which hypotheses are rejected
    
    Returns
    -------
    fdr : float
        False Discovery Rate
    """
    true_mus = np.asarray(true_mus)
    rejected = np.asarray(rejected)
    
    # Identify null hypotheses (indices where true_mus == 0)
    null_indices = np.where(true_mus == 0)[0]
    
    if rejected.ndim == 2:
        # Multiple simulations
        fdrs = []
        for sim_rejected in rejected:
            num_rejected = np.sum(sim_rejected)
            if num_rejected == 0:
                fdrs.append(0.0)
            else:
                false_discoveries = np.sum(sim_rejected[null_indices])
                fdrs.append(false_discoveries / num_rejected)
        return np.mean(fdrs)
    else:
        # Single simulation
        num_rejected = np.sum(rejected)
        if num_rejected == 0:
            return 0.0
        false_discoveries = np.sum(rejected[null_indices])
        return false_discoveries / num_rejected


def get_fwer(true_mus, rejected):
    """
    Calculate Family-Wise Error Rate.
    
    FWER = P(V >= 1) where V is number of false discoveries
    
    Parameters
    ----------
    true_mus : array-like, shape (m,)
        True mean values for each hypothesis
    rejected : array-like, shape (n_sim, m) or (m,)
        Boolean array indicating which hypotheses are rejected
    
    Returns
    -------
    fwer : float
        Family-Wise Error Rate
    """
    true_mus = np.asarray(true_mus)
    rejected = np.asarray(rejected)
    
    # Identify null hypotheses
    null_indices = np.where(true_mus == 0)[0]
    
    if len(null_indices) == 0:
        return 0.0  # No null hypotheses
    
    if rejected.ndim == 2:
        # Multiple simulations
        # FWER is proportion of simulations with at least one false discovery
        has_false_discovery = []
        for sim_rejected in rejected:
            false_discoveries = np.sum(sim_rejected[null_indices])
            has_false_discovery.append(false_discoveries > 0)
        return np.mean(has_false_discovery)
    else:
        # Single simulation
        false_discoveries = np.sum(rejected[null_indices])
        return 1.0 if false_discoveries > 0 else 0.0


