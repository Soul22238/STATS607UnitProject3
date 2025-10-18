"""
Statistical Methods being evaluated
"""

"""
Statistical Methods being evaluated
"""

import numpy as np
from scipy.stats import norm

def z_test(X, mu0=0, sigma=1, alpha=0.05):
    """
    Perform z-test for each observation in X.
    
    H0: X_i ~ N(mu0, sigma^2)
    H1: X_i ~ N(mu_i, sigma^2) where mu_i != mu0
    
    Parameters
    ----------
    X : array-like
        Observed data
    mu0 : float, default=0
        Null hypothesis mean
    sigma : float, default=1
        Known standard deviation
    alpha : float, default=0.05
        Significance level
    
    Returns
    -------
    p_values : numpy.ndarray
        Two-sided p-values for each test
    """
    X = np.asarray(X)
    
    # Calculate z-statistics
    z_stats = (X - mu0) / sigma
    
    # Calculate two-sided p-values
    p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))
    
    return p_values

def Bonferroni_correction(p_values, alpha=0.05):
    """
    Apply Bonferroni correction to p-values.
    
    Parameters
    ----------
    p_values : array-like
        P-values to correct
    alpha : float, default=0.05
        Significance level
    
    Returns
    -------
    rejected : numpy.ndarray
        Boolean array indicating which hypotheses are rejected after correction
    """
    p_values = np.asarray(p_values)
    m = len(p_values)
    corrected_alpha = alpha / m
    rejected = p_values < corrected_alpha
    return rejected

def Hochberg_correction(p_values, alpha=0.05):
    """
    Apply Hochberg's step-up procedure to p-values.
    
    Parameters
    ----------
    p_values : array-like
        P-values to correct
    alpha : float, default=0.05
        Significance level
    
    Returns
    -------
    rejected : numpy.ndarray
        Boolean array indicating which hypotheses are rejected after correction
    """
    p_values = np.asarray(p_values)
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    rejected = np.zeros(m, dtype=bool)
    for i in range(m-1, -1, -1):
        threshold = alpha / (m - i)
        if sorted_pvals[i] <= threshold:
            rejected[sorted_indices[:i+1]] = True
            break
    return rejected

def FDR_control(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg procedure for FDR control.
    
    Parameters
    ----------
    p_values : array-like
        P-values to correct
    alpha : float, default=0.05
        Significance level
    
    Returns
    -------
    rejected : numpy.ndarray
        Boolean array indicating which hypotheses are rejected after correction
    """
    p_values = np.asarray(p_values)
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    rejected = np.zeros(m, dtype=bool)
    for i in range(m):
        threshold = alpha * (i + 1) / m
        if sorted_pvals[i] <= threshold:
            rejected[sorted_indices[:i+1]] = True
        else:
            break
    return rejected

# Example usage
if __name__ == "__main__":
    # Test with some data
    X = np.array([0.5, -1.2, 2.3, 0.1, -0.3])
    
    p_values = z_test(X)
    
    print("Data:", X)
    print("P-values:", p_values)
    bonf_rejected = Bonferroni_correction(p_values)
    print("Bonferroni rejected:", bonf_rejected)
    hoch_rejected = Hochberg_correction(p_values)
    print("Hochberg rejected:", hoch_rejected)