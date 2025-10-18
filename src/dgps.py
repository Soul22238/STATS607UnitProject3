"""
Data Generation Processes (DGPs) for Simulation Studies
"""
import numpy as np
from scipy.stats import norm

def allocate_groups(total=4, mode='E'):
    """
    Allocate 'total' items into 4 groups according to one of three configurations:
    mode = 'D' → linearly decreasing (group1 > group2 > group3 > group4)
    mode = 'E' → equal distribution
    mode = 'I' → linearly increasing (group1 < group2 < group3 < group4)
    The function always returns a list of 4 integers (counts per group).
    If total < 4, only the first few groups will get items.
    """
    if total < 1:
        return [0, 0, 0, 0]
    n_groups = 4
    groups = [0]*4
    if mode.upper() == 'E':
        # Equal allocation across available groups
        base = total // n_groups
        groups = [base] * n_groups
        remainder = total % n_groups
        # Distribute remainder from left to right
        for i in range(remainder):
            groups[i] += 1

    elif mode.upper() == 'D':
        # Linearly decreasing allocation: weights 4:3:2:1
        weights = [4, 3, 2, 1]
        w_sum = sum(weights)
        groups = [round(total * w / w_sum) for w in weights]
    
        # Adjust rounding errors: prioritize keeping Group 1 (highest weight)
        diff = total - sum(groups)
        if diff > 0:
            # Need to add more: add from left (Group 1 first)
            for i in range(4):
                if diff == 0:
                    break
                groups[i % 4] += 1
                diff -= 1        
        elif diff < 0:
            # Need to remove: remove from right (Group 4 first)
            for idx in range(3, -1, -1):
                if diff == 0:
                    break
                if groups[idx] > 0:
                    groups[idx] -= 1
                    diff += 1
    elif mode.upper() == 'I':
        # Linearly increasing allocation: weights 1:2:3:4
        weights = [1, 2, 3, 4]
        w_sum = sum(weights)
        groups = [round(total * w / w_sum) for w in weights]
        # Adjust rounding errors: prioritize keeping Group 4 (highest weight)
        diff = total - sum(groups)
        if diff > 0:
            # Need to add more: add from right (Group 4 first)
            for idx in range(3, -1, -1):
                if diff == 0:
                    break
                groups[idx] += 1
                diff -= 1
        elif diff < 0:
            # Need to remove: remove from left (Group 1 first)
            for i in range(4):
                if diff == 0:
                    break
                if groups[i] > 0:
                    groups[i] -= 1
                    diff += 1
    else:
        raise ValueError("mode must be one of ['D', 'E', 'I']")
    # Adjust rounding errors so that the total sum equals 'total'
    return groups

class DGP:
    def __init__(self, m, m0, L, mode):
        self.m = m          # total number of hypotheses
        self.m0 = m0        # number of true zero null hypotheses
        self.m1 = m - m0    # number of false zero null hypotheses
        self.L = L          # signal strength parameter
        self.mode = mode    # non-null hypothesis allocation mode
        self.mus = None

    def generate_mus(self):
        # all non zero means are using 4/L, L/2, 3L/4, L 

        if self.m1 == 0:
            self.mus = np.zeros(self.m)
        else:
            # Four signal strength levels
            non_zero_group_means = np.array([self.L/4.0, self.L/2.0, 3*self.L/4.0, self.L])

            # Allocate non-null hypotheses according to mode
            non_zero_group_counts = allocate_groups(self.m1, self.mode)
            # print("non_zero_group_counts:", non_zero_group_counts)
            # Build the non-zero means array by repeating each group mean
            mus_nonnull = []
            for group_idx, count in enumerate(non_zero_group_counts):
                mus_nonnull.extend([non_zero_group_means[group_idx]] * count)

            # Create full mean vector: m0 zeros + m1 non-zeros
            self.mus = np.array([0.0] * self.m0 + mus_nonnull)
        return self.mus
    
    def generate_data(self, rng=None, seed=None):
        """
        Generate normally distributed random variables with variance 1 and means from self.mus.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator. If provided, seed is ignored.
        seed : int, optional
            Random seed for reproducibility. Only used if rng is None.

        Returns
        -------
        X : numpy.ndarray
            Array of length m with X_i ~ N(mu_i, 1)
        """
        if rng is None:
            rng = np.random.default_rng(seed=seed)
    
        # Generate mus if not already generated
        if self.mus is None:
            self.generate_mus()
        # print("Generated mus in generate_data:", self.mus)
        # Generate X_i ~ N(mu_i, 1) for each i
        X = rng.normal(loc=self.mus, scale=1.0, size=self.m)

        return X

# m = 4
# m0 = 2
# dgp = DGP(m, m0, L=5, mode='E')
# rng = np.random.default_rng(seed=607)
# mus = dgp.generate_mus()
# X = dgp.generate_data(rng=rng)
# print("Generated mus:", mus)
# print("Generated data X:", X)
# 
# X = dgp.generate_data(rng=rng)
# print("Generated mus:", mus)
# print("Generated data X:", X)