"""
dgp_functions.py
----------------
Synthetic DGP functions for each scenario in Causal Lab.
All functions share the interface fn(n, knob, true_ate, seed) -> (Y, T, X, ps).
"""

import numpy as np
from scipy.special import expit

SEED     = 89
TRUE_ATE = 2.0


def make_overlap_dgp(n=1000, gamma=1.0, true_ate=TRUE_ATE, seed=SEED):
    """
    DGP for the propensity overlap scenario.

    Parameters
    ----------
    n       : sample size
    gamma   : overlap divergence. Higher values concentrate propensity
              scores toward 0 and 1, reducing common support.
    true_ate: ground truth average treatment effect
    seed    : random seed for reproducibility

    Returns
    -------
    Y : array (n,)    observed outcomes
    T : array (n,)    binary treatment indicator
    X : array (n, 10) pre-treatment covariates, each column iid N(0,1)
    ps: array (n,)    true propensity scores
    """
    rng = np.random.default_rng(seed)

    # ten standard normal covariates
    X = rng.standard_normal((n, 10))

    # treatment probability driven by X0 (shared confounder), X4 and X5 (treatment only)
    # gamma scales the log-odds: higher gamma -> more extreme probabilities
    ps = expit(gamma * (X[:, 0] + 0.3 * X[:, 4] - 0.2 * X[:, 5]))

    # treatment drawn as independent Bernoulli trials
    T = rng.binomial(1, ps)

    # outcome driven by X0 (shared confounder), X1, X2, X3 (outcome only)
    # X0 is the only true confounder — it appears in both models
    Y = (true_ate * T
         + 3 * X[:, 0]                  # shared confounder, linear
         + 2 * X[:, 1]**2               # outcome only, nonlinear
         + 1.5 * X[:, 0] * X[:, 2]      # outcome only, interaction with confounder
         + 2 * np.sin(X[:, 3])           # outcome only, smooth nonlinearity
         + rng.standard_normal(n))

    return Y, T.astype(float), X, ps

def make_nonlinear_dgp(n=1000, alpha=0.0, true_ate=TRUE_ATE, seed=SEED):
    """
    DGP for the outcome nonlinearity scenario.

    Parameters
    ----------
    n       : sample size
    alpha   : nonlinearity intensity. 0 = purely linear outcome,
              1 = fully nonlinear outcome.
    true_ate: ground truth average treatment effect
    seed    : random seed for reproducibility

    Returns
    -------
    Y : array (n,)    observed outcomes
    T : array (n,)    binary treatment indicator
    X : array (n, 10) pre-treatment covariates, each column iid N(0,1)
    ps: array (n,)    true propensity scores
    """
    rng = np.random.default_rng(seed)

    # ten standard normal covariates
    X = rng.standard_normal((n, 10))

    # treatment assignment: fixed healthy overlap, gamma=1.0
    ps = expit(X[:, 0] + 0.3 * X[:, 4] - 0.2 * X[:, 5])
    T  = rng.binomial(1, ps)

    # linear outcome component — OLS correctly specified at alpha=0
    # X[:,0] is the shared confounder, X[:,1] and X[:,2] outcome only
    f_linear = 3 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2]

    # nonlinear outcome component — each term uses distinct covariates and transformations
    f_nonlinear = (
        2.0 * np.sin(X[:, 0])                        # oscillating in primary confounder
        + 0.4 * X[:, 1]**2                            # quadratic, always positive
        - 0.3 * X[:, 2]**3                            # cubic, sign-changing
        + 1.5 * np.exp(-0.5 * (X[:, 3]**2 + X[:, 4]**2))  # gaussian bump in 2D
        + 1.5 * np.log1p(np.abs(X[:, 5] + X[:, 6]))  # log of sum of two covariates
        + 1.0 * X[:, 7] * X[:, 8]                    # linear interaction
    )

    # alpha interpolates between linear and nonlinear surface
    Y = (true_ate * T
         + (1 - alpha) * f_linear
         + alpha * f_nonlinear
         + rng.standard_normal(n))

    return Y, T.astype(float), X, ps