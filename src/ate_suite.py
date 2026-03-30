"""
ate_suite.py
------------
Estimator suite for ATE recovery.
All estimators share the interface fn(Y, T, X, seed) -> float.
Specifications are fixed across all DGP scenarios.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from econml.dml import LinearDML
from econml.dr import LinearDRLearner

SEED = 89
PS_CLIP = (0.05, 0.95)


def ols(Y, T, X, seed=SEED):
    """
    OLS regression via statsmodels.
    Treatment effect recovered as the coefficient on T
    in a linear regression of Y on T and X.
    """
    covariates = sm.add_constant(np.column_stack([T, X]))
    model = sm.OLS(Y, covariates).fit()
    return float(model.params[1])


def ipw(Y, T, X, seed=SEED):
    """
    Horvitz-Thompson IPW estimator.
    Propensity scores estimated via regularised logistic regression (sklearn).
    Scores are clipped to PS_CLIP for numerical stability.
    """
    ps = (
        LogisticRegression(random_state=seed)
        .fit(X, T)
        .predict_proba(X)[:, 1]
    )
    ps = np.clip(ps, *PS_CLIP)
    return float(np.mean(T * Y / ps - (1 - T) * Y / (1 - ps)))


def flexible_ro(Y, T, X, seed=SEED):
    """
    T-learner using two Random Forest outcome models via sklearn.
    Separate models are fit on treated and control units respectively.
    ATE is estimated as the average difference in predicted potential
    outcomes across the full sample.
    """
    m1 = RandomForestRegressor(n_estimators=100, random_state=seed)
    m0 = RandomForestRegressor(n_estimators=100, random_state=seed)
    m1.fit(X[T == 1], Y[T == 1])
    m0.fit(X[T == 0], Y[T == 0])
    return float((m1.predict(X) - m0.predict(X)).mean())


def aipw(Y, T, X, seed=SEED):
    """
    Doubly robust AIPW estimator via EconML LinearDRLearner.
    Propensity model: logistic regression (sklearn).
    Outcome model: LassoCV (sklearn).
    Cross-fitting handled internally by EconML.
    """
    est = LinearDRLearner(
        model_propensity=LogisticRegression(random_state=seed),
        model_regression=LassoCV(cv=5, random_state=seed),
        random_state=seed
    )
    est.fit(Y, T, X=X)
    return float(est.ate(X=X))


def dml(Y, T, X, seed=SEED):
    """
    Double Machine Learning via EconML LinearDML.
    Outcome nuisance model: LassoCV (sklearn).
    Treatment nuisance model: LassoCV (sklearn).
    Cross-fitting handled internally by EconML.
    """
    est = LinearDML(
        model_y=LassoCV(cv=5, random_state=seed),
        model_t=LassoCV(cv=5, random_state=seed),
        random_state=seed
    )
    est.fit(Y, T, X=X)
    return float(est.ate(X=X))

ESTIMATORS = {
    "OLS":              ols,
    "IPW":              ipw,
    "Flexible RO": flexible_ro,
    "AIPW":             aipw,
    "DML":              dml,
}