# Causal Lab: ATE Recovery Under Controlled DGP with Observable Confounders

![Causal Lab](images/intro.png)

## Summary

- Benchmarks five causal estimators on their ability to recover a known average treatment effect (ATE = 2.0) from synthetic data where all confounders are observed by construction.
- Evaluates performance across three DGP scenarios — propensity score overlap, outcome nonlinearity, and covariate dimensionality — each with a systematic knob modulating stress intensity.
- Demonstrates that even in the selection-on-observables setting, naive methods fail in distinct and predictable ways depending on the structural feature under stress.

---

## What is the project about?

Popular causal inference methods in econometrics and biostatistics are largely
designed to solve one problem: unobserved confounders. Difference-in-differences,
regression discontinuity, instrumental variables — each is a methodological
response to the fact that we cannot observe everything that drives both treatment
and outcome. The ingenuity of these methods lies in finding structure that lets
us identify causal effects despite incomplete information.

But what happens when we observe everything? Or even much more than we need to?
This is the selection-on-observables setting, where all relevant confounders are
measured and the identification problem is, in principle, solved. Condition on $X$
and recover the treatment effect. It should be the easy case.

In practice it is not. Even with full observability, naive methods can fail e.g. through propensity score instability,
functional form misspecification or the curse of dimensionality. The machinery still matters.

This project stress tests five popular ATE estimators under controlled synthetic
DGPs where the ground truth is known. Each scenario isolates a single structural
feature and degrades it systematically while holding everything else fixed,
allowing failures to be attributed cleanly to the estimator's design rather than
incidental data problems.

> When we observe all the relevant confounders, which methods reliably recover the true treatment effect and which ones fail (and why)?

---

## Estimator Suite

Five estimators are evaluated, spanning naive baselines to doubly robust and cross-fitted methods:

| Estimator | Description |
|---|---|
| **OLS** | Linear regression via `statsmodels`. Treatment effect as coefficient on T. |
| **IPW** | Horvitz-Thompson reweighting via logistic propensity scores (`sklearn`). |
| **Flexible RO** | T-learner with two Random Forest outcome models (`sklearn`). |
| **AIPW** | Doubly robust estimator via `econml.dr.LinearDRLearner`. |
| **DML** | Double Machine Learning via `econml.dml.LinearDML` with cross-fitted LassoCV nuisance models. |

All estimators are implemented in `src/ate_suite.py` with fixed specifications across all scenarios.

---

## Scenarios and Key Findings

### Scenario 1: Overlap Degradation

Overlap is degraded by scaling the log-odds of treatment assignment by $\gamma$.
At low $\gamma$ assignment is near-random. At high $\gamma$ treated and control
units occupy largely separate regions of covariate space.

![Bias and RMSE: Overlap Scenario](images/overlap_bias_rmse.png)

- **OLS** and **DML** remain flat throughout. OLS controls for the primary
  confounder directly. DML avoids propensity weighting entirely through
  residualisation.
- **IPW** and **Flexible RO** accumulate the largest bias. IPW through weight
  explosion as propensity scores polarise. Flexible RO because its forests
  extrapolate into covariate regions they were never trained on.
- **AIPW** keeps bias near zero but RMSE grows at high $\gamma$, reflecting
  variance inflation from unstable weights rather than systematic misdirection.

---

### Scenario 2: Outcome Nonlinearity

The outcome surface interpolates between purely linear and fully nonlinear via
$\alpha$. Treatment assignment is fixed throughout.

![Bias and RMSE: Nonlinearity Scenario](images/nonlinear_bias_rmse.png)

- **OLS** accumulates bias steadily as functional form misspecification grows.
- **IPW** is largely unaffected, relying on no outcome model.
- **Flexible RO** starts with the highest bias (driven by skewed training
  subsamples) but improves as nonlinearity increases — forest flexibility
  becomes a genuine advantage that outweighs the confounding bias.
- **AIPW** holds up better than DML, with the propensity correction partially
  compensating for outcome model misspecification.
- **DML** degrades at high $\alpha$ because LassoCV cannot fit a nonlinear
  surface. Replacing it with a random forest recovers near-zero bias (see notebook).

---

### Scenario 3: High Dimensionality

The covariate space grows from $p=5$ to $p=100$ while informative covariates
stay fixed at $k=5$. Noise covariates are weakly correlated with the primary
confounder.

![Bias and RMSE: Dimensionality Scenario](images/highdim_bias_rmse.png)

- **OLS** and **DML** remain flat throughout. OLS benefits from the linear DGP.
  DML's LassoCV nuisance models down-weight noise covariates consistently.
- **IPW** drifts gradually as correlated noise confuses propensity estimation.
- **Flexible RO** accumulates bias steadily as forest splits spread across all
  covariates including noise.
- **AIPW** holds up through $p=50$ then deteriorates sharply at $p=100$.
  Neither IPW nor AIPW have explicit variable selection in their propensity
  models, leaving both exposed as the noise pool grows.

High dimensionality exposes a fundamental difference between estimators that
select variables and those that do not.

---

## Repository Structure
```
causal-lab/
├── src/
│   ├── ate_suite.py          — estimator suite, fixed specifications
│   └── dgp_functions.py      — DGP functions for all three scenarios
├── images/                   — all visualisations referenced in README and notebook
├── causal_lab.ipynb          — main analysis notebook
└── README.md
```

---

## How to Run

1. Create a Python environment (conda recommended)
2. Install dependencies: `pip install numpy pandas matplotlib scipy statsmodels scikit-learn econml joblib tqdm`
3. Run `causal_lab.ipynb` from top to bottom

---

*Knowing all the confounders does not solve your problems. It just changes which problems you have.*