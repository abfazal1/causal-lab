# Causal Lab: ATE Recovery Under Controlled DGP with Observable Confounders

![Causal Lab](images/intro.png)

## Summary

- Benchmarks five causal estimators on their ability to recover a known average treatment effect (ATE = 2.0) from synthetic data where all confounders are observed by construction.
- Evaluates performance across three DGP scenarios — propensity score overlap, outcome nonlinearity, and covariate dimensionality — each with a systematic knob modulating stress intensity.
- Demonstrates that even in the selection-on-observables setting, naive methods fail in distinct and predictable ways depending on the structural feature under stress.

---

## What is the project about?

Popular causal inference methods in econometrics and biostatistics are largely
designed to overcome the problem of unobserved confounders. For instance:
- Difference-in-differences
exploits parallel trends to difference out time-invariant unobservables. 
- Regression
discontinuity leverages quasi-random variation near a threshold. 
- Instrumental
variables find exogenous variation that bypasses the confounded channel entirely.

Each is a methodological response to the same fundamental problem: we cannot observe everything that drives both treatment and outcome.

The selection-on-observables framework sets that problem aside. No instruments,
no panel structure, no unobserved confounders — just condition on X and recover
the treatment effect. In an ideal world where all relevant confounders are
measured, this should be the easy case. In practice, even this idealised setting
produces unreliable estimates when standard methods are applied naively.

This project stress tests five popular ATE estimators under controlled synthetic DGPs where the ground truth is known. Each scenario isolates a single structural feature — propensity score overlap, outcome nonlinearity, and covariate dimensionality — and degrades each systematically while holding everything else fixed. This allows failures to be attributed cleanly to the estimator's design rather than incidental data problems.

The central question:

> When we observe all the relevant confounders, which methods reliably recover the true treatment effect — and which ones fail, and why?

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

### Scenario 1: Propensity Score Overlap

Overlap is degraded by scaling the log-odds of treatment assignment by $\gamma$. At low $\gamma$ assignment is near-random. At high $\gamma$ treated and control units occupy largely separate regions of covariate space.

![Bias and RMSE: Overlap Scenario](images/overlap_bias_rmse.png)

When overlap is healthy all five estimators recover the true ATE reasonably well. As overlap degrades, estimators fail in distinct ways.

- Flexible RO accumulates the largest bias since without any selection correction its outcome models extrapolate across covariate regions they were never trained on.
- IPW fails through weight explosion: as propensity scores polarise, a shrinking set of boundary units dominates the estimate.
- AIPW preserves near-zero bias throughout but variance inflates sharply at high $\gamma$, making it unreliable on any single dataset even when correct on average.
- OLS and DML are the most resilient. OLS because the primary confounder enters linearly and is directly controlled, DML because it avoids propensity weighting entirely through residualisation.

This demonstrates that overlap violation does not affect all estimators equally: the failure mode depends entirely on whether the estimator relies on propensity weighting, outcome modelling, or both.

---

### Scenario 2: Outcome Nonlinearity

The outcome surface interpolates between a purely linear and a fully nonlinear function via a mixing parameter $\alpha$. Treatment assignment is held fixed with healthy overlap so that functional form complexity is the only active stressor.

![Bias and RMSE: Nonlinearity Scenario](images/nonlinear_bias_rmse.png)

When the outcome is linear all estimators perform well. As $\alpha$ increases OLS accumulates bias steadily, explaining only half the outcome variance by $\alpha=1.0$.

- Flexible RO improves relative to OLS since its random forest outcome models handle complex surfaces better than a linear model.
- IPW is largely unaffected, relying on no outcome model at all.
- AIPW holds up better than DML despite also using LassoCV internally, with the propensity correction partially compensating for outcome model misspecification consistent with its doubly robust design.
- DML with linear nuisance models degrades unexpectedly, revealing that its performance is contingent on first-stage model quality. Replacing LassoCV with a random forest in the DML nuisance step recovers near-zero bias throughout.

This demonstrates that when the outcome surface is genuinely nonlinear, flexibility in the estimation procedure matters as much as the theoretical guarantees behind it. Even doubly robust and cross-fitted methods like DML can degrade if their nuisance models are too rigid to approximate the true surface.

---

### Scenario 3: Covariate Dimensionality

TBC

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