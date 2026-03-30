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

Overlap is degraded by scaling the log-odds of treatment assignment by $\gamma$. At low $\gamma$ assignment is near-random. At high $\gamma$ treated and control units occupy largely separate regions of covariate space.

![Bias and RMSE: Overlap Scenario](images/overlap_bias_rmse.png)

When overlap is healthy all five estimators recover the true ATE reasonably well. As overlap degrades, estimators fail in distinct ways.

- OLS and DML remain flat throughout.
- Flexible RO and IPW accumulate the largest bias: Flexible RO because its forests extrapolate into covariate regions
they were never trained on, IPW because extreme propensity weights concentrate influence on a shrinking set of boundary units.
- AIPW keeps bias near zero but variance grows at high $\gamma$, reflecting weight instability in the doubly robust correction rather than systematic misdirection. 

This demonstrates that the failure mode depends entirely on whether the estimator relies on propensity weighting, outcome modelling, or both.

---

### Scenario 2: Outcome Nonlinearity

The outcome surface interpolates between a purely linear and a fully nonlinear
function via $\alpha$. Treatment assignment is fixed throughout so that functional
form complexity is the only active stressor.

![Bias and RMSE: Nonlinearity Scenario](images/nonlinear_bias_rmse.png)

At $\alpha=0$ all estimators perform well on bias. As nonlinearity increases
the picture shifts in a revealing way.

- OLS accumulates bias and RMSE steadily as functional form misspecification grows.
- IPW is largely unaffected on bias, relying on no outcome model at all, though
  its RMSE stays elevated throughout due to propensity estimation variance.
- Flexible RO starts with the highest bias and RMSE at $\alpha=0$ (driven by
  skewed training subsamples) but both improve as nonlinearity increases. Forest
  flexibility becomes a genuine advantage that outweighs the confounding bias
  from skewed support, and by $\alpha=1.0$ its RMSE is among the lowest.
- AIPW holds up better than DML on bias. The propensity correction partially
  compensates for outcome model misspecification consistent with its doubly
  robust design.
- DML degrades on both bias and RMSE at high $\alpha$, revealing that its
  LassoCV nuisance model cannot fit a nonlinear outcome surface. Replacing it
  with a random forest recovers near-zero bias throughout (see Section 2c).

Notably all estimators converge to similar RMSE values around $0.15$ to $0.17$
at $\alpha=1.0$, suggesting that at full nonlinearity the remaining variance is
driven by the outcome complexity itself rather than estimator design. Even doubly
robust and cross-fitted methods degrade if their nuisance models are too rigid.

---

### Scenario 3: High Dimensionality

The covariate space grows from $p=5$ to $p=200$ while the number of truly
informative covariates stays fixed at $k=5$. Noise covariates are weakly
correlated with the primary confounder, making them look relevant to naive methods.

![Bias and RMSE: Dimensionality Scenario](images/highdim_bias_rmse.png)

At $p=5$ all estimators perform well. As dimensionality grows the picture splits.

- Flexible RO accumulates bias steadily throughout. Random forest splits spread
  across all $p$ covariates, diluting the informative signal and producing
  increasingly unreliable counterfactual predictions.
- IPW drifts gradually as the correlated noise covariates begin to confuse the
  propensity model at higher dimensions.
- AIPW holds up through $p=100$ but deteriorates sharply at $p=200$, with RMSE
  spiking to nearly $5.0$. The propensity model cannot separate true confounders
  from correlated noise at high dimensions, destabilising the IPW correction term.
- OLS and DML remain flat on both bias and RMSE throughout. OLS is well-suited
  to this linear DGP by construction. DML's LassoCV nuisance models identify and
  down-weight the noise covariates, keeping the residualised estimate clean across
  all $p$ values.

High dimensionality exposes a fundamental difference between estimators that
select variables and those that do not. Methods relying on unregularised or
poorly regularised propensity models are particularly vulnerable.

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