# Causal Lab: ATE Recovery Under Controlled DGP with Observable Confounders

![Causal Lab](images/intro.png)

## Summary

- Benchmarks five causal estimators on their ability to recover a known average treatment effect (ATE = 2.0) from synthetic data where all confounders are observed by construction.
- Evaluates performance across three DGP scenarios: (i) overlap degradation; (ii) outcome nonlinearity; and (iii) high dimensionality — each with a systematic knob modulating stress intensity.
- Demonstrates that even in the selection-on-observables setting, naive methods fail in distinct and predictable ways depending on the structural feature under stress.

---

## What is the project about?

Causal inference methods address two distinct challenges: identification and estimation. Design-based approaches such as difference-in-differences, regression discontinuity, and instrumental variables target the identification problem by recovering causal effects in the presence of unobserved confounding. 

But what happens when we observe everything? Or even much more than we need to?

This is the selection-on-observables setting, where all relevant confounders are measured and the identification problem is, in principle, solved. Conditional on $X$, treatment is as good as random, and the causal effect is identified. What remains is not an identification problem, but an estimation problem.

In practice, this “easy case” is not easy at all. Even with full observability, naive methods can fail through structural issues such as propensity score instability, functional form misspecification, or the curse of dimensionality. The choice of estimator and the assumptions it embeds still matter.

This project stress tests five popular ATE estimators under controlled synthetic DGPs where the ground truth is known. Each scenario isolates a single structural feature and degrades it systematically while holding everything else fixed, allowing failures to be attributed cleanly to the estimator’s design rather than incidental data problems.

> When identification is guaranteed, which estimators remain robust as key structural features of the data (namely overlap, functional form, and dimensionality) are systematically stressed?

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

- **OLS** and **DML** remain broadly stable throughout. OLS shows near-zero bias with only a modest increase in RMSE, while DML stays close to zero bias and exhibits similarly little deterioration as overlap worsens.
- **IPW** and **Flexible RO** accumulate the largest bias and RMSE. IPW deteriorates through weight explosion as propensity scores polarise, while Flexible RO struggles because treated and control forests are trained on increasingly separated supports, making predictions unstable in poorly represented regions.
- **AIPW** keeps bias close to zero across most of the grid, but RMSE rises sharply at high $\gamma$, reflecting variance inflation from unstable weights rather than systematic bias.

---

### Scenario 2: Outcome Nonlinearity

The outcome surface interpolates between purely linear and fully nonlinear via
$\alpha$. Treatment assignment is fixed throughout.

![Bias and RMSE: Nonlinearity Scenario](images/nonlinear_bias_rmse.png)

- **OLS** accumulates bias and RMSE steadily as functional form misspecification grows.
- **IPW** is largely unaffected, relying on no outcome model. Its bias remains low and stable throughout, while RMSE increases only modestly with $\alpha$.
- **Flexible RO** starts with the highest bias and RMSE, but improves markedly as nonlinearity increases, suggesting that forest flexibility becomes a genuine advantage once the outcome surface departs sufficiently from linearity.
- **AIPW** holds bias near zero across the grid and performs better than DML on bias at high $\alpha$, but its RMSE rises steadily and remains among the highest in the nonlinear regime.
- **DML** degrades as $\alpha$ increases because LassoCV cannot fit a nonlinear outcome surface. Bias and RMSE both rise sharply at high $\alpha$, although replacing the linear nuisance model with a random forest restores near-zero bias (see notebook).

---

### Scenario 3: High Dimensionality

The covariate space grows from $p=5$ to $p=100$ while informative covariates
stay fixed at $k=5$. Noise covariates are weakly correlated with the primary
confounder.

![Bias and RMSE: Dimensionality Scenario](images/highdim_bias_rmse.png)

- **OLS** and **DML** remain broadly flat throughout. OLS performs well because the DGP is linear and correctly specified, while DML’s LassoCV nuisance models down-weight noise covariates effectively.
- **IPW** bias and RMSE drift gradually upward as correlated noise increasingly confuses propensity estimation. Without variable selection, the propensity model absorbs noise alongside signal.
- **Flexible RO** carries the highest bias throughout, rising steadily as forest splits spread across both informative and noise covariates, diluting the signal. RMSE is systematically elevated across all $p$ values.
- **AIPW** holds up at low $p$ but deteriorates from $p=50$ onwards on both bias and RMSE, with performance worsening sharply at $p=100$. Like IPW, its propensity component has no explicit variable selection mechanism and becomes increasingly unstable as the noise pool grows.

---

## Key Takeaway

When all relevant confounders are observed, the challenge of causal inference does not disappear, it just shifts from identification to estimation.

This project makes that shift explicit by showing that estimator choice is ultimately a question of which assumptions are most credible, and which failure modes are most tolerable, in a given data environment.

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

## Disclaimer

*This project is an exploratory experiment. All results are specific to the DGP specifications, sample sizes, estimator implementations, and random seeds used here, and should not be interpreted as general or theoretical conclusions about estimator performance.*
