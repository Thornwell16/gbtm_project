# Changelog

All notable changes to AutoTraj are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## V4.0 (Unreleased) — Survey (Sampling) Weights

### New Features

- **Per-subject survey/sampling weights** — a new optional `weight_col` lets each subject carry
  an inverse-probability (or other survey) weight; the objective becomes the weighted
  pseudo-log-likelihood ℓ_w(θ) = Σ_i w_i·log P(y_i). Unlike V3.0's covariates, weights add **no
  new parameters** — every existing gradient formula applies unchanged, with each subject's
  contribution scaled by w_i before summing. Setting w_i≡1 (or omitting weight_col) reproduces
  V3.0 exactly — verified by a bit-identical regression test.
- **Weighted Huber-White sandwich SEs fall out automatically** — since the per-subject gradient
  row is already w_i-scaled, the existing sandwich "meat" Σ g_i g_iᵀ becomes Σ w_i²g_i g_iᵀ, the
  standard Binder-type weighted-pseudo-MLE variance, with no new computation required. Model-based
  (Hessian-only) SEs are retained for reference but are not a valid inference basis under
  weighting — the UI now warns when a weight column is active.
- New identifiability guards: a weight column that varies within subject, or contains zero,
  negative, or missing values, is rejected with a clear error.
- Streamlit UI: new "V4.0: Survey Weights" sidebar section for selecting a sampling-weight
  column, with input validation and an inference-guidance warning in the estimates tab.
- 4 new tests: a weighted-recovery test demonstrating weights correct sampling bias (recovers
  true population proportions from a deliberately biased/undersampled dataset better than an
  unweighted fit), a backward-compatibility test (all-ones weights ≡ unweighted), and two
  validation-guard tests. Plus a new finite-difference gradient test covering the weighted case.

### Documentation

- **MATH.md** — extended with the weighted log-likelihood, the note that all existing gradient
  formulas apply unchanged per-subject before the w_i scale, the automatic weighted-sandwich
  derivation, and documented limitations (no strata/PSU/design-effect variance; BIC/AIC use raw
  N and p, a simplification consistent with common practice, not a fully resolved question).

---

## V3.0 (Unreleased) — Covariate Architecture

### New Features

- **Multinomial baseline covariates for group membership** — mixing proportions
  generalize from a fixed constant per group to a per-subject multinomial logit
  θ_g(x_i) = Γ_g · x_i on user-supplied time-invariant covariates. Setting no
  covariates (P=0) reproduces V1.5.0 exactly.
- **Time-varying covariates (TVC) for trajectory deflection** — the trajectory
  linear predictor gains an optional Σ_q δ_{g,q}·z_{i,q,t} term, letting
  per-timepoint covariates deflect the fitted curve. Setting no TVCs (Q=0)
  reproduces V1.5.0 exactly.
- Both extensions apply uniformly across all four distributions (LOGIT, CNORM,
  Poisson, ZIP) and are compatible with the informative-dropout sub-model
  (which remains itself independent of the new covariates/TVCs by design).
- New identifiability guard: baseline covariates that vary within subject are
  rejected with a clear error (they should be supplied as a TVC instead).
- Streamlit UI: new "V3.0: Covariate Architecture" sidebar section for
  selecting baseline covariates / TVCs; parameter-estimate table, fitted
  equations, and CI bands updated to display the new Γ/δ blocks.
- 6 new parameter-recovery tests (mixing-covariate recovery, TVC recovery,
  joint recovery, backward-compatibility, null-covariate stability,
  constancy-guard) plus a new automated finite-difference gradient regression
  test covering all 4 distributions and the dropout sub-model.

### Bug Fixes

- **Fixed a subject-processing-order bug in `get_subject_assignments`** — a
  per-subject log-sum-exp intermediate was named `max_val`, silently reusing
  the same name as the CNORM upper censoring bound and corrupting it for every
  subject after the first in the loop. Affected only post-hoc CNORM subject
  assignment/adequacy reporting, not model fitting itself. Added a permanent
  order-invariance regression test.

### Documentation

- **MATH.md** — extended with the full V3.0 parameter layout, log-likelihood,
  and gradient derivations for the mixing-covariate (Γ) and TVC (δ) blocks.
  Also fixed a stale reference to "L-BFGS-B" (the engine has always used
  plain unconstrained BFGS, since all constraints are handled via
  reparameterization).

---

## V1.5.0 (2026-03-16)

### New Features

- **Poisson distribution** — count outcome support via log-link polynomial
  trajectories; full analytical gradient in the JIT kernel.
- **Zero-Inflated Poisson (ZIP) distribution** — per-group structural
  zero-inflation probability (ζ_g logit) estimated alongside trajectory betas.
- **Multiple random starting values** — configurable `n_starts` multi-start
  BFGS restarts; best NLL retained, eliminating most local-optima failures.
- **Confidence bands on trajectory plots** — diagonal-approximation delta-method
  95% CI shading on all Plotly and Matplotlib figures.
- **Full results export package** — ZIP download of all parameter tables,
  adequacy metrics, trajectory figures, and raw posterior assignment CSV.
- **Posterior probability heatmap** — diagnostic tile plot of P(group | subject)
  across all subjects and groups, coloured by assignment certainty.
- **Model equation display** — fitted polynomial equation rendered in LaTeX
  notation for each group in the estimation tab.
- **Input validation for all distribution types** — censoring bounds checked
  for CNORM; count integrality and non-negativity enforced for Poisson/ZIP;
  binary constraint enforced for LOGIT; all checked before optimization begins.
- **Comprehensive test suite — 28 tests passing** across three suites:
  parameter recovery (9), Cambridge benchmark (7), edge cases (12).

### Bug Fixes

- **BIC/AIC now reported in both Nagin and standard conventions** — Nagin BIC
  (higher = better, used for selection) and standard BIC (lower = better, used
  for reporting) now computed and displayed side-by-side.
- **CNORM sigma gradient chain rule verified** — corrected the partial derivative
  ∂ℓ/∂(log σ) to properly apply the chain rule through the inverse Mills ratio.
- **Groups sorted by ascending intercept** — `sort_groups_by_intercept` applied
  after every model fit, eliminating label-switching across restarts.
- **Overparameterized models hard-stopped** — models where polynomial order
  exceeds the number of distinct time points are now rejected before fitting
  rather than silently converging to a degenerate solution.

### Documentation

- **MATH.md** — complete technical appendix covering log-likelihood functions
  for all four distributions, analytical gradient derivations, Hessian/sandwich
  SE computation, BIC/AIC conventions, and adequacy metric formulas.
- **Function-level docstrings throughout main.py** — Google-style docstrings on
  all ~20 public functions; 50-line docstring on the JIT kernel with per-line
  mathematical annotations.
- **Module docstring** — top-of-file docstring in both main.py and app.py
  summarising parameterization, time scaling, optimization, and BIC conventions.
- **README updated** — new Key Features section, Running Tests instructions with
  Makefile targets, Mathematical Documentation section, V1.5 roadmap update.
- **CONTRIBUTING.md** — developer setup, test-running guide, code style rules.

### Infrastructure

- **GitHub Actions CI** — `test.yml` (pytest fast suite + coverage) and
  `lint.yml` (flake8 + syntax check) triggered on push and pull requests.
- **Makefile** — `make test`, `make lint`, `make coverage`, `make benchmark`,
  `make simulate`, `make clean` targets.
- **requirements-dev.txt** extended — `pytest-cov` and `flake8` added.
- **.flake8** — project-wide linter configuration with relaxed settings
  appropriate for scientific/JIT-compiled code.
- **Benchmark report generator** — `tests/generate_benchmark_report.py`
  produces `benchmark_report.md`, `benchmark_results.csv`, and four trajectory
  figures; used as core evidence for the validation paper.

---

## V1.0.0 (2026-03-03)

- Initial release with LOGIT (binary) and CNORM (censored normal / Tobit)
  outcome distributions.
- Automated exhaustive search over group count and polynomial order combinations
  (AutoTraj).
- Nagin (2005) adequacy metrics: AvePP, OCC, relative entropy.
- Huber-White robust sandwich standard errors alongside model-based Hessian SEs.
- Streamlit web application with Wide and Long format data ingestion.
- Cambridge Study of Delinquent Development sample dataset bundled.
