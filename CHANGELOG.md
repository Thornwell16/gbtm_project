# Changelog

All notable changes to AutoTraj are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## Unreleased

### Fixes

- **Critical: CNORM auto-detect bounds were silently broken in the UI.** The sidebar's CNORM
  Min/Max fields told users to "leave blank to automatically use the dataset's observed min/max,"
  but blank fields defaulted to `0.0`/`0.0` in `app.py`, not `NaN` — and the engine's auto-detect
  logic (`main.py`, all three fitting entry points) only triggers on `None`/`NaN`, never on a
  literal `0.0`. In practice, leaving the bounds blank for any CNORM outcome that doesn't naturally
  span across 0 silently fit a degenerate model (effectively all-censored at a single point), with
  no error raised. Fixed by defaulting the UI's blank state to `np.nan` in all three places
  (single-outcome, and both outcomes in Dual-Trajectory mode) so the engine's existing auto-detect
  behaves as documented. Also added the same "leave blank to auto-detect" caption to the
  Dual-Trajectory Mode CNORM fields, which previously had no such hint at all.
- **MATH.md §3e (informative dropout)** described the dropout hazard as a simple per-wave
  replacement ("last observed time gets `log P(drop)` instead of `log(1-P(drop))`"), which doesn't
  match the actual (correct, gradient-verified) implementation: every observed wave after the first
  contributes an unconditional survival factor, and — only for subjects actually lost to
  follow-up — one *additional* hazard factor is added for the interval just past the last observed
  wave, using the last observed outcome as the lag. Corrected the prose (and the matching §4f
  gradient description) to describe this standard discrete-time-survival structure accurately;
  no code changes were needed, since the implementation was already correct.
- **MATH.md §6** stated the relationship between the two BIC/AIC conventions as
  $\text{BIC}_S = -2\cdot\text{BIC}_N - p\log N$ ("not a simple sign flip") — this is algebraically
  wrong; substituting the definitions shows $\text{BIC}_S = -2\cdot\text{BIC}_N$ exactly (and
  likewise $\text{AIC}_S = -2\cdot\text{AIC}_N$), i.e. it *is* a simple scalar rescaling. Corrected
  the formula; the engine computes both conventions independently and was never affected.

### Documentation

- Removed internal version-milestone labels ("V3.0", "V4.0", "V5.0") from user-facing text
  (README roadmap, in-app sidebar/caption text, MATH.md section headers and prose) — these
  capabilities are now presented as plain features of the product rather than sequential
  milestones. This file (CHANGELOG.md) keeps its version-numbered history, per standard changelog
  convention; only the *forward-facing* narrative changed.
- Added `joint_trajectory_sample.csv`, a synthetic illustrative dataset for trying out
  Dual-Trajectory (Joint) Mode, with a "Load Joint-Trajectory Sample Data" button mirroring the
  existing Cambridge-sample-data pattern. See README's Sample Datasets section.

---

## V5.0 (Unreleased) — Joint Trajectories

### New Features

- **Dual-trajectory (Nagin-style joint) modeling** — two outcomes Y and Z, each with its own
  independent group structure (own group count, polynomial orders, distribution, optional MNAR
  dropout), linked by a joint latent-class probability matrix π_gh (K_Y × K_Z) instead of assuming
  independent group membership. Conditional on class (g,h), Y and Z are conditionally independent —
  P(y_i,z_i|g,h) = P(y_i|g)·P(z_i|h) — so the joint kernel reuses the existing single-outcome
  likelihood/gradient math for each outcome unchanged, weighted by the appropriate marginal
  posterior (P(g|i) for Y's parameters, P(h|i) for Z's).
- **Kernel refactor**: the existing single-outcome kernel's per-group likelihood and gradient loops
  were extracted into two shared `@njit` subroutines, now called by both the single-outcome kernel
  and the new joint kernel — confirmed behavior-preserving by a full regression run before any
  joint-model code was built on top.
- **Two-dimensional label-switching handling** — group-sorting by intercept is generalized to
  resort Y's and Z's groups independently, then reconstruct and re-permute the full π_gh matrix
  along both axes simultaneously (the joint mixing logits are only meaningful relative to the
  reference cell, so they cannot be permuted directly). Verified by a dedicated NLL-invariance
  test using deliberately descending-intercept parameters on both axes.
- New `run_joint_dual_trajectory_model` fitting entry point, `get_joint_subject_assignments`
  (joint + marginal posteriors and hard assignments), and joint/per-outcome-marginal model
  adequacy diagnostics (AvePP, OCC, relative entropy) via the existing, unmodified
  `calc_model_adequacy`.
- Streamlit UI: new "Dual-Trajectory (Joint) Mode" — independent Outcome Y / Outcome Z
  configuration (distribution, CNORM bounds, groups/orders, dropout), a joint probability heatmap,
  row/column-normalized conditional probability tables, a hard-assignment contingency table,
  per-outcome parameter estimates, adequacy diagnostics, and side-by-side fitted trajectory plots.
  Single Model Mode only — no combinatorial search over both outcomes' group/order grids.
- Explicit scope boundaries (deferred as future work): does not compose with V3.0's mixing
  covariates/TVC or V4.0's survey weights in this pass; requires the identical subject-ID set
  across both outcomes (partial outcome-missingness across subjects not yet supported).
- New tests: joint π_gh recovery, conditional-probability recovery, β/assignment recovery, a
  stability check under an independent (non-associated) true π_gh, and a tolerance-based collapse
  check (K_Z=1 reduces to the single-outcome Y model) — a weaker guarantee than V3.0's bit-identical
  P=0,Q=0 invariant, since it's still a distinct, jointly-optimized numerical problem.

### Documentation

- **MATH.md** — new §9 covering the conditional-independence factorization, the joint parameter
  vector layout, joint likelihood/posterior derivations, the β_Y/β_Z gradient proof (showing they
  reduce to the existing single-outcome gradient weighted by the marginal posterior), SE/BIC/AIC
  extensions, the two-dimensional label-switching argument, and the backward-compatibility framing.

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
