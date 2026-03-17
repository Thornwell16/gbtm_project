# Changelog

All notable changes to AutoTraj are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

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
