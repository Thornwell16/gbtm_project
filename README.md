# AutoTraj: Automated Group-Based Trajectory Modeling
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_STREAMLIT_URL_HERE)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/1171103165.svg)](https://doi.org/10.5281/zenodo.18866776)
[![Tests](https://github.com/Thornwell16/gbtm_project/actions/workflows/test.yml/badge.svg)](https://github.com/Thornwell16/gbtm_project/actions/workflows/test.yml)
[![Lint](https://github.com/Thornwell16/gbtm_project/actions/workflows/lint.yml/badge.svg)](https://github.com/Thornwell16/gbtm_project/actions/workflows/lint.yml)


AutoTraj is an open-source, high-performance Python engine for Group-Based Trajectory Modeling (GBTM). Designed for longitudinal epidemiology, it automates the exhaustive search, selection, and visualization of finite mixture models.

By bypassing standard interpreted Python using `numba` Just-In-Time (JIT) C-compilation and fully vectorized analytical Jacobians, AutoTraj evaluates hundreds of combinatorial polynomial grids in seconds.

## Key Features
* **Multiple Outcome Distributions:** Supports binary (LOGIT), continuous censored (CNORM / Tobit), count (Poisson), and zero-inflated count (ZIP) outcomes — enabling GBTM across a wide range of longitudinal data types.
* **Automated Heuristic Selection:** Automatically discards models with spurious subgroups or non-significant highest-order polynomial terms (Nagin & Jones, 2005).
* **Missing Not At Random (MNAR) Capability:** Integrates an optional dropout model, utilizing logistic survival equations conditioned on previous health states to account for informative attrition.
* **Robust Statistics:** Automatically computes Huber-White robust sandwich estimators for standard errors alongside model-based Hessian SEs.
* **Universal Data Formats:** Supports both Wide format (auto-pivot) and Long format data directly through the UI.
* **Publication Suite:** Instantly generates interactive Plotly graphics, exportable Matplotlib vectors, parameter estimates, AvePP/OCC adequacy metrics, and stratified baseline demographic tables.

## Mathematical Documentation
Full derivations, gradient proofs, and formula references are in [MATH.md](MATH.md).
This document is the technical appendix for the validation paper and covers:
- Log-likelihood functions for each distribution (LOGIT, CNORM, Poisson, ZIP)
- Analytical gradient derivations
- Sandwich (Huber-White) robust SE estimator
- BIC/AIC conventions (Nagin and standard)
- Model adequacy metrics (AvePP, OCC, relative entropy)

## Quick Start (Web App)
You do not need to install Python to use this engine.
1. Click the **Streamlit App** badge above to launch the web interface.
2. Click **"Load Cambridge Sample Data"** to test the engine immediately, or upload your own longitudinal dataset.
3. Select your data format (Wide vs. Long) and map your variables in the sidebar.
4. Click **Run AutoTraj Search**.

## Local Installation
If you prefer to run the engine locally:
```bash
git clone [https://github.com/Thornwell16/gbtm_project.git](https://github.com/Thornwell16/gbtm_project.git)
cd gbtm_project
pip install -r requirements.txt
streamlit run app.py
```

## Running Tests
AutoTraj ships with a comprehensive pytest suite covering parameter recovery,
the Cambridge benchmark, edge cases, and internal consistency.

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all fast tests (recommended)
make test           # or: pytest -m "not slow"

# Run with coverage report (HTML output in htmlcov/)
make coverage

# Run the full suite (includes slow autosearch tests, ~10 minutes)
pytest

# Run only the Cambridge benchmark
make benchmark      # or: pytest tests/test_cambridge_benchmark.py -v

# Run only the parameter-recovery tests
pytest tests/test_parameter_recovery.py -v -m recovery

# Run linter
make lint
```

Test suites:
| Suite | File | Description |
|-------|------|-------------|
| Parameter Recovery | `tests/test_parameter_recovery.py` | 9 tests: simulate known ground-truth, verify AutoTraj recovers parameters |
| Cambridge Benchmark | `tests/test_cambridge_benchmark.py` | 7 tests: validate against Nagin (1999) published results |
| Edge Cases | `tests/test_edge_cases.py` | 12 tests: pathological inputs, boundary conditions, reproducibility |

## Future Roadmap
AutoTraj is actively maintained. Upcoming architecture upgrades include:

* ✅ **Alternative Distributions (V2.0):** Censored Normal (Tobit) for continuous biomarkers and Poisson/ZIP for count data. — *Completed in V2.0*

* **Covariate Architecture (V3.0):** Multinomial baseline risk factors for group membership prediction and Time-Varying Covariates (TVC) for trajectory deflection.

* **Survey Weights (V4.0):** Inverse probability sampling weights for complex stratified national surveys.

* **Joint Trajectories (V5.0):** Dual-trajectory FMM architecture for modeling interacting longitudinal outcomes.

## Methodology & Attribution
Code architecture and UI generation were assisted by Large Language Models, under the strict mathematical direction and validation of the author to ensure alignment with established FMM and GBTM statistical formulas.

Built by **Donald E. Warden, PhD, MPH**

If you utilize this tool in your research, please view the included CITATION.cff for citing the software.
