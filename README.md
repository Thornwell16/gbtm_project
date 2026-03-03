# AutoTraj: Automated Group-Based Trajectory Modeling
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_STREAMLIT_URL_HERE)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

AutoTraj is an open-source, high-performance Python engine for Group-Based Trajectory Modeling (GBTM). Designed for longitudinal epidemiology, it automates the exhaustive search, selection, and visualization of finite mixture models.

By bypassing standard interpreted Python using `numba` Just-In-Time (JIT) C-compilation and fully vectorized analytical Jacobians, AutoTraj evaluates hundreds of combinatorial polynomial grids in seconds.

## Key Features (V1.0 - Logit)
* **Automated Heuristic Selection:** Automatically discards models with spurious subgroups or non-significant highest-order polynomial terms (Nagin & Jones, 2005).
* **Missing Not At Random (MNAR) Capability:** Integrates an optional dropout model, utilizing logistic survival equations conditioned on previous health states to account for informative attrition.
* **Robust Statistics:** Automatically computes Huber-White robust sandwich estimators for standard errors alongside model-based Hessian SEs.
* **Publication Suite:** Instantly generates interactive Plotly graphics, exportable Matplotlib vectors, parameter estimates, AvePP/OCC adequacy metrics, and stratified baseline demographic tables.

## Quick Start (Web App)
You do not need to install Python to use this engine. 
1. Click the **Streamlit App** badge above to launch the web interface.
2. Click **"Load Cambridge Sample Data"** to test the engine immediately, or upload your own longitudinal dataset in wide format.
3. Define your time and outcome variable prefixes.
4. Click **Run AutoTraj Search**.

## Local Installation
If you prefer to run the engine locally:
```bash
git clone [https://github.com/Thornwell16/gbtm_project.git](https://github.com/Thornwell16/gbtm_project.git)
cd gbtm_project
pip install -r requirements.txt
streamlit run app.py
