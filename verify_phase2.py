#!/usr/bin/env python3
"""
Phase 2 Verification Script for AutoTraj
=========================================
Run this AFTER completing Prompts 5-8 (Poisson, ZIP, Visualization, Diagnostics).
Run BEFORE moving to Phase 3 (Validation Suite, Prompts 9-12).

Usage:
    python verify_phase2.py

Phase 2 added:
  - Prompt 5: Poisson distribution engine
  - Prompt 6: Zero-Inflated Poisson (ZIP) engine
  - Prompt 7: Visualization & export improvements
  - Prompt 8: Diagnostic improvements

If all checks pass, you're clear to proceed to Phase 3.
"""

import sys
import os
import traceback
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    prep_trajectory_data,
    run_autotraj,
    run_single_model,
    get_subject_assignments,
    calc_model_adequacy,
    extract_flat_arrays,
    create_design_matrix_jit,
)

PASS = 0
FAIL = 0
WARN = 0
CRITICAL_FAIL = False

def check(name, condition, detail="", critical=False):
    global PASS, FAIL, CRITICAL_FAIL
    if condition:
        print(f"  ✅ PASS: {name}")
        PASS += 1
    else:
        print(f"  ❌ FAIL: {name}")
        if detail:
            print(f"         {detail}")
        FAIL += 1
        if critical:
            CRITICAL_FAIL = True

def warn(name, detail=""):
    global WARN
    print(f"  ⚠️  WARN: {name}")
    if detail:
        print(f"         {detail}")
    WARN += 1

def section(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")


# =========================================================================
# HELPER: Simulate Poisson data with known parameters
# =========================================================================
def simulate_poisson_data(n_subjects=300, n_times=10, groups=None, proportions=None, seed=42):
    """Generate Poisson trajectory data with known ground truth."""
    np.random.seed(seed)
    if groups is None:
        # 2 groups: low-count and high-count
        groups = [
            {'betas': [0.5, 0.3]},    # Group 1: exp(0.5 + 0.3t) ~ counts 1-3
            {'betas': [2.0, -0.2]},    # Group 2: exp(2.0 - 0.2t) ~ counts 5-10
        ]
    if proportions is None:
        proportions = [0.6, 0.4]

    times = np.linspace(-1, 1, n_times)
    records = []
    true_groups = {}

    for i in range(1, n_subjects + 1):
        g = np.random.choice(len(groups), p=proportions)
        true_groups[i] = g + 1
        betas = groups[g]['betas']
        for t in times:
            eta = sum(betas[p] * (t ** p) for p in range(len(betas)))
            mu = np.exp(np.clip(eta, -20, 20))
            y = np.random.poisson(mu)
            records.append({'ID': i, 'Time': t, 'Outcome': float(y)})

    df = pd.DataFrame(records).sort_values(['ID', 'Time'])
    return df, true_groups, groups, proportions


# =========================================================================
# HELPER: Simulate ZIP data with known parameters
# =========================================================================
def simulate_zip_data(n_subjects=300, n_times=10, groups=None, proportions=None,
                      zero_inflation=None, seed=42):
    """Generate ZIP trajectory data with known ground truth."""
    np.random.seed(seed)
    if groups is None:
        groups = [
            {'betas': [1.0, 0.5]},
            {'betas': [2.5, -0.3]},
        ]
    if proportions is None:
        proportions = [0.6, 0.4]
    if zero_inflation is None:
        zero_inflation = [0.3, 0.1]  # 30% and 10% structural zeros

    times = np.linspace(-1, 1, n_times)
    records = []
    true_groups = {}

    for i in range(1, n_subjects + 1):
        g = np.random.choice(len(groups), p=proportions)
        true_groups[i] = g + 1
        betas = groups[g]['betas']
        omega = zero_inflation[g]
        for t in times:
            eta = sum(betas[p] * (t ** p) for p in range(len(betas)))
            mu = np.exp(np.clip(eta, -20, 20))
            if np.random.random() < omega:
                y = 0  # structural zero
            else:
                y = np.random.poisson(mu)
            records.append({'ID': i, 'Time': t, 'Outcome': float(y)})

    df = pd.DataFrame(records).sort_values(['ID', 'Time'])
    return df, true_groups, groups, proportions, zero_inflation


# =========================================================================
# TEST 1: Poisson engine exists and basic functions are importable
# =========================================================================
section("TEST 1: Poisson Engine — Import & Function Checks")

poisson_nll_exists = False
poisson_jac_exists = False
poisson_grad_exists = False

try:
    from main import calc_poisson_dynamic_nll_jit
    poisson_nll_exists = True
    check("calc_poisson_dynamic_nll_jit exists", True)
except ImportError:
    check("calc_poisson_dynamic_nll_jit exists", False,
          "Function not found in main.py — Prompt 5 may not have been applied", critical=True)

try:
    from main import calc_poisson_dynamic_jacobian_jit
    poisson_jac_exists = True
    check("calc_poisson_dynamic_jacobian_jit exists", True)
except ImportError:
    check("calc_poisson_dynamic_jacobian_jit exists", False,
          "Function not found — needed for BFGS optimization", critical=True)

try:
    from main import calc_poisson_subject_gradients_jit
    poisson_grad_exists = True
    check("calc_poisson_subject_gradients_jit exists", True)
except ImportError:
    warn("calc_poisson_subject_gradients_jit not found",
         "May have a different name or be inlined. Robust SEs need subject-level gradients.")


# =========================================================================
# TEST 2: Poisson 1-group model on simulated data
# =========================================================================
section("TEST 2: Poisson — 1-Group Model Recovery")

if poisson_nll_exists:
    try:
        # Simple 1-group Poisson: intercept=1.5 means mu=exp(1.5)≈4.5 counts
        np.random.seed(42)
        n, T = 200, 10
        times = np.linspace(-1, 1, T)
        ids = np.repeat(np.arange(1, n+1), T)
        t_all = np.tile(times, n)
        true_beta0 = 1.5
        mu = np.exp(true_beta0 + 0.0 * t_all)
        y = np.random.poisson(mu)
        pois_df_1g = pd.DataFrame({'ID': ids, 'Time': t_all, 'Outcome': y.astype(float)})
        pois_df_1g = pois_df_1g.sort_values(['ID', 'Time'])

        m_pois1 = run_single_model(pois_df_1g, [0], use_dropout=False, dist='POISSON')

        check("Poisson 1-group converges",
              m_pois1['result'].success or m_pois1['result'].status == 2, critical=True)
        check("Poisson LL is finite", np.isfinite(m_pois1['ll']),
              f"LL = {m_pois1.get('ll', 'N/A')}")

        if m_pois1['result'].success or m_pois1['result'].status == 2:
            # Extract intercept — for 1 group, thetas are empty, intercept is first beta
            k = 1
            recovered_beta0 = m_pois1['result'].x[k - 1]  # index 0 for 1-group
            error_pct = abs(recovered_beta0 - true_beta0) / abs(true_beta0)

            check(f"Poisson intercept recovery (true={true_beta0}, got={recovered_beta0:.3f})",
                  error_pct < 0.20,
                  f"Error = {error_pct:.1%}", critical=True)

            # Mean outcome should be close to exp(intercept)
            pred_mean = np.exp(recovered_beta0)
            obs_mean = y.mean()
            check(f"Predicted mean ≈ observed mean (pred={pred_mean:.2f}, obs={obs_mean:.2f})",
                  abs(pred_mean - obs_mean) / obs_mean < 0.15)

    except Exception as e:
        check("Poisson 1-group execution", False, f"Crashed: {e}", critical=True)
        traceback.print_exc()
else:
    warn("Skipping Poisson 1-group test — NLL function not found")


# =========================================================================
# TEST 3: Poisson 2-group model recovery
# =========================================================================
section("TEST 3: Poisson — 2-Group Model Recovery")

if poisson_nll_exists:
    try:
        pois_df, true_grp, true_params, true_pis = simulate_poisson_data(
            n_subjects=400, n_times=12, seed=123
        )

        m_pois2 = run_single_model(pois_df, [1, 1], use_dropout=False, dist='POISSON')

        check("Poisson 2-group converges",
              m_pois2['result'].success or m_pois2['result'].status == 2, critical=True)
        check("Poisson 2-group LL is finite", np.isfinite(m_pois2['ll']),
              f"LL = {m_pois2.get('ll', 'N/A')}")

        if m_pois2['result'].success or m_pois2['result'].status == 2:
            pis = m_pois2['pis']
            check("Group proportions sum to ~1", abs(sum(pis) - 1.0) < 0.01,
                  f"Sum = {sum(pis):.4f}")
            check("Two distinct groups found", abs(pis[0] - pis[1]) > 0.05,
                  f"Proportions: {[f'{p:.1%}' for p in pis]}")

            # Check assignments
            assign_df = get_subject_assignments(m_pois2, pois_df)
            check("Assignments computed", len(assign_df) > 0)

            if len(assign_df) > 0:
                # Check assignment accuracy
                n_correct = 0
                n_total = 0
                for _, row in assign_df.iterrows():
                    sid = row['ID']
                    if sid in true_grp:
                        n_total += 1
                        # Groups might be in opposite order
                        if row['Assigned_Group'] == true_grp[sid]:
                            n_correct += 1
                if n_total > 0:
                    acc = n_correct / n_total
                    # If accuracy < 50%, groups are probably flipped
                    acc = max(acc, 1 - acc)
                    check(f"Assignment accuracy > 70% (got {acc:.1%})", acc > 0.70,
                          "Low accuracy may indicate parameter recovery issues")

    except Exception as e:
        check("Poisson 2-group execution", False, f"Crashed: {e}", critical=True)
        traceback.print_exc()
else:
    warn("Skipping Poisson 2-group test — NLL function not found")


# =========================================================================
# TEST 4: Poisson autosearch selects correct number of groups
# =========================================================================
section("TEST 4: Poisson — AutoSearch Model Selection")

if poisson_nll_exists:
    try:
        pois_df_search, _, _, _ = simulate_poisson_data(
            n_subjects=400, n_times=12, seed=456
        )

        top_models, all_eval = run_autotraj(
            pois_df_search, min_groups=1, max_groups=3,
            min_order=0, max_order=1, min_group_pct=5.0,
            use_dropout=False, dist='POISSON'
        )

        check("AutoSearch returns results", len(top_models) > 0,
              f"Got {len(top_models)} valid models from {len(all_eval)} evaluated")

        if len(top_models) > 0:
            best_k = len(top_models[0]['orders'])
            check(f"Best model has 2 groups (got {best_k})", best_k == 2,
                  "Data was simulated from 2 groups — BIC should prefer 2")

            if best_k != 2:
                warn("BIC did not select the true K=2",
                     "This can happen with small samples or poorly separated groups. "
                     "Check if 2-group model is in the valid set at all.")
                has_2group = any(len(m['orders']) == 2 for m in top_models)
                if has_2group:
                    warn("2-group model IS in valid set, just not ranked first")

    except Exception as e:
        check("Poisson autosearch execution", False, f"Crashed: {e}")
        traceback.print_exc()
else:
    warn("Skipping Poisson autosearch test")


# =========================================================================
# TEST 5: ZIP engine exists
# =========================================================================
section("TEST 5: ZIP Engine — Import & Function Checks")

zip_nll_exists = False
zip_jac_exists = False

try:
    from main import calc_zip_dynamic_nll_jit
    zip_nll_exists = True
    check("calc_zip_dynamic_nll_jit exists", True)
except ImportError:
    check("calc_zip_dynamic_nll_jit exists", False,
          "Function not found — Prompt 6 may not have been applied", critical=True)

try:
    from main import calc_zip_dynamic_jacobian_jit
    zip_jac_exists = True
    check("calc_zip_dynamic_jacobian_jit exists", True)
except ImportError:
    check("calc_zip_dynamic_jacobian_jit exists", False,
          "Jacobian function not found", critical=True)

# Check that ZIP is wired into run_single_model
try:
    import inspect
    src = inspect.getsource(run_single_model)
    zip_in_engine = 'ZIP' in src or 'zip' in src
    check("run_single_model handles ZIP distribution", zip_in_engine,
          "The string 'ZIP' should appear in run_single_model's source code")
except Exception:
    warn("Could not inspect run_single_model source")


# =========================================================================
# TEST 6: ZIP 1-group model
# =========================================================================
section("TEST 6: ZIP — 1-Group Model Recovery")

if zip_nll_exists:
    try:
        np.random.seed(99)
        n, T = 200, 10
        times = np.linspace(-1, 1, T)
        ids = np.repeat(np.arange(1, n+1), T)
        t_all = np.tile(times, n)
        true_beta0 = 1.0
        true_omega = 0.25  # 25% structural zeros
        mu = np.exp(true_beta0)

        y = []
        for i in range(len(t_all)):
            if np.random.random() < true_omega:
                y.append(0.0)
            else:
                y.append(float(np.random.poisson(mu)))
        y = np.array(y)

        zip_df_1g = pd.DataFrame({'ID': ids, 'Time': t_all, 'Outcome': y})
        zip_df_1g = zip_df_1g.sort_values(['ID', 'Time'])

        # Check zero fraction in data
        zero_frac = (y == 0).mean()
        print(f"  ℹ️  Data zero fraction: {zero_frac:.1%} (structural={true_omega:.0%} + Poisson zeros)")

        m_zip1 = run_single_model(zip_df_1g, [0], use_dropout=False, dist='ZIP')

        check("ZIP 1-group converges",
              m_zip1['result'].success or m_zip1['result'].status == 2, critical=True)
        check("ZIP LL is finite", np.isfinite(m_zip1['ll']),
              f"LL = {m_zip1.get('ll', 'N/A')}")

        if m_zip1['result'].success or m_zip1['result'].status == 2:
            params = m_zip1['result'].x
            print(f"  ℹ️  ZIP parameters: {[f'{p:.4f}' for p in params]}")

            # Basic sanity: LL should be reasonable (not near 0 or extremely negative)
            check("ZIP LL is reasonable", m_zip1['ll'] < -100,
                  f"LL = {m_zip1['ll']:.2f} — should be substantially negative for count data")

    except Exception as e:
        check("ZIP 1-group execution", False, f"Crashed: {e}", critical=True)
        traceback.print_exc()
else:
    warn("Skipping ZIP tests — NLL function not found")


# =========================================================================
# TEST 7: ZIP 2-group model
# =========================================================================
section("TEST 7: ZIP — 2-Group Model Recovery")

if zip_nll_exists:
    try:
        zip_df, true_grp_z, true_params_z, true_pis_z, true_zi = simulate_zip_data(
            n_subjects=400, n_times=10, seed=789
        )

        m_zip2 = run_single_model(zip_df, [1, 1], use_dropout=False, dist='ZIP')

        check("ZIP 2-group converges",
              m_zip2['result'].success or m_zip2['result'].status == 2)
        check("ZIP 2-group LL is finite", np.isfinite(m_zip2['ll']),
              f"LL = {m_zip2.get('ll', 'N/A')}")

        if m_zip2['result'].success or m_zip2['result'].status == 2:
            pis = m_zip2['pis']
            check("ZIP group proportions sum to ~1", abs(sum(pis) - 1.0) < 0.01)
            check("ZIP finds two distinct groups", abs(pis[0] - pis[1]) > 0.05,
                  f"Proportions: {[f'{p:.1%}' for p in pis]}")

    except Exception as e:
        check("ZIP 2-group execution", False, f"Crashed: {e}")
        traceback.print_exc()
else:
    warn("Skipping ZIP 2-group test")


# =========================================================================
# TEST 8: Poisson vs ZIP model comparison
# =========================================================================
section("TEST 8: ZIP Data — ZIP Model Should Beat Poisson Model")

if zip_nll_exists and poisson_nll_exists:
    try:
        # Use the ZIP-simulated data — a ZIP model should fit better than plain Poisson
        # because the data has structural zeros
        zip_df_comp, _, _, _, _ = simulate_zip_data(
            n_subjects=300, n_times=10,
            zero_inflation=[0.35, 0.15],  # heavy zero inflation
            seed=101
        )

        m_pois_on_zip = run_single_model(zip_df_comp, [1, 1], use_dropout=False, dist='POISSON')
        m_zip_on_zip = run_single_model(zip_df_comp, [1, 1], use_dropout=False, dist='ZIP')

        pois_converged = m_pois_on_zip['result'].success or m_pois_on_zip['result'].status == 2
        zip_converged = m_zip_on_zip['result'].success or m_zip_on_zip['result'].status == 2

        if pois_converged and zip_converged:
            pois_ll = m_pois_on_zip['ll']
            zip_ll = m_zip_on_zip['ll']

            check(f"ZIP LL ({zip_ll:.1f}) > Poisson LL ({pois_ll:.1f}) on ZIP data",
                  zip_ll > pois_ll,
                  "ZIP should fit better on data with structural zeros")

            # BIC comparison (both Nagin convention — higher is better)
            pois_bic = m_pois_on_zip['bic'] if 'bic' in m_pois_on_zip else m_pois_on_zip.get('bic_nagin')
            zip_bic = m_zip_on_zip['bic'] if 'bic' in m_zip_on_zip else m_zip_on_zip.get('bic_nagin')

            if pois_bic is not None and zip_bic is not None:
                check(f"ZIP BIC preferred over Poisson BIC",
                      zip_bic > pois_bic,
                      f"ZIP BIC={zip_bic:.1f}, Poisson BIC={pois_bic:.1f}")
        else:
            warn("Could not compare — one or both models failed to converge",
                 f"Poisson converged: {pois_converged}, ZIP converged: {zip_converged}")

    except Exception as e:
        check("Poisson vs ZIP comparison", False, f"Crashed: {e}")
        traceback.print_exc()
else:
    warn("Skipping Poisson vs ZIP comparison — missing engines")


# =========================================================================
# TEST 9: Existing distributions still work (regression test)
# =========================================================================
section("TEST 9: Regression — LOGIT and CNORM Still Work")

try:
    raw_df = pd.read_csv("cambridge.txt", sep=r'\s+', encoding='utf-8-sig')
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    long_df = prep_trajectory_data(raw_df, 'ID', 'C', 'T').dropna(subset=['Time', 'Outcome'])

    # LOGIT regression test
    m_logit = run_single_model(long_df, [1, 1], use_dropout=False, dist='LOGIT')
    check("LOGIT still converges after Phase 2 changes",
          m_logit['result'].success or m_logit['result'].status == 2, critical=True)
    check("LOGIT LL is finite and negative",
          np.isfinite(m_logit['ll']) and m_logit['ll'] < 0,
          f"LL = {m_logit.get('ll', 'N/A')}")

    # CNORM regression test
    np.random.seed(42)
    n = 150
    ids_c = np.repeat(np.arange(1, n+1), 8)
    t_c = np.tile(np.linspace(-1, 1, 8), n)
    y_c = np.random.normal(-0.5 + 1.0 * t_c, 0.5)
    cnorm_df = pd.DataFrame({'ID': ids_c, 'Time': t_c, 'Outcome': y_c}).sort_values(['ID', 'Time'])

    m_cnorm = run_single_model(cnorm_df, [1], use_dropout=False, dist='CNORM',
                                cnorm_min=y_c.min(), cnorm_max=y_c.max())
    check("CNORM still converges after Phase 2 changes",
          m_cnorm['result'].success or m_cnorm['result'].status == 2, critical=True)
    check("CNORM LL is finite",
          np.isfinite(m_cnorm['ll']), f"LL = {m_cnorm.get('ll', 'N/A')}")

except Exception as e:
    check("Regression tests", False, f"Crashed: {e}", critical=True)
    traceback.print_exc()


# =========================================================================
# TEST 10: Visualization functions exist (Prompt 7)
# =========================================================================
section("TEST 10: Visualization & Export Improvements (Prompt 7)")

# We can't render Streamlit from a script, but we can check for key
# indicators that Prompt 7 was applied

try:
    with open("app.py", "r") as f:
        app_source = f.read()

    # Check for confidence band code
    has_confidence = any(kw in app_source.lower() for kw in
                         ['confidence', 'ci_lower', 'ci_upper', 'fill_between',
                          'fill', 'ribbon', 'band', 'delta method', 'se_pred'])
    if has_confidence:
        check("Confidence bands code found in app.py", True)
    else:
        warn("No confidence band keywords found in app.py",
             "Prompt 7 should have added 95% CI ribbons. Check app.py manually.")

    # Check for SVG/PNG download
    has_svg_download = 'svg' in app_source.lower() and 'download' in app_source.lower()
    has_png_download = ('png' in app_source.lower() or '300' in app_source) and 'download' in app_source.lower()
    if has_svg_download or has_png_download:
        check("Plot download functionality found", True)
    else:
        warn("No SVG/PNG download keywords found",
             "Prompt 7 should have added downloadable plot exports.")

    # Check for ZIP export package
    has_zip_export = 'zipfile' in app_source.lower() or 'BytesIO' in app_source
    if has_zip_export:
        check("ZIP results package export found", True)
    else:
        warn("No zipfile/BytesIO keywords found",
             "Prompt 7 should have added a full results ZIP download.")

    # Check for model equation display
    has_equation = 'st.latex' in app_source or 'equation' in app_source.lower()
    if has_equation:
        check("Model equation display found", True)
    else:
        warn("No st.latex or equation display found",
             "Prompt 7 should have added LaTeX equation display.")

    # Check for colored spaghetti plots
    has_colored_spaghetti = ('spaghetti' in app_source.lower() and
                              ('assigned_group' in app_source.lower() or
                               'group_color' in app_source.lower() or
                               'color' in app_source.lower()))
    if has_colored_spaghetti:
        check("Colored spaghetti plot logic found", True)
    else:
        warn("Spaghetti plots may still be gray",
             "Prompt 7 should have colored individual trajectories by assigned group.")

except FileNotFoundError:
    check("app.py exists", False, "Cannot find app.py in current directory", critical=True)
except Exception as e:
    warn(f"Could not inspect app.py: {e}")


# =========================================================================
# TEST 11: Diagnostic improvements (Prompt 8)
# =========================================================================
section("TEST 11: Diagnostic Improvements (Prompt 8)")

try:
    with open("app.py", "r") as f:
        app_source = f.read()

    # Check for posterior probability heatmap
    has_heatmap = any(kw in app_source.lower() for kw in
                       ['heatmap', 'imshow', 'posterior.*matrix', 'confusion',
                        'ff.Heatmap', 'go.Heatmap'])
    if has_heatmap:
        check("Posterior probability heatmap found", True)
    else:
        warn("No heatmap keywords found in app.py",
             "Prompt 8 should have added a posterior probability heatmap.")

    # Check for observed vs estimated diagnostic
    has_obs_vs_est = any(kw in app_source.lower() for kw in
                          ['observed.*estimated', 'obs.*est', 'residual',
                           'diagnostic', 'fit.*plot'])
    if has_obs_vs_est:
        check("Observed vs estimated diagnostic found", True)
    else:
        warn("No observed-vs-estimated diagnostic keywords found",
             "Prompt 8 should have added observed vs. estimated comparison plots.")

    # Check for improved BIC elbow plot
    has_elbow_improvement = ('all_evaluated' in app_source.lower() or
                              'rejected' in app_source.lower() or
                              'scatter' in app_source.lower())
    # This is a weak check — the original already had a BIC plot
    if has_elbow_improvement:
        check("BIC plot appears enhanced", True)
    else:
        warn("BIC plot may not have been improved",
             "Prompt 8 should have added all-models scatter and rejection annotations.")

except Exception as e:
    warn(f"Could not inspect app.py for diagnostics: {e}")


# =========================================================================
# TEST 12: Streamlit app at least imports without crashing
# =========================================================================
section("TEST 12: App Import Smoke Test")

try:
    # We can't run Streamlit, but we can check that app.py at least parses
    import py_compile
    py_compile.compile("app.py", doraise=True)
    check("app.py compiles without syntax errors", True)
except py_compile.PyCompileError as e:
    check("app.py compiles", False, f"Syntax error: {e}", critical=True)
except FileNotFoundError:
    check("app.py exists", False, critical=True)

try:
    py_compile.compile("main.py", doraise=True)
    check("main.py compiles without syntax errors", True)
except py_compile.PyCompileError as e:
    check("main.py compiles", False, f"Syntax error: {e}", critical=True)
except FileNotFoundError:
    check("main.py exists", False, critical=True)


# =========================================================================
# TEST 13: All four distributions selectable in run_single_model
# =========================================================================
section("TEST 13: All Distributions Wired Into Engine")

try:
    import inspect
    src = inspect.getsource(run_single_model)

    for dist_name in ['LOGIT', 'CNORM', 'POISSON', 'ZIP']:
        found = dist_name in src
        check(f"run_single_model handles dist='{dist_name}'", found,
              f"'{dist_name}' not found in run_single_model source" if not found else "")

    # Also check run_autotraj
    src_auto = inspect.getsource(run_autotraj)
    for dist_name in ['LOGIT', 'CNORM', 'POISSON', 'ZIP']:
        found = dist_name in src_auto
        check(f"run_autotraj handles dist='{dist_name}'", found,
              f"'{dist_name}' not found in run_autotraj source" if not found else "")

except Exception as e:
    warn(f"Could not inspect engine functions: {e}")


# =========================================================================
# TEST 14: Poisson/ZIP removed from "Coming Soon" in app.py
# =========================================================================
section("TEST 14: UI Labels Updated")

try:
    with open("app.py", "r") as f:
        app_source = f.read()

    # Poisson should NOT still say "Coming"
    poisson_coming = 'POISSON' in app_source and 'Coming' in app_source.split('POISSON')[1][:50]
    if not poisson_coming:
        check("POISSON no longer labeled 'Coming Soon'", True)
    else:
        warn("POISSON may still show 'Coming Soon' in dropdown",
             "Prompt 5 should have removed this label.")

    # Check the error block that stops Poisson/ZIP from running
    has_stop_block = 'POISSON' in app_source and 'st.stop' in app_source
    if has_stop_block:
        # Check if the stop block still blocks POISSON
        # This is hard to detect precisely, so just warn
        warn("Found st.stop() near POISSON — verify it no longer blocks Poisson execution",
             "The original code had a hard stop for Poisson/ZIP. It should be removed for Poisson now.")
    else:
        check("No execution block for POISSON", True)

except Exception as e:
    warn(f"Could not inspect UI labels: {e}")


# =========================================================================
# SUMMARY
# =========================================================================
print(f"\n{'='*70}")
print("PHASE 2 VERIFICATION SUMMARY")
print(f"{'='*70}")
print(f"  ✅ Passed:   {PASS}")
print(f"  ❌ Failed:   {FAIL}")
print(f"  ⚠️  Warnings: {WARN}")
print()

if CRITICAL_FAIL:
    print("🛑 CRITICAL FAILURES DETECTED")
    print("One or more core engine functions are missing or crashing.")
    print("DO NOT proceed to Phase 3. Fix these issues first:")
    print("  - If Poisson functions don't exist: re-run Prompt 5")
    print("  - If ZIP functions don't exist: re-run Prompt 6")
    print("  - If LOGIT/CNORM broke: check for regressions in main.py")
    print()
    print("Tip: paste the failure output into Claude Code with:")
    print('  "These tests are failing after Phase 2. Debug and fix: [paste output]"')
elif FAIL == 0:
    print("🎉 ALL CHECKS PASSED — Clear to proceed to Phase 3.")
    print()
    print("Before starting Phase 3, commit your work:")
    print('  git add -A && git commit -m "Phase 2 complete: Poisson, ZIP, viz, diagnostics"')
    print('  git push origin main')
    print()
    print("Phase 3 (Prompts 9-12) builds the validation test suite.")
    print("Start with Prompt 9: Simulation Framework.")
elif FAIL <= 3 and not CRITICAL_FAIL:
    print("⚠️  MOSTLY PASSING — Review failures above.")
    print()
    print("If failures are in visualization/UI checks (Tests 10-11, 14):")
    print("  These are keyword-based heuristics and may be false negatives.")
    print("  Manually open the app (streamlit run app.py) and verify visually.")
    print("  If it looks right, proceed to Phase 3.")
    print()
    print("If failures are in Poisson/ZIP math (Tests 2-8):")
    print("  DO NOT proceed. Fix the engine first.")
else:
    print("🛑 MULTIPLE FAILURES — Fix before proceeding to Phase 3.")
    print("Phase 3 builds tests that depend on all four distributions working.")

sys.exit(0 if FAIL == 0 else 1)