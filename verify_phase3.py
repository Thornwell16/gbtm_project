#!/usr/bin/env python3
"""
Phase 3 Verification Script for AutoTraj
=========================================
Run this AFTER completing Prompts 9-12 (Simulation Framework, Parameter
Recovery, Cambridge Benchmarks, Edge Cases).

This script does THREE things:
  1. Verifies the test infrastructure exists and runs
  2. Runs key validation checks directly (independent of pytest)
  3. Generates and SAVES reusable test datasets for the paper and Streamlit

Saved datasets go into test_datasets/ — use these for:
  - The validation paper's results section
  - Manual testing in the Streamlit app
  - Reproducible demonstrations

Usage:
    python verify_phase3.py

If all checks pass, proceed to Phase 4 (Prompts 13-16: polish & publication).
"""

import sys
import os
import traceback
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    prep_trajectory_data,
    run_autotraj,
    run_single_model,
    get_subject_assignments,
    calc_model_adequacy,
    extract_flat_arrays,
)

PASS = 0
FAIL = 0
WARN = 0
CRITICAL_FAIL = False

DATASET_DIR = Path("test_datasets")
DATASET_DIR.mkdir(exist_ok=True)

RESULTS_LOG = []  # Collect results for summary CSV


def check(name, condition, detail="", critical=False):
    global PASS, FAIL, CRITICAL_FAIL
    if condition:
        print(f"  ✅ PASS: {name}")
        PASS += 1
        RESULTS_LOG.append({"Test": name, "Result": "PASS", "Detail": ""})
    else:
        print(f"  ❌ FAIL: {name}")
        if detail:
            print(f"         {detail}")
        FAIL += 1
        RESULTS_LOG.append({"Test": name, "Result": "FAIL", "Detail": detail})
        if critical:
            CRITICAL_FAIL = True

def warn(name, detail=""):
    global WARN
    print(f"  ⚠️  WARN: {name}")
    if detail:
        print(f"         {detail}")
    WARN += 1
    RESULTS_LOG.append({"Test": name, "Result": "WARN", "Detail": detail})

def section(title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")


def save_dataset(df, name, metadata=None):
    """Save a dataset as CSV with an accompanying JSON metadata file."""
    csv_path = DATASET_DIR / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    if metadata:
        meta_path = DATASET_DIR / f"{name}_metadata.json"
        # Convert numpy types to native Python for JSON serialization
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                clean_meta[k] = v.tolist()
            elif isinstance(v, (np.integer,)):
                clean_meta[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean_meta[k] = float(v)
            else:
                clean_meta[k] = v
        with open(meta_path, 'w') as f:
            json.dump(clean_meta, f, indent=2, default=str)
    print(f"  💾 Saved: {csv_path}")


# =========================================================================
# TEST 1: Test infrastructure files exist
# =========================================================================
section("TEST 1: Test Infrastructure Exists")

tests_dir = Path("tests")
check("tests/ directory exists", tests_dir.is_dir(),
      "Prompt 9 should have created tests/", critical=True)

expected_files = {
    "tests/__init__.py": "Package init",
    "tests/simulate.py": "Simulation framework (Prompt 9)",
    "tests/test_parameter_recovery.py": "Parameter recovery tests (Prompt 10)",
    "tests/test_cambridge_benchmark.py": "Cambridge benchmarks (Prompt 11)",
    "tests/test_edge_cases.py": "Edge case tests (Prompt 12)",
}

for fpath, desc in expected_files.items():
    exists = Path(fpath).is_file()
    check(f"{fpath} exists ({desc})", exists,
          f"File not found — re-run the relevant prompt" if not exists else "")
    if not exists and 'simulate' in fpath:
        CRITICAL_FAIL = True


# =========================================================================
# TEST 2: Simulation framework imports and functions exist
# =========================================================================
section("TEST 2: Simulation Framework Functions")

sim_logit = None
sim_cnorm = None
sim_poisson = None
sim_zip = None
sim_dropout = None

try:
    from tests.simulate import simulate_logit_trajectories
    sim_logit = simulate_logit_trajectories
    check("simulate_logit_trajectories importable", True)
except (ImportError, ModuleNotFoundError) as e:
    check("simulate_logit_trajectories importable", False, str(e), critical=True)

try:
    from tests.simulate import simulate_cnorm_trajectories
    sim_cnorm = simulate_cnorm_trajectories
    check("simulate_cnorm_trajectories importable", True)
except (ImportError, ModuleNotFoundError) as e:
    check("simulate_cnorm_trajectories importable", False, str(e))

try:
    from tests.simulate import simulate_poisson_trajectories
    sim_poisson = simulate_poisson_trajectories
    check("simulate_poisson_trajectories importable", True)
except (ImportError, ModuleNotFoundError) as e:
    check("simulate_poisson_trajectories importable", False, str(e))

try:
    from tests.simulate import simulate_zip_trajectories
    sim_zip = simulate_zip_trajectories
    check("simulate_zip_trajectories importable", True)
except (ImportError, ModuleNotFoundError) as e:
    check("simulate_zip_trajectories importable", False, str(e))

try:
    from tests.simulate import simulate_dropout_data
    sim_dropout = simulate_dropout_data
    check("simulate_dropout_data importable", True)
except (ImportError, ModuleNotFoundError) as e:
    warn("simulate_dropout_data not importable", str(e))


# =========================================================================
# TEST 3: Generate and save LOGIT benchmark dataset
# =========================================================================
section("TEST 3: LOGIT Benchmark — Generate, Save, Recover")

if sim_logit:
    try:
        logit_params = [
            {'betas': [-2.5, 0.3]},   # Group 1: low, slowly rising
            {'betas': [0.8, -0.5]},    # Group 2: high, falling
        ]
        logit_pis = [0.65, 0.35]

        long_df, true_assign = sim_logit(
            n_subjects=500,
            time_points=np.linspace(-1.1, 1.1, 23),
            group_params=logit_params,
            group_proportions=logit_pis,
            missing_rate=0.0,
            seed=42
        )

        check("LOGIT simulation produces data", len(long_df) > 0,
              f"Got {len(long_df)} rows, {long_df['ID'].nunique()} subjects")
        check("LOGIT data has correct columns",
              all(c in long_df.columns for c in ['ID', 'Time', 'Outcome']))
        check("LOGIT outcomes are binary",
              set(long_df['Outcome'].dropna().unique()).issubset({0, 1, 0.0, 1.0}))

        # Save for paper and Streamlit
        save_dataset(long_df, "benchmark_logit_2group", metadata={
            "distribution": "LOGIT",
            "n_subjects": 500,
            "n_timepoints": 23,
            "true_groups": 2,
            "true_proportions": logit_pis,
            "true_params": [p['betas'] for p in logit_params],
            "description": "2-group LOGIT benchmark for validation paper. "
                           "Group 1 (65%): low offending, slight rise. "
                           "Group 2 (35%): high offending, decline.",
            "format": "Long (ID, Time, Outcome)",
            "seed": 42
        })

        # Save true assignments
        assign_true_df = pd.DataFrame([
            {"ID": k, "True_Group": v} for k, v in true_assign.items()
        ])
        save_dataset(assign_true_df, "benchmark_logit_2group_true_assignments", metadata={
            "description": "True group assignments for LOGIT benchmark"
        })

        # Run AutoTraj and check recovery
        m_logit = run_single_model(long_df, [1, 1], use_dropout=False, dist='LOGIT')
        converged = m_logit['result'].success or m_logit['result'].status == 2
        check("LOGIT benchmark model converges", converged, critical=True)

        if converged:
            pis = m_logit['pis']
            # Check proportions (allow for label flip)
            pi_error = min(
                abs(pis[0] - logit_pis[0]) + abs(pis[1] - logit_pis[1]),
                abs(pis[0] - logit_pis[1]) + abs(pis[1] - logit_pis[0])
            )
            check(f"LOGIT proportions within 15pp of truth (error={pi_error:.2f})",
                  pi_error < 0.15,
                  f"Estimated: {[f'{p:.1%}' for p in pis]}, True: {[f'{p:.1%}' for p in logit_pis]}")

            # Check assignment accuracy
            a_df = get_subject_assignments(m_logit, long_df)
            n_correct = 0
            n_total = 0
            for _, row in a_df.iterrows():
                sid = row['ID']
                if sid in true_assign:
                    n_total += 1
                    if row['Assigned_Group'] == true_assign[sid]:
                        n_correct += 1
            if n_total > 0:
                acc = max(n_correct / n_total, 1 - n_correct / n_total)
                check(f"LOGIT assignment accuracy > 75% (got {acc:.1%})", acc > 0.75)

    except Exception as e:
        check("LOGIT benchmark", False, f"Crashed: {e}", critical=True)
        traceback.print_exc()
else:
    warn("Skipping LOGIT benchmark — simulate function not found")


# =========================================================================
# TEST 4: Generate and save CNORM benchmark dataset
# =========================================================================
section("TEST 4: CNORM Benchmark — Generate, Save, Recover")

if sim_cnorm:
    try:
        cnorm_params = [
            {'betas': [2.0, 0.8]},    # Group 1: starts at 2, rising
            {'betas': [6.0, -0.5]},    # Group 2: starts at 6, falling
        ]
        cnorm_pis = [0.55, 0.45]
        true_sigma = 1.2

        long_df_c, true_assign_c = sim_cnorm(
            n_subjects=500,
            time_points=np.linspace(-1, 1, 15),
            group_params=cnorm_params,
            group_proportions=cnorm_pis,
            sigma=true_sigma,
            cnorm_min=0.0,
            cnorm_max=10.0,
            seed=123
        )

        check("CNORM simulation produces data", len(long_df_c) > 0)
        check("CNORM outcomes are continuous",
              len(long_df_c['Outcome'].unique()) > 10,
              f"Only {len(long_df_c['Outcome'].unique())} unique values — may not be continuous")

        save_dataset(long_df_c, "benchmark_cnorm_2group", metadata={
            "distribution": "CNORM",
            "n_subjects": 500,
            "n_timepoints": 15,
            "true_groups": 2,
            "true_proportions": cnorm_pis,
            "true_params": [p['betas'] for p in cnorm_params],
            "true_sigma": true_sigma,
            "cnorm_min": 0.0,
            "cnorm_max": 10.0,
            "description": "2-group CNORM benchmark. Continuous outcome censored at 0-10. "
                           "Group 1 (55%): rising from 2. Group 2 (45%): falling from 6.",
            "seed": 123
        })

        # Run and check sigma recovery
        m_cnorm = run_single_model(long_df_c, [1, 1], use_dropout=False, dist='CNORM',
                                    cnorm_min=0.0, cnorm_max=10.0)
        converged = m_cnorm['result'].success or m_cnorm['result'].status == 2
        check("CNORM benchmark converges", converged)

        if converged:
            recovered_sigma = np.exp(m_cnorm['result'].x[-1])
            sigma_err = abs(recovered_sigma - true_sigma) / true_sigma
            check(f"CNORM sigma recovery (true={true_sigma}, got={recovered_sigma:.3f}, err={sigma_err:.1%})",
                  sigma_err < 0.25,
                  "If > 25% error, the CNORM gradient may still have issues")

    except Exception as e:
        check("CNORM benchmark", False, f"Crashed: {e}")
        traceback.print_exc()
else:
    warn("Skipping CNORM benchmark — simulate function not found")


# =========================================================================
# TEST 5: Generate and save Poisson benchmark dataset
# =========================================================================
section("TEST 5: Poisson Benchmark — Generate, Save, Recover")

if sim_poisson:
    try:
        pois_params = [
            {'betas': [0.5, 0.4]},    # Group 1: low counts, rising
            {'betas': [2.2, -0.3]},    # Group 2: high counts, falling
        ]
        pois_pis = [0.6, 0.4]

        long_df_p, true_assign_p = sim_poisson(
            n_subjects=500,
            time_points=np.linspace(-1, 1, 12),
            group_params=pois_params,
            group_proportions=pois_pis,
            seed=456
        )

        check("Poisson simulation produces data", len(long_df_p) > 0)
        check("Poisson outcomes are non-negative integers",
              (long_df_p['Outcome'] >= 0).all() and
              (long_df_p['Outcome'] == long_df_p['Outcome'].astype(int)).all())

        mean_count = long_df_p['Outcome'].mean()
        print(f"  ℹ️  Mean count: {mean_count:.2f}, Max: {long_df_p['Outcome'].max():.0f}")

        save_dataset(long_df_p, "benchmark_poisson_2group", metadata={
            "distribution": "POISSON",
            "n_subjects": 500,
            "n_timepoints": 12,
            "true_groups": 2,
            "true_proportions": pois_pis,
            "true_params": [p['betas'] for p in pois_params],
            "description": "2-group Poisson benchmark. Count outcomes. "
                           "Group 1 (60%): low counts ~1-3. Group 2 (40%): high counts ~5-12.",
            "seed": 456
        })

        m_pois = run_single_model(long_df_p, [1, 1], use_dropout=False, dist='POISSON')
        converged = m_pois['result'].success or m_pois['result'].status == 2
        check("Poisson benchmark converges", converged)

        if converged:
            pis = m_pois['pis']
            pi_error = min(
                abs(pis[0] - pois_pis[0]) + abs(pis[1] - pois_pis[1]),
                abs(pis[0] - pois_pis[1]) + abs(pis[1] - pois_pis[0])
            )
            check(f"Poisson proportions within 15pp (error={pi_error:.2f})",
                  pi_error < 0.15,
                  f"Estimated: {[f'{p:.1%}' for p in pis]}")

    except Exception as e:
        check("Poisson benchmark", False, f"Crashed: {e}")
        traceback.print_exc()
else:
    warn("Skipping Poisson benchmark — simulate function not found")


# =========================================================================
# TEST 6: Generate and save ZIP benchmark dataset
# =========================================================================
section("TEST 6: ZIP Benchmark — Generate, Save, Recover")

if sim_zip:
    try:
        zip_params = [
            {'betas': [1.0, 0.3]},
            {'betas': [2.5, -0.2]},
        ]
        zip_pis = [0.55, 0.45]
        zip_zi = [0.30, 0.10]

        long_df_z, true_assign_z = sim_zip(
            n_subjects=500,
            time_points=np.linspace(-1, 1, 12),
            group_params=zip_params,
            group_proportions=zip_pis,
            zero_inflation_rates=zip_zi,
            seed=789
        )

        check("ZIP simulation produces data", len(long_df_z) > 0)

        zero_frac = (long_df_z['Outcome'] == 0).mean()
        print(f"  ℹ️  Zero fraction: {zero_frac:.1%}")
        check("ZIP data has excess zeros (> 20%)", zero_frac > 0.20,
              f"Zero fraction = {zero_frac:.1%}. With 30%/10% inflation, expect > 20%")

        save_dataset(long_df_z, "benchmark_zip_2group", metadata={
            "distribution": "ZIP",
            "n_subjects": 500,
            "n_timepoints": 12,
            "true_groups": 2,
            "true_proportions": zip_pis,
            "true_params": [p['betas'] for p in zip_params],
            "true_zero_inflation": zip_zi,
            "description": "2-group ZIP benchmark. Group 1 (55%): 30% zero inflation. "
                           "Group 2 (45%): 10% zero inflation.",
            "seed": 789
        })

        m_zip = run_single_model(long_df_z, [1, 1], use_dropout=False, dist='ZIP')
        converged = m_zip['result'].success or m_zip['result'].status == 2
        check("ZIP benchmark converges", converged)

    except Exception as e:
        check("ZIP benchmark", False, f"Crashed: {e}")
        traceback.print_exc()
else:
    warn("Skipping ZIP benchmark — simulate function not found")


# =========================================================================
# TEST 7: Generate datasets for special scenarios (paper & demo)
# =========================================================================
section("TEST 7: Special Scenario Datasets — Generate & Save")

# --- 7a: 3-group LOGIT (for demonstrating complex models) ---
if sim_logit:
    try:
        logit_3g_params = [
            {'betas': [-3.5, 0.1]},        # Non-offender: very low, flat
            {'betas': [-0.5, 1.2, -0.8]},   # Adolescence-peaked: quadratic hump
            {'betas': [0.5, 0.3]},           # Chronic: moderate, slowly rising
        ]
        logit_3g_pis = [0.55, 0.30, 0.15]

        df_3g, assign_3g = sim_logit(
            n_subjects=800,
            time_points=np.linspace(-1.1, 1.1, 23),
            group_params=logit_3g_params,
            group_proportions=logit_3g_pis,
            seed=2026
        )

        save_dataset(df_3g, "demo_logit_3group", metadata={
            "distribution": "LOGIT",
            "n_subjects": 800,
            "n_timepoints": 23,
            "true_groups": 3,
            "true_proportions": logit_3g_pis,
            "true_params": [p['betas'] for p in logit_3g_params],
            "description": "3-group LOGIT demo: Non-offender (55%), "
                           "Adolescence-peaked quadratic (30%), Chronic (15%). "
                           "Good for demonstrating automated model selection.",
            "seed": 2026
        })
        check("3-group LOGIT demo dataset saved", True)

    except Exception as e:
        warn(f"3-group LOGIT demo generation failed: {e}")

# --- 7b: Missing data scenario ---
if sim_logit:
    try:
        df_miss, assign_miss = sim_logit(
            n_subjects=400,
            time_points=np.linspace(-1.1, 1.1, 23),
            group_params=[{'betas': [-2.0, 0.5]}, {'betas': [0.5, -0.3]}],
            group_proportions=[0.65, 0.35],
            missing_rate=0.25,
            seed=999
        )

        obs_per_subj = df_miss.groupby('ID').size()
        actual_miss = 1 - len(df_miss) / (400 * 23)

        save_dataset(df_miss, "demo_logit_missing_25pct", metadata={
            "distribution": "LOGIT",
            "n_subjects": 400,
            "missing_rate": 0.25,
            "actual_missing_pct": float(actual_miss),
            "description": "2-group LOGIT with 25% MCAR missing data. "
                           "For demonstrating FIML and dropout model.",
            "seed": 999
        })
        check(f"Missing data dataset saved ({actual_miss:.0%} missing)", True)

    except Exception as e:
        warn(f"Missing data demo generation failed: {e}")

# --- 7c: Dropout scenario ---
if sim_dropout:
    try:
        df_drop, assign_drop = sim_dropout(
            n_subjects=400,
            time_points=np.linspace(-1.1, 1.1, 23),
            group_params=[{'betas': [-2.0, 0.4]}, {'betas': [0.8, -0.2]}],
            group_proportions=[0.6, 0.4],
            dropout_gammas=[
                {'gamma': [-2.0, 0.5, 1.0]},   # Group 1 dropout
                {'gamma': [-1.5, 0.3, 0.8]},    # Group 2 dropout
            ],
            seed=777
        )

        obs_per_subj = df_drop.groupby('ID').size()
        mean_obs = obs_per_subj.mean()

        save_dataset(df_drop, "demo_logit_informative_dropout", metadata={
            "distribution": "LOGIT",
            "n_subjects": 400,
            "n_timepoints": 23,
            "mean_observations_per_subject": float(mean_obs),
            "description": "2-group LOGIT with MNAR informative dropout. "
                           "Subjects with positive outcomes drop out faster. "
                           "For demonstrating the dropout model toggle.",
            "seed": 777
        })
        check(f"Dropout dataset saved (mean obs/subject: {mean_obs:.1f})", True)

    except Exception as e:
        warn(f"Dropout demo generation failed: {e}")
else:
    warn("Skipping dropout dataset — simulate_dropout_data not found")

# --- 7d: Large Poisson dataset (for speed demos) ---
if sim_poisson:
    try:
        df_big, _ = sim_poisson(
            n_subjects=2000,
            time_points=np.linspace(-1, 1, 15),
            group_params=[
                {'betas': [0.3, 0.2]},
                {'betas': [1.5, 0.0]},
                {'betas': [2.8, -0.4]},
            ],
            group_proportions=[0.5, 0.3, 0.2],
            seed=555
        )

        save_dataset(df_big, "demo_poisson_3group_large", metadata={
            "distribution": "POISSON",
            "n_subjects": 2000,
            "n_timepoints": 15,
            "true_groups": 3,
            "description": "Large 3-group Poisson dataset (N=2000). "
                           "For speed benchmarking and demonstrating scalability.",
            "seed": 555
        })
        check("Large Poisson demo dataset saved (N=2000)", True)

    except Exception as e:
        warn(f"Large Poisson demo generation failed: {e}")


# =========================================================================
# TEST 8: pytest infrastructure runs
# =========================================================================
section("TEST 8: pytest Runs Without Import Errors")

try:
    import subprocess
    # Just do a collect-only to verify tests are discoverable without running them
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        capture_output=True, text=True, timeout=60
    )

    if result.returncode == 0:
        # Count collected tests
        lines = result.stdout.strip().split('\n')
        test_count_line = [l for l in lines if 'test' in l.lower() and 'selected' in l.lower()]
        check("pytest discovers tests successfully", True)
        print(f"  ℹ️  pytest output:\n{result.stdout.strip()}")
    elif 'no tests' in result.stdout.lower() or result.returncode == 5:
        warn("pytest found no tests",
             "Test files may exist but have no test_ functions, or pytest can't import them.\n"
             f"         stdout: {result.stdout.strip()}\n"
             f"         stderr: {result.stderr.strip()}")
    else:
        check("pytest collection", False,
              f"Exit code {result.returncode}\n"
              f"         stdout: {result.stdout.strip()[:300]}\n"
              f"         stderr: {result.stderr.strip()[:300]}")

except subprocess.TimeoutExpired:
    warn("pytest collection timed out (60s)")
except FileNotFoundError:
    warn("pytest not installed — run: pip install pytest")
except Exception as e:
    warn(f"Could not run pytest: {e}")


# =========================================================================
# TEST 9: Run actual pytest and capture results
# =========================================================================
section("TEST 9: pytest Execution (Full Suite)")

try:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-x"],
        capture_output=True, text=True, timeout=600  # 10 min timeout
    )

    # Parse results
    stdout = result.stdout
    lines = stdout.strip().split('\n')

    passed = sum(1 for l in lines if ' PASSED' in l)
    failed = sum(1 for l in lines if ' FAILED' in l)
    errors = sum(1 for l in lines if ' ERROR' in l)
    skipped = sum(1 for l in lines if ' SKIPPED' in l)

    total = passed + failed + errors
    print(f"  ℹ️  pytest results: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")

    check(f"pytest passes ({passed}/{total} tests)",
          failed == 0 and errors == 0,
          f"Failures/errors detected. Last 20 lines:\n" +
          '\n'.join(f"         {l}" for l in lines[-20:]) if (failed > 0 or errors > 0) else "")

    if failed > 0 or errors > 0:
        # Show the failure details
        print(f"\n  📋 pytest output (last 30 lines):")
        for line in lines[-30:]:
            print(f"     {line}")

    if passed > 0 and failed == 0:
        check(f"All {passed} tests passing", True)

except subprocess.TimeoutExpired:
    warn("pytest timed out after 10 minutes",
         "Some tests may be too slow. Check for infinite loops or very large simulations.")
except FileNotFoundError:
    warn("pytest not found — install with: pip install pytest")
except Exception as e:
    warn(f"Could not run pytest: {e}")


# =========================================================================
# TEST 10: Cambridge benchmark specifically
# =========================================================================
section("TEST 10: Cambridge Benchmark — Direct Verification")

try:
    raw_df = pd.read_csv("cambridge.txt", sep=r'\s+', encoding='utf-8-sig')
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    long_df_cam = prep_trajectory_data(raw_df, 'ID', 'C', 'T').dropna(subset=['Time', 'Outcome'])

    # Run autosearch
    top_models, all_eval = run_autotraj(
        long_df_cam, min_groups=1, max_groups=4,
        min_order=0, max_order=3, min_group_pct=5.0,
        use_dropout=False, dist='LOGIT'
    )

    n_evaluated = len(all_eval)
    n_valid = len(top_models)
    print(f"  ℹ️  Evaluated {n_evaluated} models, {n_valid} valid")

    check("Cambridge autosearch completes", n_evaluated > 0, critical=True)
    check("At least one valid model found", n_valid > 0, critical=True)

    if n_valid > 0:
        best = top_models[0]
        best_k = len(best['orders'])
        best_bic = best['bic']
        print(f"  ℹ️  Best model: {best_k}-group {best['orders']}, BIC={best_bic:.2f}")

        check(f"Best model has 2+ groups (got {best_k})", best_k >= 2,
              "Cambridge data clearly has subgroups — 1-group shouldn't win")

        # Check adequacy of best model
        a_df = get_subject_assignments(best, long_df_cam)
        adq, ent = calc_model_adequacy(a_df, best['pis'],
                                        [f"Group {i+1}" for i in range(best_k)])
        check(f"Best model relative entropy > 0.5 (got {ent:.3f})", ent > 0.5)

        # Save the Cambridge results for the paper
        eval_df = pd.DataFrame(all_eval)
        save_dataset(eval_df, "cambridge_all_models_evaluated", metadata={
            "description": "All models evaluated on Cambridge data by AutoTraj autosearch. "
                           "Groups 1-4, Orders 0-3. For validation paper model comparison table.",
            "n_evaluated": n_evaluated,
            "n_valid": n_valid,
            "best_model_groups": best_k,
            "best_model_orders": best['orders'],
            "best_bic": float(best_bic),
        })

        # Save the best model's adequacy metrics
        save_dataset(adq, "cambridge_best_model_adequacy", metadata={
            "description": "Adequacy metrics (AvePP, OCC) for best Cambridge model.",
            "relative_entropy": float(ent),
        })

except Exception as e:
    check("Cambridge benchmark", False, f"Crashed: {e}", critical=True)
    traceback.print_exc()


# =========================================================================
# TEST 11: BIC selects correct K on simulated data
# =========================================================================
section("TEST 11: BIC Model Selection — Does It Pick the Right K?")

selection_results = []

if sim_logit:
    for true_k, seed in [(1, 1001), (2, 1002), (3, 1003)]:
        try:
            if true_k == 1:
                params = [{'betas': [-1.5, 0.3]}]
                pis = [1.0]
            elif true_k == 2:
                params = [{'betas': [-2.5, 0.2]}, {'betas': [0.5, -0.4]}]
                pis = [0.6, 0.4]
            else:
                params = [{'betas': [-3.0, 0.1]}, {'betas': [-0.5, 0.8, -0.6]}, {'betas': [0.5, 0.2]}]
                pis = [0.50, 0.30, 0.20]

            df_sel, _ = sim_logit(
                n_subjects=600, time_points=np.linspace(-1, 1, 15),
                group_params=params, group_proportions=pis, seed=seed
            )

            top, _ = run_autotraj(
                df_sel, min_groups=1, max_groups=4,
                min_order=0, max_order=2, min_group_pct=5.0,
                use_dropout=False, dist='LOGIT'
            )

            if len(top) > 0:
                selected_k = len(top[0]['orders'])
                correct = selected_k == true_k
                selection_results.append({
                    "True_K": true_k, "Selected_K": selected_k, "Correct": correct
                })
                check(f"True K={true_k}: BIC selects K={selected_k}",
                      correct,
                      "" if correct else f"Mismatch — may be acceptable if K={selected_k} is close")
            else:
                warn(f"True K={true_k}: No valid models found")

        except Exception as e:
            warn(f"BIC selection test K={true_k} failed: {e}")

    if selection_results:
        n_correct = sum(1 for r in selection_results if r['Correct'])
        total = len(selection_results)
        print(f"\n  ℹ️  Model selection: {n_correct}/{total} correct")
        save_dataset(pd.DataFrame(selection_results), "bic_model_selection_results", metadata={
            "description": "BIC model selection accuracy across simulated datasets. "
                           "For validation paper's model selection evaluation."
        })


# =========================================================================
# SUMMARY
# =========================================================================
section("FILES SAVED")
saved_files = sorted(DATASET_DIR.glob("*"))
for f in saved_files:
    size = f.stat().st_size
    print(f"  📁 {f.name} ({size:,} bytes)")
print(f"\n  Total: {len(saved_files)} files in {DATASET_DIR}/")

# Save the full results log
results_df = pd.DataFrame(RESULTS_LOG)
results_df.to_csv(DATASET_DIR / "verification_results.csv", index=False)
print(f"  📁 verification_results.csv (test log)")

print(f"\n{'='*70}")
print("PHASE 3 VERIFICATION SUMMARY")
print(f"{'='*70}")
print(f"  ✅ Passed:   {PASS}")
print(f"  ❌ Failed:   {FAIL}")
print(f"  ⚠️  Warnings: {WARN}")
print()

if CRITICAL_FAIL:
    print("🛑 CRITICAL FAILURES — Do not proceed to Phase 4.")
    print("Fix the issues above first. Common problems:")
    print("  - tests/simulate.py doesn't exist → re-run Prompt 9")
    print("  - Parameter recovery failing → check main.py for regressions")
    print("  - Cambridge search crashes → check run_autotraj didn't break")
elif FAIL == 0:
    print("🎉 ALL CHECKS PASSED — Clear to proceed to Phase 4.")
    print()
    print("Your test_datasets/ folder now contains:")
    print("  - 4 benchmark datasets (LOGIT, CNORM, Poisson, ZIP) with ground truth")
    print("  - 3 demo datasets (3-group, missing data, dropout)")
    print("  - 1 large Poisson dataset for speed testing")
    print("  - Cambridge analysis results")
    print("  - BIC model selection results")
    print()
    print("These datasets serve triple duty:")
    print("  1. Evidence for the validation paper")
    print("  2. Test fixtures for the pytest suite")
    print("  3. Demo data you can load in Streamlit for screenshots/videos")
    print()
    print("Commit:")
    print('  git add -A')
    print('  git commit -m "Phase 3 complete: validation suite, test datasets, benchmarks"')
    print('  git push origin main')
    print()
    print("Then proceed to Phase 4 (Prompts 13-16): documentation, CI, release prep.")
elif FAIL <= 3 and not CRITICAL_FAIL:
    print("⚠️  MOSTLY PASSING — Review failures.")
    print()
    print("If pytest failures are in edge cases (Test 9) but benchmarks pass (Tests 3-6):")
    print("  Edge case failures are lower priority. Note them, proceed to Phase 4,")
    print("  and fix them during polish.")
    print()
    print("If benchmark parameter recovery is failing (Tests 3-6):")
    print("  DO NOT proceed. Debug the engine first.")
else:
    print("🛑 MULTIPLE FAILURES — Fix before proceeding.")

sys.exit(0 if FAIL == 0 else 1)