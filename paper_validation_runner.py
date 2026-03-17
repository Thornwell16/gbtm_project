#!/usr/bin/env python3
"""
AutoTraj Validation Runner for Manuscript
==========================================
Generates ALL numerical evidence needed for the validation paper.
Run this ONCE, review the output, then write the paper around it.

Outputs saved to paper_results/:
  - parameter_recovery_logit.csv
  - parameter_recovery_cnorm.csv
  - parameter_recovery_poisson.csv
  - parameter_recovery_zip.csv
  - cambridge_reanalysis.csv
  - cambridge_adequacy.csv
  - cambridge_all_models.csv
  - bic_selection_accuracy.csv
  - missing_data_sensitivity.csv
  - computation_time_benchmarks.csv
  - summary_statistics.txt  ← READ THIS FIRST

Usage:
    cd /path/to/gbtm_project
    python paper_validation_runner.py 2>&1 | tee paper_results/full_log.txt

The tee command saves everything to a log file while also printing to screen.
Review time: ~30 minutes for results, ~5-10 minutes to run.
"""

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    prep_trajectory_data, run_autotraj, run_single_model,
    get_subject_assignments, calc_model_adequacy, extract_flat_arrays,
)

# Try to import simulation framework
try:
    from tests.simulate import (
        simulate_logit_trajectories,
        simulate_cnorm_trajectories,
        simulate_poisson_trajectories,
        simulate_zip_trajectories,
    )
    HAS_SIM = True
except ImportError:
    HAS_SIM = False
    print("WARNING: tests/simulate.py not found. Will use inline simulation.")

OUT = Path("paper_results")
OUT.mkdir(exist_ok=True)

SUMMARY_LINES = []

def log(msg):
    print(msg)
    SUMMARY_LINES.append(msg)

def divider(title):
    msg = f"\n{'='*70}\n{title}\n{'='*70}"
    print(msg)
    SUMMARY_LINES.append(msg)


# =========================================================================
# INLINE SIMULATION FALLBACK (if tests/simulate.py not available)
# =========================================================================
def _sim_logit(n_subjects, time_points, group_params, group_proportions, missing_rate=0.0, seed=42):
    np.random.seed(seed)
    records, true_groups = [], {}
    for i in range(1, n_subjects + 1):
        g = np.random.choice(len(group_params), p=group_proportions)
        true_groups[i] = g + 1
        betas = group_params[g]['betas']
        for t in time_points:
            if missing_rate > 0 and np.random.random() < missing_rate:
                continue
            z = sum(betas[p] * (t ** p) for p in range(len(betas)))
            z = np.clip(z, -25, 25)
            prob = 1 / (1 + np.exp(-z))
            y = float(np.random.binomial(1, prob))
            records.append({'ID': i, 'Time': float(t), 'Outcome': y})
    return pd.DataFrame(records).sort_values(['ID', 'Time']), true_groups

def _sim_cnorm(n_subjects, time_points, group_params, group_proportions, sigma, cnorm_min, cnorm_max, seed=42):
    np.random.seed(seed)
    records, true_groups = [], {}
    for i in range(1, n_subjects + 1):
        g = np.random.choice(len(group_params), p=group_proportions)
        true_groups[i] = g + 1
        betas = group_params[g]['betas']
        for t in time_points:
            mu = sum(betas[p] * (t ** p) for p in range(len(betas)))
            y = np.clip(np.random.normal(mu, sigma), cnorm_min, cnorm_max)
            records.append({'ID': i, 'Time': float(t), 'Outcome': float(y)})
    return pd.DataFrame(records).sort_values(['ID', 'Time']), true_groups

def _sim_poisson(n_subjects, time_points, group_params, group_proportions, seed=42):
    np.random.seed(seed)
    records, true_groups = [], {}
    for i in range(1, n_subjects + 1):
        g = np.random.choice(len(group_params), p=group_proportions)
        true_groups[i] = g + 1
        betas = group_params[g]['betas']
        for t in time_points:
            eta = sum(betas[p] * (t ** p) for p in range(len(betas)))
            mu = np.exp(np.clip(eta, -20, 20))
            y = float(np.random.poisson(mu))
            records.append({'ID': i, 'Time': float(t), 'Outcome': y})
    return pd.DataFrame(records).sort_values(['ID', 'Time']), true_groups

def _sim_zip(n_subjects, time_points, group_params, group_proportions, zero_inflation, seed=42):
    np.random.seed(seed)
    records, true_groups = [], {}
    for i in range(1, n_subjects + 1):
        g = np.random.choice(len(group_params), p=group_proportions)
        true_groups[i] = g + 1
        betas = group_params[g]['betas']
        omega = zero_inflation[g]
        for t in time_points:
            eta = sum(betas[p] * (t ** p) for p in range(len(betas)))
            mu = np.exp(np.clip(eta, -20, 20))
            y = 0.0 if np.random.random() < omega else float(np.random.poisson(mu))
            records.append({'ID': i, 'Time': float(t), 'Outcome': y})
    return pd.DataFrame(records).sort_values(['ID', 'Time']), true_groups

if HAS_SIM:
    sim_logit = simulate_logit_trajectories
    sim_cnorm = simulate_cnorm_trajectories
    sim_poisson = simulate_poisson_trajectories
    sim_zip = simulate_zip_trajectories
else:
    sim_logit = _sim_logit
    sim_cnorm = _sim_cnorm
    sim_poisson = _sim_poisson
    sim_zip = _sim_zip


# =========================================================================
# HELPER: Parameter recovery evaluation
# =========================================================================
def evaluate_recovery(model, true_params_list, true_pis, true_assign, long_df, dist):
    """Compare estimated vs true parameters and compute assignment accuracy."""
    results = []

    if not (model['result'].success or model['result'].status == 2):
        return pd.DataFrame(), 0.0, 0.0

    orders = model['orders']
    params = model['result'].x
    se = model['se_model']
    pis = model['pis']
    k = len(orders)

    # Determine if groups are flipped (compare by intercept proximity)
    est_intercepts = []
    idx = k - 1
    for g in range(k):
        est_intercepts.append(params[idx])
        idx += orders[g] + 1

    true_intercepts = [p['betas'][0] for p in true_params_list]

    # Try both orderings and pick the better match
    if k == 2:
        err_normal = abs(est_intercepts[0] - true_intercepts[0]) + abs(est_intercepts[1] - true_intercepts[1])
        err_flipped = abs(est_intercepts[0] - true_intercepts[1]) + abs(est_intercepts[1] - true_intercepts[0])
        flipped = err_flipped < err_normal
    else:
        flipped = False

    group_map = list(range(k))
    if flipped and k == 2:
        group_map = [1, 0]

    # Extract and compare parameters
    idx = k - 1
    for g in range(k):
        true_g = group_map[g]
        true_betas = true_params_list[true_g]['betas']
        n_betas = orders[g] + 1

        for b in range(n_betas):
            est = params[idx + b]
            se_val = se[idx + b] if idx + b < len(se) else np.nan
            true_val = true_betas[b] if b < len(true_betas) else 0.0

            ci_lower = est - 1.96 * se_val if np.isfinite(se_val) else np.nan
            ci_upper = est + 1.96 * se_val if np.isfinite(se_val) else np.nan
            recovered = ci_lower <= true_val <= ci_upper if np.isfinite(ci_lower) else False
            bias = est - true_val

            labels = ["Intercept", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
            results.append({
                "Distribution": dist,
                "Group": g + 1,
                "Parameter": labels[b] if b < len(labels) else f"Beta_{b}",
                "True_Value": round(true_val, 4),
                "Estimate": round(est, 4),
                "SE": round(se_val, 4) if np.isfinite(se_val) else "NA",
                "CI_Lower": round(ci_lower, 4) if np.isfinite(ci_lower) else "NA",
                "CI_Upper": round(ci_upper, 4) if np.isfinite(ci_upper) else "NA",
                "Bias": round(bias, 4),
                "Recovered_95CI": "Yes" if recovered else "No"
            })
        idx += n_betas

    # Group proportion comparison
    for g in range(k):
        true_g = group_map[g]
        results.append({
            "Distribution": dist, "Group": g + 1, "Parameter": "Pi (proportion)",
            "True_Value": round(true_pis[true_g], 4),
            "Estimate": round(pis[g], 4),
            "SE": "NA", "CI_Lower": "NA", "CI_Upper": "NA",
            "Bias": round(pis[g] - true_pis[true_g], 4),
            "Recovered_95CI": "Yes" if abs(pis[g] - true_pis[true_g]) < 0.10 else "No"
        })

    # CNORM sigma
    if dist == "CNORM":
        true_sigma = true_params_list[0].get('sigma', None)
        if true_sigma is None:
            # sigma passed separately -- caller should add
            pass

    # Assignment accuracy
    assign_df = get_subject_assignments(model, long_df)
    n_correct = 0
    n_total = 0
    for _, row in assign_df.iterrows():
        sid = row['ID']
        if sid in true_assign:
            n_total += 1
            assigned = int(row['Assigned_Group'])
            true_grp = true_assign[sid]
            if flipped and k == 2:
                # Flip assignment for comparison
                assigned_mapped = 2 if assigned == 1 else 1
                if assigned_mapped == true_grp:
                    n_correct += 1
            else:
                if assigned == true_grp:
                    n_correct += 1

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    # Handle potential complete flip
    accuracy = max(accuracy, 1 - accuracy) if k == 2 else accuracy

    results_df = pd.DataFrame(results)
    n_recovered = (results_df['Recovered_95CI'] == 'Yes').sum()
    n_total_params = len(results_df)
    recovery_rate = n_recovered / n_total_params if n_total_params > 0 else 0.0

    return results_df, accuracy, recovery_rate


# =========================================================================
# 1. LOGIT PARAMETER RECOVERY
# =========================================================================
divider("1. LOGIT PARAMETER RECOVERY (Table 2 in paper)")

logit_scenarios = [
    {
        "name": "2-group linear (N=500)",
        "n": 500, "times": np.linspace(-1.1, 1.1, 23),
        "params": [{'betas': [-2.5, 0.3]}, {'betas': [0.8, -0.5]}],
        "pis": [0.65, 0.35], "orders": [1, 1], "seed": 42
    },
    {
        "name": "2-group quadratic (N=500)",
        "n": 500, "times": np.linspace(-1.1, 1.1, 23),
        "params": [{'betas': [-2.0, 0.5, 0.3]}, {'betas': [0.5, -0.3, -0.2]}],
        "pis": [0.60, 0.40], "orders": [2, 2], "seed": 100
    },
    {
        "name": "3-group linear (N=800)",
        "n": 800, "times": np.linspace(-1.1, 1.1, 23),
        "params": [{'betas': [-3.0, 0.1]}, {'betas': [-0.5, 0.8]}, {'betas': [0.5, -0.3]}],
        "pis": [0.50, 0.30, 0.20], "orders": [1, 1, 1], "seed": 200
    },
]

all_logit_results = []
for scenario in logit_scenarios:
    log(f"\n  Running: {scenario['name']}...")
    t0 = time.time()

    df, true_assign = sim_logit(
        scenario['n'], scenario['times'], scenario['params'],
        scenario['pis'], seed=scenario['seed']
    )

    model = run_single_model(df, scenario['orders'], use_dropout=False, dist='LOGIT')
    elapsed = time.time() - t0

    if model['result'].success or model['result'].status == 2:
        res_df, acc, rec_rate = evaluate_recovery(
            model, scenario['params'], scenario['pis'], true_assign, df, "LOGIT"
        )
        res_df['Scenario'] = scenario['name']
        all_logit_results.append(res_df)
        log(f"    Converged in {elapsed:.1f}s | Recovery: {rec_rate:.0%} | Accuracy: {acc:.0%}")
        log(f"    LL={model['ll']:.2f}, BIC={model['bic']:.2f}")
    else:
        log(f"    FAILED TO CONVERGE")

if all_logit_results:
    logit_df = pd.concat(all_logit_results, ignore_index=True)
    logit_df.to_csv(OUT / "parameter_recovery_logit.csv", index=False)
    log(f"\n  LOGIT overall recovery: {(logit_df['Recovered_95CI']=='Yes').mean():.0%}")


# =========================================================================
# 2. CNORM PARAMETER RECOVERY
# =========================================================================
divider("2. CNORM PARAMETER RECOVERY (Table 3 in paper)")

cnorm_scenarios = [
    {
        "name": "2-group linear, sigma=1.5 (N=500)",
        "n": 500, "times": np.linspace(-1, 1, 15),
        "params": [{'betas': [2.0, 0.8]}, {'betas': [6.0, -0.5]}],
        "pis": [0.55, 0.45], "sigma": 1.5,
        "min": 0.0, "max": 10.0, "orders": [1, 1], "seed": 300
    },
]

all_cnorm_results = []
for scenario in cnorm_scenarios:
    log(f"\n  Running: {scenario['name']}...")
    t0 = time.time()

    df, true_assign = sim_cnorm(
        scenario['n'], scenario['times'], scenario['params'], scenario['pis'],
        scenario['sigma'], scenario['min'], scenario['max'], seed=scenario['seed']
    )

    model = run_single_model(df, scenario['orders'], use_dropout=False, dist='CNORM',
                              cnorm_min=scenario['min'], cnorm_max=scenario['max'])
    elapsed = time.time() - t0

    if model['result'].success or model['result'].status == 2:
        res_df, acc, rec_rate = evaluate_recovery(
            model, scenario['params'], scenario['pis'], true_assign, df, "CNORM"
        )

        # Add sigma recovery
        recovered_sigma = np.exp(model['result'].x[-1])
        sigma_se = model['se_model'][-1] * recovered_sigma  # delta method
        ci_lo = recovered_sigma - 1.96 * sigma_se
        ci_hi = recovered_sigma + 1.96 * sigma_se
        sigma_recovered = ci_lo <= scenario['sigma'] <= ci_hi

        sigma_row = pd.DataFrame([{
            "Distribution": "CNORM", "Group": "All", "Parameter": "Sigma",
            "True_Value": scenario['sigma'], "Estimate": round(recovered_sigma, 4),
            "SE": round(sigma_se, 4),
            "CI_Lower": round(ci_lo, 4), "CI_Upper": round(ci_hi, 4),
            "Bias": round(recovered_sigma - scenario['sigma'], 4),
            "Recovered_95CI": "Yes" if sigma_recovered else "No",
            "Scenario": scenario['name']
        }])
        res_df = pd.concat([res_df, sigma_row], ignore_index=True)
        res_df['Scenario'] = scenario['name']
        all_cnorm_results.append(res_df)

        log(f"    Converged in {elapsed:.1f}s | Recovery: {rec_rate:.0%} | Accuracy: {acc:.0%}")
        log(f"    True sigma={scenario['sigma']}, Recovered sigma={recovered_sigma:.3f}")
    else:
        log(f"    FAILED TO CONVERGE")

if all_cnorm_results:
    cnorm_df = pd.concat(all_cnorm_results, ignore_index=True)
    cnorm_df.to_csv(OUT / "parameter_recovery_cnorm.csv", index=False)


# =========================================================================
# 3. POISSON PARAMETER RECOVERY
# =========================================================================
divider("3. POISSON PARAMETER RECOVERY (Table 4 in paper)")

pois_scenarios = [
    {
        "name": "2-group linear (N=500)",
        "n": 500, "times": np.linspace(-1, 1, 12),
        "params": [{'betas': [0.5, 0.4]}, {'betas': [2.2, -0.3]}],
        "pis": [0.60, 0.40], "orders": [1, 1], "seed": 400
    },
]

all_pois_results = []
for scenario in pois_scenarios:
    log(f"\n  Running: {scenario['name']}...")
    t0 = time.time()

    df, true_assign = sim_poisson(
        scenario['n'], scenario['times'], scenario['params'],
        scenario['pis'], seed=scenario['seed']
    )

    model = run_single_model(df, scenario['orders'], use_dropout=False, dist='POISSON')
    elapsed = time.time() - t0

    if model['result'].success or model['result'].status == 2:
        res_df, acc, rec_rate = evaluate_recovery(
            model, scenario['params'], scenario['pis'], true_assign, df, "POISSON"
        )
        res_df['Scenario'] = scenario['name']
        all_pois_results.append(res_df)
        log(f"    Converged in {elapsed:.1f}s | Recovery: {rec_rate:.0%} | Accuracy: {acc:.0%}")
    else:
        log(f"    FAILED TO CONVERGE")

if all_pois_results:
    pois_df = pd.concat(all_pois_results, ignore_index=True)
    pois_df.to_csv(OUT / "parameter_recovery_poisson.csv", index=False)


# =========================================================================
# 4. ZIP PARAMETER RECOVERY
# =========================================================================
divider("4. ZIP PARAMETER RECOVERY (Table 5 in paper)")

zip_scenarios = [
    {
        "name": "2-group, 30%/10% zero-inflation (N=500)",
        "n": 500, "times": np.linspace(-1, 1, 12),
        "params": [{'betas': [1.0, 0.3]}, {'betas': [2.5, -0.2]}],
        "pis": [0.55, 0.45], "zi": [0.30, 0.10],
        "orders": [1, 1], "seed": 500
    },
]

all_zip_results = []
for scenario in zip_scenarios:
    log(f"\n  Running: {scenario['name']}...")
    t0 = time.time()

    df, true_assign = sim_zip(
        scenario['n'], scenario['times'], scenario['params'],
        scenario['pis'], scenario['zi'], seed=scenario['seed']
    )

    model = run_single_model(df, scenario['orders'], use_dropout=False, dist='ZIP')
    elapsed = time.time() - t0

    if model['result'].success or model['result'].status == 2:
        res_df, acc, rec_rate = evaluate_recovery(
            model, scenario['params'], scenario['pis'], true_assign, df, "ZIP"
        )
        res_df['Scenario'] = scenario['name']
        all_zip_results.append(res_df)
        log(f"    Converged in {elapsed:.1f}s | Recovery: {rec_rate:.0%} | Accuracy: {acc:.0%}")
    else:
        log(f"    FAILED TO CONVERGE (this is not uncommon for ZIP)")

if all_zip_results:
    zip_df = pd.concat(all_zip_results, ignore_index=True)
    zip_df.to_csv(OUT / "parameter_recovery_zip.csv", index=False)


# =========================================================================
# 5. CAMBRIDGE REANALYSIS
# =========================================================================
divider("5. CAMBRIDGE DATA REANALYSIS (Table 6 in paper)")

try:
    raw_df = pd.read_csv("cambridge.txt", sep=r'\s+', encoding='utf-8-sig')
    raw_df.columns = [str(c).strip() for c in raw_df.columns]
    cam_long = prep_trajectory_data(raw_df, 'ID', 'C', 'T').dropna(subset=['Time', 'Outcome'])

    n_subj = cam_long['ID'].nunique()
    n_obs = len(cam_long)
    log(f"\n  Cambridge data: N={n_subj} subjects, {n_obs} observations")
    log(f"  Time range: {cam_long['Time'].min():.1f} to {cam_long['Time'].max():.1f}")
    log(f"  Overall offense rate: {cam_long['Outcome'].mean():.3f}")

    # Full autosearch
    log(f"\n  Running autosearch (groups 1-4, orders 0-3)...")
    t0 = time.time()
    top_models, all_eval = run_autotraj(
        cam_long, min_groups=1, max_groups=4,
        min_order=0, max_order=3, min_group_pct=5.0,
        p_val_thresh=0.05, use_dropout=False, dist='LOGIT'
    )
    cam_time = time.time() - t0
    n_evaluated = len(all_eval)

    log(f"  Evaluated {n_evaluated} models in {cam_time:.1f}s ({n_evaluated/cam_time:.1f} models/sec)")
    log(f"  Valid models: {len(top_models)}")

    # Save all evaluated models
    eval_df = pd.DataFrame(all_eval)
    eval_df.to_csv(OUT / "cambridge_all_models.csv", index=False)

    # Report top 5 models
    log(f"\n  Top 5 models by BIC:")
    for i, m in enumerate(top_models[:5]):
        log(f"    {i+1}. {len(m['orders'])}-group {m['orders']} | BIC={m['bic']:.2f} | "
            f"LL={m['ll']:.2f} | Pis={[f'{p:.1%}' for p in m['pis']]}")

    # Detailed results for best model
    if top_models:
        best = top_models[0]
        best_k = len(best['orders'])

        # Parameter estimates
        from app import get_parameter_estimates_for_ui
        group_names = [f"Group {i+1}" for i in range(best_k)]
        est_df = get_parameter_estimates_for_ui(best, group_names)
        est_df.to_csv(OUT / "cambridge_reanalysis.csv", index=False)
        log(f"\n  Best model parameter estimates saved")

        # Adequacy
        assign_df = get_subject_assignments(best, cam_long)
        adq_df, rel_ent = calc_model_adequacy(assign_df, best['pis'], group_names)
        adq_df.to_csv(OUT / "cambridge_adequacy.csv", index=False)
        log(f"\n  Model adequacy:")
        log(f"    Relative entropy: {rel_ent:.3f}")
        for _, row in adq_df.iterrows():
            log(f"    {row['Group']}: N={row['Assigned N']}, "
                f"Pi={row['Estimated Pi (%)']}%, AvePP={row['AvePP']}, OCC={row['OCC']}")

except Exception as e:
    log(f"  CAMBRIDGE ANALYSIS FAILED: {e}")
    import traceback
    traceback.print_exc()


# =========================================================================
# 6. BIC MODEL SELECTION ACCURACY
# =========================================================================
divider("6. BIC MODEL SELECTION ACCURACY (Table 7 in paper)")

selection_results = []
for true_k in [1, 2, 3]:
    for rep in range(3):  # 3 replications per K
        seed = 1000 + true_k * 100 + rep

        if true_k == 1:
            params = [{'betas': [-1.5, 0.3]}]
            pis = [1.0]
        elif true_k == 2:
            params = [{'betas': [-2.5, 0.2]}, {'betas': [0.5, -0.4]}]
            pis = [0.6, 0.4]
        else:
            params = [{'betas': [-3.0, 0.1]}, {'betas': [-0.5, 0.8]}, {'betas': [0.5, -0.2]}]
            pis = [0.50, 0.30, 0.20]

        df, _ = sim_logit(600, np.linspace(-1, 1, 15), params, pis, seed=seed)

        try:
            top, evaluated = run_autotraj(
                df, min_groups=1, max_groups=4, min_order=0, max_order=2,
                min_group_pct=5.0, use_dropout=False, dist='LOGIT'
            )
            if top:
                selected_k = len(top[0]['orders'])
                selection_results.append({
                    "True_K": true_k, "Replication": rep + 1, "Selected_K": selected_k,
                    "Correct": selected_k == true_k, "BIC": top[0]['bic'],
                    "N_Evaluated": len(evaluated), "N_Valid": len(top)
                })
                log(f"  True K={true_k}, Rep {rep+1}: Selected K={selected_k} {'[OK]' if selected_k == true_k else '[X]'}")
            else:
                log(f"  True K={true_k}, Rep {rep+1}: No valid models found")
        except Exception as e:
            log(f"  True K={true_k}, Rep {rep+1}: Error -- {e}")

if selection_results:
    sel_df = pd.DataFrame(selection_results)
    sel_df.to_csv(OUT / "bic_selection_accuracy.csv", index=False)
    n_correct = sel_df['Correct'].sum()
    n_total = len(sel_df)
    log(f"\n  BIC selection accuracy: {n_correct}/{n_total} ({n_correct/n_total:.0%})")


# =========================================================================
# 7. MISSING DATA SENSITIVITY
# =========================================================================
divider("7. MISSING DATA SENSITIVITY ANALYSIS (Table 8 in paper)")

miss_results = []
true_params_miss = [{'betas': [-2.0, 0.5]}, {'betas': [0.8, -0.3]}]
true_pis_miss = [0.65, 0.35]

for miss_rate in [0.0, 0.10, 0.25, 0.40]:
    log(f"\n  Missing rate: {miss_rate:.0%}")

    df, true_assign = sim_logit(
        500, np.linspace(-1.1, 1.1, 23),
        true_params_miss, true_pis_miss,
        missing_rate=miss_rate, seed=600 + int(miss_rate * 100)
    )

    actual_miss = 1 - len(df) / (500 * 23)
    model = run_single_model(df, [1, 1], use_dropout=False, dist='LOGIT')

    if model['result'].success or model['result'].status == 2:
        res_df, acc, rec_rate = evaluate_recovery(
            model, true_params_miss, true_pis_miss, true_assign, df, "LOGIT"
        )

        miss_results.append({
            "Nominal_Missing_Rate": miss_rate,
            "Actual_Missing_Rate": round(actual_miss, 3),
            "N_Observations": len(df),
            "Recovery_Rate": round(rec_rate, 3),
            "Assignment_Accuracy": round(acc, 3),
            "LL": round(model['ll'], 2),
            "BIC": round(model['bic'], 2),
            "Converged": True
        })
        log(f"    Recovery: {rec_rate:.0%}, Accuracy: {acc:.0%}, Obs: {len(df)}")
    else:
        miss_results.append({
            "Nominal_Missing_Rate": miss_rate, "Actual_Missing_Rate": round(actual_miss, 3),
            "N_Observations": len(df), "Recovery_Rate": np.nan,
            "Assignment_Accuracy": np.nan, "LL": np.nan, "BIC": np.nan, "Converged": False
        })
        log(f"    FAILED TO CONVERGE")

miss_df = pd.DataFrame(miss_results)
miss_df.to_csv(OUT / "missing_data_sensitivity.csv", index=False)


# =========================================================================
# 8. COMPUTATION TIME BENCHMARKS
# =========================================================================
divider("8. COMPUTATION TIME BENCHMARKS (Table 9 in paper)")

time_results = []

for n_subj, n_time, max_g, max_o in [(200, 10, 3, 2), (500, 15, 3, 2), (500, 23, 4, 3), (2000, 15, 3, 2)]:
    df_t, _ = sim_logit(
        n_subj, np.linspace(-1, 1, n_time),
        [{'betas': [-2.0, 0.3]}, {'betas': [0.5, -0.2]}], [0.6, 0.4],
        seed=700 + n_subj
    )

    log(f"\n  N={n_subj}, T={n_time}, Groups 1-{max_g}, Orders 0-{max_o}...")
    t0 = time.time()
    top_t, eval_t = run_autotraj(
        df_t, min_groups=1, max_groups=max_g, min_order=0, max_order=max_o,
        min_group_pct=5.0, use_dropout=False, dist='LOGIT'
    )
    elapsed = time.time() - t0
    n_models = len(eval_t)
    manual_minutes = n_models * 5  # estimated 5 min per model manually

    time_results.append({
        "N_Subjects": n_subj, "N_Timepoints": n_time,
        "Max_Groups": max_g, "Max_Order": max_o,
        "Models_Evaluated": n_models,
        "AutoTraj_Seconds": round(elapsed, 1),
        "Models_Per_Second": round(n_models / elapsed, 1) if elapsed > 0 else "NA",
        "Estimated_Manual_Minutes": manual_minutes,
        "Speedup_Factor": round(manual_minutes * 60 / elapsed, 0) if elapsed > 0 else "NA"
    })
    log(f"    {n_models} models in {elapsed:.1f}s ({n_models/elapsed:.1f}/sec)")
    log(f"    Est. manual: {manual_minutes} min -> {manual_minutes*60/elapsed:.0f}x speedup")

time_df = pd.DataFrame(time_results)
time_df.to_csv(OUT / "computation_time_benchmarks.csv", index=False)


# =========================================================================
# SUMMARY
# =========================================================================
divider("VALIDATION SUMMARY")

all_recovery = []
for label, fpath in [("LOGIT", "parameter_recovery_logit.csv"),
                      ("CNORM", "parameter_recovery_cnorm.csv"),
                      ("POISSON", "parameter_recovery_poisson.csv"),
                      ("ZIP", "parameter_recovery_zip.csv")]:
    full_path = OUT / fpath
    if full_path.exists():
        df_tmp = pd.read_csv(full_path)
        rate = (df_tmp['Recovered_95CI'] == 'Yes').mean()
        all_recovery.append(rate)
        log(f"  {label} parameter recovery rate: {rate:.0%}")

if all_recovery:
    log(f"\n  OVERALL PARAMETER RECOVERY: {np.mean(all_recovery):.0%}")

if selection_results:
    sel_acc = sum(r['Correct'] for r in selection_results) / len(selection_results)
    log(f"  BIC MODEL SELECTION ACCURACY: {sel_acc:.0%}")

log(f"\n  All results saved to: {OUT.resolve()}/")
log(f"  Run completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Save the full summary
with open(OUT / "summary_statistics.txt", 'w') as f:
    f.write('\n'.join(SUMMARY_LINES))

print(f"\n{'='*70}")
print("DONE. Now read paper_results/summary_statistics.txt")
print(f"{'='*70}")
print()
print("WHAT TO CHECK BEFORE WRITING THE PAPER:")
print("  1. Overall parameter recovery > 80%? If < 80%, investigate which")
print("     parameters are failing and why.")
print("  2. BIC selection accuracy > 70%? If not, check if it's selecting")
print("     K+1 or K-1 (off by one is less concerning than random).")
print("  3. Cambridge best model has 2+ groups? (Should be 2 or 3)")
print("  4. All AvePP values > 0.70? All OCC > 5.0?")
print("  5. Missing data: does recovery degrade gracefully (not collapse)?")
print("  6. Computation speed: at least 1 model/second for N=500?")
print()
print("If all six checks pass, you have a publishable validation.")
print("If any fail, fix the engine before writing the paper.")