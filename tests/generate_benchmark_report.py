#!/usr/bin/env python3
"""
tests/generate_benchmark_report.py
====================================
Standalone benchmark script for the AutoTraj validation paper.

Runs four benchmarks and produces:
  benchmark_report.md      — formatted markdown with all tables
  benchmark_results.csv    — raw numerical results
  benchmark_figures/       — trajectory plots for each benchmark

Usage
-----
    python tests/generate_benchmark_report.py

The script is NOT a pytest file and imports nothing from pytest.
"""

from __future__ import annotations

import os
import sys
import time
import datetime
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    prep_trajectory_data,
    run_single_model,
    run_autotraj,
    get_subject_assignments,
    calc_model_adequacy,
)
from tests.simulate import (
    simulate_logit_trajectories,
    simulate_cnorm_trajectories,
    simulate_poisson_trajectories,
)


# ── output paths ─────────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMBRIDGE_PATH = os.path.join(_REPO, "cambridge.txt")
FIGURES_DIR    = os.path.join(_REPO, "benchmark_figures")
REPORT_MD      = os.path.join(_REPO, "benchmark_report.md")
REPORT_CSV     = os.path.join(_REPO, "benchmark_results.csv")

# ── run settings ─────────────────────────────────────────────────────────────
N_SUBJECTS  = 500
TIME_POINTS = np.linspace(-1.0, 1.0, 10)
N_STARTS    = 5     # multi-start restarts per model fit
SEED        = 42

# palette used for trajectory plots (group 1, group 2, group 3)
COLORS = ["#2196F3", "#F44336", "#4CAF50"]


# ── parameter-extraction helper ───────────────────────────────────────────────

def _extract_beta_rows(model: dict, dist: str = "LOGIT") -> List[dict]:
    """Return one dict per estimable parameter with estimate, SE, and labels.

    Covers:
    - beta_p (group g) for all groups and polynomial orders
    - sigma  (CNORM only, transformed from log scale via delta method)

    The 'param_key' field is used to match against true values.
    """
    k      = len(model["orders"])
    params = model["result"].x
    se     = model["se_model"]
    orders = model["orders"]
    pis    = model["pis"]

    rows: List[dict] = []
    idx = k - 1     # beta block starts after (k-1) theta params

    for g in range(k):
        order = orders[g]
        pi_g  = float(pis[g])
        for p in range(order + 1):
            name  = f"beta{p}_g{g + 1}"
            label = f"β_{p} Group {g + 1} ({pi_g * 100:.0f}%)"
            est   = float(params[idx])
            se_v  = float(se[idx]) if idx < len(se) else np.nan
            rows.append({
                "param_key": name,
                "label":     label,
                "estimate":  est,
                "se":        se_v,
            })
            idx += 1

    # CNORM sigma (on original scale, SE via delta method)
    if dist == "CNORM":
        log_sig_idx = len(params) - 1
        log_sig_est = float(params[log_sig_idx])
        sigma_est   = float(np.exp(log_sig_est))
        se_log      = float(se[log_sig_idx]) if log_sig_idx < len(se) else np.nan
        se_sigma    = sigma_est * se_log        # delta method: SE(e^x) ≈ e^x * SE(x)
        rows.append({
            "param_key": "sigma",
            "label":     "σ (residual SD)",
            "estimate":  sigma_est,
            "se":        se_sigma,
        })

    return rows


def _build_recovery_df(
    param_rows: List[dict],
    true_values: Dict[str, float],
) -> pd.DataFrame:
    """Build recovery table: one row per parameter, with 95% CI and recovery flag."""
    records = []
    for row in param_rows:
        key   = row["param_key"]
        est   = row["estimate"]
        se_v  = row["se"]
        ci_lo = est - 1.96 * se_v if np.isfinite(se_v) else np.nan
        ci_hi = est + 1.96 * se_v if np.isfinite(se_v) else np.nan
        true  = true_values.get(key)

        if true is not None and np.isfinite(ci_lo):
            recovered = "YES" if ci_lo <= true <= ci_hi else "NO"
        else:
            recovered = "N/A"

        records.append({
            "Parameter":         row["label"],
            "True Value":        f"{true:.4f}"  if true is not None else "N/A",
            "AutoTraj Estimate": f"{est:.4f}",
            "Std Error":         f"{se_v:.4f}"  if np.isfinite(se_v) else "N/A",
            "95% CI":            f"[{ci_lo:.4f}, {ci_hi:.4f}]" if np.isfinite(ci_lo) else "N/A",
            "Recovery":          recovered,
            # raw values for CSV / summary stats
            "_true":             true,
            "_est":              est,
            "_ci_lo":            ci_lo,
            "_ci_hi":            ci_hi,
            "_recovered_bool":   (recovered == "YES"),
        })
    return pd.DataFrame(records)


def _assignment_accuracy(model: dict, df: pd.DataFrame, truth: dict) -> float:
    """Group assignment accuracy (best over all label permutations)."""
    assignments  = get_subject_assignments(model, df)
    true_map     = truth["assignments"]
    pred         = assignments.set_index("ID")["Assigned_Group"]
    true_series  = pd.Series(true_map)
    common       = pred.index.intersection(true_series.index)
    pred_c, true_c = pred.loc[common], true_series.loc[common]
    k = int(pred_c.max())
    best_acc = 0.0
    for perm in permutations(range(1, k + 1)):
        mapping  = {orig: new for orig, new in enumerate(perm, 1)}
        remapped = pred_c.map(mapping)
        best_acc = max(best_acc, float((remapped == true_c).mean()))
    return best_acc


def _run_bic_selection(
    df: pd.DataFrame,
    dist: str,
    cnorm_min: float = 0.0,
    cnorm_max: float = 0.0,
) -> Tuple[Optional[int], List[dict]]:
    """Run AutoTraj search (k=1..3, order=0..2) and return best k by Nagin BIC."""
    valid, _ = run_autotraj(
        df,
        min_groups=1, max_groups=3,
        min_order=0,  max_order=2,
        min_group_pct=5.0, p_val_thresh=0.05,
        dist=dist,
        cnorm_min=cnorm_min, cnorm_max=cnorm_max,
        n_starts=N_STARTS,
    )
    if not valid:
        return None, []
    best_k = len(max(valid, key=lambda m: m["bic_nagin"])["orders"])
    return best_k, valid


# ── figure helpers ────────────────────────────────────────────────────────────

def _make_figure(
    title: str,
    model: dict,
    dist: str,
    time_points: np.ndarray,
    group_params: Optional[List[dict]] = None,
    cnorm_bounds: Tuple[float, float] = (0.0, 1.0),
    output_path: str = "",
) -> None:
    """Plot estimated trajectories (solid) and, if provided, true trajectories (dashed)."""
    k      = len(model["orders"])
    orders = model["orders"]
    params = model["result"].x
    se     = model["se_model"]
    pis    = model["pis"]
    t_plot = np.linspace(time_points.min(), time_points.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    legend_handles = []

    for g in range(k):
        color = COLORS[g % len(COLORS)]
        pi_g  = float(pis[g])

        # locate beta block for group g
        b_start = (k - 1) + sum(orders[j] + 1 for j in range(g))
        betas_est = params[b_start: b_start + orders[g] + 1]

        eta = sum(betas_est[p] * t_plot ** p for p in range(len(betas_est)))
        if dist == "LOGIT":
            y_est = 1.0 / (1.0 + np.exp(-eta))
            ylabel = "P(outcome = 1)"
        elif dist == "CNORM":
            y_est = np.clip(eta, cnorm_bounds[0], cnorm_bounds[1])
            ylabel = "Outcome (continuous)"
        else:  # POISSON / ZIP
            y_est = np.exp(np.clip(eta, -10, 10))
            ylabel = "Expected count"

        ax.plot(t_plot, y_est, color=color, lw=2.5,
                label=f"G{g + 1} est. (π={pi_g:.0%})")

        # Optional: rough 95% CI band using diagonal SE approximation
        se_eta = np.sqrt(sum(
            (se[b_start + p] * t_plot ** p) ** 2 for p in range(len(betas_est))
        ))
        if dist == "LOGIT":
            lo = 1.0 / (1.0 + np.exp(-(eta - 1.96 * se_eta)))
            hi = 1.0 / (1.0 + np.exp(-(eta + 1.96 * se_eta)))
        elif dist == "CNORM":
            lo = np.clip(eta - 1.96 * se_eta, cnorm_bounds[0], cnorm_bounds[1])
            hi = np.clip(eta + 1.96 * se_eta, cnorm_bounds[0], cnorm_bounds[1])
        else:
            lo = np.exp(np.clip(eta - 1.96 * se_eta, -10, 10))
            hi = np.exp(np.clip(eta + 1.96 * se_eta, -10, 10))
        ax.fill_between(t_plot, lo, hi, color=color, alpha=0.12)
        legend_handles.append(mpatches.Patch(color=color,
                                              label=f"G{g + 1} est. (π={pi_g:.0%})"))

        # true trajectories (dashed)
        if group_params is not None and g < len(group_params):
            # match by intercept to find which true group aligns to this estimated group
            # (handled by caller providing group_params in sorted order)
            betas_true = group_params[g]["betas"]
            eta_true   = sum(betas_true[p] * t_plot ** p for p in range(len(betas_true)))
            if dist == "LOGIT":
                y_true = 1.0 / (1.0 + np.exp(-eta_true))
            elif dist == "CNORM":
                y_true = np.clip(eta_true, cnorm_bounds[0], cnorm_bounds[1])
            else:
                y_true = np.exp(np.clip(eta_true, -10, 10))
            ax.plot(t_plot, y_true, color=color, lw=1.5, ls="--")

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # custom legend adding "--- True" entry
    if group_params is not None:
        legend_handles.append(
            plt.Line2D([0], [0], color="grey", lw=1.5, ls="--", label="True trajectory")
        )
    ax.legend(handles=legend_handles, fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── benchmark functions ───────────────────────────────────────────────────────

def run_cambridge_benchmark() -> dict:
    """Cambridge Study of Delinquent Development — canonical GBTM dataset.

    Fits a 2-group [2,2] LOGIT model to the real Cambridge data and reports
    parameter estimates, SEs, CIs, and Nagin adequacy metrics.
    No true values are available (real data), so the Recovery column is N/A.
    """
    print("\n[1/4] Cambridge benchmark (real data, N=195, T=23)...")
    t0 = time.time()

    wide = pd.read_csv(CAMBRIDGE_PATH, sep=r"\s+")
    df   = prep_trajectory_data(wide)

    model = run_single_model(df, orders_list=[2, 2],
                             dist="LOGIT", n_starts=N_STARTS)

    param_rows = _extract_beta_rows(model, dist="LOGIT")
    rec_df     = _build_recovery_df(param_rows, true_values={})

    k          = len(model["orders"])
    group_names = [f"Group {g + 1}" for g in range(k)]
    assignments = get_subject_assignments(model, df)
    adeq_df, rel_entropy = calc_model_adequacy(assignments, model["pis"], group_names)

    elapsed = time.time() - t0
    print(f"    LL={model['ll']:.3f}  BIC(Nagin)={model['bic_nagin']:.3f}  "
          f"H_rel={rel_entropy:.4f}  [{elapsed:.1f}s]")

    return {
        "name":          "Cambridge LOGIT [2,2]",
        "dist":          "LOGIT",
        "n_subjects":    df["ID"].nunique(),
        "n_timepoints":  df.groupby("ID").size().max(),
        "ll":            model["ll"],
        "bic_nagin":     model["bic_nagin"],
        "bic_selected_k": 2,
        "true_k":        None,
        "model":         model,
        "df":            df,
        "recovery_df":   rec_df,
        "adequacy_df":   adeq_df,
        "rel_entropy":   rel_entropy,
        "assign_acc":    None,   # no ground truth
        "elapsed":       elapsed,
        "group_params_sorted": None,
        "time_points":   np.sort(df["Time"].unique()),
    }


def run_logit_benchmark() -> dict:
    """LOGIT simulation: 2 groups, linear trajectories, N=500."""
    print("\n[2/4] LOGIT simulation benchmark (N=500, T=10, k=2)...")
    t0 = time.time()

    # Group 1 (40%): rising high-risk  — lower intercept → sorted first
    # Group 2 (60%): flat low-risk     — higher intercept → sorted second
    true_group_params = [
        {"betas": [-2.0,  3.5]},   # sorted group 1 after sort_by_intercept
        {"betas": [-0.5,  0.0]},   # sorted group 2
    ]
    true_proportions = [0.40, 0.60]   # after sorting by intercept

    # Simulate using the original (pre-sort) order then fit
    df, truth = simulate_logit_trajectories(
        n_subjects=N_SUBJECTS,
        time_points=TIME_POINTS,
        group_params=[
            {"betas": [-0.5,  0.0]},   # simulation group 1 (60%)
            {"betas": [-2.0,  3.5]},   # simulation group 2 (40%)
        ],
        group_proportions=[0.60, 0.40],
        seed=SEED,
    )

    model = run_single_model(df, orders_list=[1, 1],
                             dist="LOGIT", n_starts=N_STARTS)

    param_rows = _extract_beta_rows(model, dist="LOGIT")

    # True values in sorted order (sort_groups_by_intercept → ascending β_0)
    true_values = {
        "beta0_g1": -2.0, "beta1_g1":  3.5,
        "beta0_g2": -0.5, "beta1_g2":  0.0,
    }
    rec_df = _build_recovery_df(param_rows, true_values)

    bic_k, _ = _run_bic_selection(df, dist="LOGIT")
    acc       = _assignment_accuracy(model, df, truth)
    elapsed   = time.time() - t0

    print(f"    LL={model['ll']:.3f}  BIC(Nagin)={model['bic_nagin']:.3f}  "
          f"BIC selects k={bic_k}  AssignAcc={acc:.1%}  [{elapsed:.1f}s]")

    return {
        "name":          "LOGIT Simulation [1,1]",
        "dist":          "LOGIT",
        "n_subjects":    N_SUBJECTS,
        "n_timepoints":  len(TIME_POINTS),
        "ll":            model["ll"],
        "bic_nagin":     model["bic_nagin"],
        "bic_selected_k": bic_k,
        "true_k":        2,
        "model":         model,
        "df":            df,
        "recovery_df":   rec_df,
        "adequacy_df":   None,
        "rel_entropy":   None,
        "assign_acc":    acc,
        "elapsed":       elapsed,
        "group_params_sorted": true_group_params,
        "time_points":   TIME_POINTS,
    }


def run_cnorm_benchmark() -> dict:
    """CNORM simulation: 2 groups, linear trajectories, sigma=0.8, N=500."""
    print("\n[3/4] CNORM simulation benchmark (N=500, T=10, k=2, sigma=0.8)...")
    t0 = time.time()

    true_sigma = 0.8
    cnorm_min, cnorm_max = 0.0, 6.0

    # Well-separated groups (confirmed by test_parameter_recovery)
    # Group 1 (40%): declining   mu = 1.0 - 2.5*t  → lower intercept → sorted first
    # Group 2 (60%): flat-high   mu = 4.5           → higher intercept → sorted second
    true_group_params = [
        {"betas": [1.0, -2.5]},
        {"betas": [4.5,  0.0]},
    ]
    true_proportions = [0.40, 0.60]

    df, truth = simulate_cnorm_trajectories(
        n_subjects=N_SUBJECTS,
        time_points=TIME_POINTS,
        group_params=[
            {"betas": [4.5,  0.0]},   # simulation group 1 (60%)
            {"betas": [1.0, -2.5]},   # simulation group 2 (40%)
        ],
        group_proportions=[0.60, 0.40],
        sigma=true_sigma,
        cnorm_min=cnorm_min,
        cnorm_max=cnorm_max,
        seed=SEED,
    )

    model = run_single_model(df, orders_list=[1, 1], dist="CNORM",
                             cnorm_min=cnorm_min, cnorm_max=cnorm_max,
                             n_starts=N_STARTS)

    param_rows = _extract_beta_rows(model, dist="CNORM")

    true_values = {
        "beta0_g1":  1.0, "beta1_g1": -2.5,
        "beta0_g2":  4.5, "beta1_g2":  0.0,
        "sigma":     true_sigma,
    }
    rec_df = _build_recovery_df(param_rows, true_values)

    bic_k, _ = _run_bic_selection(df, dist="CNORM",
                                   cnorm_min=cnorm_min, cnorm_max=cnorm_max)
    acc       = _assignment_accuracy(model, df, truth)
    elapsed   = time.time() - t0

    print(f"    LL={model['ll']:.3f}  BIC(Nagin)={model['bic_nagin']:.3f}  "
          f"BIC selects k={bic_k}  AssignAcc={acc:.1%}  [{elapsed:.1f}s]")

    return {
        "name":          "CNORM Simulation [1,1]",
        "dist":          "CNORM",
        "n_subjects":    N_SUBJECTS,
        "n_timepoints":  len(TIME_POINTS),
        "ll":            model["ll"],
        "bic_nagin":     model["bic_nagin"],
        "bic_selected_k": bic_k,
        "true_k":        2,
        "model":         model,
        "df":            df,
        "recovery_df":   rec_df,
        "adequacy_df":   None,
        "rel_entropy":   None,
        "assign_acc":    acc,
        "elapsed":       elapsed,
        "group_params_sorted": true_group_params,
        "cnorm_bounds":  (cnorm_min, cnorm_max),
        "time_points":   TIME_POINTS,
    }


def run_poisson_benchmark() -> dict:
    """Poisson simulation: 2 groups, linear trajectories, N=500."""
    print("\n[4/4] Poisson simulation benchmark (N=500, T=10, k=2)...")
    t0 = time.time()

    # Group 1 (40%): low-count  log(mu) = 0.5 + 0.3*t  → lower intercept → sorted first
    # Group 2 (60%): high-count log(mu) = 2.0 - 0.3*t  → higher intercept → sorted second
    true_group_params = [
        {"betas": [0.5,  0.3]},
        {"betas": [2.0, -0.3]},
    ]
    true_proportions = [0.40, 0.60]

    df, truth = simulate_poisson_trajectories(
        n_subjects=N_SUBJECTS,
        time_points=TIME_POINTS,
        group_params=[
            {"betas": [2.0, -0.3]},   # simulation group 1 (60%)
            {"betas": [0.5,  0.3]},   # simulation group 2 (40%)
        ],
        group_proportions=[0.60, 0.40],
        seed=SEED,
    )

    model = run_single_model(df, orders_list=[1, 1],
                             dist="POISSON", n_starts=N_STARTS)

    param_rows = _extract_beta_rows(model, dist="POISSON")

    true_values = {
        "beta0_g1": 0.5, "beta1_g1":  0.3,
        "beta0_g2": 2.0, "beta1_g2": -0.3,
    }
    rec_df = _build_recovery_df(param_rows, true_values)

    bic_k, _ = _run_bic_selection(df, dist="POISSON")
    acc       = _assignment_accuracy(model, df, truth)
    elapsed   = time.time() - t0

    print(f"    LL={model['ll']:.3f}  BIC(Nagin)={model['bic_nagin']:.3f}  "
          f"BIC selects k={bic_k}  AssignAcc={acc:.1%}  [{elapsed:.1f}s]")

    return {
        "name":          "Poisson Simulation [1,1]",
        "dist":          "POISSON",
        "n_subjects":    N_SUBJECTS,
        "n_timepoints":  len(TIME_POINTS),
        "ll":            model["ll"],
        "bic_nagin":     model["bic_nagin"],
        "bic_selected_k": bic_k,
        "true_k":        2,
        "model":         model,
        "df":            df,
        "recovery_df":   rec_df,
        "adequacy_df":   None,
        "rel_entropy":   None,
        "assign_acc":    acc,
        "elapsed":       elapsed,
        "group_params_sorted": true_group_params,
        "time_points":   TIME_POINTS,
    }


# ── output writers ─────────────────────────────────────────────────────────────

def _df_to_md_table(df: pd.DataFrame, display_cols: List[str]) -> str:
    """Render a DataFrame subset as a GitHub-flavoured markdown table."""
    sub  = df[display_cols]
    rows = []
    rows.append("| " + " | ".join(display_cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
    for _, row in sub.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in display_cols) + " |")
    return "\n".join(rows)


def _group_prop_table(model: dict) -> str:
    """Markdown table of estimated group proportions."""
    k    = len(model["orders"])
    pis  = model["pis"]
    rows = ["| Group | Est. π (%) |", "| --- | --- |"]
    for g in range(k):
        rows.append(f"| Group {g + 1} | {pis[g] * 100:.1f}% |")
    return "\n".join(rows)


def write_markdown_report(benchmarks: List[dict]) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = []
    lines += [
        "# AutoTraj Benchmark Report",
        "",
        f"*Generated: {ts}*",
        "",
        "This report provides parameter recovery evidence for the AutoTraj validation paper.",
        "Simulated benchmarks compare AutoTraj estimates against known ground-truth parameters.",
        "The Cambridge benchmark validates against the canonical Nagin (1999) real-world dataset.",
        "",
        "---",
        "",
    ]

    # Overall summary table
    lines += ["## Executive Summary", ""]
    lines += ["| Benchmark | N | T | Dist | LL | BIC(Nagin) | True k | BIC k | Assign Acc | β Recovery |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for bm in benchmarks:
        rec  = bm["recovery_df"]
        mask = rec["_recovered_bool"] & rec["_true"].notna()
        n_checked   = int(mask.sum())
        total_check = int(rec["_true"].notna().sum())
        pct_rec = f"{n_checked / total_check:.0%}" if total_check > 0 else "N/A"
        acc_str = f"{bm['assign_acc']:.1%}" if bm["assign_acc"] is not None else "N/A"
        true_k  = str(bm["true_k"]) if bm["true_k"] is not None else "N/A"
        sel_k   = str(bm["bic_selected_k"]) if bm["bic_selected_k"] is not None else "N/A"
        lines.append(
            f"| {bm['name']} | {bm['n_subjects']} | {bm['n_timepoints']} "
            f"| {bm['dist']} | {bm['ll']:.1f} | {bm['bic_nagin']:.1f} "
            f"| {true_k} | {sel_k} | {acc_str} | {pct_rec} ({n_checked}/{total_check}) |"
        )
    lines += ["", "---", ""]

    # Per-benchmark sections
    for bm in benchmarks:
        rec = bm["recovery_df"]
        lines += [f"## {bm['name']}", ""]

        lines += [
            "### Dataset Summary",
            "",
            f"- **N subjects:** {bm['n_subjects']}",
            f"- **Time points:** {bm['n_timepoints']}",
            f"- **Distribution:** {bm['dist']}",
            f"- **Log-likelihood:** {bm['ll']:.4f}",
            f"- **BIC (Nagin):** {bm['bic_nagin']:.4f}",
            "",
        ]

        lines += ["### Estimated Group Proportions", "", _group_prop_table(bm["model"]), ""]

        lines += [
            "### Parameter Recovery Table",
            "",
            "> 95% CI = Estimate ± 1.96 × SE(model-based Hessian). "
            "Recovery = YES if true value falls within 95% CI.",
            "",
        ]
        display_cols = ["Parameter", "True Value", "AutoTraj Estimate", "Std Error", "95% CI", "Recovery"]
        lines += [_df_to_md_table(rec, display_cols), ""]

        # recovery summary
        total = int(rec["_true"].notna().sum())
        n_rec = int((rec["_recovered_bool"] & rec["_true"].notna()).sum())
        if total > 0:
            lines += [f"**Recovery: {n_rec}/{total} parameters ({n_rec / total:.0%})**", ""]

        # adequacy metrics (Cambridge only)
        if bm["adequacy_df"] is not None:
            lines += [
                "### Adequacy Metrics (Nagin 2005 thresholds: AvePP > 0.70, OCC > 5.0, H_rel > 0.50)",
                "",
                _df_to_md_table(
                    bm["adequacy_df"],
                    ["Group", "Assigned N", "Estimated Pi (%)", "AvePP", "OCC"],
                ),
                "",
                f"**Relative entropy (H_rel): {bm['rel_entropy']:.4f}**",
                "",
            ]

        if bm["assign_acc"] is not None:
            lines += [f"**Group assignment accuracy: {bm['assign_acc']:.1%}**", ""]

        # figure reference
        safe_name = bm["name"].replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")
        fig_name  = f"{safe_name}.png"
        lines += [
            "### Trajectory Figure",
            "",
            f"![{bm['name']}](benchmark_figures/{fig_name})",
            "",
            "---",
            "",
        ]

    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  Markdown report -> {REPORT_MD}")


def write_csv_results(benchmarks: List[dict]) -> None:
    rows = []
    for bm in benchmarks:
        rec = bm["recovery_df"]
        for _, r in rec.iterrows():
            rows.append({
                "benchmark":    bm["name"],
                "dist":         bm["dist"],
                "n_subjects":   bm["n_subjects"],
                "n_timepoints": bm["n_timepoints"],
                "parameter":    r["Parameter"],
                "true_value":   r["_true"],
                "estimate":     r["_est"],
                "ci_lo":        r["_ci_lo"],
                "ci_hi":        r["_ci_hi"],
                "recovered":    r["Recovery"],
                "ll":           bm["ll"],
                "bic_nagin":    bm["bic_nagin"],
                "assign_acc":   bm["assign_acc"],
                "true_k":       bm["true_k"],
                "bic_k":        bm["bic_selected_k"],
            })
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False)
    print(f"  CSV results      -> {REPORT_CSV}")


def write_figures(benchmarks: List[dict]) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for bm in benchmarks:
        safe_name = bm["name"].replace(" ", "_").replace("[", "").replace("]", "").replace(",", "")
        out_path  = os.path.join(FIGURES_DIR, f"{safe_name}.png")
        cnorm_bounds = bm.get("cnorm_bounds", (0.0, 1.0))
        _make_figure(
            title       = bm["name"],
            model       = bm["model"],
            dist        = bm["dist"],
            time_points = bm["time_points"],
            group_params = bm["group_params_sorted"],
            cnorm_bounds = cnorm_bounds,
            output_path  = out_path,
        )
    print(f"  Figures          -> {FIGURES_DIR}/")


# ── console summary ───────────────────────────────────────────────────────────

def print_summary(benchmarks: List[dict]) -> None:
    """Print the paper summary line and per-benchmark detail."""
    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)

    total_params   = 0
    total_recovered = 0
    k_correct      = 0
    k_total        = 0

    for bm in benchmarks:
        rec  = bm["recovery_df"]
        has_truth = rec["_true"].notna()
        n_chk = int(has_truth.sum())
        n_rec = int((rec["_recovered_bool"] & has_truth).sum())
        total_params    += n_chk
        total_recovered += n_rec

        if bm["true_k"] is not None:
            k_total += 1
            if bm["bic_selected_k"] == bm["true_k"]:
                k_correct += 1

        # mean absolute bias (only params with true values)
        if n_chk > 0:
            truth_vals = rec.loc[has_truth, "_true"].values.astype(float)
            est_vals   = rec.loc[has_truth, "_est"].values.astype(float)
            mab = float(np.mean(np.abs(est_vals - truth_vals)))
        else:
            mab = float("nan")

        acc_str  = f"{bm['assign_acc']:.1%}" if bm["assign_acc"] is not None else "N/A"
        pct_rec  = f"{n_rec / n_chk:.0%}" if n_chk > 0 else "N/A"
        k_str    = (f"k_BIC={bm['bic_selected_k']}, k_true={bm['true_k']}"
                    if bm["true_k"] is not None else "real data (no true k)")
        mab_str  = f"{mab:.4f}" if np.isfinite(mab) else "N/A"

        print(f"\n  {bm['name']}")
        print(f"    Recovery: {n_rec}/{n_chk} parameters ({pct_rec})")
        print(f"    Mean |bias|:  {mab_str}")
        print(f"    Assign acc:   {acc_str}")
        print(f"    BIC selection: {k_str}")
        if bm.get("rel_entropy") is not None:
            print(f"    Rel. entropy: {bm['rel_entropy']:.4f}")

    pct_total = f"{total_recovered / total_params:.0%}" if total_params > 0 else "N/A"
    print("\n" + "-" * 65)
    print(
        f"BENCHMARK SUMMARY: {total_recovered}/{total_params} parameters recovered "
        f"({pct_total}), K correct in {k_correct}/{k_total} cases"
    )
    print("=" * 65)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"AutoTraj Benchmark Report Generator")
    print(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"n_starts={N_STARTS}, N_subjects={N_SUBJECTS}, seed={SEED}")

    t_total = time.time()

    benchmarks = [
        run_cambridge_benchmark(),
        run_logit_benchmark(),
        run_cnorm_benchmark(),
        run_poisson_benchmark(),
    ]

    print("\nWriting outputs...")
    write_markdown_report(benchmarks)
    write_csv_results(benchmarks)
    write_figures(benchmarks)

    print_summary(benchmarks)

    total_elapsed = time.time() - t_total
    print(f"\nTotal runtime: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
