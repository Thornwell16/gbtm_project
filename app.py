"""
app.py — AutoTraj Streamlit Web Interface
==========================================
Interactive front-end for the AutoTraj GBTM engine (main.py).  Provides a
point-and-click workflow covering data loading, model configuration, result
visualisation, and export.

Layout overview
---------------
The app is organised into a sidebar (configuration) and five result tabs:

  tab_viz  — Trajectory Plot
      Interactive Plotly or Matplotlib trajectory curves with optional 95%
      delta-method confidence bands, individual-subject spaghetti overlays,
      fitted model equations (LaTeX), and download buttons (SVG / PNG / CSV).

  tab_est  — Parameter Estimates
      Full coefficient table (Estimate, model SE, robust SE, T-stat, P-value)
      for all trajectory betas, dropout gammas (if fitted), CNORM sigma, and
      ZIP zeta parameters.

  tab_adq  — Adequacy Diagnostics
      Per-group AvePP bar chart, posterior probability heatmap, observed vs.
      estimated overlay, residual histogram + Q-Q plot + outlier table, BIC
      elbow plot, and per-group entropy decomposition.

  tab_char — Group Characteristics
      Posterior-weighted baseline demographic table (TableOne), sorted by
      group assignment probability.

  tab_comp — Model Comparison
      Interactive BIC elbow plot (all evaluated models), per-group membership
      statistics, and a full results ZIP export.

Key helper functions
--------------------
  _beta_start_indices      : Index mapping from orders_list to params vector.
  _compute_ci_band         : Diagonal delta-method 95% CI on response scale.
  get_parameter_estimates_for_ui : Build parameter table DataFrame.
  _build_equation_latex    : LaTeX string for one group's fitted equation.
  _posterior_heatmap       : E[P(g'|i) | assigned group = g] heatmap.
  _entropy_decomposition   : Per-group relative entropy contributions.
  _obs_vs_est_figure       : Posterior-weighted observed vs. estimated plot.
  _residual_analysis       : Histogram, Q-Q, and outlier detection.
  _make_model_summary_txt  : Plain-text model summary for ZIP export.

References
----------
See MATH.md for all formula derivations.  See main.py module docstring for
the mathematical model and optimisation details.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import zipfile
import pickle
from types import SimpleNamespace
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, probplot

try:
    from tableone import TableOne
    HAS_TABLEONE = True
except ImportError:
    HAS_TABLEONE = False

from main import (
    prep_trajectory_data,
    run_autotraj,
    run_single_model,
    calc_logit_prob_jit,
    create_design_matrix_jit,
    get_subject_assignments,
    calc_model_adequacy,
    run_joint_dual_trajectory_model,
    get_joint_subject_assignments,
    calc_joint_model_adequacy,
    _joint_layout,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _beta_start_indices(orders_list, n_mix=1):
    """Return list of (start_idx, n_betas) tuples for each group's beta block.

    n_mix (V3.0): mixing-covariate block width per group (P+1, incl. intercept).
    Default 1 = intercept-only (V1.5.0-equivalent), matching the old `k - 1`
    Gamma/theta block size.
    """
    k = len(orders_list)
    idx = (k - 1) * n_mix
    out = []
    for g in range(k):
        n = orders_list[g] + 1
        out.append((idx, n))
        idx += n
    return out


def _delta_start_index(orders_list, n_mix=1):
    """Return the flat-vector index where the TVC (delta) block begins (V3.0).

    The delta block has fixed width n_tvc per group (all k groups, no
    reference-group exclusion) — this returns only the *start* index; callers
    slice model_dict['n_tvc'] * k entries from there, or per-group offsets of
    `delta_start + g * n_tvc`.
    """
    k = len(orders_list)
    num_betas = sum(o + 1 for o in orders_list)
    return (k - 1) * n_mix + num_betas


def _compute_ci_band(smooth_times, g_betas, order, se_model, beta_start, n_betas, dist_type, z=1.96, eta_offset=0.0):
    """Delta-method 95% CI band for a single group trajectory.

    Uses the diagonal of the model covariance (se_model^2) — a valid approximation
    when off-diagonal beta covariances are small, and avoids the complexity of
    re-permuting the full matrix after label-sorting.

    Parameters
    ----------
    smooth_times : array of evaluation points
    g_betas      : (n_betas,) fitted coefficients
    order        : polynomial order
    se_model     : full se_model array (sorted, aligned with result.x)
    beta_start   : index of this group's first beta in se_model
    n_betas      : number of betas for this group
    dist_type    : 'LOGIT' | 'CNORM' | 'POISSON' | 'ZIP'
    z            : critical value (default 1.96 for 95 %)
    eta_offset   : constant added to eta (V3.0: TVC deflection evaluated at the
                   sample-mean TVC level, e.g. delta_g . mean(TVC); its own
                   sampling variance is not incorporated into the band, a
                   deliberate simplification — the band still reflects beta
                   uncertainty exactly, just centred at a different eta).

    Returns
    -------
    lo, hi : arrays of lower / upper CI on the *response* scale
    """
    X       = create_design_matrix_jit(smooth_times, order)         # (T, n_betas)
    se_beta = se_model[beta_start:beta_start + n_betas]             # (n_betas,)
    # Var(X @ beta) ≈ sum_p X_p^2 * se_p^2  (diagonal delta method)
    var_eta = (X ** 2) @ (se_beta ** 2)
    var_eta = np.clip(var_eta, 0.0, None)
    se_eta  = np.sqrt(var_eta)

    eta = X @ g_betas + eta_offset

    if dist_type == 'LOGIT':
        lo = 1.0 / (1.0 + np.exp(-(eta - z * se_eta)))
        hi = 1.0 / (1.0 + np.exp(-(eta + z * se_eta)))
    elif dist_type == 'POISSON':
        lo = np.exp(eta - z * se_eta)
        hi = np.exp(eta + z * se_eta)
    else:  # CNORM, ZIP: CI on linear predictor
        lo = eta - z * se_eta
        hi = eta + z * se_eta

    return lo, hi


def get_parameter_estimates_for_ui(model_dict, group_names=None):
    orders     = model_dict['orders']
    params     = model_dict['result'].x
    se_model   = model_dict['se_model']
    se_robust  = model_dict['se_robust']
    use_dropout = model_dict['use_dropout']
    dof        = model_dict['dof']
    model_type = model_dict.get('dist', 'LOGIT')
    n_mix      = model_dict.get('n_mix', 1)
    n_tvc      = model_dict.get('n_tvc', 0)
    baseline_cov_names = model_dict.get('baseline_cov_cols') or []
    tvc_names          = model_dict.get('tvc_cols') or []

    k = len(orders)
    if group_names is None or len(group_names) != k:
        group_names = [f"Group {g+1}" for g in range(k)]

    data = []

    def _row(component, group, parameter, est, err_m, err_r):
        t_stat = est / err_m if err_m > 0 else 0
        p_val  = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
        return {
            "Component": component, "Group": str(group), "Parameter": parameter,
            "Estimate": round(est, 5), "Standard Error": round(err_m, 5),
            "Robust SE": round(err_r, 5),
            "T for H0: Param=0": round(t_stat, 3),
            "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001",
        }

    # V3.0: Gamma (mixing covariate) rows — one per non-reference group x covariate.
    # Row 0 (reference group) is implicit-zero and not shown.
    mix_labels = ["Intercept"] + list(baseline_cov_names)
    for g in range(1, k):
        for p in range(n_mix):
            idx = (g - 1) * n_mix + p
            label = mix_labels[p] if p < len(mix_labels) else f"Covariate {p}"
            data.append(_row(
                "Mixing Covariate", str(group_names[g]), f"Gamma: {label}",
                params[idx], se_model[idx], se_robust[idx],
            ))

    current_beta_idx  = (k - 1) * n_mix
    delta_start_idx    = (k - 1) * n_mix + sum(o + 1 for o in orders)
    current_gamma_idx = delta_start_idx + k * n_tvc
    labels       = ["Intercept", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
    gamma_labels = ["Dropout: Intercept", "Dropout: Time", "Dropout: Prev Outcome"]

    for g in range(k):
        n_betas = orders[g] + 1
        for b_idx in range(n_betas):
            est   = params[current_beta_idx + b_idx]
            err_m = se_model[current_beta_idx + b_idx]
            err_r = se_robust[current_beta_idx + b_idx]
            data.append(_row("Trajectory", group_names[g], labels[b_idx], est, err_m, err_r))
        current_beta_idx += n_betas

        # V3.0: delta (TVC) rows — fixed n_tvc-width block per group.
        for q in range(n_tvc):
            idx = delta_start_idx + g * n_tvc + q
            label = tvc_names[q] if q < len(tvc_names) else f"TVC {q}"
            data.append(_row(
                "TVC Deflection", str(group_names[g]), f"Delta: {label}",
                params[idx], se_model[idx], se_robust[idx],
            ))

        if use_dropout:
            for gam_idx in range(3):
                est   = params[current_gamma_idx + gam_idx]
                err_m = se_model[current_gamma_idx + gam_idx]
                err_r = se_robust[current_gamma_idx + gam_idx]
                data.append(_row("Dropout", group_names[g], gamma_labels[gam_idx], est, err_m, err_r))
            current_gamma_idx += 3

    if model_type == 'CNORM':
        sigma_idx = len(params) - 1
        est   = np.exp(params[sigma_idx])
        err_m = se_model[sigma_idx] * est
        err_r = se_robust[sigma_idx] * est
        t_stat = est / err_m if err_m > 0 else 0
        p_val  = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
        data.append({
            "Component": "Variance", "Group": "All Groups",
            "Parameter": "Sigma (Standard Deviation)",
            "Estimate": round(est, 5), "Standard Error": round(err_m, 5),
            "Robust SE": round(err_r, 5),
            "T for H0: Param=0": round(t_stat, 3),
            "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
        })

    if model_type == 'ZIP':
        zeta_start_idx = len(params) - k
        for g in range(k):
            est   = params[zeta_start_idx + g]
            err_m = se_model[zeta_start_idx + g]
            err_r = se_robust[zeta_start_idx + g]
            omega = 1.0 / (1.0 + np.exp(-est))
            t_stat = est / err_m if err_m > 0 else 0
            p_val  = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
            data.append({
                "Component": "Zero Inflation", "Group": str(group_names[g]),
                "Parameter": f"Zeta (logit of \u03c9,  \u03c9={omega:.3f})",
                "Estimate": round(est, 5), "Standard Error": round(err_m, 5),
                "Robust SE": round(err_r, 5),
                "T for H0: Param=0": round(t_stat, 3),
                "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
            })

    return pd.DataFrame(data)


def get_joint_parameter_estimates_for_ui(model_dict):
    """Parameter table for a V5.0 joint dual-trajectory model.

    Mirrors get_parameter_estimates_for_ui but for the joint layout
    (Theta_joint | Y-BLOCK | Z-BLOCK) — no mixing covariates/TVC, since V5.0
    doesn't compose with those in this pass (explicit scope boundary).
    """
    orders_y, orders_z = model_dict['orders_y'], model_dict['orders_z']
    k_y, k_z = len(orders_y), len(orders_z)
    params = model_dict['result'].x
    se_model, se_robust = model_dict['se_model'], model_dict['se_robust']
    use_dropout_y, use_dropout_z = model_dict['use_dropout_y'], model_dict['use_dropout_z']
    dist_y, dist_z = model_dict.get('dist_y', 'LOGIT'), model_dict.get('dist_z', 'LOGIT')
    dof = model_dict['dof']

    n_theta, y_beta_start, z_beta_start, num_betas_y, num_betas_z, num_params = _joint_layout(
        k_y, k_z, orders_y, orders_z, use_dropout_y, dist_y, use_dropout_z, dist_z
    )

    data = []
    labels = ["Intercept", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
    gamma_labels = ["Dropout: Intercept", "Dropout: Time", "Dropout: Prev Outcome"]

    def _row(component, group, parameter, idx):
        est, err_m, err_r = params[idx], se_model[idx], se_robust[idx]
        t_stat = est / err_m if err_m > 0 else 0
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
        return {
            "Component": component, "Group": str(group), "Parameter": parameter,
            "Estimate": round(est, 5), "Standard Error": round(err_m, 5),
            "Robust SE": round(err_r, 5),
            "T for H0: Param=0": round(t_stat, 3),
            "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001",
        }

    idx = 0
    for g in range(k_y):
        for h in range(k_z):
            if g == 0 and h == 0:
                continue
            data.append(_row("Joint Mixing", f"Y{g+1}/Z{h+1}", "Theta (log-odds vs. Y1/Z1)", idx))
            idx += 1

    def _outcome_rows(outcome_label, k, orders, use_dropout, dist, block_start):
        cur = block_start
        for g in range(k):
            n_betas = orders[g] + 1
            for b in range(n_betas):
                data.append(_row(f"{outcome_label} Trajectory", f"Group {g+1}", labels[b], cur + b))
            cur += n_betas
            if use_dropout:
                for gi in range(3):
                    data.append(_row(f"{outcome_label} Dropout", f"Group {g+1}", gamma_labels[gi], cur + gi))
                cur += 3
        if dist == 'CNORM':
            est = np.exp(params[cur])
            err_m, err_r = se_model[cur] * est, se_robust[cur] * est
            t_stat = est / err_m if err_m > 0 else 0
            p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
            data.append({
                "Component": f"{outcome_label} Variance", "Group": "All Groups",
                "Parameter": "Sigma (Standard Deviation)",
                "Estimate": round(est, 5), "Standard Error": round(err_m, 5),
                "Robust SE": round(err_r, 5),
                "T for H0: Param=0": round(t_stat, 3),
                "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001",
            })
            cur += 1
        elif dist == 'ZIP':
            for g in range(k):
                data.append(_row(f"{outcome_label} Zero Inflation", f"Group {g+1}", "Zeta (logit of omega)", cur + g))
            cur += k
        return cur

    _outcome_rows("Y", k_y, orders_y, use_dropout_y, dist_y, y_beta_start)
    _outcome_rows("Z", k_z, orders_z, use_dropout_z, dist_z, z_beta_start)

    return pd.DataFrame(data)


def _build_equation_latex(g_betas, order, dist_type, group_name, g_idx, winning_result, winning_orders,
                           g_delta=None, tvc_names=None):
    """Return a LaTeX string for one group's fitted equation.

    g_delta/tvc_names (V3.0): if TVCs are present, append their deflection
    terms symbolically (using the covariate's name, not a numeric value,
    since the term's contribution varies per subject/time).
    """
    terms = []
    poly_terms = []
    coeff_labels = ["", "t", "t^2", "t^3", "t^4", "t^5"]
    for p in range(order + 1):
        c = g_betas[p]
        sign = "+" if c >= 0 and p > 0 else ""
        coeff_str = f"{sign}{c:.3f}"
        if p == 0:
            poly_terms.append(coeff_str)
        else:
            poly_terms.append(f"{coeff_str}{coeff_labels[p]}")
    poly = " ".join(poly_terms)

    if g_delta is not None and len(g_delta) > 0:
        for q, d in enumerate(g_delta):
            name = tvc_names[q] if tvc_names and q < len(tvc_names) else f"z_{{{q+1}}}"
            safe_name = str(name).replace("_", r"\_")
            sign = "+" if d >= 0 else ""
            poly += rf" {sign}{d:.3f}\cdot\text{{{safe_name}}}"

    if dist_type == 'LOGIT':
        lhs = r"\text{logit}(p)"
    elif dist_type == 'CNORM':
        lhs = r"\mu"
    elif dist_type in ('POISSON', 'ZIP'):
        lhs = r"\log(\mu)"
    else:
        lhs = r"\mu"

    if dist_type == 'ZIP':
        k = len(winning_orders)
        zeta_g = winning_result.x[len(winning_result.x) - k + g_idx]
        omega_g = 1.0 / (1.0 + np.exp(-zeta_g))
        extra = rf"\quad \omega={omega_g:.3f}"
    else:
        extra = ""

    return rf"\text{{{group_name}}}: \; {lhs} = {poly}{extra}"


def _build_mixing_equation_latex(gamma_matrix, group_names, baseline_cov_names):
    """Return LaTeX strings for the V3.0 mixing-covariate equation, one line
    per non-reference group: theta_g(x) = Gamma_g0 + Gamma_g1*x1 + ...

    gamma_matrix: (k, n_mix) array, row 0 (reference) is all zeros and skipped.
    Returns [] when there are no baseline covariates (n_mix == 1) — callers
    should skip rendering this equation entirely in that case, since it adds
    no information beyond the group proportion table.
    """
    k, n_mix = gamma_matrix.shape
    if n_mix <= 1:
        return []
    labels = ["1"] + [str(c).replace("_", r"\_") for c in baseline_cov_names]
    lines = []
    for g in range(1, k):
        terms = []
        for p in range(n_mix):
            c = gamma_matrix[g, p]
            sign = "+" if c >= 0 and p > 0 else ""
            coeff_str = f"{sign}{c:.3f}"
            if p == 0:
                terms.append(coeff_str)
            else:
                terms.append(rf"{coeff_str}\cdot\text{{{labels[p]}}}")
        lines.append(rf"\text{{{group_names[g]}}}: \; \theta = {' '.join(terms)}")
    return lines


def _make_model_summary_txt(winning_model, group_names, rel_entropy):
    """Return a human-readable model summary string."""
    lines = []
    lines.append("=" * 60)
    lines.append("GBTM MODEL SUMMARY — AutoTraj")
    lines.append("=" * 60)
    orders = winning_model['orders']
    dist   = winning_model.get('dist', 'LOGIT')
    k      = len(orders)
    lines.append(f"Distribution : {dist}")
    lines.append(f"Groups       : {k}")
    lines.append(f"Orders       : {orders}")
    lines.append(f"LL           : {winning_model['ll']:.4f}")
    lines.append(f"BIC (Nagin)  : {winning_model['bic_nagin']:.4f}")
    lines.append(f"BIC (Std)    : {winning_model['bic_standard']:.4f}")
    lines.append(f"AIC (Nagin)  : {winning_model['aic_nagin']:.4f}")
    lines.append(f"AIC (Std)    : {winning_model['aic_standard']:.4f}")
    lines.append(f"Rel. Entropy : {rel_entropy:.4f}")
    lines.append("")
    lines.append("Group Membership Probabilities:")
    for g in range(k):
        lines.append(f"  {group_names[g]}: {winning_model['pis'][g]*100:.1f}%")
    lines.append("")
    lines.append("Parameter Estimates (Trajectory Betas):")
    params    = winning_model['result'].x
    se_model  = winning_model['se_model']
    beta_info = _beta_start_indices(orders, n_mix=winning_model.get('n_mix', 1))
    for g in range(k):
        start, n = beta_info[g]
        lines.append(f"  {group_names[g]}:")
        for p in range(n):
            lines.append(f"    beta_{p} = {params[start+p]:.5f}  (SE={se_model[start+p]:.5f})")
    return "\n".join(lines)


def _generate_plain_language_summary(winning_model, group_names, long_df, adq_df, rel_entropy, dist_type):
    """Return a rule-based, plain-English interpretation of a fitted GBTM model.

    Classifies each group's trajectory by its predicted level (relative to
    the other groups) and direction (stable/increasing/decreasing, comparing
    the fitted value at the first vs. last observed time point), then
    summarizes overall group separation using the standard Nagin (2005)
    adequacy thresholds (relative entropy >= 0.50, AvePP >= 0.70). This is a
    deterministic heuristic, not a model-generated narrative — it always
    describes exactly what the fitted parameters say, nothing more.
    """
    orders = winning_model['orders']
    result = winning_model['result']
    pis = winning_model['pis']
    k = len(orders)
    n_mix = winning_model.get('n_mix', 1)
    beta_info = _beta_start_indices(orders, n_mix=n_mix)
    unique_times = np.sort(long_df['Time'].unique())
    t_start, t_end = float(unique_times[0]), float(unique_times[-1])

    group_stats = []
    for g in range(k):
        beta_start, n_betas = beta_info[g]
        g_betas = result.x[beta_start:beta_start + n_betas]
        X_endpoints = create_design_matrix_jit(np.array([t_start, t_end], dtype=np.float64), orders[g])
        eta_endpoints = X_endpoints @ g_betas
        if dist_type == 'LOGIT':
            val_endpoints = 1.0 / (1.0 + np.exp(-np.clip(eta_endpoints, -25, 25)))
        elif dist_type in ('POISSON', 'ZIP'):
            val_endpoints = np.exp(np.clip(eta_endpoints, -20, 20))
        else:
            val_endpoints = eta_endpoints
        val_start, val_end = float(val_endpoints[0]), float(val_endpoints[1])
        group_stats.append({
            'name': group_names[g], 'pct': pis[g] * 100,
            'val_start': val_start, 'val_end': val_end,
            'mean_val': (val_start + val_end) / 2.0,
        })

    sorted_by_level = sorted(group_stats, key=lambda gs: gs['mean_val'])
    n = len(sorted_by_level)
    for i, gs in enumerate(sorted_by_level):
        if n == 1: gs['level'] = "a single-level"
        elif i == 0: gs['level'] = "the lowest-level"
        elif i == n - 1: gs['level'] = "the highest-level"
        else: gs['level'] = "an intermediate-level"

    all_vals = [gs['val_start'] for gs in group_stats] + [gs['val_end'] for gs in group_stats]
    overall_range = max(all_vals) - min(all_vals)
    threshold = max(overall_range * 0.1, 1e-9)
    for gs in group_stats:
        delta = gs['val_end'] - gs['val_start']
        if abs(delta) < threshold:
            gs['direction'] = "stable"
        elif delta > 0:
            gs['direction'] = "increasing"
        else:
            gs['direction'] = "decreasing"

    group_stats.sort(key=lambda gs: -gs['pct'])

    lines = [f"This model identified **{k} distinct trajectory group{'s' if k != 1 else ''}** in the data:"]
    for gs in group_stats:
        lines.append(f"- **{gs['name']}** ({gs['pct']:.1f}% of subjects) followed {gs['level']}, **{gs['direction']}** pattern over the observed time period.")

    entropy_verdict = "good" if rel_entropy >= 0.80 else ("adequate" if rel_entropy >= 0.50 else "weak")
    lines.append("")
    lines.append(
        f"Overall group separation was **{entropy_verdict}** (relative entropy = {rel_entropy:.2f}; "
        f"the conventional adequacy threshold, per Nagin 2005, is ≥ 0.50)."
    )

    if adq_df is not None and len(adq_df) > 0 and 'AvePP' in adq_df.columns:
        try:
            ave_pps = [float(v) for v in adq_df['AvePP'] if v != "N/A"]
            if ave_pps:
                min_avepp = min(ave_pps)
                if min_avepp >= 0.70:
                    lines.append(f"Classification confidence was strong for every group (lowest average posterior probability = {min_avepp:.2f}, above the recommended 0.70 threshold).")
                else:
                    lines.append(f"Classification confidence fell below the recommended 0.70 threshold for at least one group (lowest average posterior probability = {min_avepp:.2f}) — interpret hard group assignments for that group with caution.")
        except (ValueError, TypeError):
            pass

    return "\n".join(lines)


def _build_html_report(winning_model, group_names, estimates_df, adq_df, rel_entropy,
                        summary_txt, equations, png_bytes=None, plain_summary=None):
    """Return a single self-contained HTML report string bundling the model
    summary, parameter table, adequacy table, fitted equations, and (if
    available) the trajectory plot as an embedded base64 PNG — suitable for
    archiving or sharing without needing to reopen AutoTraj.
    """
    import html as _html
    import base64

    orders = winning_model['orders']
    dist = winning_model.get('dist', 'LOGIT')
    k = len(orders)

    img_tag = ""
    if png_bytes is not None:
        b64 = base64.b64encode(png_bytes).decode('ascii')
        img_tag = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;" alt="Trajectory plot"/>'

    eq_html = "\n".join(f"<div>\\[{e}\\]</div>" for e in equations)

    plain_summary_html = ""
    if plain_summary:
        import re as _re
        body_lines = []
        for line in plain_summary.split("\n"):
            escaped = _html.escape(line)
            escaped = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
            if escaped.startswith("- "):
                body_lines.append(f"<li>{escaped[2:]}</li>")
            elif escaped.strip() == "":
                body_lines.append("")
            else:
                body_lines.append(f"<p>{escaped}</p>")
        plain_summary_html = (
            '<div style="background:#f0f6fa;border-left:4px solid #2B6083;padding:1rem 1.2rem;">'
            + "\n".join(body_lines) + "</div>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>AutoTraj Model Report</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
h1 {{ border-bottom: 3px solid #2B6083; padding-bottom: 0.3rem; }}
h2 {{ color: #2B6083; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #2B6083; color: white; }}
tr:nth-child(even) {{ background: #f7f7f7; }}
pre {{ background: #f4f4f4; padding: 1rem; overflow-x: auto; }}
.meta {{ color: #666; font-size: 0.85rem; }}
</style></head>
<body>
<h1>AutoTraj Model Report</h1>
<p class="meta">Generated by AutoTraj &mdash; {dist} distribution, {k} group(s), orders {orders}.</p>

<h2>Plain-Language Summary</h2>
{plain_summary_html if plain_summary_html else "<p><em>Not available.</em></p>"}

<h2>Model Summary</h2>
<pre>{_html.escape(summary_txt)}</pre>

<h2>Fitted Trajectories</h2>
{img_tag if img_tag else "<p><em>Plot not available.</em></p>"}

<h2>Fitted Model Equations</h2>
{eq_html}

<h2>Parameter Estimates</h2>
{estimates_df.to_html(index=False, border=0)}

<h2>Model Adequacy Diagnostics (Nagin, 2005)</h2>
<p class="meta">Relative Entropy: {rel_entropy:.3f}</p>
{adq_df.to_html(index=False, border=0)}

<p class="meta">Suggested Citation: Warden, D. E. (2026). AutoTraj: Automated Group-Based Trajectory Modeling Engine [Software]. GitHub. https://github.com/Thornwell16/gbtm_project</p>
</body></html>"""


def _build_pdf_report(winning_model, group_names, estimates_df, adq_df, rel_entropy,
                       summary_txt, equations, png_bytes=None, plain_summary=None):
    """Return PDF bytes for a shareable model report, built with reportlab
    (pure-Python, no external binary/system dependency — unlike wkhtmltopdf
    or weasyprint, this works identically on Windows/macOS/Linux/Streamlit
    Cloud with no extra system packages).

    Mirrors _build_html_report's content (plain-language summary, model
    summary, trajectory plot, equations, parameter/adequacy tables) in a
    print-friendly layout for journal supplementary materials.
    """
    import re as _re
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, Preformatted,
    )

    orders = winning_model['orders']
    dist = winning_model.get('dist', 'LOGIT')
    k = len(orders)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle('AT_H1', parent=styles['Heading1'], textColor=colors.HexColor('#2B6083'))
    h2 = ParagraphStyle('AT_H2', parent=styles['Heading2'], textColor=colors.HexColor('#2B6083'), spaceBefore=14)
    body = styles['BodyText']
    meta = ParagraphStyle('AT_Meta', parent=styles['BodyText'], textColor=colors.grey, fontSize=8.5)

    def _clean_latex(s):
        s = _re.sub(r"\\text\{([^}]*)\}", r"\1", s)
        s = s.replace("\\quad", "    ").replace("\\;", " ").replace("\\,", " ")
        s = s.replace("\\cdot", "*")
        return s

    def _md_inline(s):
        # Minimal **bold** -> <b> conversion for reportlab's mini-HTML markup.
        return _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

    def _df_to_table(df, max_rows=40):
        data = [list(df.columns)] + df.head(max_rows).astype(str).values.tolist()
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2B6083')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f7f7')]),
        ]))
        return t

    story = [
        Paragraph("AutoTraj Model Report", h1),
        Paragraph(f"Generated by AutoTraj &mdash; {dist} distribution, {k} group(s), orders {orders}.", meta),
        Spacer(1, 10),
    ]

    if plain_summary:
        story.append(Paragraph("Plain-Language Summary", h2))
        for line in plain_summary.split("\n"):
            if line.strip() == "":
                continue
            text = _md_inline(line[2:] if line.startswith("- ") else line)
            story.append(Paragraph(("&bull; " + text) if line.startswith("- ") else text, body))
        story.append(Spacer(1, 8))

    story.append(Paragraph("Model Summary", h2))
    story.append(Preformatted(summary_txt, styles['Code']))

    if png_bytes is not None:
        story.append(Paragraph("Fitted Trajectories", h2))
        try:
            story.append(RLImage(io.BytesIO(png_bytes), width=6.5 * inch, height=4.06 * inch))
        except Exception:
            story.append(Paragraph("<i>Plot could not be embedded.</i>", body))

    if equations:
        story.append(Paragraph("Fitted Model Equations", h2))
        for eq in equations:
            story.append(Paragraph(_clean_latex(eq), styles['Code']))

    story.append(Paragraph("Parameter Estimates", h2))
    story.append(_df_to_table(estimates_df))

    story.append(Paragraph("Model Adequacy Diagnostics (Nagin, 2005)", h2))
    story.append(Paragraph(f"Relative Entropy: {rel_entropy:.3f}", meta))
    story.append(Spacer(1, 4))
    story.append(_df_to_table(adq_df))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Suggested Citation: Warden, D. E. (2026). AutoTraj: Automated Group-Based Trajectory "
        "Modeling Engine [Software]. GitHub. https://github.com/Thornwell16/gbtm_project", meta,
    ))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    doc.build(story)
    return buf.getvalue()


# ── diagnostic helpers ────────────────────────────────────────────────────────

def _posterior_heatmap(assignments_df, k, group_names):
    """Return (matrix, fig) for average posterior probability heatmap.

    matrix[r, c] = mean P(group c+1) among subjects assigned to group r+1.
    Diagonal values should be high (>0.7) for a well-separated model.
    """
    prob_cols = [f'Group_{g+1}_Prob' for g in range(k)]
    matrix = np.zeros((k, k))
    for r in range(k):
        mask = assignments_df['Assigned_Group'] == r + 1
        if mask.sum() > 0:
            matrix[r, :] = assignments_df.loc[mask, prob_cols].mean().values

    text = [[f"{matrix[r, c]:.3f}" for c in range(k)] for r in range(k)]
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"{gn} (Est.)" for gn in group_names],
        y=[f"{gn} (Assigned)" for gn in group_names],
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        zmin=0, zmax=1,
        colorbar=dict(title="Avg PP"),
    ))
    fig.update_layout(
        xaxis_title="Posterior probability of group →",
        yaxis_title="Assigned group ↓",
        template="plotly_white",
        height=300 + 60 * k,
    )
    return matrix, fig


def _entropy_decomposition(assignments_df, pis, k, group_names):
    """Per-group relative entropy contribution.

    A group with relative entropy near 1.0 is cleanly separated;
    near 0 means subjects in that group have diffuse posteriors.
    """
    prob_cols = [f'Group_{g+1}_Prob' for g in range(k)]
    rows = []
    for g in range(k):
        mask = assignments_df['Assigned_Group'] == g + 1
        n    = mask.sum()
        if n > 0 and k > 1:
            grp_probs = assignments_df.loc[mask, prob_cols].values
            ent_sum   = np.sum(grp_probs * np.log(np.clip(grp_probs, 1e-15, 1.0)))
            re_g      = 1.0 + ent_sum / (n * np.log(k))
        else:
            re_g = 1.0 if k == 1 else np.nan
        rows.append({
            "Group": group_names[g],
            "N Assigned": int(n),
            "Est. Pi (%)": round(pis[g] * 100, 1),
            "Group Rel. Entropy": round(re_g, 4) if not np.isnan(re_g) else "N/A",
        })
    return pd.DataFrame(rows)


def _obs_vs_est_figure(long_df, assignments_df, winning_model, group_names, dist_type):
    """Observed group means vs model-estimated trajectory at each unique time point.

    Observed means use posterior-weighted averaging for a rigorous comparison.
    """
    orders         = winning_model['orders']
    winning_result = winning_model['result']
    beta_info      = _beta_start_indices(orders, n_mix=winning_model.get('n_mix', 1))
    k              = len(orders)
    prob_cols      = [f'Group_{g+1}_Prob' for g in range(k)]
    colors         = ['#2B6083', '#B5373A', '#D4A843', '#2E7D52', '#7B4F8A', '#C97B2A']

    # Merge posterior weights onto long_df
    merged = pd.merge(long_df, assignments_df[['ID'] + prob_cols], on='ID', how='left')
    unique_times = np.sort(long_df['Time'].unique())

    fig = go.Figure()

    for g in range(k):
        beta_start, n_betas = beta_info[g]
        g_betas = winning_result.x[beta_start:beta_start + n_betas]
        color   = colors[g % len(colors)]

        # Posterior-weighted observed mean at each time point
        obs_vals = []
        for t in unique_times:
            t_mask   = merged['Time'] == t
            weights  = merged.loc[t_mask, f'Group_{g+1}_Prob'].values
            outcomes = merged.loc[t_mask, 'Outcome'].values
            w_sum    = weights.sum()
            obs_vals.append(np.dot(weights, outcomes) / w_sum if w_sum > 0 else np.nan)
        obs_vals = np.array(obs_vals)

        # Model-estimated at those exact time points
        X_times = create_design_matrix_jit(unique_times.astype(np.float64), orders[g])
        eta     = X_times @ g_betas
        if dist_type == 'LOGIT':
            est_vals = 1.0 / (1.0 + np.exp(-np.clip(eta, -25, 25)))
        elif dist_type in ('POISSON', 'ZIP'):
            est_vals = np.exp(np.clip(eta, -20, 20))
        else:
            est_vals = eta

        fig.add_trace(go.Scatter(
            x=unique_times, y=obs_vals,
            mode='markers', name=f'{group_names[g]} Obs.',
            marker=dict(color=color, size=9, symbol='circle'),
        ))
        fig.add_trace(go.Scatter(
            x=unique_times, y=est_vals,
            mode='lines', name=f'{group_names[g]} Est.',
            line=dict(color=color, width=2.5),
        ))

    fig.update_layout(template="plotly_white", height=400,
                      xaxis_title="Time", yaxis_title="Outcome")
    return fig


def _residual_analysis(long_df, assignments_df, winning_model, group_names, dist_type):
    """Compute per-observation residuals; return summary DataFrame and figures.

    Returns
    -------
    resid_df    : per-subject DataFrame with mean residual and outlier flag
    fig_hist    : Plotly histogram of mean residuals
    fig_qq      : Matplotlib QQ figure (CNORM only, else None)
    """
    orders         = winning_model['orders']
    winning_result = winning_model['result']
    beta_info      = _beta_start_indices(orders, n_mix=winning_model.get('n_mix', 1))

    all_obs_resid = []   # flat list of all observation-level residuals
    subj_records  = []

    for _, arow in assignments_df.iterrows():
        sid  = arow['ID']
        g    = int(arow['Assigned_Group']) - 1

        subj_df   = long_df[long_df['ID'] == sid].sort_values('Time')
        times_arr = subj_df['Time'].values.astype(np.float64)
        y_arr     = subj_df['Outcome'].values.astype(np.float64)

        beta_start, n_betas = beta_info[g]
        g_betas = winning_result.x[beta_start:beta_start + n_betas]

        X_s = create_design_matrix_jit(times_arr, orders[g])
        eta = X_s @ g_betas

        if dist_type == 'LOGIT':
            pred = 1.0 / (1.0 + np.exp(-np.clip(eta, -25, 25)))
        elif dist_type in ('POISSON', 'ZIP'):
            pred = np.exp(np.clip(eta, -20, 20))
        else:
            pred = eta

        resid     = y_arr - pred
        mean_resid = float(np.mean(resid))
        all_obs_resid.extend(resid.tolist())
        subj_records.append({'ID': sid, 'Assigned_Group': g + 1,
                             'Mean_Residual': mean_resid})

    resid_arr = np.array([r['Mean_Residual'] for r in subj_records])
    mu_r, sd_r = resid_arr.mean(), resid_arr.std()
    threshold  = mu_r + 2.5 * sd_r

    resid_df = pd.DataFrame(subj_records)
    resid_df['Outlier'] = resid_df['Mean_Residual'].abs() > abs(threshold)

    # Histogram
    colors_map = {r['Assigned_Group']: f"Group {r['Assigned_Group']}"
                  for r in subj_records}
    fig_hist = go.Figure()
    palette  = ['#2B6083', '#B5373A', '#D4A843', '#2E7D52', '#7B4F8A', '#C97B2A']
    for g_num in sorted(resid_df['Assigned_Group'].unique()):
        sub = resid_df[resid_df['Assigned_Group'] == g_num]['Mean_Residual']
        gname = group_names[g_num - 1] if g_num - 1 < len(group_names) else f"Group {g_num}"
        fig_hist.add_trace(go.Histogram(
            x=sub, name=gname, opacity=0.7, nbinsx=30,
            marker_color=palette[(g_num - 1) % len(palette)],
        ))
    fig_hist.add_vline(x=mu_r, line_dash="dash", line_color="black",
                       annotation_text=f"Mean={mu_r:.3f}")
    fig_hist.update_layout(
        barmode='overlay', template="plotly_white",
        xaxis_title="Mean Residual per Subject", yaxis_title="Count", height=360,
    )

    # QQ plot (CNORM only)
    fig_qq = None
    if dist_type == 'CNORM':
        fig_qq, ax_qq = plt.subplots(figsize=(5, 4))
        fig_qq.patch.set_facecolor('none')
        ax_qq.patch.set_facecolor('none')
        res_pp = probplot(np.array(all_obs_resid), dist="norm")
        ax_qq.plot(res_pp[0][0], res_pp[0][1], 'o', alpha=0.4, color='steelblue',
                   markersize=3, label='Residuals')
        ax_qq.plot(res_pp[0][0], res_pp[1][1] + res_pp[1][0] * res_pp[0][0],
                   'r-', linewidth=1.5, label='Normal ref.')
        ax_qq.set_xlabel("Theoretical quantiles")
        ax_qq.set_ylabel("Sample quantiles")
        ax_qq.set_title("Q-Q Plot of Residuals")
        ax_qq.legend(frameon=False, fontsize=8)
        plt.tight_layout()

    return resid_df, fig_hist, fig_qq


# ── Streamlit app ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="AutoTraj | GBTM Engine", layout="wide")

# ── Brand Theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ===== CSS Design Tokens ===== */
/* All theme-sensitive colors are defined here as custom properties.
   Brand accents (gold, blue, red, green) are the same in both themes.
   Backgrounds, text, and borders flip between light and dark.          */
:root {
    /* Backgrounds */
    --at-bg-page:    #FEFCF9;
    --at-bg-cream:   #FAF6F0;
    --at-bg-card:    #E8F1F5;
    --at-bg-surface: #FFFFFF;
    --at-bg-sidebar: #1B3A4B;
    --at-bg-input:   #243F52;
    /* Text */
    --at-text-pri:   #1A1A2E;
    --at-text-mut:   #5A6977;
    --at-text-deep:  #1B3A4B;
    /* Borders */
    --at-border:     #E0E0E0;
    --at-border-sub: #E8F1F5;
    /* Message tints (low-opacity backgrounds) */
    --at-msg-ok:     rgba(46, 125, 82,  0.07);
    --at-msg-warn:   rgba(212, 168, 67, 0.08);
    --at-msg-err:    rgba(181, 55,  58, 0.07);
    --at-msg-info:   rgba(43,  96,  131,0.07);
    /* Brand accents — identical in light and dark */
    --at-gold:       #D4A843;
    --at-gold-warm:  #C49A2E;
    --at-gold-dim:   #C8B87A;
    --at-blue:       #2B6083;
    --at-navy:       #1B3A4B;
    --at-red:        #B5373A;
    --at-green:      #2E7D52;
}

/* ===== Dark Mode — system media query ===== */
@media (prefers-color-scheme: dark) {
    :root {
        --at-bg-page:    #0E1117;
        --at-bg-cream:   #1A1D23;
        --at-bg-card:    #1B2838;
        --at-bg-surface: #151920;
        --at-bg-sidebar: #0A1929;
        --at-bg-input:   #1A2535;
        --at-text-pri:   #E0E0E0;
        --at-text-mut:   #8899AA;
        --at-text-deep:  #C5D5E5;
        --at-border:     #2A3040;
        --at-border-sub: #252B35;
        /* Richer tints so messages read against dark bg */
        --at-msg-ok:     rgba(46, 125, 82,  0.20);
        --at-msg-warn:   rgba(212, 168, 67, 0.18);
        --at-msg-err:    rgba(181, 55,  58, 0.20);
        --at-msg-info:   rgba(43,  96,  131,0.20);
    }
}

/* ===== Dark Mode — Streamlit in-app theme toggle ===== */
/* Streamlit sets data-theme="dark" on <html> when the user
   switches via Settings → Theme → Dark.                    */
[data-theme="dark"] {
    --at-bg-page:    #0E1117;
    --at-bg-cream:   #1A1D23;
    --at-bg-card:    #1B2838;
    --at-bg-surface: #151920;
    --at-bg-sidebar: #0A1929;
    --at-bg-input:   #1A2535;
    --at-text-pri:   #E0E0E0;
    --at-text-mut:   #8899AA;
    --at-text-deep:  #C5D5E5;
    --at-border:     #2A3040;
    --at-border-sub: #252B35;
    --at-msg-ok:     rgba(46, 125, 82,  0.20);
    --at-msg-warn:   rgba(212, 168, 67, 0.18);
    --at-msg-err:    rgba(181, 55,  58, 0.20);
    --at-msg-info:   rgba(43,  96,  131,0.20);
}

/* ===== Base & Page ===== */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--at-bg-page);
}
.main .block-container {
    background-color: var(--at-bg-page);
    padding-top: 1.5rem;
}

/* ===== Sidebar ===== */
/* The sidebar is always a dark panel — gold/white text on navy bg —
   so sidebar brand/label colors are hardcoded, not variable.        */
[data-testid="stSidebar"] {
    background-color: var(--at-bg-sidebar) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] label {
    color: #D4A843 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #FFFFFF !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
    background-color: var(--at-bg-input) !important;
    color: #FFFFFF !important;
    border-color: #2B6083 !important;
}

/* ===== Buttons ===== */
/* Primary — gold bg, dark navy text: striking on both themes */
button[data-testid="baseButton-primary"],
[data-testid="stBaseButton-primary"] {
    background-color: var(--at-gold) !important;
    color: var(--at-navy) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
}
button[data-testid="baseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background-color: var(--at-gold-warm) !important;
}
/* Secondary / download — transparent bg, gold border */
button[data-testid="baseButton-secondary"],
[data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] button,
.stDownloadButton > button {
    background-color: transparent !important;
    color: var(--at-blue) !important;
    border: 1.5px solid var(--at-gold) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
button[data-testid="baseButton-secondary"]:hover,
[data-testid="stDownloadButton"] button:hover {
    background-color: var(--at-bg-card) !important;
    color: var(--at-text-deep) !important;
}

/* ===== Metrics ===== */
[data-testid="metric-container"],
[data-testid="stMetric"] {
    background-color: var(--at-bg-card) !important;
    border-left: 4px solid var(--at-gold) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricValue"] > div,
[data-testid="stMetricValue"] {
    color: var(--at-text-deep) !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] {
    color: var(--at-text-mut) !important;
    font-size: 0.85rem !important;
}

/* ===== Tabs ===== */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid var(--at-border-sub);
    gap: 2px;
}
[data-testid="stTabs"] button[role="tab"] {
    color: var(--at-text-mut) !important;
    font-weight: 500;
    padding: 10px 18px;
    border-radius: 6px 6px 0 0;
    background: transparent !important;
    border: none !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--at-text-deep) !important;
    font-weight: 700 !important;
    border-bottom: 3px solid var(--at-gold) !important;
    background-color: var(--at-bg-card) !important;
}

/* ===== File Uploader ===== */
[data-testid="stFileUploader"] section {
    border: 2px dashed var(--at-gold) !important;
    border-radius: 8px !important;
    background-color: var(--at-bg-cream) !important;
}

/* ===== Messages ===== */
[data-testid="stNotification"][kind="success"], .stSuccess {
    border-left: 4px solid var(--at-green) !important;
    background-color: var(--at-msg-ok) !important;
    border-radius: 4px !important;
}
[data-testid="stNotification"][kind="warning"], .stWarning {
    border-left: 4px solid var(--at-gold) !important;
    background-color: var(--at-msg-warn) !important;
    border-radius: 4px !important;
}
[data-testid="stNotification"][kind="error"], .stError {
    border-left: 4px solid var(--at-red) !important;
    background-color: var(--at-msg-err) !important;
    border-radius: 4px !important;
}
[data-testid="stNotification"][kind="info"], .stInfo {
    border-left: 4px solid var(--at-blue) !important;
    background-color: var(--at-msg-info) !important;
    border-radius: 4px !important;
}

/* ===== Dividers ===== */
hr {
    border-color: var(--at-gold) !important;
    opacity: 0.35;
}

/* ===== Expanders ===== */
[data-testid="stExpander"] summary span {
    color: var(--at-blue) !important;
    font-weight: 600 !important;
}

/* ===== Custom Components ===== */
.autotraj-header {
    padding: 0 0 14px 0;
    margin-bottom: 12px;
    border-bottom: 2px solid var(--at-gold);
}
.autotraj-header h1 {
    color: var(--at-blue);
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.autotraj-header p {
    color: var(--at-text-mut);
    font-size: 1.0rem;
    margin: 0;
}
/* Sidebar brand — hardcoded gold/cream: sidebar is always dark */
.sidebar-brand {
    padding: 8px 0 12px 0;
    border-bottom: 1px solid rgba(212,168,67,0.35);
    margin-bottom: 8px;
}
.sidebar-brand .title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #D4A843;
    letter-spacing: -0.3px;
    line-height: 1.1;
    display: block;
}
.sidebar-brand .motto {
    font-size: 0.76rem;
    font-style: italic;
    color: #C8B87A;
    margin-top: 3px;
    display: block;
}
.sidebar-section-header {
    color: #D4A843 !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    border-bottom: 1px solid rgba(212,168,67,0.3) !important;
    padding-bottom: 4px !important;
    margin: 16px 0 8px 0 !important;
    display: block;
}
.model-explorer-card {
    background-color: var(--at-bg-card);
    border-left: 4px solid var(--at-gold);
    border-radius: 6px;
    padding: 14px 18px 8px 18px;
    margin-bottom: 8px;
}
.model-explorer-card h4 {
    color: var(--at-text-deep);
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 4px 0;
}
.model-explorer-card p {
    color: var(--at-text-mut);
    font-size: 0.85rem;
    margin: 0;
}
.app-footer {
    text-align: center;
    color: var(--at-text-mut);
    font-size: 0.80rem;
    padding: 20px 0 4px 0;
    margin-top: 28px;
    border-top: 1px solid var(--at-border-sub);
}
</style>
""", unsafe_allow_html=True)

if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
    st.session_state.top_models  = None
    st.session_state.all_evaluated = None
    st.session_state.run_time    = 0
    st.session_state.long_df     = None
    st.session_state.raw_df      = None
    st.session_state.use_sample_data = False
    st.session_state.use_joint_sample_data = False

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="title">AutoTraj</span>
        <span class="motto">Sapientia Veritatem Parit</span>
    </div>
    """, unsafe_allow_html=True)
    app_mode = st.radio("Navigation", ["AutoTraj Search", "Single Model Mode", "Dual-Trajectory (Joint) Mode", "About & Docs"])
    st.markdown('<span class="sidebar-section-header">&#160;</span>', unsafe_allow_html=True)

    if app_mode in ("AutoTraj Search", "Single Model Mode"):
        st.markdown('<span class="sidebar-section-header">1. Data Format</span>', unsafe_allow_html=True)
        data_format = st.radio("Select Data Structure:", ["Wide Format", "Long Format"], horizontal=True)

        st.markdown('<span class="sidebar-section-header">2. Data Mapping</span>', unsafe_allow_html=True)
        if data_format == "Wide Format":
            col_id, col_out, col_time = st.columns(3)
            with col_id:   id_col      = st.text_input("ID",           value="ID")
            with col_out:  outcome_col = st.text_input("Out. Prefix",  value="C")
            with col_time: time_col    = st.text_input("Time Prefix",  value="T")
        else:
            col_id, col_out, col_time = st.columns(3)
            with col_id:   id_col      = st.text_input("ID Col",   value="ID")
            with col_out:  outcome_col = st.text_input("Out. Col", value="Outcome")
            with col_time: time_col    = st.text_input("Time Col", value="Time")

        st.markdown('<span class="sidebar-section-header">3. Model Distribution</span>', unsafe_allow_html=True)
        selected_dist = st.selectbox("Select Outcome Type:", [
            "LOGIT (Binary)",
            "CNORM (Continuous/Tobit)",
            "POISSON (Count)",
            "ZIP (Zero-Inflated Poisson)",
        ])
        dist_flag = selected_dist.split(" ")[0]

        cnorm_min = np.nan
        cnorm_max = np.nan
        if dist_flag == "CNORM":
            st.markdown("*CNORM Scale Limits (Optional)*")
            st.info("Leave blank to automatically use the dataset's observed min/max.")
            col_c1, col_c2 = st.columns(2)
            c_min_in = col_c1.text_input("Minimum", value="")
            c_max_in = col_c2.text_input("Maximum", value="")
            if c_min_in.strip() != "": cnorm_min = float(c_min_in)
            if c_max_in.strip() != "": cnorm_max = float(c_max_in)

        st.markdown('<span class="sidebar-section-header">4. Engine Options</span>', unsafe_allow_html=True)
        use_dropout    = st.checkbox("Include MNAR Dropout Model", value=False)
        default_starts = 3 if app_mode == "AutoTraj Search" else 5
        n_starts = st.number_input(
            "Multi-Start Restarts", min_value=1, max_value=20, value=default_starts,
            help="Number of random starting points per model. More starts reduce local-optima risk."
        )
        manual_min_per_model = st.number_input(
            "Est. Manual Time per Model (min)", min_value=0.5, max_value=60.0, value=5.0, step=0.5,
            help="Rough assumption for how long it takes to specify, run, and check ONE model by "
                 "hand in SAS/Stata/R (edit syntax, rerun, inspect BIC/significance/group size, "
                 "decide accept/reject) — this is an editable estimate, not a benchmark. Adjust it "
                 "to match your own workflow speed; it only drives the 'Manual Proc Time' comparison "
                 "shown after fitting."
        )

        if app_mode == "AutoTraj Search":
            st.markdown('<span class="sidebar-section-header">5. Search Grid</span>', unsafe_allow_html=True)
            group_range = st.slider("Min & Max Groups",           1, 8, (1, 3))
            order_range = st.slider("Min & Max Polynomial Order", 0, 5, (0, 2))

            st.markdown('<span class="sidebar-section-header">6. Heuristic Rules</span>', unsafe_allow_html=True)
            min_pct = st.slider("Min Group Size (%)", 1.0, 15.0, 5.0, 0.5)
            p_val   = st.number_input("P-Value Threshold", value=0.05, format="%.3f")

        elif app_mode == "Single Model Mode":
            st.markdown('<span class="sidebar-section-header">5. Model Specifications</span>', unsafe_allow_html=True)
            k_single = st.number_input("Number of Groups", min_value=1, max_value=8, value=2)

            orders_single = []
            cols_ord = st.columns(2)
            for i in range(k_single):
                with cols_ord[i % 2]:
                    o = st.number_input(f"Group {i+1} Order", min_value=0, max_value=5, value=1, key=f"o_{i}")
                    orders_single.append(o)

        zip_iorder = 0  # no longer used; kept for API compatibility

    elif app_mode == "Dual-Trajectory (Joint) Mode":
        st.caption(
            "Two outcomes (Y, Z) linked by a joint latent-class probability "
            "matrix, instead of assuming independent group membership. Long format "
            "only — single model specification (no AutoTraj-style search over both "
            "outcomes' group/order grids)."
        )
        st.markdown('<span class="sidebar-section-header">1. Data Mapping (Long Format)</span>', unsafe_allow_html=True)
        joint_id_col = st.text_input("ID Col", value="ID", key="joint_id_col")
        col_jy1, col_jy2 = st.columns(2)
        with col_jy1: joint_outcome_y_col = st.text_input("Outcome Y Col", value="Aggression", key="joint_out_y")
        with col_jy2: joint_time_y_col = st.text_input("Time Y Col", value="Age", key="joint_time_y")
        col_jz1, col_jz2 = st.columns(2)
        with col_jz1: joint_outcome_z_col = st.text_input("Outcome Z Col", value="EmotionalSymptoms", key="joint_out_z")
        with col_jz2: joint_time_z_col = st.text_input("Time Z Col", value="Age", key="joint_time_z")

        st.markdown('<span class="sidebar-section-header">2. Outcome Y</span>', unsafe_allow_html=True)
        joint_dist_y = st.selectbox("Distribution:", ["LOGIT", "CNORM", "POISSON", "ZIP"], key="joint_dist_y")
        joint_cnorm_min_y, joint_cnorm_max_y = np.nan, np.nan
        if joint_dist_y == "CNORM":
            st.caption("Leave blank to automatically use the dataset's observed min/max.")
            c1, c2 = st.columns(2)
            miny = c1.text_input("Min", value="", key="joint_cmin_y")
            maxy = c2.text_input("Max", value="", key="joint_cmax_y")
            if miny.strip() != "": joint_cnorm_min_y = float(miny)
            if maxy.strip() != "": joint_cnorm_max_y = float(maxy)
        joint_dropout_y = st.checkbox("Include MNAR Dropout (Y)", value=False, key="joint_drop_y")
        joint_k_y = st.number_input("Number of Groups (Y)", min_value=1, max_value=6, value=2, key="joint_k_y")
        joint_orders_y = [
            st.number_input(f"Y Group {i+1} Order", min_value=0, max_value=5, value=1, key=f"joint_oy_{i}")
            for i in range(joint_k_y)
        ]

        st.markdown('<span class="sidebar-section-header">3. Outcome Z</span>', unsafe_allow_html=True)
        joint_dist_z = st.selectbox("Distribution:", ["LOGIT", "CNORM", "POISSON", "ZIP"], key="joint_dist_z")
        joint_cnorm_min_z, joint_cnorm_max_z = np.nan, np.nan
        if joint_dist_z == "CNORM":
            st.caption("Leave blank to automatically use the dataset's observed min/max.")
            c1, c2 = st.columns(2)
            minz = c1.text_input("Min", value="", key="joint_cmin_z")
            maxz = c2.text_input("Max", value="", key="joint_cmax_z")
            if minz.strip() != "": joint_cnorm_min_z = float(minz)
            if maxz.strip() != "": joint_cnorm_max_z = float(maxz)
        joint_dropout_z = st.checkbox("Include MNAR Dropout (Z)", value=False, key="joint_drop_z")
        joint_k_z = st.number_input("Number of Groups (Z)", min_value=1, max_value=6, value=2, key="joint_k_z")
        joint_orders_z = [
            st.number_input(f"Z Group {i+1} Order", min_value=0, max_value=5, value=1, key=f"joint_oz_{i}")
            for i in range(joint_k_z)
        ]

        st.markdown('<span class="sidebar-section-header">4. Engine Options</span>', unsafe_allow_html=True)
        joint_n_starts = st.number_input(
            "Multi-Start Restarts", min_value=1, max_value=20, value=5, key="joint_n_starts",
            help="Number of random starting points. More starts reduce local-optima risk."
        )

# ── About page ───────────────────────────────────────────────────────────────

if app_mode == "About & Docs":
    st.markdown("""
    <div class="autotraj-header">
        <h1>AutoTraj</h1>
        <p>Automated Group-Based Trajectory Modeling Engine &nbsp;&mdash;&nbsp; <em>Sapientia Veritatem Parit</em></p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(r"""
    **Overview**
    AutoTraj is a high-performance engine for Group-Based Trajectory Modeling (GBTM), a specialized application of finite mixture modeling utilized to identify latent subpopulations following distinct developmental trajectories over time. It automates the exhaustive search, selection, and visualization of these models by leveraging a fully vectorized, C-compiled analytical Jacobian engine to rapidly evaluate combinatorial polynomial grids.

    **Methodology & Missing Data**
    By default, the engine utilizes Full Information Maximum Likelihood (FIML), which provides unbiased parameter estimates under the assumption that missing data is Missing At Random (MAR).

    To account for informative attrition (Missing Not At Random - MNAR), users can toggle the **Dropout Model**. This fits a joint likelihood model integrating a logistic survival equation conditioned on the subject's previous health state:
    """)

    st.latex(r"P(Dropout_{it} = 1 | g) = \frac{1}{1 + e^{-(\gamma_{0g} + \gamma_{1g} t + \gamma_{2g} y_{i, t-1})}}")

    st.markdown(r"""
    **Mathematical Safeguards & Model Identifiability**
    Unlike standard statistical packages that may output estimates for overparameterized or unidentifiable models, AutoTraj utilizes strict mathematical exclusion criteria during the automated search phase. By actively calculating the condition number of the scaled Hessian matrix, the engine automatically rejects models that produce singular information matrices (flat likelihood surfaces) or degenerate standard errors, protecting against artificial significance caused by algorithmic bounds.

    **Robust Standard Errors**
    In addition to model-based standard errors derived from the exact numerical Hessian (Observed Information Matrix), AutoTraj natively computes Huber-White sandwich estimators. This is achieved by cross-multiplying the analytical subject-level gradient vectors against the inverse Hessian, providing standard errors robust to minor model misspecifications and heteroskedasticity.

    **Fit Statistics & Optimization**
    Calculations align precisely with standard epidemiological conventions. Significance is calculated using the Student's T-distribution ($DF = N_{obs} - p$) to match standard statistical reporting in developmental models. Models are optimized and selected using the Bayesian Information Criterion (BIC). Two conventions are reported:

    *Nagin / Proc Traj convention (Jones & Nagin, 2001) — higher (less negative) = better fit:*
    * **AIC (Nagin):** $LL - p$
    * **BIC (Nagin):** $LL - 0.5 \cdot p \cdot \ln(N)$

    *Standard convention — lower = better fit:*
    * **AIC (Standard):** $-2 \cdot LL + 2p$
    * **BIC (Standard):** $-2 \cdot LL + p \cdot \ln(N)$

    ---
    **Suggested Citation**
    Warden, D. E. (2026). AutoTraj: Automated Group-Based Trajectory Modeling Engine [Software]. GitHub. https://github.com/Thornwell16/gbtm_project

    **References**
    * Haviland, A. M., Jones, B. L., & Nagin, D. S. (2011). Group-based trajectory modeling: extended statistical and survival analysis capabilities. *Sociological Methods & Research*, 40(3), 485-492.
    * Jones, B. L., Nagin, D. S., & Roeder, K. (2001). A SAS procedure based on mixture models for estimating developmental trajectories. *Sociological Methods & Research*, 29(3), 374-393.
    * Nagin, D. S. (1999). Analyzing developmental trajectories: a semiparametric, group-based approach. *Psychological Methods*, 4(2), 139-157.
    """)
    st.divider()
    st.markdown('<div class="app-footer">AutoTraj &nbsp;&middot;&nbsp; Built by Donald E. Warden, PhD, MPH &nbsp;&middot;&nbsp; <em>Sapientia Veritatem Parit</em> &nbsp;&middot;&nbsp; MIT License</div>', unsafe_allow_html=True)

# ── Dual-Trajectory (Joint) mode ───────────────────────────────────────────────

elif app_mode == "Dual-Trajectory (Joint) Mode":
    st.markdown("""
    <div class="autotraj-header">
        <h1>AutoTraj</h1>
        <p>Dual-Trajectory (Joint) Mode &nbsp;&mdash;&nbsp; Two outcomes linked by a joint latent-class probability matrix</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🚀 New here? 60-second walkthrough"):
        st.markdown("""
1. **Load data** — click **"Load Joint-Trajectory Sample Data"** below to try it instantly with a
   simulated dataset, or upload your own long-format file (one row per subject-timepoint, with
   separate outcome/time columns for each of the two outcomes).
2. **Check the Data Quality Preview** — after loading, expand it to confirm subject counts, wave
   counts, and outcome ranges look right (especially CNORM min/max, if used).
3. **Map columns & configure both outcomes** in the sidebar — ID, Outcome/Time columns for Y and Z,
   each outcome's distribution, number of groups, and polynomial order.
4. **Click "Fit Joint Dual-Trajectory Model."** The key output is the **π matrix** — it shows
   whether the two outcomes' latent classes are associated (e.g. "does high aggression co-occur
   with high emotional symptoms more than chance?"), not just each outcome's own trajectories.
5. **Export** parameter estimates, assignments, and the π/contingency tables from the Export row.
        """)

    uploaded_file_j = st.file_uploader(
        "Upload Long-Format Dataset (.csv, .txt, .xlsx)", type=["csv", "txt", "xlsx"], key="joint_uploader"
    )
    st.markdown(
        "*Or, just here to try out dual-trajectory modeling? Click below to load a simulated "
        "illustrative dataset (synthetic — not real published data) tracking two co-developing "
        "adolescent outcomes.*"
    )
    if st.button("Load Joint-Trajectory Sample Data", use_container_width=False):
        st.session_state.use_joint_sample_data = True

    if uploaded_file_j is not None:
        st.session_state.use_joint_sample_data = False

    raw_df_j = None
    if uploaded_file_j is not None:
        try:
            file_name_j = uploaded_file_j.name.lower()
            if file_name_j.endswith('.csv'):
                raw_df_j = pd.read_csv(uploaded_file_j, encoding='utf-8-sig')
            elif file_name_j.endswith('.txt'):
                raw_df_j = pd.read_csv(uploaded_file_j, sep=r'\s+', encoding='utf-8-sig')
            elif file_name_j.endswith('.xlsx'):
                raw_df_j = pd.read_excel(uploaded_file_j, engine='openpyxl')
            raw_df_j.columns = [str(c).strip() for c in raw_df_j.columns]
            st.success("Custom file uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif st.session_state.use_joint_sample_data:
        try:
            raw_df_j = pd.read_csv("joint_trajectory_sample.csv", encoding='utf-8-sig')
            raw_df_j.columns = [str(c).strip() for c in raw_df_j.columns]
            st.success(
                "Joint-trajectory sample dataset loaded! (700 simulated subjects, ages 10-16. "
                "Use ID='ID', Outcome Y='Aggression', Time Y='Age', Outcome Z='EmotionalSymptoms', "
                "Time Z='Age' — Outcome Y is LOGIT, Outcome Z is CNORM with bounds 0-10.)"
            )
        except Exception as e:
            st.error("Could not locate joint_trajectory_sample.csv in the repository.")

    if raw_df_j is not None:
        required_cols_j = [joint_id_col, joint_outcome_y_col, joint_time_y_col, joint_outcome_z_col, joint_time_z_col]
        missing_j = [c for c in required_cols_j if c not in raw_df_j.columns]
        if missing_j:
            st.error(f"Column(s) not found in uploaded file: {', '.join(missing_j)}. Available columns: {', '.join(raw_df_j.columns)}")
        else:
            df_y_j = raw_df_j[[joint_id_col, joint_time_y_col, joint_outcome_y_col]].rename(
                columns={joint_id_col: 'ID', joint_time_y_col: 'Time', joint_outcome_y_col: 'Outcome'}
            ).dropna()
            df_z_j = raw_df_j[[joint_id_col, joint_time_z_col, joint_outcome_z_col]].rename(
                columns={joint_id_col: 'ID', joint_time_z_col: 'Time', joint_outcome_z_col: 'Outcome'}
            ).dropna()

            ids_y_j, ids_z_j = set(df_y_j['ID'].unique()), set(df_z_j['ID'].unique())
            if ids_y_j != ids_z_j:
                only_y_j, only_z_j = sorted(ids_y_j - ids_z_j), sorted(ids_z_j - ids_y_j)
                msg = "Outcome Y and Outcome Z must share the identical subject-ID set after dropping missing rows. "
                if only_y_j: msg += f"IDs only in Y ({len(only_y_j)}): {only_y_j[:10]}. "
                if only_z_j: msg += f"IDs only in Z ({len(only_z_j)}): {only_z_j[:10]}."
                st.error(msg)
            else:
                st.success(f"Loaded {df_y_j['ID'].nunique()} subjects — {len(df_y_j)} Y-observations, {len(df_z_j)} Z-observations.")

                with st.expander("Data Quality Preview", expanded=False):
                    obs_y = df_y_j.groupby('ID').size()
                    obs_z = df_z_j.groupby('ID').size()
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("Subjects", f"{df_y_j['ID'].nunique():,}")
                    pc2.metric("Y Waves (median)", f"{obs_y.median():.0f}")
                    pc3.metric("Z Waves (median)", f"{obs_z.median():.0f}")
                    pc4.metric("Y Outcome Mean", f"{df_y_j['Outcome'].mean():.2f}")
                    st.caption(
                        f"Outcome Y range: [{df_y_j['Outcome'].min():.2f}, {df_y_j['Outcome'].max():.2f}] — "
                        f"Outcome Z range: [{df_z_j['Outcome'].min():.2f}, {df_z_j['Outcome'].max():.2f}]. "
                        "If either outcome is CNORM, set its Min/Max bounds to (at least) these ranges, "
                        "or leave blank to auto-detect from the data."
                    )

                if st.button("Fit Joint Dual-Trajectory Model", type="primary"):
                    joint_fit_log = []
                    with st.spinner("Fitting joint model (multi-start BFGS)... this may take a while."):
                        try:
                            model_j = run_joint_dual_trajectory_model(
                                df_y_j, df_z_j, orders_y=joint_orders_y, orders_z=joint_orders_z,
                                dist_y=joint_dist_y, dist_z=joint_dist_z,
                                use_dropout_y=joint_dropout_y, use_dropout_z=joint_dropout_z,
                                cnorm_min_y=joint_cnorm_min_y, cnorm_max_y=joint_cnorm_max_y,
                                cnorm_min_z=joint_cnorm_min_z, cnorm_max_z=joint_cnorm_max_z,
                                n_starts=joint_n_starts, log_callback=joint_fit_log.append,
                            )
                            st.session_state.joint_model = model_j
                            st.session_state.joint_df_y = df_y_j
                            st.session_state.joint_df_z = df_z_j
                            st.session_state.joint_fit_log = joint_fit_log
                        except Exception as e:
                            st.error(f"Model fitting failed: {e}")
                    if joint_fit_log:
                        with st.expander(f"Fit Log ({len(joint_fit_log)} multi-start resolutions)"):
                            st.code("\n".join(joint_fit_log), language=None)
                            st.session_state.joint_model = None

                if st.session_state.get("joint_model") is not None:
                    model_j = st.session_state.joint_model
                    df_y_j = st.session_state.joint_df_y
                    df_z_j = st.session_state.joint_df_z

                    if model_j['pis_joint'] is None:
                        st.error("Model did not converge to a valid solution. Try increasing restarts or simplifying the group/order specification.")
                    else:
                        pis_joint  = model_j['pis_joint']
                        k_y, k_z   = model_j['k_y'], model_j['k_z']
                        group_names_y = [f"Y-Group {g+1}" for g in range(k_y)]
                        group_names_z = [f"Z-Group {h+1}" for h in range(k_z)]

                        st.divider()
                        st.subheader("Model Fit")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Log-Likelihood", f"{model_j['ll']:.2f}")
                        c2.metric("BIC (Nagin)", f"{model_j['bic']:.2f}")
                        c3.metric("AIC (Nagin)", f"{model_j['aic']:.2f}")
                        c4.metric("Condition Number", f"{model_j['cond_num']:.1e}")
                        if model_j['cond_num'] > 1e10:
                            st.warning("High condition number — model may be weakly identified. Consider fewer groups or lower polynomial orders.")

                        st.subheader("Joint Latent-Class Probability Matrix (πₘₕ)")
                        st.caption(
                            "Cell (g,h) = P(subject is simultaneously in Y-Group g AND Z-Group h). "
                            "If Y and Z were independent, each cell would equal the product of its row and "
                            "column marginals — deviations from that indicate the two outcomes co-develop."
                        )
                        fig_pi = go.Figure(go.Heatmap(
                            z=pis_joint, x=group_names_z, y=group_names_y,
                            colorscale='Blues', text=np.round(pis_joint, 3), texttemplate="%{text}",
                            zmin=0, colorbar=dict(title="P(g,h)"),
                        ))
                        fig_pi.update_layout(height=320 + 40 * max(k_y, k_z), template="plotly_white")
                        st.plotly_chart(fig_pi, use_container_width=True)

                        col_cond1, col_cond2 = st.columns(2)
                        with col_cond1:
                            st.markdown("**P(Z-Group | Y-Group)** — row-normalized")
                            p_h_given_g = pis_joint / pis_joint.sum(axis=1, keepdims=True)
                            st.dataframe(pd.DataFrame(np.round(p_h_given_g, 3), index=group_names_y, columns=group_names_z), use_container_width=True)
                        with col_cond2:
                            st.markdown("**P(Y-Group | Z-Group)** — column-normalized")
                            p_g_given_h = pis_joint / pis_joint.sum(axis=0, keepdims=True)
                            st.dataframe(pd.DataFrame(np.round(p_g_given_h, 3), index=group_names_y, columns=group_names_z), use_container_width=True)

                        assignments_df_j = get_joint_subject_assignments(model_j, df_y_j, df_z_j)

                        st.subheader("Hard-Assignment Contingency Table")
                        st.caption("Empirical cross-check: counts of subjects by their most-likely Y-group and Z-group (computed independently of the fitted πₘₕ, from marginal posteriors).")
                        contingency_j = pd.crosstab(
                            assignments_df_j['Assigned_Group_Y'].map(lambda g: group_names_y[g - 1]),
                            assignments_df_j['Assigned_Group_Z'].map(lambda h: group_names_z[h - 1]),
                        )
                        st.dataframe(contingency_j, use_container_width=True)

                        joint_adq_df, joint_rel_entropy, y_adq_df, y_rel_entropy, z_adq_df, z_rel_entropy = calc_joint_model_adequacy(
                            assignments_df_j, pis_joint, group_names_y, group_names_z
                        )
                        st.subheader("Model Adequacy Diagnostics (Nagin, 2005)")
                        tab_adq_j, tab_adq_y, tab_adq_z = st.tabs(["Joint", "Y-Marginal", "Z-Marginal"])
                        with tab_adq_j:
                            st.dataframe(joint_adq_df, use_container_width=True)
                            st.caption(f"Relative Entropy: {joint_rel_entropy:.3f}")
                        with tab_adq_y:
                            st.dataframe(y_adq_df, use_container_width=True)
                            st.caption(f"Relative Entropy: {y_rel_entropy:.3f}")
                        with tab_adq_z:
                            st.dataframe(z_adq_df, use_container_width=True)
                            st.caption(f"Relative Entropy: {z_rel_entropy:.3f}")

                        st.subheader("Parameter Estimates")
                        st.caption("Theta rows are joint mixing log-odds relative to reference cell Y-Group 1/Z-Group 1. Model-based SE is shown for reference; Robust SE (Huber-White sandwich) is the recommended inference basis.")
                        param_df_j = get_joint_parameter_estimates_for_ui(model_j)
                        st.dataframe(param_df_j, use_container_width=True)

                        st.subheader("Export")
                        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
                        with exp_col1:
                            st.download_button(
                                "Joint π Matrix (CSV)",
                                pd.DataFrame(pis_joint, index=group_names_y, columns=group_names_z).to_csv().encode('utf-8'),
                                file_name="joint_pi_matrix.csv", mime="text/csv",
                            )
                        with exp_col2:
                            st.download_button(
                                "Contingency Table (CSV)",
                                contingency_j.to_csv().encode('utf-8'),
                                file_name="joint_contingency_table.csv", mime="text/csv",
                            )
                        with exp_col3:
                            st.download_button(
                                "Parameter Estimates (CSV)",
                                param_df_j.to_csv(index=False).encode('utf-8'),
                                file_name="joint_parameter_estimates.csv", mime="text/csv",
                            )
                        with exp_col4:
                            st.download_button(
                                "Subject Assignments (CSV)",
                                assignments_df_j.to_csv(index=False).encode('utf-8'),
                                file_name="joint_subject_assignments.csv", mime="text/csv",
                            )

                        st.subheader("Fitted Trajectories")
                        n_theta_j, y_beta_start_j, z_beta_start_j, num_betas_y_j, num_betas_z_j, _ = _joint_layout(
                            k_y, k_z, model_j['orders_y'], model_j['orders_z'],
                            model_j['use_dropout_y'], model_j['dist_y'], model_j['use_dropout_z'], model_j['dist_z']
                        )
                        col_traj_y, col_traj_z = st.columns(2)
                        with col_traj_y:
                            st.markdown("**Outcome Y**")
                            fake_x_y = np.concatenate([np.zeros(k_y - 1), model_j['result'].x[y_beta_start_j:z_beta_start_j]])
                            model_y_view = {'orders': model_j['orders_y'], 'result': SimpleNamespace(x=fake_x_y), 'n_mix': 1}
                            assignments_y_view = assignments_df_j.rename(columns={
                                **{f'Y_Group_{g+1}_Prob': f'Group_{g+1}_Prob' for g in range(k_y)},
                                'Assigned_Group_Y': 'Assigned_Group',
                            })
                            st.plotly_chart(
                                _obs_vs_est_figure(df_y_j, assignments_y_view, model_y_view, group_names_y, model_j['dist_y']),
                                use_container_width=True,
                            )
                        with col_traj_z:
                            st.markdown("**Outcome Z**")
                            fake_x_z = np.concatenate([np.zeros(k_z - 1), model_j['result'].x[z_beta_start_j:]])
                            model_z_view = {'orders': model_j['orders_z'], 'result': SimpleNamespace(x=fake_x_z), 'n_mix': 1}
                            assignments_z_view = assignments_df_j.rename(columns={
                                **{f'Z_Group_{h+1}_Prob': f'Group_{h+1}_Prob' for h in range(k_z)},
                                'Assigned_Group_Z': 'Assigned_Group',
                            })
                            st.plotly_chart(
                                _obs_vs_est_figure(df_z_j, assignments_z_view, model_z_view, group_names_z, model_j['dist_z']),
                                use_container_width=True,
                            )

# ── Main app ──────────────────────────────────────────────────────────────────

else:
    _mode_subtitle = {
        "AutoTraj Search":   "Automated exhaustive search across groups and polynomial orders",
        "Single Model Mode": "Fit and inspect a single user-specified model",
    }.get(app_mode, "")
    st.markdown(f"""
    <div class="autotraj-header">
        <h1>AutoTraj</h1>
        <p>{app_mode} &nbsp;&mdash;&nbsp; {_mode_subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🚀 New here? 60-second walkthrough"):
        if app_mode == "AutoTraj Search":
            st.markdown("""
1. **Load data** — click **"Load Cambridge Sample Data"** below to try it instantly, or upload your
   own file (wide or long format).
2. **Check the Data Quality Preview** — expand it after loading to confirm subject/observation
   counts and wave balance look right before running a potentially slow search.
3. **Select data format & map your columns** in the sidebar (ID, Outcome, Time).
4. **Set the search range** — min/max groups and polynomial orders. Wider ranges take longer;
   the app will warn you if the estimated number of model fits is large.
5. **Click "Run AutoTraj Search."** AutoTraj fits every (groups × order) combination, rejects
   poorly-identified or non-significant models automatically, and ranks the rest by BIC.
6. **Explore results** — use the Model Explorer to browse alternatives, "Compare Top Models" for a
   side-by-side table, and the Export row to download parameters, plots, or a shareable HTML report.
            """)
        else:
            st.markdown("""
1. **Load data** — click **"Load Cambridge Sample Data"** below to try it instantly, or upload your
   own file (wide or long format).
2. **Check the Data Quality Preview** — expand it after loading to sanity-check your data.
3. **Map your columns** and specify a fixed number of groups + polynomial order per group in the
   sidebar (use this mode when you already know the model you want, unlike AutoTraj Search's
   exhaustive grid).
4. **Click "Run Single Model."**
5. **Inspect results** — parameter estimates, fitted equations, adequacy diagnostics, and the
   Export row (CSV, plots, or a shareable HTML report).
            """)

    uploaded_file = st.file_uploader("Upload Dataset (.csv, .txt, .xlsx, .sas7bdat)", type=["csv", "txt", "xlsx", "sas7bdat"])
    st.markdown("*Or, just here to try out the engine? Click below to load sample data (Nagin, 1999).*")

    if st.button("Load Cambridge Sample Data", use_container_width=False):
        st.session_state.use_sample_data = True

    with st.expander("Reload a previously saved model"):
        st.caption(
            "Upload a `.atmodel` file downloaded from the \"Save Fitted Model\" button below a "
            "previous run — reloads its results instantly without refitting."
        )
        saved_model_file = st.file_uploader("Saved model file (.atmodel)", type=["atmodel"], key="saved_model_uploader")
        if saved_model_file is not None:
            try:
                bundle = pickle.load(saved_model_file)
                st.session_state.run_complete  = True
                st.session_state.top_models    = bundle['top_models']
                st.session_state.all_evaluated = bundle['all_evaluated']
                st.session_state.long_df       = bundle['long_df']
                st.session_state.raw_df        = bundle['long_df']
                st.session_state.use_dropout   = bundle['use_dropout']
                st.session_state.run_time      = 0.0
                st.success(f"Reloaded a saved {bundle.get('app_mode', 'model')} fit — scroll down to view results.")
            except Exception as e:
                st.error(f"Could not load this file — it may not be a valid AutoTraj saved-model file. ({e})")

    if uploaded_file is not None:
        st.session_state.use_sample_data = False

    raw_df = None
    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name.lower()
            if file_name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            elif file_name.endswith('.txt'):
                raw_df = pd.read_csv(uploaded_file, sep=r'\s+', encoding='utf-8-sig')
            elif file_name.endswith('.xlsx'):
                raw_df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_name.endswith('.sas7bdat'):
                raw_df = pd.read_sas(uploaded_file, format='sas7bdat', encoding='utf-8')
            raw_df.columns = [str(c).strip() for c in raw_df.columns]
            st.success("Custom file uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}. If uploading SAS or Excel files, ensure 'pyreadstat' and 'openpyxl' are installed.")
    elif st.session_state.use_sample_data:
        try:
            raw_df = pd.read_csv("cambridge.txt", sep=r'\s+', encoding='utf-8-sig')
            raw_df.columns = [str(c).strip() for c in raw_df.columns]
            st.success("Cambridge sample dataset loaded! (Note: Sample data is in Wide format. Use ID='ID', Out='C', Time='T')")
        except Exception as e:
            st.error("Could not locate cambridge.txt in the repository.")

    baseline_cov_cols = []
    tvc_cols = []
    weight_col = None
    if raw_df is not None:
        with st.expander("Data Quality Preview", expanded=False):
            try:
                if data_format == "Wide Format" or st.session_state.use_sample_data:
                    _preview_df = prep_trajectory_data(raw_df.copy(), id_col, outcome_col, time_col).dropna(subset=['Time', 'Outcome'])
                elif id_col in raw_df.columns and outcome_col in raw_df.columns and time_col in raw_df.columns:
                    _preview_df = raw_df.rename(columns={id_col: 'ID', outcome_col: 'Outcome', time_col: 'Time'})[['ID', 'Time', 'Outcome']].copy()
                    _preview_df['Time'] = pd.to_numeric(_preview_df['Time'], errors='coerce')
                    _preview_df['Outcome'] = pd.to_numeric(_preview_df['Outcome'], errors='coerce')
                    _preview_df = _preview_df.dropna()
                else:
                    _preview_df = None

                if _preview_df is not None and len(_preview_df) > 0:
                    obs_per_subj = _preview_df.groupby('ID').size()
                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("Subjects", f"{_preview_df['ID'].nunique():,}")
                    pc2.metric("Observations", f"{len(_preview_df):,}")
                    pc3.metric("Unique Time Points", _preview_df['Time'].nunique())
                    pc4.metric("Obs / Subject (median)", f"{obs_per_subj.median():.0f}")
                    max_waves = int(obs_per_subj.max())
                    balanced_pct = (obs_per_subj == max_waves).mean() * 100
                    st.caption(
                        f"{balanced_pct:.0f}% of subjects have the maximum observed wave count "
                        f"({max_waves}) — the rest have fewer (dropout / unbalanced design, "
                        f"handled automatically via FIML, or via the MNAR Dropout Model option)."
                    )
                    fig_obs = go.Figure(go.Histogram(x=obs_per_subj.values, nbinsx=min(20, max(max_waves, 1))))
                    fig_obs.update_layout(
                        height=220, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Observations per subject", yaxis_title="N subjects",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_obs, use_container_width=True)
                else:
                    st.info("Map the ID / Outcome / Time columns above to preview data quality.")
            except Exception:
                st.info("Could not generate a preview yet — check the column mapping above.")

        if data_format == "Wide Format" or st.session_state.use_sample_data:
            # outcome_col/time_col are stub *prefixes* in wide format (e.g. "C" -> C1..C23),
            # so exclude by prefix, not exact match, or every reshaped column leaks through.
            candidate_cols = [
                c for c in raw_df.columns
                if str(c) != str(id_col) and not str(c).startswith((str(outcome_col), str(time_col)))
            ]
        else:
            reserved_cols = {str(id_col), str(outcome_col), str(time_col)}
            candidate_cols = [c for c in raw_df.columns if str(c) not in reserved_cols]

        with st.expander("Covariate Architecture (optional)"):
            st.markdown(
                "**Baseline covariates** predict group membership (multinomial "
                "logit on mixing proportions). Must be time-invariant (constant "
                "per subject) — works for both Wide and Long format, since "
                "wide-format extra columns carry through unchanged."
            )
            baseline_cov_cols = st.multiselect(
                "Baseline covariates for group membership:", candidate_cols,
                key="baseline_cov_cols_select",
            )
            if data_format == "Long Format":
                st.markdown(
                    "**Time-varying covariates (TVC)** deflect the trajectory "
                    "equation itself and may vary within subject over time."
                )
                tvc_cols = st.multiselect(
                    "Time-varying covariates for trajectory:",
                    [c for c in candidate_cols if c not in baseline_cov_cols],
                    key="tvc_cols_select",
                )
            else:
                st.caption(
                    "Time-varying covariates currently require Long format input."
                )

        with st.expander("Survey Weights (optional)"):
            st.markdown(
                "**Sampling weight** (e.g. inverse-probability-of-selection weight) applied "
                "per subject. Must be time-invariant and strictly positive. Robust "
                "(Huber-White sandwich) standard errors become the valid basis for inference "
                "once a weight is used — model-based SEs are shown for reference only."
            )
            weight_col_choice = st.selectbox(
                "Sampling weight column:", ["(None)"] + candidate_cols,
                key="weight_col_select",
            )
            weight_col = None if weight_col_choice == "(None)" else weight_col_choice

        button_label = "Run AutoTraj Search" if app_mode == "AutoTraj Search" else "Run Single Model"

        if st.button(button_label, type="primary", use_container_width=True):

            if data_format == "Wide Format" or st.session_state.use_sample_data:
                if id_col not in raw_df.columns:
                    st.error(f"🚨 **Data Mapping Error:** The ID column '{id_col}' was not found. Available columns: {', '.join(raw_df.columns[:5])}...")
                    st.stop()
                if not any(str(c).startswith(outcome_col) for c in raw_df.columns):
                    st.error(f"🚨 **Data Mapping Error:** No columns found starting with Outcome Prefix '{outcome_col}'.")
                    st.stop()
                if not any(str(c).startswith(time_col) for c in raw_df.columns):
                    st.error(f"🚨 **Data Mapping Error:** No columns found starting with Time Prefix '{time_col}'.")
                    st.stop()
            else:
                if id_col not in raw_df.columns or outcome_col not in raw_df.columns or time_col not in raw_df.columns:
                    st.error(f"🚨 **Data Mapping Error:** One or more columns ({id_col}, {outcome_col}, {time_col}) not found.")
                    st.stop()

            start_time = time.time()
            fit_log_lines = []
            def _log_cb(msg):
                fit_log_lines.append(msg)

            progress_bar = None
            if app_mode == "AutoTraj Search":
                progress_bar = st.progress(0.0, text="Starting AutoTraj Search...")

            def _progress_cb(current, total, orders_list):
                if progress_bar is not None:
                    progress_bar.progress(
                        current / total,
                        text=f"Evaluated {current}/{total} models — last: {len(orders_list)}-group {orders_list}",
                    )

            with st.spinner("AutoTraj Engine Running..."):

                if data_format == "Wide Format" or st.session_state.use_sample_data:
                    long_df = prep_trajectory_data(raw_df, id_col, outcome_col, time_col).dropna(subset=['Time', 'Outcome'])
                else:
                    weight_cols_list = [weight_col] if weight_col else []
                    keep_cols = ['ID', 'Time', 'Outcome'] + list(baseline_cov_cols) + list(tvc_cols) + weight_cols_list
                    long_df = raw_df.rename(columns={id_col: 'ID', outcome_col: 'Outcome', time_col: 'Time'})
                    long_df = long_df[keep_cols].dropna(subset=['Time', 'Outcome'] + list(tvc_cols))
                    long_df['Time']    = pd.to_numeric(long_df['Time'])
                    long_df['Outcome'] = pd.to_numeric(long_df['Outcome'])
                    long_df = long_df.sort_values(by=['ID', 'Time'])

                # ── INPUT VALIDATION ──────────────────────────────────────────

                obs_counts     = long_df.groupby('ID').size()
                single_obs_ids = obs_counts[obs_counts < 2].index.tolist()
                if single_obs_ids:
                    long_df = long_df[~long_df['ID'].isin(single_obs_ids)].copy()
                    preview = single_obs_ids[:5]
                    extra   = f" … and {len(single_obs_ids) - 5} more" if len(single_obs_ids) > 5 else ""
                    st.info(f"Removed {len(single_obs_ids)} subject(s) with only 1 observation (IDs: {preview}{extra}).")

                n_subjects_val = long_df['ID'].nunique()
                if n_subjects_val < 30:
                    st.warning(f"⚠️ Only {n_subjects_val} subjects remain after filtering. Results may be unreliable (recommended n ≥ 30).")

                if n_subjects_val > 20000 or len(long_df) > 500000:
                    st.warning(
                        f"⚠️ Large dataset ({n_subjects_val:,} subjects, {len(long_df):,} observations). "
                        "Fitting may take a while, especially with many multi-start restarts — consider "
                        "starting with fewer restarts to sanity-check the specification first."
                    )

                if app_mode == "AutoTraj Search":
                    n_combos_est = sum(
                        (order_range[1] - order_range[0] + 1) ** kk
                        for kk in range(group_range[0], group_range[1] + 1)
                    )
                    est_fits = n_combos_est * n_starts
                    if est_fits > 1500:
                        st.warning(
                            f"⚠️ This search will evaluate {n_combos_est} model specification(s) × "
                            f"{n_starts} restart(s) ≈ {est_fits:,} total optimizer runs. This may take "
                            "several minutes or more — consider narrowing the group/order range, or "
                            "reducing restarts, especially for a first pass."
                        )

                if dist_flag == 'LOGIT':
                    invalid_mask = ~long_df['Outcome'].isin([0.0, 1.0])
                    if invalid_mask.any():
                        bad_vals = sorted(long_df.loc[invalid_mask, 'Outcome'].unique().tolist())[:10]
                        st.error(f"🚨 **LOGIT requires binary outcomes (0 or 1).** Found non-binary values: {bad_vals}.")
                        st.stop()

                if dist_flag == 'CNORM':
                    if not pd.api.types.is_numeric_dtype(long_df['Outcome']):
                        st.error("🚨 **CNORM requires a numeric outcome.**")
                        st.stop()
                    elif long_df['Outcome'].dropna().apply(lambda x: float(x) == int(x)).all():
                        st.warning("⚠️ All Outcome values appear to be whole numbers. If binary, consider LOGIT instead.")

                if weight_col:
                    if not pd.api.types.is_numeric_dtype(long_df[weight_col]):
                        st.error(f"🚨 **Sampling weight column '{weight_col}' must be numeric.**")
                        st.stop()
                    elif (long_df[weight_col] <= 0).any() or long_df[weight_col].isna().any():
                        st.error(f"🚨 **Sampling weight column '{weight_col}' must be strictly positive for every subject** (found zero, negative, or missing values).")
                        st.stop()

                n_timepoints      = len(long_df['Time'].unique())
                max_order_attempted = max(orders_single) if app_mode == "Single Model Mode" else order_range[1]
                if max_order_attempted >= n_timepoints:
                    st.error(
                        f"🚨 **Unidentifiable Model:** Order {max_order_attempted} requires {max_order_attempted + 1} "
                        f"params per group but only {n_timepoints} unique time point(s) exist. "
                        f"Reduce order to at most {n_timepoints - 1}."
                    )
                    st.stop()

                # ── RUN MODEL ─────────────────────────────────────────────────

                if app_mode == "AutoTraj Search":
                    top_models, all_evaluated = run_autotraj(
                        long_df, min_groups=group_range[0], max_groups=group_range[1],
                        min_order=order_range[0], max_order=order_range[1],
                        min_group_pct=min_pct, p_val_thresh=p_val, use_dropout=use_dropout,
                        dist=dist_flag, cnorm_min=cnorm_min, cnorm_max=cnorm_max,
                        zip_iorder=0, n_starts=n_starts,
                        baseline_cov_cols=baseline_cov_cols, tvc_cols=tvc_cols,
                        weight_col=weight_col,
                        progress_callback=_progress_cb, log_callback=_log_cb,
                    )
                else:
                    single_res = run_single_model(
                        long_df, orders_single, zip_iorder=0,
                        use_dropout=use_dropout, dist=dist_flag,
                        cnorm_min=cnorm_min, cnorm_max=cnorm_max, n_starts=n_starts,
                        baseline_cov_cols=baseline_cov_cols, tvc_cols=tvc_cols,
                        weight_col=weight_col, log_callback=_log_cb,
                    )
                    top_models   = [single_res] if single_res['result'].success or single_res['result'].status == 2 else []
                    all_evaluated = []

            if progress_bar is not None:
                progress_bar.empty()
            if fit_log_lines:
                with st.expander(f"Fit Log ({len(fit_log_lines)} multi-start resolutions)"):
                    st.code("\n".join(fit_log_lines), language=None)

            st.session_state.run_complete  = True
            st.session_state.top_models    = top_models
            st.session_state.all_evaluated = all_evaluated
            st.session_state.run_time      = time.time() - start_time
            st.session_state.long_df       = long_df
            st.session_state.raw_df        = raw_df
            st.session_state.use_dropout   = use_dropout

    # ── RESULTS ───────────────────────────────────────────────────────────────

    if st.session_state.run_complete:
        top_models    = st.session_state.top_models
        all_evaluated = st.session_state.all_evaluated
        long_df       = st.session_state.long_df
        raw_df        = st.session_state.raw_df
        use_dropout_state = st.session_state.use_dropout
        run_time_val  = st.session_state.run_time

        if top_models:
            st.divider()

            save_bundle = pickle.dumps({
                'top_models': top_models, 'all_evaluated': all_evaluated,
                'long_df': long_df, 'use_dropout': use_dropout_state, 'app_mode': app_mode,
            })
            st.download_button(
                "💾 Save Fitted Model", save_bundle, file_name="autotraj_model.atmodel",
                mime="application/octet-stream",
                help="Downloads everything needed to reload these results instantly later, via "
                     "the 'Reload a previously saved model' expander above — no refitting required.",
            )

            if len(top_models) > 1 and app_mode == "AutoTraj Search":
                st.markdown("""
                <div class="model-explorer-card">
                    <h4>Model Explorer</h4>
                    <p>Select a valid model below to explore its trajectories, parameters, and diagnostics.</p>
                </div>
                """, unsafe_allow_html=True)
                model_choices = [f"Rank {i+1} | {len(m['orders'])}-Group {m['orders']} | BIC: {m['bic']:.2f}" for i, m in enumerate(top_models[:10])]
                selected_model_str = st.selectbox("Select a valid model to visualize:", model_choices, label_visibility="collapsed")
                selected_rank = int(selected_model_str.split("|")[0].replace("Rank ", "").strip()) - 1
                winning_model = top_models[selected_rank]

                with st.expander(f"📊 Compare Top {min(len(top_models), 10)} Models", expanded=False):
                    st.caption(
                        "Shortlist view: only models that passed the heuristic rejection rules, "
                        "ranked by BIC. For every specification tried — including rejected and "
                        "non-converged ones — see the 'BIC Search Diagnostics' tab below."
                    )
                    comparison_rows = []
                    for i, m in enumerate(top_models[:10]):
                        comparison_rows.append({
                            "Rank": i + 1, "Groups": len(m['orders']), "Orders": str(m['orders']),
                            "LL": round(m['ll'], 2), "BIC (Nagin)": round(m['bic'], 2),
                            "BIC (Standard)": round(m['bic_standard'], 2), "AIC (Nagin)": round(m['aic'], 2),
                            "Min Group %": round(m['min_pct'], 1) if pd.notnull(m['min_pct']) else "N/A",
                            "Condition #": f"{m['cond_num']:.1e}",
                        })
                    comparison_df = pd.DataFrame(comparison_rows)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Comparison Table (CSV)", comparison_df.to_csv(index=False).encode('utf-8'),
                        file_name="model_comparison.csv", mime="text/csv", key="dl_comparison_table",
                    )

                    pin_choices = st.multiselect(
                        "Pin models to compare trajectories side by side:", model_choices,
                        default=model_choices[:min(2, len(model_choices))], key="pin_models_select",
                    )
                    if pin_choices:
                        pin_cols = st.columns(len(pin_choices))
                        for col, choice_str in zip(pin_cols, pin_choices):
                            with col:
                                rank = int(choice_str.split("|")[0].replace("Rank ", "").strip()) - 1
                                pinned_model = top_models[rank]
                                st.markdown(f"**Rank {rank + 1}** ({len(pinned_model['orders'])}-Group {pinned_model['orders']})")
                                pinned_assignments = get_subject_assignments(pinned_model, long_df)
                                pinned_group_names = [f"Group {g+1}" for g in range(len(pinned_model['orders']))]
                                st.plotly_chart(
                                    _obs_vs_est_figure(long_df, pinned_assignments, pinned_model, pinned_group_names, pinned_model.get('dist', 'LOGIT')),
                                    use_container_width=True, key=f"pin_fig_{rank}",
                                )
            else:
                winning_model = top_models[0]
                st.subheader("🏆 Model Results")

            winning_orders  = winning_model['orders']
            winning_result  = winning_model['result']
            winning_pis_raw = winning_model['pis']
            dist_type       = winning_model.get('dist', 'LOGIT')
            se_model_arr    = winning_model['se_model']   # aligned with result.x after label sort

            if winning_model.get('cond_num', 0) > 1e10 or np.any(winning_model['se_model'] < 1e-3) or np.any(winning_model['se_model'] > 50):
                st.warning("⚠️ **Warning: Unidentifiable Model Detected.** Standard errors are degenerate. Consider reducing groups.")

            n_eval        = len(all_evaluated) if all_evaluated else 1
            mps           = n_eval / run_time_val if run_time_val > 0 else 0
            manual_mins   = n_eval * manual_min_per_model
            manual_str    = f"~{manual_mins:.0f} min" if manual_mins < 60 else f"~{manual_mins/60:.1f} hrs"
            time_saved_min = manual_mins - (run_time_val / 60.0)
            saved_str      = f"~{time_saved_min:.0f} min saved" if time_saved_min < 60 else f"~{time_saved_min/60:.1f} hrs saved"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("BIC (Nagin)",    f"{winning_model['bic_nagin']:.2f}")
            col2.metric("BIC (Standard)", f"{winning_model['bic_standard']:.2f}")
            col3.metric("AIC (Nagin)",    f"{winning_model['aic_nagin']:.2f}")
            col4.metric("AIC (Standard)", f"{winning_model['aic_standard']:.2f}")
            st.caption("Nagin convention: higher (less negative) = better fit. Standard convention: lower = better.")
            col5, col6, col7 = st.columns(3)
            col5.metric("Log-Likelihood", f"{winning_model['ll']:.2f}")
            col6.metric("Engine Time",    f"{run_time_val:.2f}s", f"{n_eval} models | {mps:.1f}/sec", delta_color="off")
            col7.metric(
                "Manual Proc Time (est.)", manual_str, saved_str,
                delta_color="normal" if time_saved_min > 0 else "off",
                help=f"Based on your 'Est. Manual Time per Model' assumption of {manual_min_per_model:.1f} "
                     f"min × {n_eval} model(s) evaluated — an editable estimate (see the sidebar), not a "
                     "benchmark. Compares against this run's actual Engine Time.",
            )

            st.markdown("##### ✏️ Customize Plot Labels & Group Names")
            col_lbl1, col_lbl2 = st.columns(2)
            x_axis_label = col_lbl1.text_input("X-Axis Label", value="Time Period")
            if dist_type == 'LOGIT':
                default_y_label = "Probability of Outcome"
            elif dist_type in ('ZIP', 'POISSON'):
                default_y_label = "Expected Count"
            else:
                default_y_label = "Outcome Score"
            y_axis_label = col_lbl2.text_input("Y-Axis Label", value=default_y_label)

            cols_gn = st.columns(len(winning_orders))
            group_names = []
            for g in range(len(winning_orders)):
                name = cols_gn[g].text_input(f"Group {g+1} Label", value=f"Group {g+1}")
                group_names.append(name)

            assignments_df = get_subject_assignments(winning_model, long_df)

            adq_df_summary, rel_entropy_summary = calc_model_adequacy(assignments_df, winning_pis_raw, group_names)
            plain_summary_txt = _generate_plain_language_summary(
                winning_model, group_names, long_df, adq_df_summary, rel_entropy_summary, dist_type
            )
            with st.container(border=True):
                st.markdown("##### 🗒️ Plain-Language Summary")
                st.markdown(plain_summary_txt)
                st.caption("Auto-generated directly from the fitted parameters and adequacy diagnostics — not an AI-written narrative.")

            st.divider()
            st.subheader("Publication Suite")

            tab_viz, tab_est, tab_adq, tab_char, tab_comp, tab_export = st.tabs([
                "Trajectories", "Parameter Estimates", "Model Adequacy",
                "Baseline Characteristics", "BIC Search Diagnostics", "Reports & Exports"
            ])

            # ── VISUALIZATION TAB ─────────────────────────────────────────────

            with tab_viz:
                col_viz1, col_viz2 = st.columns([3, 1])
                with col_viz2:
                    viz_style     = st.selectbox("Graphic Style:", [
                        "Interactive Web (Plotly)",
                        "Publication: Grayscale (Matplotlib)",
                        "Publication: Color (Matplotlib)"
                    ])
                    st.markdown("**Plot Elements:**")
                    show_spaghetti = st.checkbox("Individual Trajectories",        value=False)
                    show_smooth    = st.checkbox("Estimated Curves (Smoothed)",    value=True)
                    show_ci        = st.checkbox("95% Confidence Bands",           value=True)
                    show_obs       = st.checkbox("Observed Averages",              value=True)

                actual_times = long_df['Time'].values
                smooth_times = np.linspace(min(actual_times), max(actual_times), 200)

                merged_for_plot = pd.merge(long_df, assignments_df[['ID', 'Assigned_Group']], on='ID')
                obs_means       = merged_for_plot.groupby(['Assigned_Group', 'Time'])['Outcome'].mean().reset_index()

                # Pre-compute beta indices once (used by both Plotly and Matplotlib)
                beta_info   = _beta_start_indices(winning_orders, n_mix=winning_model.get('n_mix', 1))
                k_plot      = len(winning_orders)
                tvc_names_plot = winning_model.get('tvc_cols') or []
                n_tvc_plot     = winning_model.get('n_tvc', 0)
                delta_start_plot = _delta_start_index(winning_orders, n_mix=winning_model.get('n_mix', 1))
                if n_tvc_plot > 0:
                    tvc_means_plot = long_df[tvc_names_plot].mean().values
                else:
                    tvc_means_plot = np.zeros(0)

                _BRAND_COLORS    = ['#2B6083', '#B5373A', '#D4A843', '#2E7D52', '#7B4F8A', '#C97B2A']
                plotly_colors    = _BRAND_COLORS
                mpl_colors_color = _BRAND_COLORS
                mpl_colors_gray  = ['black', 'dimgray', 'darkgray', 'lightgray', 'slategray', 'silver']

                # ── helper: TVC deflection evaluated at the sample-mean TVC level
                # (V3.0). Returns 0.0 when no TVCs are present.
                def _tvc_offset(g_idx):
                    if n_tvc_plot == 0:
                        return 0.0
                    g_delta = winning_result.x[delta_start_plot + g_idx * n_tvc_plot: delta_start_plot + (g_idx + 1) * n_tvc_plot]
                    return float(np.dot(g_delta, tvc_means_plot))

                # ── helper: compute trajectory curve for one group
                # V3.0: if TVCs are present, the curve is plotted "at mean TVC
                # level" — a constant offset (delta_g . mean(TVC)) is added to
                # eta, since the trajectory is no longer a function of t alone.
                def _group_curve(g_idx):
                    beta_start, n_betas = beta_info[g_idx]
                    g_betas = winning_result.x[beta_start:beta_start + n_betas]
                    order   = winning_orders[g_idx]
                    X_smooth = create_design_matrix_jit(smooth_times, order)
                    tvc_offset = _tvc_offset(g_idx)
                    if dist_type == 'LOGIT':
                        y_hat = calc_logit_prob_jit(g_betas, X_smooth) if tvc_offset == 0.0 else \
                            1.0 / (1.0 + np.exp(-(X_smooth @ g_betas + tvc_offset)))
                    elif dist_type == 'POISSON':
                        y_hat = np.exp(X_smooth @ g_betas + tvc_offset)
                    elif dist_type == 'ZIP':
                        lam     = np.exp(X_smooth @ g_betas + tvc_offset)
                        zeta_g  = winning_result.x[len(winning_result.x) - k_plot + g_idx]
                        omega_g = 1.0 / (1.0 + np.exp(-zeta_g))
                        y_hat   = lam * (1.0 - omega_g)
                    else:
                        y_hat = X_smooth @ g_betas + tvc_offset
                    return g_betas, order, y_hat, beta_start, n_betas

                with col_viz1:
                    if "Plotly" in viz_style:
                        fig = go.Figure()
                        light_colors = ['rgba(43,96,131,0.15)',  'rgba(181,55,58,0.15)',
                                        'rgba(212,168,67,0.15)', 'rgba(46,125,82,0.15)',
                                        'rgba(123,79,138,0.15)', 'rgba(201,123,42,0.15)']

                        if show_spaghetti:
                            id_group_map = assignments_df.set_index('ID')['Assigned_Group'].to_dict()
                            sample_ids = long_df['ID'].drop_duplicates().sample(
                                n=min(100, long_df['ID'].nunique()), random_state=42
                            )
                            for s_id in sample_ids:
                                sub_df = long_df[long_df['ID'] == s_id]
                                g_num  = id_group_map.get(s_id, 1)
                                cidx   = (g_num - 1) % len(plotly_colors)
                                r, gr, b = int(plotly_colors[cidx][1:3], 16), int(plotly_colors[cidx][3:5], 16), int(plotly_colors[cidx][5:7], 16)
                                col_light = f'rgba({r},{gr},{b},0.12)'
                                fig.add_trace(go.Scatter(
                                    x=sub_df['Time'], y=sub_df['Outcome'],
                                    mode='lines', opacity=1.0,
                                    line=dict(color=col_light, width=1),
                                    hoverinfo='skip', showlegend=False
                                ))

                        for g in range(k_plot):
                            g_betas, order, y_hat, beta_start, n_betas = _group_curve(g)
                            color = plotly_colors[g % len(plotly_colors)]

                            if show_ci:
                                lo, hi = _compute_ci_band(
                                    smooth_times, g_betas, order, se_model_arr,
                                    beta_start, n_betas, dist_type, eta_offset=_tvc_offset(g)
                                )
                                fig.add_trace(go.Scatter(
                                    x=np.concatenate([smooth_times, smooth_times[::-1]]),
                                    y=np.concatenate([hi, lo[::-1]]),
                                    fill='toself', fillcolor=light_colors[g % len(light_colors)],
                                    line=dict(color='rgba(0,0,0,0)'),
                                    hoverinfo='skip', showlegend=False
                                ))

                            if show_smooth:
                                fig.add_trace(go.Scatter(
                                    x=smooth_times, y=y_hat, mode='lines',
                                    line=dict(color=color, width=4, dash='dot' if show_obs else 'solid'),
                                    name=f'{group_names[g]} (Est.)'
                                ))

                            if show_obs:
                                g_obs = obs_means[obs_means['Assigned_Group'] == g + 1]
                                fig.add_trace(go.Scatter(
                                    x=g_obs['Time'], y=g_obs['Outcome'],
                                    mode='lines+markers+text',
                                    text=[f"{g+1}"] * len(g_obs), textposition="top center",
                                    line=dict(color=color, width=2),
                                    name=f'{group_names[g]} (Obs.)'
                                ))

                        y_range_val = [-0.1, 1.1] if dist_type == 'LOGIT' else None
                        fig.update_layout(
                            yaxis_title=y_axis_label, xaxis_title=x_axis_label,
                            yaxis_range=y_range_val,
                            template="plotly_white",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#7F8C9A', size=13),
                            xaxis=dict(gridcolor='rgba(128,128,128,0.15)', title_font=dict(color='#7F8C9A', size=13)),
                            yaxis=dict(gridcolor='rgba(128,128,128,0.15)', title_font=dict(color='#7F8C9A', size=13)),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        # ── Matplotlib ──────────────────────────────────────────
                        colors = mpl_colors_gray if "Grayscale" in viz_style else mpl_colors_color
                        fig_mpl, ax = plt.subplots(figsize=(8, 5))
                        fig_mpl.patch.set_facecolor('none')
                        ax.patch.set_facecolor('none')

                        if show_spaghetti:
                            id_group_map = assignments_df.set_index('ID')['Assigned_Group'].to_dict()
                            sample_ids = long_df['ID'].drop_duplicates().sample(
                                n=min(100, long_df['ID'].nunique()), random_state=42
                            )
                            for s_id in sample_ids:
                                sub_df = long_df[long_df['ID'] == s_id]
                                g_num  = id_group_map.get(s_id, 1)
                                cidx   = (g_num - 1) % len(colors)
                                ax.plot(sub_df['Time'], sub_df['Outcome'],
                                        color=colors[cidx], alpha=0.12, linewidth=0.8)

                        for g in range(k_plot):
                            g_betas, order, y_hat, beta_start, n_betas = _group_curve(g)
                            color = colors[g % len(colors)]

                            if show_ci:
                                lo, hi = _compute_ci_band(
                                    smooth_times, g_betas, order, se_model_arr,
                                    beta_start, n_betas, dist_type, eta_offset=_tvc_offset(g)
                                )
                                ax.fill_between(smooth_times, lo, hi, color=color, alpha=0.15)

                            if show_smooth:
                                lw = 2.5 if not show_obs else 1.5
                                ls = '--' if show_obs else '-'
                                ax.plot(smooth_times, y_hat, linewidth=lw, color=color,
                                        linestyle=ls, label=f'{group_names[g]} (Est.)')

                            if show_obs:
                                g_obs = obs_means[obs_means['Assigned_Group'] == g + 1]
                                ax.plot(g_obs['Time'], g_obs['Outcome'], color=color,
                                        marker='o', linewidth=2, label=f'{group_names[g]} (Obs.)')
                                for _, row in g_obs.iterrows():
                                    ax.text(row['Time'], row['Outcome'] + 0.02, str(g + 1),
                                            color=color, ha='center', fontsize=8)

                        if dist_type == 'LOGIT':
                            ax.set_ylim(-0.1, 1.1)
                        ax.set_ylabel(y_axis_label)
                        ax.set_xlabel(x_axis_label)
                        ax.legend(frameon=False)
                        plt.tight_layout()
                        st.pyplot(fig_mpl)

                # ── MODEL EQUATIONS ───────────────────────────────────────────
                st.markdown("**Fitted Model Equations**")
                report_equations = []
                for g in range(k_plot):
                    beta_start, n_betas = beta_info[g]
                    g_betas = winning_result.x[beta_start:beta_start + n_betas]
                    g_delta = (
                        winning_result.x[delta_start_plot + g * n_tvc_plot: delta_start_plot + (g + 1) * n_tvc_plot]
                        if n_tvc_plot > 0 else None
                    )
                    eq = _build_equation_latex(
                        g_betas, winning_orders[g], dist_type,
                        group_names[g], g, winning_result, winning_orders,
                        g_delta=g_delta, tvc_names=tvc_names_plot,
                    )
                    st.latex(eq)
                    report_equations.append(eq)

                # V3.0: mixing-covariate equation (theta_g(x) = Gamma_g . x), only
                # shown when baseline covariates were used for group membership.
                n_mix_plot = winning_model.get('n_mix', 1)
                if n_mix_plot > 1:
                    gamma_block = winning_result.x[0:(k_plot - 1) * n_mix_plot].reshape(k_plot - 1, n_mix_plot)
                    gamma_matrix_plot = np.vstack([np.zeros((1, n_mix_plot)), gamma_block])
                    mix_eqs = _build_mixing_equation_latex(
                        gamma_matrix_plot, group_names, winning_model.get('baseline_cov_cols') or []
                    )
                    if mix_eqs:
                        st.markdown("**Group Membership Equation** (multinomial logit on mixing proportions)")
                        for eq in mix_eqs:
                            st.latex(eq)

                # ── DOWNLOAD BUTTONS ──────────────────────────────────────────
                st.markdown("**Download Plot**")
                dl_col1, dl_col2, dl_col3 = st.columns(3)

                # SVG
                buf_svg = io.BytesIO()
                try:
                    fig_mpl.savefig(buf_svg, format='svg', bbox_inches='tight')
                    buf_svg.seek(0)
                    dl_col1.download_button("📥 SVG (Vector)", data=buf_svg,
                                            file_name="trajectory_plot.svg", mime="image/svg+xml")
                except Exception:
                    dl_col1.caption("SVG unavailable (Plotly mode)")

                # PNG 300 DPI
                buf_png = io.BytesIO()
                try:
                    fig_mpl.savefig(buf_png, format='png', dpi=300, bbox_inches='tight')
                    buf_png.seek(0)
                    dl_col2.download_button("📥 PNG 300 DPI", data=buf_png,
                                            file_name="trajectory_plot.png", mime="image/png")
                except Exception:
                    dl_col2.caption("PNG unavailable (Plotly mode)")

                dl_col3.download_button(
                    label="📥 Observed Averages (CSV)",
                    data=obs_means.to_csv(index=False).encode('utf-8'),
                    file_name='trajectory_observed_averages.csv', mime='text/csv'
                )

            # ── EXACT ESTIMATES TAB ───────────────────────────────────────────

            with tab_est:
                if winning_model.get('weight_col'):
                    st.warning(
                        f"⚠️ **Survey weights active** (column: `{winning_model['weight_col']}`). "
                        "Use the **Robust SE** column as the valid basis for inference — "
                        "**Standard Error** (model-based) is shown for reference only and is not "
                        "a consistent variance estimator under survey weighting."
                    )
                estimates_df = get_parameter_estimates_for_ui(winning_model, group_names)
                st.dataframe(estimates_df, use_container_width=True, hide_index=True)
                st.download_button(
                    label="📥 Download Parameter Estimates (CSV)",
                    data=estimates_df.to_csv(index=False).encode('utf-8'),
                    file_name='trajectory_parameters.csv', mime='text/csv'
                )

            # ── ADEQUACY TAB ──────────────────────────────────────────────────

            with tab_adq:
                adq_df, rel_entropy = calc_model_adequacy(assignments_df, winning_pis_raw, group_names)

                # ── Summary row ───────────────────────────────────────────────
                st.metric(label="Relative Entropy (0-1)", value=f"{rel_entropy:.3f}",
                          help="Values closer to 1 indicate better group separation. Rule of thumb: ≥ 0.70 is good.")
                st.dataframe(adq_df, use_container_width=True, hide_index=True)
                st.divider()

                # ── Entropy decomposition ─────────────────────────────────────
                st.markdown("#### Per-Group Entropy")
                st.caption(
                    "Group-level relative entropy: how cleanly each assigned subgroup is separated. "
                    "Values near 1.0 = clean assignment; near 0 = diffuse posteriors within that group."
                )
                k_adq = len(winning_orders)
                ent_df = _entropy_decomposition(assignments_df, winning_pis_raw, k_adq, group_names)
                st.dataframe(ent_df, use_container_width=True, hide_index=True)

                # Bar chart of per-group entropy
                if k_adq > 1:
                    palette_adq = ['#2B6083', '#B5373A', '#D4A843', '#2E7D52', '#7B4F8A', '#C97B2A']
                    ent_vals = [
                        float(row["Group Rel. Entropy"]) if row["Group Rel. Entropy"] != "N/A" else 0.0
                        for _, row in ent_df.iterrows()
                    ]
                    fig_ent = go.Figure(go.Bar(
                        x=group_names,
                        y=ent_vals,
                        marker_color=[palette_adq[g % len(palette_adq)] for g in range(k_adq)],
                        text=[f"{v:.3f}" for v in ent_vals],
                        textposition='outside',
                    ))
                    fig_ent.add_hline(y=0.7, line_dash="dot", line_color="gray",
                                      annotation_text="0.70 threshold", annotation_position="right")
                    fig_ent.update_layout(
                        yaxis=dict(range=[0, 1.1], title="Relative Entropy"),
                        xaxis_title="Group",
                        template="plotly_white", height=320,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_ent, use_container_width=True)
                st.divider()

                # ── Posterior probability heatmap ─────────────────────────────
                st.markdown("#### Posterior Probability Matrix")
                st.caption(
                    "Rows = assigned group, Columns = posterior probability of each group. "
                    "Diagonal (self-assignment probability) should be > 0.70 for a well-identified model."
                )
                _, fig_heat = _posterior_heatmap(assignments_df, k_adq, group_names)
                st.plotly_chart(fig_heat, use_container_width=True)
                st.divider()

                # ── Observed vs Estimated ─────────────────────────────────────
                st.markdown("#### Observed vs. Estimated Trajectories")
                st.caption(
                    "Points = posterior-weighted observed group means at each time point. "
                    "Lines = model-estimated trajectory. Close alignment indicates good model fit."
                )
                fig_ove = _obs_vs_est_figure(long_df, assignments_df, winning_model, group_names, dist_type)
                st.plotly_chart(fig_ove, use_container_width=True)
                st.divider()

                # ── Residual analysis ─────────────────────────────────────────
                st.markdown("#### Residual Analysis")
                st.caption("Residual = observed − model-predicted value for each subject's assigned group.")

                resid_df, fig_hist_r, fig_qq_r = _residual_analysis(
                    long_df, assignments_df, winning_model, group_names, dist_type
                )

                outlier_n = int(resid_df['Outlier'].sum())
                resid_mu  = resid_df['Mean_Residual'].mean()
                resid_sd  = resid_df['Mean_Residual'].std()

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Mean Residual",  f"{resid_mu:.4f}", help="Should be near 0 for unbiased fit.")
                rc2.metric("SD of Residuals", f"{resid_sd:.4f}")
                rc3.metric("Outlier Subjects (|resid| > 2.5 SD)", str(outlier_n))

                st.plotly_chart(fig_hist_r, use_container_width=True)

                if fig_qq_r is not None:
                    st.markdown("**Q-Q Plot (CNORM residuals)**")
                    st.pyplot(fig_qq_r)

                if outlier_n > 0:
                    st.markdown("**Flagged Outlier Subjects**")
                    outlier_tbl = (
                        resid_df[resid_df['Outlier']]
                        .assign(
                            **{'Assigned Group': resid_df['Assigned_Group'].map(
                                lambda g: group_names[g - 1] if g - 1 < len(group_names) else f"Group {g}"
                            )}
                        )[['ID', 'Assigned Group', 'Mean_Residual']]
                        .sort_values('Mean_Residual', key=np.abs, ascending=False)
                        .rename(columns={'Mean_Residual': 'Mean Residual'})
                    )
                    st.dataframe(outlier_tbl, use_container_width=True, hide_index=True)

            # ── SAMPLE CHARACTERISTICS TAB ────────────────────────────────────

            with tab_char:
                if HAS_TABLEONE:
                    if data_format == "Wide Format" or st.session_state.use_sample_data:
                        potential_covariates = [col for col in raw_df.columns.tolist()
                                                if not col.startswith((outcome_col, time_col))]
                        selected_vars   = st.multiselect(
                            "Additional descriptive variables (not used in model):",
                            potential_covariates,
                            help="Purely descriptive — for baseline-characteristics reporting only. "
                                 "This does NOT feed the model; use the 'Covariate Architecture' "
                                 "expander in the sidebar to add covariates that affect group membership "
                                 "or the trajectory equation.",
                        )
                        categorical_vars = st.multiselect("Which of these are categorical?", selected_vars)
                        if selected_vars and st.button("Generate Table 1"):
                            merged_df = pd.merge(raw_df, assignments_df[['ID', 'Assigned_Group']],
                                                 left_on=id_col, right_on='ID')
                            group_map = {i + 1: name for i, name in enumerate(group_names)}
                            merged_df['Assigned_Group'] = merged_df['Assigned_Group'].map(group_map)
                            mytable = TableOne(merged_df, columns=selected_vars,
                                              categorical=categorical_vars,
                                              groupby="Assigned_Group", pval=True)
                            st.markdown(mytable.to_html(), unsafe_allow_html=True)
                    else:
                        st.info("Table 1 requires wide-format data. Join the exported assignments CSV to your baseline data.")
                else:
                    st.warning("Run `pip install tableone` to enable this feature.")

            # ── MODEL COMPARISON TAB ──────────────────────────────────────────

            with tab_comp:
                if app_mode == "AutoTraj Search" and all_evaluated:
                    # ── BIC elbow plot ─────────────────────────────────────────
                    st.markdown("#### BIC Elbow Plot")
                    st.caption(
                        "All evaluated models are shown. "
                        "Green = valid, orange = rejected by heuristic rules, "
                        "gray = failed convergence. "
                        "The best BIC per group count is connected by the elbow line."
                    )

                    best_per_k = {}
                    for m in all_evaluated:
                        if m['Status'] != "Failed Convergence" and not np.isnan(m['BIC (Nagin)']):
                            kk = m['Groups']
                            if kk not in best_per_k or m['BIC (Nagin)'] > best_per_k[kk]['BIC (Nagin)']:
                                best_per_k[kk] = m

                    # Categorise every evaluated model for scatter colouring
                    STATUS_COLOR = {
                        'Valid':              '#2E7D52',  # brand success green
                        'Failed Convergence': '#aaaaaa',  # gray
                    }
                    _DEFAULT_REJECTED = '#B5373A'         # brand red for rejected

                    # Separate traces by category for a clean legend
                    cat_data: dict[str, list] = {
                        'Valid': {'x': [], 'y': [], 'text': []},
                        'Rejected': {'x': [], 'y': [], 'text': []},
                        'Failed': {'x': [], 'y': [], 'text': []},
                    }
                    for m in all_evaluated:
                        bic_val = m['BIC (Nagin)']
                        hover   = (
                            f"Groups: {m['Groups']}<br>"
                            f"Orders: {m['Orders']}<br>"
                            f"BIC (Nagin): {round(bic_val, 2) if pd.notnull(bic_val) else 'N/A'}<br>"
                            f"Status: {m['Status']}"
                        )
                        if not pd.notnull(bic_val):
                            cat_data['Failed']['x'].append(m['Groups'])
                            cat_data['Failed']['y'].append(np.nan)
                            cat_data['Failed']['text'].append(hover)
                        elif m['Status'] == 'Valid':
                            cat_data['Valid']['x'].append(m['Groups'])
                            cat_data['Valid']['y'].append(bic_val)
                            cat_data['Valid']['text'].append(hover)
                        else:
                            cat_data['Rejected']['x'].append(m['Groups'])
                            cat_data['Rejected']['y'].append(bic_val)
                            cat_data['Rejected']['text'].append(hover)

                    fig_bic = go.Figure()

                    # Background scatter: rejected
                    if cat_data['Rejected']['x']:
                        fig_bic.add_trace(go.Scatter(
                            x=cat_data['Rejected']['x'],
                            y=cat_data['Rejected']['y'],
                            mode='markers',
                            name='Rejected',
                            marker=dict(color=_DEFAULT_REJECTED, size=7, opacity=0.5,
                                        symbol='circle-open'),
                            hovertext=cat_data['Rejected']['text'],
                            hoverinfo='text',
                        ))

                    # Valid models scatter
                    if cat_data['Valid']['x']:
                        fig_bic.add_trace(go.Scatter(
                            x=cat_data['Valid']['x'],
                            y=cat_data['Valid']['y'],
                            mode='markers',
                            name='Valid',
                            marker=dict(color=STATUS_COLOR['Valid'], size=9, opacity=0.7),
                            hovertext=cat_data['Valid']['text'],
                            hoverinfo='text',
                        ))

                    # Best-per-k elbow line
                    if best_per_k:
                        ks_line   = sorted(best_per_k.keys())
                        bics_line = [best_per_k[kk]['BIC (Nagin)'] for kk in ks_line]
                        hover_line = [
                            (f"Groups: {kk}<br>Orders: {best_per_k[kk]['Orders']}<br>"
                             f"BIC (Nagin): {round(best_per_k[kk]['BIC (Nagin)'], 2)}")
                            for kk in ks_line
                        ]
                        fig_bic.add_trace(go.Scatter(
                            x=ks_line, y=bics_line,
                            mode='lines+markers',
                            name='Best per k',
                            line=dict(color='#2B6083', width=3),
                            marker=dict(color='#2B6083', size=12,
                                        symbol='diamond', line=dict(color='white', width=1.5)),
                            hovertext=hover_line,
                            hoverinfo='text',
                        ))

                    fig_bic.update_layout(
                        xaxis=dict(title="Number of Groups", tickmode='linear', tick0=1, dtick=1),
                        yaxis_title="BIC (Nagin) — higher = better",
                        template="plotly_white",
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        height=420,
                    )
                    st.plotly_chart(fig_bic, use_container_width=True)

                    # ── Full comparison table ──────────────────────────────────
                    st.markdown("#### All Evaluated Models")
                    comp_df = pd.DataFrame(all_evaluated)
                    for col in ['BIC (Nagin)', 'BIC (Standard)', 'AIC (Nagin)', 'AIC (Standard)']:
                        comp_df[col] = comp_df[col].apply(lambda x: round(x, 2) if pd.notnull(x) else "NaN")
                    comp_df['Min_Group_%'] = comp_df['Min_Group_%'].apply(
                        lambda x: round(x, 1) if pd.notnull(x) else "NaN")
                    st.dataframe(comp_df, hide_index=True, use_container_width=True)
                elif app_mode == "Single Model Mode":
                    st.info("Model Comparison is only available in AutoTraj Search mode.")

            # ── REPORTS & EXPORTS TAB ─────────────────────────────────────────

            with tab_export:
                st.caption(
                    "Every export below reflects the model currently selected above (Model "
                    "Explorer / Publication Suite), including the Plain-Language Summary shown "
                    "at the top of the page."
                )

                export_col1, export_col2, export_col3, export_col4 = st.columns(4)

                with export_col1:
                    st.download_button(
                        label="📥 Download Posterior Probabilities (CSV)",
                        data=assignments_df.to_csv(index=False).encode('utf-8'),
                        file_name='gbtm_trajectory_assignments.csv', mime='text/csv'
                    )

                with export_col2:
                    # ── Full Results Package (ZIP) ──────────────────────────
                    adq_df_exp, rel_entropy_exp = calc_model_adequacy(
                        assignments_df, winning_pis_raw, group_names)
                    estimates_df_exp = get_parameter_estimates_for_ui(winning_model, group_names)
                    summary_txt = _make_model_summary_txt(winning_model, group_names, rel_entropy_exp)

                    # Render plot to bytes
                    buf_svg_exp = io.BytesIO()
                    buf_png_exp = io.BytesIO()
                    try:
                        fig_mpl.savefig(buf_svg_exp, format='svg', bbox_inches='tight')
                        fig_mpl.savefig(buf_png_exp, format='png', dpi=300, bbox_inches='tight')
                        buf_svg_exp.seek(0)
                        buf_png_exp.seek(0)
                        plot_bytes_available = True
                    except Exception:
                        plot_bytes_available = False

                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("parameter_estimates.csv",    estimates_df_exp.to_csv(index=False))
                        zf.writestr("posterior_assignments.csv",  assignments_df.to_csv(index=False))
                        zf.writestr("adequacy_metrics.csv",       adq_df_exp.to_csv(index=False))
                        zf.writestr("model_summary.txt",          summary_txt)
                        zf.writestr("plain_language_summary.md",  plain_summary_txt)
                        if all_evaluated:
                            comp_df_exp = pd.DataFrame(all_evaluated)
                            zf.writestr("model_comparison.csv",  comp_df_exp.to_csv(index=False))
                        if plot_bytes_available:
                            zf.writestr("trajectory_plot.svg",   buf_svg_exp.read())
                            zf.writestr("trajectory_plot.png",   buf_png_exp.read())
                    zip_buf.seek(0)

                    st.download_button(
                        label="📦 Download Full Results Package (.zip)",
                        data=zip_buf,
                        file_name='gbtm_results_package.zip',
                        mime='application/zip'
                    )

                with export_col3:
                    report_html = _build_html_report(
                        winning_model, group_names, estimates_df_exp, adq_df_exp, rel_entropy_exp,
                        summary_txt, report_equations,
                        png_bytes=buf_png_exp.getvalue() if plot_bytes_available else None,
                        plain_summary=plain_summary_txt,
                    )
                    st.download_button(
                        label="📄 Generate HTML Report",
                        data=report_html.encode('utf-8'),
                        file_name='gbtm_model_report.html', mime='text/html',
                        help="A single shareable HTML file with the model summary, equations, "
                             "parameter table, adequacy diagnostics, and trajectory plot.",
                    )

                with export_col4:
                    try:
                        report_pdf = _build_pdf_report(
                            winning_model, group_names, estimates_df_exp, adq_df_exp, rel_entropy_exp,
                            summary_txt, report_equations,
                            png_bytes=buf_png_exp.getvalue() if plot_bytes_available else None,
                            plain_summary=plain_summary_txt,
                        )
                        st.download_button(
                            label="📑 Generate PDF Report",
                            data=report_pdf,
                            file_name='gbtm_model_report.pdf', mime='application/pdf',
                            help="A print-ready PDF version of the same report — suitable for "
                                 "journal supplementary materials.",
                        )
                    except Exception as e:
                        st.caption(f"PDF report unavailable: {e}")

        else:
            st.error("Model Failed to Converge or was rejected based on heuristic rules.")

    st.markdown(
        '<div class="app-footer">AutoTraj &nbsp;&middot;&nbsp;'
        'Built by Donald E. Warden, PhD, MPH &nbsp;&middot;&nbsp; '
        '<em>Sapientia Veritatem Parit</em></div>',
        unsafe_allow_html=True,
    )
