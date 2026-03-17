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
    calc_model_adequacy
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _beta_start_indices(orders_list):
    """Return list of (start_idx, n_betas) tuples for each group's beta block."""
    k = len(orders_list)
    idx = k - 1
    out = []
    for g in range(k):
        n = orders_list[g] + 1
        out.append((idx, n))
        idx += n
    return out


def _compute_ci_band(smooth_times, g_betas, order, se_model, beta_start, n_betas, dist_type, z=1.96):
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

    eta = X @ g_betas

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

    k = len(orders)
    if group_names is None or len(group_names) != k:
        group_names = [f"Group {g+1}" for g in range(k)]

    data = []
    current_beta_idx  = k - 1
    current_gamma_idx = (k - 1) + sum([o + 1 for o in orders])
    labels       = ["Intercept", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
    gamma_labels = ["Dropout: Intercept", "Dropout: Time", "Dropout: Prev Outcome"]

    for g in range(k):
        n_betas = orders[g] + 1
        for b_idx in range(n_betas):
            est   = params[current_beta_idx + b_idx]
            err_m = se_model[current_beta_idx + b_idx]
            err_r = se_robust[current_beta_idx + b_idx]
            t_stat = est / err_m if err_m > 0 else 0
            p_val  = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
            data.append({
                "Component": "Trajectory", "Group": str(group_names[g]),
                "Parameter": labels[b_idx],
                "Estimate": round(est, 5), "Standard Error": round(err_m, 5),
                "Robust SE": round(err_r, 5),
                "T for H0: Param=0": round(t_stat, 3),
                "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
            })
        current_beta_idx += n_betas

        if use_dropout:
            for gam_idx in range(3):
                est   = params[current_gamma_idx + gam_idx]
                err_m = se_model[current_gamma_idx + gam_idx]
                err_r = se_robust[current_gamma_idx + gam_idx]
                t_stat = est / err_m if err_m > 0 else 0
                p_val  = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
                data.append({
                    "Component": "Dropout", "Group": str(group_names[g]),
                    "Parameter": gamma_labels[gam_idx],
                    "Estimate": round(est, 5), "Standard Error": round(err_m, 5),
                    "Robust SE": round(err_r, 5),
                    "T for H0: Param=0": round(t_stat, 3),
                    "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
                })
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


def _build_equation_latex(g_betas, order, dist_type, group_name, g_idx, winning_result, winning_orders):
    """Return a LaTeX string for one group's fitted equation."""
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
    beta_info = _beta_start_indices(orders)
    for g in range(k):
        start, n = beta_info[g]
        lines.append(f"  {group_names[g]}:")
        for p in range(n):
            lines.append(f"    beta_{p} = {params[start+p]:.5f}  (SE={se_model[start+p]:.5f})")
    return "\n".join(lines)


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
    beta_info      = _beta_start_indices(orders)
    k              = len(orders)
    prob_cols      = [f'Group_{g+1}_Prob' for g in range(k)]
    colors         = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']

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
    beta_info      = _beta_start_indices(orders)

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
    palette  = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
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

if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
    st.session_state.top_models  = None
    st.session_state.all_evaluated = None
    st.session_state.run_time    = 0
    st.session_state.long_df     = None
    st.session_state.raw_df      = None
    st.session_state.use_sample_data = False

with st.sidebar:
    st.title("AutoTraj")
    app_mode = st.radio("Navigation", ["AutoTraj Search", "Single Model Mode", "About & Docs"])
    st.markdown("---")

    if app_mode != "About & Docs":
        st.markdown("**1. Data Format**")
        data_format = st.radio("Select Data Structure:", ["Wide Format", "Long Format"], horizontal=True)

        st.markdown("**2. Data Mapping**")
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

        st.markdown("**3. Model Distribution**")
        selected_dist = st.selectbox("Select Outcome Type:", [
            "LOGIT (Binary)",
            "CNORM (Continuous/Tobit)",
            "POISSON (Count)",
            "ZIP (Zero-Inflated Poisson)",
        ])
        dist_flag = selected_dist.split(" ")[0]

        cnorm_min = 0.0
        cnorm_max = 0.0
        if dist_flag == "CNORM":
            st.markdown("*CNORM Scale Limits (Optional)*")
            st.info("Leave blank to automatically use the dataset's observed min/max.")
            col_c1, col_c2 = st.columns(2)
            c_min_in = col_c1.text_input("Minimum", value="")
            c_max_in = col_c2.text_input("Maximum", value="")
            if c_min_in.strip() != "": cnorm_min = float(c_min_in)
            if c_max_in.strip() != "": cnorm_max = float(c_max_in)

        st.markdown("**4. Engine Options**")
        use_dropout    = st.checkbox("Include MNAR Dropout Model", value=False)
        default_starts = 3 if app_mode == "AutoTraj Search" else 5
        n_starts = st.number_input(
            "Multi-Start Restarts", min_value=1, max_value=20, value=default_starts,
            help="Number of random starting points per model. More starts reduce local-optima risk."
        )

        if app_mode == "AutoTraj Search":
            st.markdown("**5. Search Grid**")
            group_range = st.slider("Min & Max Groups",           1, 8, (1, 3))
            order_range = st.slider("Min & Max Polynomial Order", 0, 5, (0, 2))

            st.markdown("**6. Heuristic Rules**")
            min_pct = st.slider("Min Group Size (%)", 1.0, 15.0, 5.0, 0.5)
            p_val   = st.number_input("P-Value Threshold", value=0.05, format="%.3f")

        elif app_mode == "Single Model Mode":
            st.markdown("**5. Model Specifications**")
            k_single = st.number_input("Number of Groups", min_value=1, max_value=8, value=2)

            orders_single = []
            cols_ord = st.columns(2)
            for i in range(k_single):
                with cols_ord[i % 2]:
                    o = st.number_input(f"Group {i+1} Order", min_value=0, max_value=5, value=1, key=f"o_{i}")
                    orders_single.append(o)

        zip_iorder = 0  # no longer used; kept for API compatibility

# ── About page ───────────────────────────────────────────────────────────────

if app_mode == "About & Docs":
    st.header("About AutoTraj")
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
    st.markdown("---")
    st.markdown("© 2026 Donald E. Warden, PhD, MPH. Licensed under the MIT License.")

# ── Main app ──────────────────────────────────────────────────────────────────

else:
    st.title(f"GBTM Engine: {app_mode}")

    uploaded_file = st.file_uploader("Upload Dataset (.csv, .txt, .xlsx, .sas7bdat)", type=["csv", "txt", "xlsx", "sas7bdat"])
    st.markdown("*Or, just here to try out the engine? Click below to load sample data (Nagin, 1999).*")

    if st.button("Load Cambridge Sample Data", use_container_width=False):
        st.session_state.use_sample_data = True

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

    if raw_df is not None:

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
            with st.spinner("Executing C-Compiled Math Engine..."):

                if data_format == "Wide Format" or st.session_state.use_sample_data:
                    long_df = prep_trajectory_data(raw_df, id_col, outcome_col, time_col).dropna(subset=['Time', 'Outcome'])
                else:
                    long_df = raw_df.rename(columns={id_col: 'ID', outcome_col: 'Outcome', time_col: 'Time'})
                    long_df = long_df[['ID', 'Time', 'Outcome']].dropna(subset=['Time', 'Outcome'])
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
                        zip_iorder=0, n_starts=n_starts
                    )
                else:
                    single_res = run_single_model(
                        long_df, orders_single, zip_iorder=0,
                        use_dropout=use_dropout, dist=dist_flag,
                        cnorm_min=cnorm_min, cnorm_max=cnorm_max, n_starts=n_starts
                    )
                    top_models   = [single_res] if single_res['result'].success or single_res['result'].status == 2 else []
                    all_evaluated = []

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

            if len(top_models) > 1 and app_mode == "AutoTraj Search":
                st.markdown("#### 🔍 Model Explorer")
                model_choices = [f"Rank {i+1} | {len(m['orders'])}-Group {m['orders']} | BIC: {m['bic']:.2f}" for i, m in enumerate(top_models[:10])]
                selected_model_str = st.selectbox("Select a valid model to visualize:", model_choices, label_visibility="collapsed")
                selected_rank = int(selected_model_str.split("|")[0].replace("Rank ", "").strip()) - 1
                winning_model = top_models[selected_rank]
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

            n_eval      = len(all_evaluated) if all_evaluated else 1
            mps         = n_eval / run_time_val if run_time_val > 0 else 0
            manual_mins = n_eval * 5
            manual_str  = f"~{manual_mins} mins" if manual_mins < 60 else f"~{manual_mins/60:.1f} hrs"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("BIC (Nagin)",    f"{winning_model['bic_nagin']:.2f}")
            col2.metric("BIC (Standard)", f"{winning_model['bic_standard']:.2f}")
            col3.metric("AIC (Nagin)",    f"{winning_model['aic_nagin']:.2f}")
            col4.metric("AIC (Standard)", f"{winning_model['aic_standard']:.2f}")
            st.caption("Nagin convention: higher (less negative) = better fit. Standard convention: lower = better.")
            col5, col6, col7 = st.columns(3)
            col5.metric("Log-Likelihood", f"{winning_model['ll']:.2f}")
            col6.metric("Engine Time",    f"{run_time_val:.2f}s", f"{n_eval} models | {mps:.1f}/sec", delta_color="off")
            col7.metric("Manual Proc Time", manual_str, "vs. SAS Syntax", delta_color="off")

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

            st.divider()
            st.subheader("Publication Suite")

            tab_viz, tab_est, tab_adq, tab_char, tab_comp = st.tabs([
                "Visualization", "Exact Estimates", "Adequacy Metrics",
                "Sample Characteristics", "Model Comparison"
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
                beta_info   = _beta_start_indices(winning_orders)
                k_plot      = len(winning_orders)

                plotly_colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
                mpl_colors_color = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#8338EC', '#FB5607']
                mpl_colors_gray  = ['black', 'dimgray', 'darkgray', 'lightgray', 'slategray', 'silver']

                # ── helper: compute trajectory curve for one group
                def _group_curve(g_idx):
                    beta_start, n_betas = beta_info[g_idx]
                    g_betas = winning_result.x[beta_start:beta_start + n_betas]
                    order   = winning_orders[g_idx]
                    X_smooth = create_design_matrix_jit(smooth_times, order)
                    if dist_type == 'LOGIT':
                        y_hat = calc_logit_prob_jit(g_betas, X_smooth)
                    elif dist_type == 'POISSON':
                        y_hat = np.exp(X_smooth @ g_betas)
                    elif dist_type == 'ZIP':
                        lam     = np.exp(X_smooth @ g_betas)
                        zeta_g  = winning_result.x[len(winning_result.x) - k_plot + g_idx]
                        omega_g = 1.0 / (1.0 + np.exp(-zeta_g))
                        y_hat   = lam * (1.0 - omega_g)
                    else:
                        y_hat = X_smooth @ g_betas
                    return g_betas, order, y_hat, beta_start, n_betas

                with col_viz1:
                    if "Plotly" in viz_style:
                        fig = go.Figure()
                        light_colors = ['rgba(239,85,59,0.15)', 'rgba(0,204,150,0.15)',
                                        'rgba(171,99,250,0.15)', 'rgba(255,161,90,0.15)',
                                        'rgba(25,211,243,0.15)', 'rgba(255,102,146,0.15)']

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
                                    beta_start, n_betas, dist_type
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
                            yaxis_range=y_range_val, template="plotly_white"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        # ── Matplotlib ──────────────────────────────────────────
                        colors = mpl_colors_gray if "Grayscale" in viz_style else mpl_colors_color
                        fig_mpl, ax = plt.subplots(figsize=(8, 5))

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
                                    beta_start, n_betas, dist_type
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
                for g in range(k_plot):
                    beta_start, n_betas = beta_info[g]
                    g_betas = winning_result.x[beta_start:beta_start + n_betas]
                    eq = _build_equation_latex(
                        g_betas, winning_orders[g], dist_type,
                        group_names[g], g, winning_result, winning_orders
                    )
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
                    palette_adq = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
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
                        selected_vars   = st.multiselect("Variables to include:", potential_covariates)
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
                        'Valid':            '#2ca02c',   # green
                        'Failed Convergence': '#aaaaaa', # gray
                    }
                    _DEFAULT_REJECTED = '#ff7f0e'        # orange for all rejection reasons

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
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(color='#1f77b4', size=12,
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

            # ── EXPORT SECTION ────────────────────────────────────────────────

            st.divider()
            st.subheader("Export")

            export_col1, export_col2 = st.columns(2)

            with export_col1:
                st.download_button(
                    label="📥 Download Posterior Probabilities (CSV)",
                    data=assignments_df.to_csv(index=False).encode('utf-8'),
                    file_name='gbtm_trajectory_assignments.csv', mime='text/csv'
                )

            with export_col2:
                # ── Full Results Package (ZIP) ──────────────────────────────
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

        else:
            st.error("Model Failed to Converge or was rejected based on heuristic rules.")
