import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

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

def get_parameter_estimates_for_ui(model_dict, group_names=None):
    orders = model_dict['orders']
    params = model_dict['result'].x
    se_model = model_dict['se_model']
    se_robust = model_dict['se_robust']
    use_dropout = model_dict['use_dropout']
    dof = model_dict['dof']
    model_type = model_dict.get('dist', 'LOGIT')
    
    k = len(orders)
    if group_names is None or len(group_names) != k:
        group_names = [f"Group {g+1}" for g in range(k)]
        
    data = []
    current_beta_idx = k - 1
    current_gamma_idx = (k - 1) + sum([o + 1 for o in orders])
    labels = ["Intercept", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
    gamma_labels = ["Dropout: Intercept", "Dropout: Time", "Dropout: Prev Outcome"]
    
    for g in range(k):
        n_betas = orders[g] + 1
        for b_idx in range(n_betas):
            est = params[current_beta_idx + b_idx]
            err_m = se_model[current_beta_idx + b_idx]
            err_r = se_robust[current_beta_idx + b_idx]
            
            t_stat = est / err_m if err_m > 0 else 0
            p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
            
            data.append({
                "Component": "Trajectory", "Group": str(group_names[g]), "Parameter": labels[b_idx],
                "Estimate": round(est, 5), "Standard Error": round(err_m, 5), "Robust SE": round(err_r, 5),
                "T for H0: Param=0": round(t_stat, 3),
                "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
            })
        current_beta_idx += n_betas
        
        if use_dropout:
            for gam_idx in range(3):
                est = params[current_gamma_idx + gam_idx]
                err_m = se_model[current_gamma_idx + gam_idx]
                err_r = se_robust[current_gamma_idx + gam_idx]
                
                t_stat = est / err_m if err_m > 0 else 0
                p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
                
                data.append({
                    "Component": "Dropout", "Group": str(group_names[g]), "Parameter": gamma_labels[gam_idx],
                    "Estimate": round(est, 5), "Standard Error": round(err_m, 5), "Robust SE": round(err_r, 5),
                    "T for H0: Param=0": round(t_stat, 3),
                    "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
                })
            current_gamma_idx += 3
            
    if model_type == 'CNORM':
        sigma_idx = len(params) - 1
        est = np.exp(params[sigma_idx]) 
        err_m = se_model[sigma_idx] * est 
        err_r = se_robust[sigma_idx] * est
        
        t_stat = est / err_m if err_m > 0 else 0
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
        
        data.append({
            "Component": "Variance", "Group": "All Groups", "Parameter": "Sigma (Standard Deviation)",
            "Estimate": round(est, 5), "Standard Error": round(err_m, 5), "Robust SE": round(err_r, 5),
            "T for H0: Param=0": round(t_stat, 3),
            "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
        })
        
    if model_type == 'ZIP':
        zip_iorder = model_dict.get('zip_iorder', 0)
        tau_start_idx = len(params) - (zip_iorder + 1)
        tau_labels = ["Alpha 0 (Intercept)", "Alpha 1 (Linear)", "Alpha 2 (Quadratic)", "Alpha 3 (Cubic)", "Alpha 4", "Alpha 5"]
        
        for p in range(zip_iorder + 1):
            est = params[tau_start_idx + p]
            err_m = se_model[tau_start_idx + p]
            err_r = se_robust[tau_start_idx + p]
            
            t_stat = est / err_m if err_m > 0 else 0
            p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=dof))
            
            data.append({
                "Component": "Zero Inflation", "Group": "All Groups (Global)", "Parameter": tau_labels[p] if p < len(tau_labels) else f"Alpha {p}",
                "Estimate": round(est, 5), "Standard Error": round(err_m, 5), "Robust SE": round(err_r, 5),
                "T for H0: Param=0": round(t_stat, 3),
                "Prob > |T|": f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
            })
            
    return pd.DataFrame(data)

st.set_page_config(page_title="AutoTraj | GBTM Engine", layout="wide")

if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
    st.session_state.top_models = None
    st.session_state.all_evaluated = None
    st.session_state.run_time = 0
    st.session_state.long_df = None
    st.session_state.raw_df = None
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
            with col_id: id_col = st.text_input("ID", value="ID")
            with col_out: outcome_col = st.text_input("Out. Prefix", value="C")
            with col_time: time_col = st.text_input("Time Prefix", value="T")
        else:
            col_id, col_out, col_time = st.columns(3)
            with col_id: id_col = st.text_input("ID Col", value="ID")
            with col_out: outcome_col = st.text_input("Out. Col", value="Outcome")
            with col_time: time_col = st.text_input("Time Col", value="Time")
            
        st.markdown("**3. Model Distribution**")
        selected_dist = st.selectbox("Select Outcome Type:", ["LOGIT (Binary)", "CNORM (Continuous/Tobit)", "ZIP (Zero-Inflated Poisson)"])
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
        use_dropout = st.checkbox("Include MNAR Dropout Model", value=False)
        default_starts = 3 if app_mode == "AutoTraj Search" else 5
        n_starts = st.number_input(
            "Multi-Start Restarts", min_value=1, max_value=20, value=default_starts,
            help="Number of random starting points tried per model. More starts reduce the risk of local maxima but increase runtime."
        )

        if app_mode == "AutoTraj Search":
            st.markdown("**5. Search Grid**")
            group_range = st.slider("Min & Max Groups", 1, 8, (1, 3))
            order_range = st.slider("Min & Max Polynomial Order", 0, 5, (0, 2))

            zip_iorder = 0
            if dist_flag == "ZIP":
                st.markdown("**Zero-Inflation (Global)**")
                zip_iorder = st.number_input("I-Order (Applies to all groups)", min_value=0, max_value=5, value=0)

            st.markdown("**6. Heuristic Rules**")
            min_pct = st.slider("Min Group Size (%)", 1.0, 15.0, 5.0, 0.5)
            p_val = st.number_input("P-Value Threshold", value=0.05, format="%.3f")

        elif app_mode == "Single Model Mode":
            st.markdown("**5. Model Specifications**")
            k_single = st.number_input("Number of Groups", min_value=1, max_value=8, value=2)

            zip_iorder = 0
            if dist_flag == "ZIP":
                st.markdown("**Zero-Inflation (Global)**")
                zip_iorder = st.number_input("I-Order (Applies to all groups)", min_value=0, max_value=5, value=0)
                st.markdown("**Trajectory Orders ($\mu$)**")

            orders_single = []
            cols_ord = st.columns(2)
            for i in range(k_single):
                with cols_ord[i % 2]:
                    o = st.number_input(f"Group {i+1} Order", min_value=0, max_value=5, value=1, key=f"o_{i}")
                    orders_single.append(o)

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
                    st.error(f"🚨 **Data Mapping Error:** The ID column '{id_col}' was not found. Please update the 'ID' text box in the sidebar. Available columns: {', '.join(raw_df.columns[:5])}...")
                    st.stop()
                if not any(str(c).startswith(outcome_col) for c in raw_df.columns):
                    st.error(f"🚨 **Data Mapping Error:** No columns found starting with Outcome Prefix '{outcome_col}'. Please update the sidebar.")
                    st.stop()
                if not any(str(c).startswith(time_col) for c in raw_df.columns):
                    st.error(f"🚨 **Data Mapping Error:** No columns found starting with Time Prefix '{time_col}'. Please update the sidebar.")
                    st.stop()
            else:
                if id_col not in raw_df.columns or outcome_col not in raw_df.columns or time_col not in raw_df.columns:
                    st.error(f"🚨 **Data Mapping Error:** One or more specified columns ({id_col}, {outcome_col}, {time_col}) were not found. Please check your sidebar inputs.")
                    st.stop()

            start_time = time.time()
            with st.spinner("Executing C-Compiled Math Engine..."):
                
                if data_format == "Wide Format" or st.session_state.use_sample_data:
                    long_df = prep_trajectory_data(raw_df, id_col, outcome_col, time_col).dropna(subset=['Time', 'Outcome'])
                else:
                    long_df = raw_df.rename(columns={id_col: 'ID', outcome_col: 'Outcome', time_col: 'Time'})
                    long_df = long_df[['ID', 'Time', 'Outcome']].dropna(subset=['Time', 'Outcome'])
                    long_df['Time'] = pd.to_numeric(long_df['Time'])
                    long_df['Outcome'] = pd.to_numeric(long_df['Outcome'])
                    long_df = long_df.sort_values(by=['ID', 'Time'])
                
                # ------------------------------------------------------------------
                # INPUT VALIDATION
                # ------------------------------------------------------------------

                # Check 4: remove subjects with fewer than 2 observations
                obs_counts = long_df.groupby('ID').size()
                single_obs_ids = obs_counts[obs_counts < 2].index.tolist()
                if single_obs_ids:
                    long_df = long_df[~long_df['ID'].isin(single_obs_ids)].copy()
                    preview = single_obs_ids[:5]
                    extra   = f" … and {len(single_obs_ids) - 5} more" if len(single_obs_ids) > 5 else ""
                    st.info(f"Removed {len(single_obs_ids)} subject(s) with only 1 observation (IDs: {preview}{extra}). These subjects cannot contribute to trajectory estimation.")

                # Check 3: minimum sample size
                n_subjects_val = long_df['ID'].nunique()
                if n_subjects_val < 30:
                    st.warning(f"⚠️ Only {n_subjects_val} subjects remain after filtering. Results may be unreliable with very small samples (recommended n ≥ 30).")

                # Check 1: LOGIT requires binary outcomes
                if dist_flag == 'LOGIT':
                    invalid_mask = ~long_df['Outcome'].isin([0.0, 1.0])
                    if invalid_mask.any():
                        bad_vals = sorted(long_df.loc[invalid_mask, 'Outcome'].unique().tolist())[:10]
                        n_bad    = int(invalid_mask.sum())
                        st.error(
                            f"🚨 **LOGIT requires binary outcomes (0 or 1).** "
                            f"Found {n_bad} row(s) with non-binary values: {bad_vals}. "
                            f"Please select CNORM or ZIP, or recode your outcome as 0/1."
                        )
                        st.stop()

                # Check 2: CNORM requires numeric outcomes; warn if all-integer
                if dist_flag == 'CNORM':
                    if not pd.api.types.is_numeric_dtype(long_df['Outcome']):
                        st.error("🚨 **CNORM requires a numeric outcome.** The Outcome column contains non-numeric values. Please check your data mapping.")
                        st.stop()
                    elif long_df['Outcome'].dropna().apply(lambda x: float(x) == int(x)).all():
                        st.warning("⚠️ All Outcome values appear to be whole numbers. If your outcome is binary (0/1), consider using LOGIT instead of CNORM.")

                # Check 5: polynomial order vs. unique time points — hard stop
                n_timepoints = len(long_df['Time'].unique())
                max_order_attempted = max(orders_single) if app_mode == "Single Model Mode" else order_range[1]
                if max_order_attempted >= n_timepoints:
                    st.error(
                        f"🚨 **Unidentifiable Model:** Polynomial order {max_order_attempted} "
                        f"requires {max_order_attempted + 1} parameters per group but the data has only "
                        f"{n_timepoints} unique time point(s), leaving 0 degrees of freedom. "
                        f"Please reduce the maximum polynomial order to at most {n_timepoints - 1}."
                    )
                    st.stop()

                # ------------------------------------------------------------------

                if app_mode == "AutoTraj Search":
                    top_models, all_evaluated = run_autotraj(
                        long_df, min_groups=group_range[0], max_groups=group_range[1],
                        min_order=order_range[0], max_order=order_range[1],
                        min_group_pct=min_pct, p_val_thresh=p_val, use_dropout=use_dropout,
                        dist=dist_flag, cnorm_min=cnorm_min, cnorm_max=cnorm_max, zip_iorder=zip_iorder, n_starts=n_starts
                    )
                else:
                    single_res = run_single_model(long_df, orders_single, zip_iorder=zip_iorder, use_dropout=use_dropout, dist=dist_flag, cnorm_min=cnorm_min, cnorm_max=cnorm_max, n_starts=n_starts)
                    top_models = [single_res] if single_res['result'].success or single_res['result'].status == 2 else []
                    all_evaluated = []
            
            st.session_state.run_complete = True
            st.session_state.top_models = top_models
            st.session_state.all_evaluated = all_evaluated
            st.session_state.run_time = time.time() - start_time
            st.session_state.long_df = long_df
            st.session_state.raw_df = raw_df
            st.session_state.use_dropout = use_dropout

    if st.session_state.run_complete:
        top_models = st.session_state.top_models
        all_evaluated = st.session_state.all_evaluated
        long_df = st.session_state.long_df
        raw_df = st.session_state.raw_df
        use_dropout_state = st.session_state.use_dropout
        run_time_val = st.session_state.run_time
        
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
                
            winning_orders = winning_model['orders']
            winning_result = winning_model['result']
            winning_pis_raw = winning_model['pis'] 
            dist_type = winning_model.get('dist', 'LOGIT')
            
            if winning_model.get('cond_num', 0) > 1e10 or np.any(winning_model['se_model'] < 1e-3) or np.any(winning_model['se_model'] > 50):
                st.warning("⚠️ **Warning: Unidentifiable Model Detected.** The information matrix is singular or standard errors are degenerate. The model has been overparameterized, making estimates and p-values unreliable. Consider reducing the number of groups.")
            
            n_eval = len(all_evaluated) if all_evaluated else 1
            mps = n_eval / run_time_val if run_time_val > 0 else 0
            
            manual_mins = n_eval * 5
            manual_str = f"~{manual_mins} mins" if manual_mins < 60 else f"~{manual_mins/60:.1f} hrs"
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("BIC (Nagin)", f"{winning_model['bic_nagin']:.2f}")
            col2.metric("BIC (Standard)", f"{winning_model['bic_standard']:.2f}")
            col3.metric("AIC (Nagin)", f"{winning_model['aic_nagin']:.2f}")
            col4.metric("AIC (Standard)", f"{winning_model['aic_standard']:.2f}")
            st.caption("Nagin convention: higher (less negative) = better fit. Standard convention: lower = better.")
            col5, col6, col7 = st.columns(3)
            col5.metric("Log-Likelihood", f"{winning_model['ll']:.2f}")
            col6.metric("Engine Time", f"{run_time_val:.2f}s", f"{n_eval} models | {mps:.1f}/sec", delta_color="off")
            col7.metric("Manual Proc Time", manual_str, "vs. SAS Syntax", delta_color="off")
            
            st.markdown("##### ✏️ Customize Plot Labels & Group Names")
            col_lbl1, col_lbl2 = st.columns(2)
            x_axis_label = col_lbl1.text_input("X-Axis Label", value="Time Period")
            
            if dist_type == 'LOGIT':
                default_y_label = "Probability of Outcome"
            elif dist_type == 'ZIP':
                default_y_label = "Expected Count"
            else:
                default_y_label = "Outcome Score"
                
            y_axis_label = col_lbl2.text_input("Y-Axis Label", value=default_y_label)
            
            cols = st.columns(len(winning_orders))
            group_names = []
            for g in range(len(winning_orders)):
                name = cols[g].text_input(f"Group {g+1} Label", value=f"Group {g+1}")
                group_names.append(name)
                
            assignments_df = get_subject_assignments(winning_model, long_df)
            
            st.divider()
            st.subheader("Publication Suite")
            
            tab_viz, tab_est, tab_adq, tab_char, tab_comp = st.tabs([
                "Visualization", 
                "Exact Estimates", 
                "Adequacy Metrics", 
                "Sample Characteristics",
                "Model Comparison"
            ])
            
            with tab_viz:
                col_viz1, col_viz2 = st.columns([3, 1])
                with col_viz2:
                    viz_style = st.selectbox("Graphic Style:", ["Interactive Web (Plotly)", "Publication: Grayscale (Matplotlib)", "Publication: Color (Matplotlib)"])
                    st.markdown("**Plot Elements:**")
                    show_spaghetti = st.checkbox("Individual Trajectories", value=False)
                    show_smooth = st.checkbox("Estimated Curves (Smoothed)", value=True)
                    show_obs = st.checkbox("Observed Averages", value=True)
                
                actual_times = long_df['Time'].values
                smooth_times = np.linspace(min(actual_times), max(actual_times), 100)
                current_idx = len(winning_orders) - 1
                
                merged_for_plot = pd.merge(long_df, assignments_df[['ID', 'Assigned_Group']], on='ID')
                obs_means = merged_for_plot.groupby(['Assigned_Group', 'Time'])['Outcome'].mean().reset_index()
                
                with col_viz1:
                    if "Plotly" in viz_style:
                        fig = go.Figure()
                        colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
                        
                        if show_spaghetti:
                            sample_ids = long_df['ID'].drop_duplicates().sample(n=min(100, len(long_df['ID'].unique())), random_state=42)
                            for s_id in sample_ids:
                                sub_df = long_df[long_df['ID'] == s_id]
                                fig.add_trace(go.Scatter(x=sub_df['Time'], y=sub_df['Outcome'], mode='lines', opacity=0.08, line=dict(color='gray'), hoverinfo='skip', showlegend=False))
                        
                        for g in range(len(winning_orders)):
                            n_betas = winning_orders[g] + 1
                            g_betas = winning_result.x[current_idx : current_idx + n_betas]
                            current_idx += n_betas
                            
                            if show_smooth:
                                X_smooth = create_design_matrix_jit(smooth_times, winning_orders[g])
                                if dist_type == 'LOGIT':
                                    g_probs = calc_logit_prob_jit(g_betas, X_smooth)
                                elif dist_type == 'ZIP':
                                    lam = np.exp(X_smooth @ g_betas) 
                                    zip_i = winning_model.get('zip_iorder', 0)
                                    alphas = winning_result.x[-(zip_i + 1):]
                                    tau = create_design_matrix_jit(smooth_times, zip_i) @ alphas
                                    rho = 1.0 / (1.0 + np.exp(-tau))
                                    g_probs = lam * (1.0 - rho) # Expected count
                                else:
                                    g_probs = X_smooth @ g_betas 
                                fig.add_trace(go.Scatter(x=smooth_times, y=g_probs, mode='lines', line=dict(color=colors[g%len(colors)], width=4, dash='dot' if show_obs else 'solid'), name=f'{group_names[g]} (Est.)'))
                            
                            if show_obs:
                                g_obs = obs_means[obs_means['Assigned_Group'] == g+1]
                                fig.add_trace(go.Scatter(x=g_obs['Time'], y=g_obs['Outcome'], mode='lines+markers+text', text=[f"{g+1}"]*len(g_obs), textposition="top center", line=dict(color=colors[g%len(colors)], width=2), name=f'{group_names[g]} (Obs.)'))
                        
                        y_range_val = [-0.1, 1.1] if dist_type == 'LOGIT' else None
                        fig.update_layout(yaxis_title=y_axis_label, xaxis_title=x_axis_label, yaxis_range=y_range_val, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = ['black', 'dimgray', 'darkgray'] if "Grayscale" in viz_style else ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
                        
                        if show_spaghetti:
                            sample_ids = long_df['ID'].drop_duplicates().sample(n=min(100, len(long_df['ID'].unique())), random_state=42)
                            for s_id in sample_ids:
                                sub_df = long_df[long_df['ID'] == s_id]
                                ax.plot(sub_df['Time'], sub_df['Outcome'], color='gray', alpha=0.08, linewidth=1)
                        
                        for g in range(len(winning_orders)):
                            n_betas = winning_orders[g] + 1
                            g_betas = winning_result.x[current_idx : current_idx + n_betas]
                            current_idx += n_betas
                            
                            if show_smooth:
                                X_smooth = create_design_matrix_jit(smooth_times, winning_orders[g])
                                if dist_type == 'LOGIT':
                                    g_probs = calc_logit_prob_jit(g_betas, X_smooth)
                                elif dist_type == 'ZIP':
                                    lam = np.exp(X_smooth @ g_betas)
                                    zip_i = winning_model.get('zip_iorder', 0)
                                    alphas = winning_result.x[-(zip_i + 1):]
                                    tau = create_design_matrix_jit(smooth_times, zip_i) @ alphas
                                    rho = 1.0 / (1.0 + np.exp(-tau))
                                    g_probs = lam * (1.0 - rho)
                                else:
                                    g_probs = X_smooth @ g_betas
                                ax.plot(smooth_times, g_probs, linewidth=2.5 if not show_obs else 1.5, color=colors[g%len(colors)], linestyle='--' if show_obs else '-', label=f'{group_names[g]} (Est.)')
                            
                            if show_obs:
                                g_obs = obs_means[obs_means['Assigned_Group'] == g+1]
                                ax.plot(g_obs['Time'], g_obs['Outcome'], color=colors[g%len(colors)], marker='o', linewidth=2, label=f'{group_names[g]} (Obs.)')
                                for _, row in g_obs.iterrows():
                                    ax.text(row['Time'], row['Outcome'] + 0.02, str(g+1), color=colors[g%len(colors)], ha='center')
                                    
                        if dist_type == 'LOGIT': ax.set_ylim(-0.1, 1.1)
                        ax.set_ylabel(y_axis_label)
                        ax.set_xlabel(x_axis_label)
                        ax.legend(frameon=False)
                        st.pyplot(fig)
                
                st.download_button(label="📥 Download Observed Averages (CSV)", data=obs_means.to_csv(index=False).encode('utf-8'), file_name='trajectory_observed_averages.csv', mime='text/csv')
                    
            with tab_est:
                estimates_df = get_parameter_estimates_for_ui(winning_model, group_names)
                st.dataframe(estimates_df, use_container_width=True, hide_index=True)
                csv_est = estimates_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Parameter Estimates Table", data=csv_est, file_name='trajectory_parameters.csv', mime='text/csv')
                
            with tab_adq:
                adq_df, rel_entropy = calc_model_adequacy(assignments_df, winning_pis_raw, group_names)
                st.metric(label="Relative Entropy (0-1)", value=f"{rel_entropy:.3f}", help="Measures how well the model separates the subpopulations. Values closer to 1 indicate better separation.")
                st.dataframe(adq_df, use_container_width=True, hide_index=True)

            with tab_char:
                if HAS_TABLEONE:
                    if data_format == "Wide Format" or st.session_state.use_sample_data:
                        potential_covariates = [col for col in raw_df.columns.tolist() if not col.startswith((outcome_col, time_col))]
                        selected_vars = st.multiselect("Variables to include:", potential_covariates)
                        categorical_vars = st.multiselect("Which of these are categorical?", selected_vars)
                        if selected_vars and st.button("Generate Table 1"):
                            merged_df = pd.merge(raw_df, assignments_df[['ID', 'Assigned_Group']], left_on=id_col, right_on='ID')
                            group_map = {i+1: name for i, name in enumerate(group_names)}
                            merged_df['Assigned_Group'] = merged_df['Assigned_Group'].map(group_map)
                            mytable = TableOne(merged_df, columns=selected_vars, categorical=categorical_vars, groupby="Assigned_Group", pval=True)
                            st.markdown(mytable.to_html(), unsafe_allow_html=True)
                    else:
                        st.info("Baseline characteristics table generation requires wide-format data to prevent duplicating subjects across timepoints. Please join the exported posterior assignments CSV to your baseline demographics dataset.")
                else: st.warning("Please run `pip install tableone` in your terminal to enable this feature.")

            with tab_comp:
                if app_mode == "AutoTraj Search" and all_evaluated:
                    best_per_k = {}
                    for m in all_evaluated:
                        if m['Status'] != "Failed Convergence" and not np.isnan(m['BIC (Nagin)']):
                            k = m['Groups']
                            if k not in best_per_k or m['BIC (Nagin)'] > best_per_k[k]['BIC (Nagin)']:
                                best_per_k[k] = m

                    ks = sorted(list(best_per_k.keys()))
                    bics = [best_per_k[k]['BIC (Nagin)'] for k in ks]
                    
                    fig_bic = go.Figure()
                    fig_bic.add_trace(go.Scatter(x=ks, y=bics, mode='lines+markers', marker=dict(size=10, color='#1f77b4'), line=dict(width=3)))
                    fig_bic.update_layout(
                        xaxis_title="Number of Groups", 
                        yaxis_title="BIC (Lower on graph = Closer to 0)", 
                        xaxis=dict(tickmode='linear', tick0=1, dtick=1), 
                        yaxis=dict(autorange="reversed"), 
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_bic, use_container_width=True)
                    
                    comp_df = pd.DataFrame(all_evaluated)
                    for col in ['BIC (Nagin)', 'BIC (Standard)', 'AIC (Nagin)', 'AIC (Standard)']:
                        comp_df[col] = comp_df[col].apply(lambda x: round(x, 2) if pd.notnull(x) else "NaN")
                    comp_df['Min_Group_%'] = comp_df['Min_Group_%'].apply(lambda x: round(x, 1) if pd.notnull(x) else "NaN")
                    st.dataframe(comp_df, hide_index=True)

            st.divider()
            st.subheader("Export Core Data")
            st.download_button(label="📥 Download Posterior Probabilities (CSV)", data=assignments_df.to_csv(index=False).encode('utf-8'), file_name='gbtm_trajectory_assignments.csv', mime='text/csv')
            
        else:
            st.error("Model Failed to Converge or was rejected based on heuristic rules.")