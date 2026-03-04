import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, norm

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
    get_subject_assignments
)

def get_parameter_estimates_for_ui(model_dict, group_names=None):
    orders = model_dict['orders']
    params = model_dict['result'].x
    se_model = model_dict['se_model']
    se_robust = model_dict['se_robust']
    use_dropout = model_dict['use_dropout']
    dof = model_dict['dof']
    
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
        
        st.markdown("**3. Engine Options**")
        use_dropout = st.checkbox("Include MNAR Dropout Model", value=False)
        
        if app_mode == "AutoTraj Search":
            st.markdown("**4. Search Grid**")
            group_range = st.slider("Min & Max Groups", 1, 8, (1, 3))
            order_range = st.slider("Min & Max Polynomial Order", 0, 5, (0, 2))
            
            st.markdown("**5. Heuristic Rules**")
            min_pct = st.slider("Min Group Size (%)", 1.0, 15.0, 5.0, 0.5)
            p_val = st.number_input("P-Value Threshold", value=0.05, format="%.3f")
            
        elif app_mode == "Single Model Mode":
            st.markdown("**4. Model Specifications**")
            k_single = st.number_input("Number of Groups", min_value=1, max_value=8, value=2)
            orders_single = []
            cols_ord = st.columns(2)
            for i in range(k_single):
                with cols_ord[i % 2]:
                    o = st.number_input(f"Group {i+1} Order", min_value=0, max_value=5, value=1)
                    orders_single.append(o)

if app_mode == "About & Docs":
    st.header("About AutoTraj")
    st.markdown(r"""
    **Overview**
    AutoTraj is a high-performance engine for Group-Based Trajectory Modeling (GBTM), a specialized application of finite mixture modeling utilized to identify latent subpopulations following distinct developmental trajectories over time.
    """)
    st.markdown("---")
    st.markdown("© 2026 Donald E. Warden, PhD, MPH. Licensed under the MIT License.")

else:
    st.title(f"GBTM Engine: {app_mode}")
    
    uploaded_file = st.file_uploader("Upload Dataset (.csv or .txt)", type=["csv", "txt"])
    st.markdown("*Or, just here to try out the engine? Click below to load sample data (Nagin, 1999).*")
    
    if st.button("Load Cambridge Sample Data", use_container_width=False):
        st.session_state.use_sample_data = True
        
    if uploaded_file is not None:
        st.session_state.use_sample_data = False 

    raw_df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            else:
                raw_df = pd.read_csv(uploaded_file, sep=r'\s+', encoding='utf-8-sig')
            raw_df.columns = [str(c).strip() for c in raw_df.columns]
            st.success("Custom file uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif st.session_state.use_sample_data:
        try:
            raw_df = pd.read_csv("cambridge.txt", sep=r'\s+', encoding='utf-8-sig')
            raw_df.columns = [str(c).strip() for c in raw_df.columns]
            st.success("Cambridge sample dataset loaded! (Note: Sample data is in Wide format)")
        except Exception as e:
            st.error("Could not locate cambridge.txt in the repository.")

    if raw_df is not None:
        button_label = "Run AutoTraj Search" if app_mode == "AutoTraj Search" else "Run Single Model"
        
        if st.button(button_label, type="primary", use_container_width=True):
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
                
                if app_mode == "AutoTraj Search":
                    top_models, all_evaluated = run_autotraj(
                        long_df, min_groups=group_range[0], max_groups=group_range[1],
                        min_order=order_range[0], max_order=order_range[1],
                        min_group_pct=min_pct, p_val_thresh=p_val, use_dropout=use_dropout
                    )
                else:
                    single_res = run_single_model(long_df, orders_single, use_dropout=use_dropout)
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
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("BIC (N=Subj)", f"{winning_model['bic']:.2f}")
            col2.metric("BIC (N=Obs)", f"{winning_model['bic_obs']:.2f}")
            col3.metric("AIC", f"{winning_model['aic']:.2f}")
            col4.metric("Log-Likelihood", f"{winning_model['ll']:.2f}")
            col5.metric("Engine Time", f"{st.session_state.run_time:.2f}s")
            
            st.markdown("##### ✏️ Customize Group Labels")
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
                                g_probs = calc_logit_prob_jit(g_betas, X_smooth)
                                fig.add_trace(go.Scatter(x=smooth_times, y=g_probs, mode='lines', line=dict(color=colors[g%len(colors)], width=4, dash='dot' if show_obs else 'solid'), name=f'{group_names[g]} (Est.)'))
                            
                            if show_obs:
                                g_obs = obs_means[obs_means['Assigned_Group'] == g+1]
                                fig.add_trace(go.Scatter(x=g_obs['Time'], y=g_obs['Outcome'], mode='lines+markers+text', text=[f"{g+1}"]*len(g_obs), textposition="top center", line=dict(color=colors[g%len(colors)], width=2), name=f'{group_names[g]} (Obs.)'))
                                
                        fig.update_layout(yaxis_title="Probability", xaxis_title="Time", yaxis_range=[-0.1, 1.1], template="plotly_white")
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
                                g_probs = calc_logit_prob_jit(g_betas, X_smooth)
                                ax.plot(smooth_times, g_probs, linewidth=2.5 if not show_obs else 1.5, color=colors[g%len(colors)], linestyle='--' if show_obs else '-', label=f'{group_names[g]} (Est.)')
                            
                            if show_obs:
                                g_obs = obs_means[obs_means['Assigned_Group'] == g+1]
                                ax.plot(g_obs['Time'], g_obs['Outcome'], color=colors[g%len(colors)], marker='o', linewidth=2, label=f'{group_names[g]} (Obs.)')
                                for _, row in g_obs.iterrows():
                                    ax.text(row['Time'], row['Outcome'] + 0.02, str(g+1), color=colors[g%len(colors)], ha='center')
                                    
                        ax.set_ylim(-0.1, 1.1)
                        ax.set_ylabel("Probability of Outcome")
                        ax.set_xlabel("Time Period")
                        ax.legend(frameon=False)
                        st.pyplot(fig)
                
                st.download_button(label="📥 Download Observed Averages (CSV)", data=obs_means.to_csv(index=False).encode('utf-8'), file_name='trajectory_observed_averages.csv', mime='text/csv')
                    
            with tab_est:
                estimates_df = get_parameter_estimates_for_ui(winning_model, group_names)
                st.dataframe(estimates_df, use_container_width=True, hide_index=True)
                csv_est = estimates_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download Parameter Estimates Table", data=csv_est, file_name='trajectory_parameters.csv', mime='text/csv')
                
            with tab_adq:
                st.dataframe(calc_model_adequacy(assignments_df, winning_pis_raw, group_names), use_container_width=True, hide_index=True)

            with tab_comp:
                if app_mode == "AutoTraj Search" and all_evaluated:
                    st.markdown("##### 📈 BIC Curve (Optimal Model per Group Size)")
                    
                    best_per_k = {}
                    for m in all_evaluated:
                        if m['Status'] != "Failed Convergence" and not np.isnan(m['BIC']):
                            k = m['Groups']
                            if k not in best_per_k or m['BIC'] > best_per_k[k]['BIC']:
                                best_per_k[k] = m
                    
                    ks = sorted(list(best_per_k.keys()))
                    bics = [best_per_k[k]['BIC'] for k in ks]
                    
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
                    
                    st.markdown("##### Full Exhaustive Search Results")
                    comp_df = pd.DataFrame(all_evaluated)
                    comp_df['BIC'] = comp_df['BIC'].apply(lambda x: round(x, 2) if pd.notnull(x) else "NaN")
                    comp_df['AIC'] = comp_df['AIC'].apply(lambda x: round(x, 2) if pd.notnull(x) else "NaN")
                    comp_df['Min_Group_%'] = comp_df['Min_Group_%'].apply(lambda x: round(x, 1) if pd.notnull(x) else "NaN")
                    st.dataframe(comp_df, hide_index=True)

            st.divider()
            st.subheader("Export Core Data")
            st.download_button(label="📥 Download Posterior Probabilities (CSV)", data=assignments_df.to_csv(index=False).encode('utf-8'), file_name='gbtm_trajectory_assignments.csv', mime='text/csv')
            
        else:
            st.error("Model Failed to Converge or was rejected based on heuristic rules.")