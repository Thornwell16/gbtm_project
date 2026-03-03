import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
    get_parameter_estimates,
    calc_model_adequacy
)

st.set_page_config(page_title="AutoTraj | GBTM Engine", layout="wide")

if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
    st.session_state.top_models = None
    st.session_state.run_time = 0
    st.session_state.long_df = None
    st.session_state.raw_df = None

# ==========================================
# SIDEBAR NAVIGATION & SETTINGS
# ==========================================
with st.sidebar:
    st.title("AutoTraj")
    app_mode = st.radio("Navigation", ["AutoTraj Search", "Single Model Mode", "About & Docs"])
    
    st.markdown("---")
    
    if app_mode != "About & Docs":
        st.markdown("**1. Data Mapping**")
        id_col = st.text_input("ID Column", value="ID")
        outcome_col = st.text_input("Outcome Prefix", value="C")
        time_col = st.text_input("Time Prefix", value="T")
        
        if app_mode == "AutoTraj Search":
            st.markdown("<br>**2. Search Grid**", unsafe_allow_html=True)
            group_range = st.slider("Min & Max Groups", 1, 8, (1, 3))
            order_range = st.slider("Min & Max Polynomial Order", 0, 5, (0, 2))
            
            st.markdown("<br>**3. Heuristic Rules**", unsafe_allow_html=True)
            min_pct = st.slider("Minimum Group Size (%)", 1.0, 15.0, 5.0, 0.5)
            p_val = st.number_input("P-Value Threshold", value=0.05, format="%.3f")
            
        elif app_mode == "Single Model Mode":
            st.markdown("<br>**2. Model Specifications**", unsafe_allow_html=True)
            k_single = st.number_input("Number of Groups", min_value=1, max_value=8, value=2)
            orders_single = []
            for i in range(k_single):
                o = st.number_input(f"Group {i+1} Polynomial Order", min_value=0, max_value=5, value=1)
                orders_single.append(o)

# ==========================================
# PAGE ROUTING
# ==========================================
if app_mode == "About & Docs":
    st.header("About AutoTraj")
    st.markdown("""
    **Overview**
    AutoTraj automates the exhaustive search, selection, and visualization of finite mixture models for longitudinal data. It utilizes a fully vectorized, C-compiled analytical Jacobian engine to rapidly evaluate combinatorial polynomial grids.
    
    **Methodology & Missing Data**
    Unlike standard FIML engines, AutoTraj incorporates a true **Missing Not At Random (MNAR)** dropout model. The likelihood function simultaneously evaluates the probability of the outcome and the probability of subject attrition, conditional on their previous health state:
    
    `P(Dropout) = Logit(γ0 + γ1*Time + γ2*Previous_Outcome)`
    
    **Selection Criteria**
    * **Group Size Penalty:** Discards any model containing a spurious trajectory group smaller than the user-defined threshold.
    * **Polynomial Significance:** Adheres strictly to Nagin & Jones (2005); models lacking significance in their highest-order polynomial term are rejected.
    * **Optimization:** Determines the global optimal model using the Bayesian Information Criterion (BIC).
    """)
    st.markdown("---")
    st.markdown("© 2026 Dr. Don Warden (Westat / Texas A&M University-Corpus Christi). Licensed under the MIT License.")

else:
    st.title(f"GBTM Engine: {app_mode}")
    uploaded_file = st.file_uploader("Upload Wide-Format Dataset (.csv or .txt)", type=["csv", "txt"])

    if uploaded_file is not None:
        try: raw_df = pd.read_csv(uploaded_file, sep=r'\s+')
        except Exception: raw_df = pd.read_csv(uploaded_file)
            
        st.success("File uploaded successfully!")
        
        button_label = "Run AutoTraj Search" if app_mode == "AutoTraj Search" else "Run Single Model"
        
        if st.button(button_label, type="primary", use_container_width=True):
            start_time = time.time()
            with st.spinner("Executing C-Compiled Math Engine..."):
                long_df = prep_trajectory_data(raw_df, id_col, outcome_col, time_col).dropna(subset=['Time', 'Outcome'])
                
                if app_mode == "AutoTraj Search":
                    top_models = run_autotraj(
                        long_df, min_groups=group_range[0], max_groups=group_range[1],
                        min_order=order_range[0], max_order=order_range[1],
                        min_group_pct=min_pct, p_val_thresh=p_val
                    )
                else:
                    single_res = run_single_model(long_df, orders_single)
                    top_models = [single_res] if single_res['result'].success or single_res['result'].status == 2 else []
            
            st.session_state.run_complete = True
            st.session_state.top_models = top_models
            st.session_state.run_time = time.time() - start_time
            st.session_state.long_df = long_df
            st.session_state.raw_df = raw_df

    if st.session_state.run_complete:
        top_models = st.session_state.top_models
        long_df = st.session_state.long_df
        raw_df = st.session_state.raw_df
        
        if top_models:
            st.divider()
            
            # --- MODEL EXPLORER DROPDOWN ---
            if len(top_models) > 1 and app_mode == "AutoTraj Search":
                st.markdown("#### 🔍 Model Explorer")
                model_choices = [f"Rank {i+1} | {len(m['orders'])}-Group {m['orders']} | BIC: {m['bic']:.2f}" for i, m in enumerate(top_models)]
                selected_model_str = st.selectbox("Select a model to visualize:", model_choices)
                selected_rank = int(selected_model_str.split("|")[0].replace("Rank ", "").strip()) - 1
                winning_model = top_models[selected_rank]
            else:
                winning_model = top_models[0]
                st.subheader("🏆 Model Results")
                
            winning_orders = winning_model['orders']
            winning_result = winning_model['result']
            winning_pis_raw = winning_model['pis'] 
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Groups", len(winning_orders))
            col2.metric("Polynomial Orders", str(winning_orders))
            col3.metric("Engine Time", f"{st.session_state.run_time:.2f} sec")
            col4.metric("BIC (Subject-Level)", f"{winning_model['bic']:.2f}")
            
            # --- CUSTOM GROUP LABELS ---
            st.markdown("##### ✏️ Customize Group Labels")
            cols = st.columns(len(winning_orders))
            group_names = []
            for g in range(len(winning_orders)):
                name = cols[g].text_input(f"Group {g+1} Label", value=f"Group {g+1}")
                group_names.append(name)
                
            assignments_df = get_subject_assignments(winning_result, long_df, winning_orders)
            
            st.divider()
            st.subheader("Publication Suite")
            tab1, tab2, tab3, tab4 = st.tabs(["Visualization", "Exact Estimates", "Adequacy Metrics", "Demographics (Table 1)"])
            
            with tab1:
                col_viz1, col_viz2 = st.columns([3, 1])
                with col_viz2:
                    viz_style = st.selectbox("Graphic Style:", ["Interactive Web (Plotly)", "Publication: Grayscale (Matplotlib)", "Publication: Color (Matplotlib)"])
                    show_spaghetti = st.checkbox("Overlay Individual Trajectories", value=False)
                
                actual_times = long_df['Time'].values
                smooth_times = np.linspace(min(actual_times), max(actual_times), 100)
                current_idx = len(winning_orders) - 1
                
                with col_viz1:
                    if "Plotly" in viz_style:
                        fig = go.Figure()
                        colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
                        if show_spaghetti:
                            sample_ids = long_df['ID'].drop_duplicates().sample(n=min(50, len(long_df['ID'].unique())), random_state=42)
                            for s_id in sample_ids:
                                sub_df = long_df[long_df['ID'] == s_id]
                                fig.add_trace(go.Scatter(x=sub_df['Time'], y=sub_df['Outcome'], mode='lines', opacity=0.08, line=dict(color='gray'), hoverinfo='skip', showlegend=False))
                        else:
                            jittered_y = long_df['Outcome'].values + np.random.normal(0, 0.02, size=len(long_df))
                            fig.add_trace(go.Scatter(x=actual_times, y=jittered_y, mode='markers', marker=dict(color='gray', opacity=0.1), name='Observations', hoverinfo='skip'))
                        
                        for g in range(len(winning_orders)):
                            n_betas = winning_orders[g] + 1
                            g_betas = winning_result.x[current_idx : current_idx + n_betas]
                            current_idx += n_betas
                            X_smooth = create_design_matrix_jit(smooth_times, winning_orders[g])
                            g_probs = calc_logit_prob_jit(g_betas, X_smooth)
                            fig.add_trace(go.Scatter(x=smooth_times, y=g_probs, mode='lines', line=dict(color=colors[g%len(colors)], width=4), name=f'{group_names[g]} ({winning_pis_raw[g]*100:.1f}%)'))
                        fig.update_layout(yaxis_title="Probability", xaxis_title="Time", yaxis_range=[-0.1, 1.1], template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = ['black', 'dimgray', 'darkgray'] if "Grayscale" in viz_style else ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
                        linestyles = ['-', '--', '-.', ':']
                        if show_spaghetti:
                            sample_ids = long_df['ID'].drop_duplicates().sample(n=min(50, len(long_df['ID'].unique())), random_state=42)
                            for s_id in sample_ids:
                                sub_df = long_df[long_df['ID'] == s_id]
                                ax.plot(sub_df['Time'], sub_df['Outcome'], color='gray', alpha=0.08, linewidth=1)
                        
                        for g in range(len(winning_orders)):
                            n_betas = winning_orders[g] + 1
                            g_betas = winning_result.x[current_idx : current_idx + n_betas]
                            current_idx += n_betas
                            X_smooth = create_design_matrix_jit(smooth_times, winning_orders[g])
                            g_probs = calc_logit_prob_jit(g_betas, X_smooth)
                            ax.plot(smooth_times, g_probs, linewidth=2.5, color=colors[g%len(colors)], linestyle=linestyles[g%len(linestyles)], label=f'{group_names[g]} ({winning_pis_raw[g]*100:.1f}%)')
                        ax.set_ylim(-0.1, 1.1)
                        ax.set_ylabel("Probability of Outcome")
                        ax.set_xlabel("Time Period")
                        ax.legend(frameon=False)
                        st.pyplot(fig)
                    
            with tab2:
                st.dataframe(get_parameter_estimates(winning_result, winning_orders, group_names), use_container_width=True, hide_index=True)
                
            with tab3:
                st.dataframe(calc_model_adequacy(assignments_df, winning_pis_raw, group_names), use_container_width=True, hide_index=True)

            with tab4:
                if HAS_TABLEONE:
                    potential_covariates = [col for col in raw_df.columns.tolist() if not col.startswith((outcome_col, time_col))]
                    selected_vars = st.multiselect("Variables to include:", potential_covariates)
                    categorical_vars = st.multiselect("Which of these are categorical?", selected_vars)
                    if selected_vars and st.button("Generate Table 1"):
                        merged_df = pd.merge(raw_df, assignments_df[['ID', 'Assigned_Group']], left_on=id_col, right_on='ID')
                        group_map = {i+1: name for i, name in enumerate(group_names)}
                        merged_df['Assigned_Group'] = merged_df['Assigned_Group'].map(group_map)
                        
                        mytable = TableOne(merged_df, columns=selected_vars, categorical=categorical_vars, groupby="Assigned_Group", pval=True)
                        st.markdown(mytable.to_html(), unsafe_allow_html=True)
                else: st.warning("Please run `pip install tableone` in your terminal to enable this feature.")

            st.divider()
            st.subheader("Export Core Data")
            st.download_button(label="📥 Download Posterior Probabilities (CSV)", data=assignments_df.to_csv(index=False).encode('utf-8'), file_name='gbtm_trajectory_assignments.csv', mime='text/csv')
            
            st.markdown("---")
            st.markdown("© 2026 Dr. Don Warden. Licensed under the MIT License.")
            
        else:
            st.error("Model Failed to Converge or was rejected based on rules.")