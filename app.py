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
    calc_logit_prob_jit, 
    create_design_matrix_jit, 
    get_subject_assignments, 
    get_parameter_estimates,
    calc_model_adequacy
)

st.set_page_config(page_title="GBTM AutoTraj Engine", layout="wide")
st.title("AutoTRAJ: Automated Group-Based Trajectory Modeling (GBTM)")
st.markdown("Automated trajectory selection, visualization, and parameter extraction.")

with st.expander("📖 About AutoTraj & Instructions", expanded=False):
    st.markdown("""
    **Overview**
    This application automates the exhaustive search, selection, and visualization of finite mixture models for longitudinal data. It utilizes a custom-built, C-compiled analytical Jacobian engine to rapidly evaluate combinatorial polynomial grids.
    
    **Methodology**
    * **Selection:** The engine adheres to established GBTM heuristics, discarding models that produce spurious groups (size < user-defined threshold) or lack significance in their highest-order polynomial term.
    * **Adequacy:** Model fit is evaluated using Average Posterior Probability (AvePP) and Odds of Correct Classification (OCC).
    
    **Instructions**
    1. Upload your data in **wide format** (.csv or .txt).
    2. Map your column prefixes in the sidebar (e.g., if your time variables are T1, T2, T3, enter 'T').
    3. Set your search grid and statistical thresholds.
    4. Click Run. The engine will evaluate all permutations and return the mathematically optimal model.
    
    **Attribution**
    Built by Donald E. Warden, PhD, MPH
    """)

if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False
    st.session_state.top_models = None
    st.session_state.run_time = 0
    st.session_state.long_df = None
    st.session_state.raw_df = None

with st.sidebar:
    st.header("1. Data Mapping")
    id_col = st.text_input("Subject ID Column Name", value="ID")
    outcome_col = st.text_input("Outcome Variable Prefix", value="C")
    time_col = st.text_input("Time Variable Prefix", value="T")
    st.divider()
    st.header("2. Search Grid")
    group_range = st.slider("Min & Max Groups", 1, 8, (1, 3))
    order_range = st.slider("Min & Max Polynomial Order", 0, 5, (0, 2))
    st.divider()
    st.header("3. Selection Rules")
    min_pct = st.slider("Minimum Group Size (%)", 1.0, 15.0, 5.0, 0.5)
    p_val = st.number_input("Highest-Order P-Value Threshold", value=0.05, format="%.3f")

uploaded_file = st.file_uploader("Upload Wide-Format Dataset (CSV or TXT)", type=["csv", "txt"])

if uploaded_file is not None:
    try: raw_df = pd.read_csv(uploaded_file, sep=r'\s+')
    except Exception: raw_df = pd.read_csv(uploaded_file)
        
    st.success("File uploaded successfully!")
    
    if st.button("Run AutoTraj Search", type="primary", use_container_width=True):
        start_time = time.time()
        with st.spinner("Executing Fully Vectorized C-Compiled Search..."):
            long_df = prep_trajectory_data(raw_df, id_col, outcome_col, time_col).dropna(subset=['Time', 'Outcome'])
            top_models = run_autotraj(
                long_df, min_groups=group_range[0], max_groups=group_range[1],
                min_order=order_range[0], max_order=order_range[1],
                min_group_pct=min_pct, p_val_thresh=p_val
            )
        
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
        winning_model = top_models[0]
        winning_orders = winning_model['orders']
        winning_result = winning_model['result']
        winning_pis_raw = winning_model['pis'] 
        assignments_df = get_subject_assignments(winning_result, long_df, winning_orders)
        
        st.divider()
        st.subheader("🏆 Optimal Trajectory Model Found")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Optimal Groups", len(winning_orders))
        col2.metric("Polynomial Orders", str(winning_orders))
        col3.metric("Engine Time", f"{st.session_state.run_time:.2f} sec")
        col4.metric("BIC (Subject-Level)", f"{winning_model['bic']:.2f}")
        
        st.divider()
        st.subheader("Publication Suite")
        
        # We now have 5 tabs to include the internal documentation
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Visualization", 
            "Exact Estimates", 
            "Adequacy Metrics", 
            "Demographics (Table 1)",
            "About & Docs"
        ])
        
        with tab1:
            col_viz1, col_viz2 = st.columns([3, 1])
            with col_viz2:
                viz_style = st.selectbox("Select Graphic Style:", ["Interactive Web (Plotly)", "Publication: Grayscale (Matplotlib)", "Publication: Color (Matplotlib)"])
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
                        fig.add_trace(go.Scatter(x=smooth_times, y=g_probs, mode='lines', line=dict(color=colors[g%len(colors)], width=4), name=f'Group {g+1} ({winning_pis_raw[g]*100:.1f}%)'))
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
                        ax.plot(smooth_times, g_probs, linewidth=2.5, color=colors[g%len(colors)], linestyle=linestyles[g%len(linestyles)], label=f'Group {g+1} ({winning_pis_raw[g]*100:.1f}%)')
                    ax.set_ylim(-0.1, 1.1)
                    ax.set_ylabel("Probability of Outcome")
                    ax.set_xlabel("Time Period")
                    ax.legend(frameon=False)
                    st.pyplot(fig)
                
        with tab2:
            st.markdown("##### Parameter Estimates, Standard Errors, and P-Values")
            st.dataframe(get_parameter_estimates(winning_result, winning_orders), use_container_width=True, hide_index=True)
            
        with tab3:
            st.markdown("##### Model Adequacy")
            st.dataframe(calc_model_adequacy(assignments_df, winning_pis_raw), use_container_width=True, hide_index=True)

        with tab4:
            st.markdown("##### Baseline Demographics (Table 1)")
            if HAS_TABLEONE:
                potential_covariates = [col for col in raw_df.columns.tolist() if not col.startswith((outcome_col, time_col))]
                selected_vars = st.multiselect("Variables to include:", potential_covariates)
                categorical_vars = st.multiselect("Which of these are categorical?", selected_vars)
                if selected_vars and st.button("Generate Table 1"):
                    merged_df = pd.merge(raw_df, assignments_df[['ID', 'Assigned_Group']], left_on=id_col, right_on='ID')
                    mytable = TableOne(merged_df, columns=selected_vars, categorical=categorical_vars, groupby="Assigned_Group", pval=True)
                    st.markdown(mytable.to_html(), unsafe_allow_html=True)
            else: st.warning("Please run `pip install tableone` in your terminal to enable this feature.")
            
        with tab5:
            st.markdown("### AutoTraj: Automated Group-Based Trajectory Modeling")
            st.markdown("""
            **Engine Architecture**
            AutoTraj replaces traditional manual heuristic step-down approaches with a fully vectorized, C-compiled exhaustive search grid. 
            
            By utilizing the `numba` library, the Python interpreter is bypassed, allowing the Negative Log-Likelihood and Analytical Jacobian to be calculated directly at the machine-code level. This allows for the evaluation of hundreds of complex finite mixture models in seconds.
            
            **Selection Criteria**
            * **Group Size Penalty:** Discards any model containing a spurious trajectory group smaller than the user-defined threshold (default 5.0%).
            * **Polynomial Significance:** Adheres strictly to Nagin & Jones (2005) methodology; if the highest-order polynomial term of any group is non-significant, the model is rejected in favor of a trimmed alternative.
            * **Optimization:** Determines the global optimal model using the Bayesian Information Criterion (BIC).
                        
            Copyright \u00A9 2026 Donald E. Warden 
            """)

        st.divider()
        col_down1, col_down2 = st.columns(2)
        with col_down1:
            st.subheader("Export Core Data")
            st.download_button(label="📥 Download Posterior Probabilities (CSV)", data=assignments_df.to_csv(index=False).encode('utf-8'), file_name='gbtm_trajectory_assignments.csv', mime='text/csv')
        with col_down2:
            if len(top_models) > 1:
                st.subheader("Alternative Valid Models")
                runners_up = [{"Rank": i+1, "Orders": str(m['orders']), "BIC": round(m['bic'], 2), "Smallest Group (%)": round(m['min_pct'], 1)} for i, m in enumerate(top_models[1:5], 1)]
                st.dataframe(pd.DataFrame(runners_up), hide_index=True)
    else:
        st.error("No valid models found. Try lowering the Minimum Group Size or raising the P-Value threshold.")