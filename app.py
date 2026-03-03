import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Import our engine from main.py
from main import prep_trajectory_data, run_autotraj, calc_logit_prob_matrix, create_design_matrix

st.set_page_config(page_title="GBTM AutoTraj Engine", layout="wide")
st.title("Group-Based Trajectory Modeling (GBTM)")
st.markdown("Automated trajectory selection and finite mixture modeling.")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("1. Data Mapping")
    id_col = st.text_input("Subject ID Column Name", value="ID")
    outcome_col = st.text_input("Outcome Variable Prefix", value="C", help="e.g., 'C' for C1, C2, C3...")
    time_col = st.text_input("Time Variable Prefix", value="T", help="e.g., 'T' for T1, T2, T3...")
    
    st.divider()
    
    st.header("2. Search Grid")
    group_range = st.slider("Min & Max Groups", 1, 6, (1, 3))
    order_range = st.slider("Min & Max Polynomial Order", 0, 5, (0, 2), help="0=Intercept, 1=Linear, 2=Quadratic...")
    
    st.divider()
    
    st.header("3. Selection Algorithm Rules")
    min_pct = st.slider("Minimum Group Size (%)", 1.0, 15.0, 5.0, 0.5, help="Disqualifies models with spurious small groups.")
    p_val = st.number_input("Highest-Order P-Value Threshold", value=0.05, format="%.3f")

# ==========================================
# MAIN INTERACTION AREA
# ==========================================
uploaded_file = st.file_uploader("Upload Wide-Format Dataset (CSV or TXT)", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file, sep=r'\s+')
    except:
        raw_df = pd.read_csv(uploaded_file)
        
    st.success("File uploaded successfully!")
    with st.expander("Preview Raw Data"):
        st.dataframe(raw_df.head())
    
    if st.button("Run AutoTraj Search", type="primary", use_container_width=True):
        
        # Start the stopwatch!
        start_time = time.time()
        
        with st.spinner("Pivoting data and executing analytical gradients..."):
            # 1. Prep data using the dynamic UI column names
            long_df = prep_trajectory_data(raw_df, id_col, outcome_col, time_col).dropna(subset=['Time', 'Outcome'])
            
            # 2. Run the exhaustive search using the UI rules
            winning_orders, winning_result = run_autotraj(
                long_df, 
                min_groups=group_range[0], max_groups=group_range[1],
                min_order=order_range[0], max_order=order_range[1],
                min_group_pct=min_pct, p_val_thresh=p_val
            )
        
        # Stop the stopwatch!
        run_time_seconds = time.time() - start_time
        
        # Calculate the gimmick (Assuming 3 minutes per manual test in SAS/R)
        total_models_tested = 0
        for k in range(group_range[0], group_range[1] + 1):
            total_models_tested += (order_range[1] - order_range[0] + 1) ** k
        manual_time_minutes = total_models_tested * 3
        
        if winning_result is not None:
            # --- THE DASHBOARD RESULTS ---
            st.divider()
            st.subheader("🏆 Optimal Trajectory Model Found")
            
            # Top-level metrics columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Optimal Groups", len(winning_orders))
            col2.metric("Polynomial Orders", str(winning_orders))
            col3.metric("AutoTraj Engine Time", f"{run_time_seconds:.2f} sec")
            col4.metric("Estimated Manual Time", f"{manual_time_minutes} min", delta="Massive time saved!", delta_color="normal")
            
            # --- THE VISUALIZATION ---
            st.subheader("Trajectory Visualization")
            
            # Create a Matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot raw data as a faint scatter plot
            actual_times = long_df['Time'].values
            actual_outcomes = long_df['Outcome'].values
            jittered_y = actual_outcomes + np.random.normal(0, 0.02, size=len(actual_outcomes))
            ax.scatter(actual_times, jittered_y, alpha=0.05, color='gray')
            
            # Plot the winning curves
            smooth_times = np.linspace(min(actual_times), max(actual_times), 100)
            current_idx = len(winning_orders) - 1 # Skip thetas
            
            colors = ['#FF4B4B', '#0068C9', '#83C9FF', '#FFABAB', '#29B09D', '#7DE2D1']
            
            for g in range(len(winning_orders)):
                n_betas = winning_orders[g] + 1
                g_betas = winning_result.x[current_idx : current_idx + n_betas]
                current_idx += n_betas
                
                # Generate matrix for the smooth line
                X_smooth = create_design_matrix(smooth_times, winning_orders[g])
                g_probs = calc_logit_prob_matrix(g_betas, X_smooth)
                
                ax.plot(smooth_times, g_probs, linewidth=3, color=colors[g], label=f"Group {g+1}")

            ax.set_ylim(-0.1, 1.1)
            ax.set_ylabel("Probability of Outcome")
            ax.set_xlabel("Time Period")
            ax.legend()
            
            # Render the Matplotlib figure directly onto the Streamlit webpage!
            st.pyplot(fig)
            
        else:
            st.error("No model met the specified criteria. Try lowering the Minimum Group Size or raising the P-Value threshold.")