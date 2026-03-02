import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

def load_cambridge_data():
    """
    Loads the Cambridge dataset used for testing logit trajectory models.
    Reference: https://www.andrew.cmu.edu/user/bjj/traj/
    """
    df = pd.read_csv("cambridge.txt", sep=r'\s+')
    return df

def prep_trajectory_data(df):
    """Pivots wide-format subject data into long-format observation data."""
    long_df = pd.wide_to_long(df, stubnames=['C', 'T'], i='ID', j='Measurement_Period').reset_index()
    long_df = long_df.rename(columns={'C': 'Outcome', 'T': 'Time'})
    long_df = long_df.sort_values(by=['ID', 'Measurement_Period'])
    return long_df

def create_design_matrix(times, order):
    """Creates a dynamic design matrix (X) for any polynomial order."""
    return np.column_stack([times**p for p in range(order + 1)])

def calc_logit_prob_matrix(betas, X):
    """Calculates probabilities for ALL observations simultaneously."""
    z = np.dot(X, betas)
    return 1 / (1 + np.exp(-z))

def calc_log_likelihood_matrix(betas, X, actual_outcomes):
    """Calculates the Bernoulli Log-Likelihood for an array of observations."""
    p = np.clip(calc_logit_prob_matrix(betas, X), 1e-10, 1 - 1e-10)
    ll_array = (actual_outcomes * np.log(p)) + ((1 - actual_outcomes) * np.log(1 - p))
    return np.sum(ll_array)

def calc_dynamic_nll(params, df, orders):
    """Calculates the Negative Log-Likelihood for ANY number of groups/orders."""
    k = len(orders)
    
    # Parse Thetas
    thetas = np.zeros(k)
    if k > 1:
        thetas[1:] = params[0 : k-1]
    log_pis = thetas - logsumexp(thetas)
    
    # Parse Betas
    group_betas = []
    current_idx = k - 1
    for order in orders:
        n_betas = order + 1
        group_betas.append(params[current_idx : current_idx + n_betas])
        current_idx += n_betas

    total_ll = 0
    grouped = df.groupby('ID')
    
    for subject_id, subject_data in grouped:
        times = subject_data['Time'].values
        outcomes = subject_data['Outcome'].values
        subject_group_lls = []
        
        for g in range(k):
            X = create_design_matrix(times, orders[g])
            ll_g = calc_log_likelihood_matrix(group_betas[g], X, outcomes)
            subject_group_lls.append(log_pis[g] + ll_g)
            
        total_ll += logsumexp(subject_group_lls)
        
    return -1 * total_ll

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Load and prep data
    raw_wide = load_cambridge_data()
    long_data = prep_trajectory_data(raw_wide).dropna(subset=['Time', 'Outcome'])

    # 2. Define Model Architecture
    model_orders = [1, 2, 3] 
    k = len(model_orders)
    num_params = (k - 1) + sum([order + 1 for order in model_orders])
    initial_guess = np.zeros(num_params)

    print(f"Running dynamic {k}-Group optimization...")
    print(f"Model architecture (orders): {model_orders}")

    # 3. Run Optimizer
    result = minimize(
        calc_dynamic_nll, 
        initial_guess, 
        args=(long_data, model_orders),
        method='BFGS'
    )

    # 4. Results
    print(f"\n--- {k}-Group Dynamic Model Results ---")
    print(f"Optimization Success: {result.success}")
    print(f"Maximum Log-Likelihood: {-1 * result.fun:.4f}")