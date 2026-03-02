import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
import itertools

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

def extract_and_print_metrics(result, df, orders):
    """
    Translates raw output into AIC, BICs, Group Sizes, and P-Values.
    """
    k = len(orders)
    params = result.x
    max_ll = -1 * result.fun
    
    print("\n" + "="*60)
    print(f"MODEL METRICS: {k}-GROUP ({orders})")
    print("="*60)
    
    # 1. Information Criteria (The Deciders)
    n_subjects = df['ID'].nunique()
    n_obs = len(df)
    p_params = len(params)
    
    aic = (-2 * max_ll) + (2 * p_params)
    bic_subj = (-2 * max_ll) + (p_params * np.log(n_subjects))
    bic_obs = (-2 * max_ll) + (p_params * np.log(n_obs))
    
    print(f"Maximum Log-Likelihood: {max_ll:.4f}")
    print(f"AIC:                    {aic:.4f}")
    print(f"BIC (N={n_subjects} subjects):  {bic_subj:.4f}")
    print(f"BIC (N={n_obs} obs):       {bic_obs:.4f}")
    print("-" * 60)
    
    # 2. Group Sizes
    thetas = np.zeros(k)
    if k > 1:
        thetas[1:] = params[0 : k-1]
    pis = np.exp(thetas - logsumexp(thetas))
    
    # 3. Standard Errors and P-values
    # SciPy's BFGS method approximates the inverse Hessian (variance-covariance matrix)
    try:
        se = np.sqrt(np.diag(result.hess_inv))
    except:
        # Fallback if the matrix is singular/fails
        se = np.full(p_params, np.nan) 
        
    current_idx = k - 1
    for g in range(k):
        n_betas = orders[g] + 1
        group_betas = params[current_idx : current_idx + n_betas]
        group_se = se[current_idx : current_idx + n_betas]
        current_idx += n_betas
        
        print(f"GROUP {g+1} (Size: {pis[g] * 100:.1f}%)")
        print(f"{'Parameter':<12} | {'Estimate':<10} | {'Std Err':<10} | {'P-Value':<10}")
        print("-" * 55)
        
        beta_labels = ["Intercept", "Linear", "Quadratic", "Cubic", "Quartic"]
        
        for b_idx in range(n_betas):
            estimate = group_betas[b_idx]
            error = group_se[b_idx]
            
            # Calculate Z-score and two-tailed P-value
            z_score = estimate / error if error > 0 else 0
            p_val = 2 * (1 - norm.cdf(abs(z_score)))
            
            print(f"{beta_labels[b_idx]:<12} | {estimate:>10.4f} | {error:>10.4f} | {p_val:>10.4f}")
        print()
        
def run_autotraj(df, max_groups=2, max_order=2):
    """
    Exhaustively searches for the best trajectory model.
    Now outputs BIC, Min Group Size, and Highest-Order Significance on a single line!
    """
    best_bic = np.inf
    best_orders = None
    best_result = None
    
    n_subjects = df['ID'].nunique()
    
    print("\n" + "*"*80)
    print("STARTING AUTOTRAJ EXHAUSTIVE SEARCH")
    print("*"*80)
    
    for k in range(1, max_groups + 1):
        order_combinations = list(itertools.product(range(1, max_order + 1), repeat=k))
        
        for orders in order_combinations:
            orders_list = list(orders)
            
            # Format the print statement so it stays on one line while computing
            print(f"Testing {k}-Group {orders_list}...".ljust(25), end=" ", flush=True)
            
            num_params = (k - 1) + sum([order + 1 for order in orders_list])
            initial_guess = np.zeros(num_params)
            
            result = minimize(
                calc_dynamic_nll, 
                initial_guess, 
                args=(df, orders_list),
                method='BFGS'
            )
            
            if result.success or result.status == 2:
                # 1. Calculate BIC
                max_ll = -1 * result.fun
                bic = (-2 * max_ll) + (num_params * np.log(n_subjects))
                
                # 2. Check Group Sizes
                thetas = np.zeros(k)
                if k > 1:
                    thetas[1:] = result.x[0 : k-1]
                pis = np.exp(thetas - logsumexp(thetas))
                min_group_size = np.min(pis) * 100
                
                # 3. Check P-values for highest orders
                try:
                    se = np.sqrt(np.diag(result.hess_inv))
                except:
                    se = np.full(num_params, np.nan)
                    
                highest_order_pvals = []
                all_significant = True
                current_idx = k - 1
                
                for g in range(k):
                    n_betas = orders_list[g] + 1
                    group_betas = result.x[current_idx : current_idx + n_betas]
                    group_se = se[current_idx : current_idx + n_betas]
                    current_idx += n_betas
                    
                    # The highest order is always the LAST beta in the group's array
                    highest_est = group_betas[-1]
                    highest_se = group_se[-1]
                    
                    z_score = highest_est / highest_se if highest_se > 0 else 0
                    p_val = 2 * (1 - norm.cdf(abs(z_score)))
                    highest_order_pvals.append(p_val)
                    
                    if p_val >= 0.05:
                        all_significant = False
                
                # Format the output for easy reading
                p_val_str = "[" + ", ".join([f"{p:.3f}" for p in highest_order_pvals]) + "]"
                
                if min_group_size < 5.0:
                    print(f"REJECTED: Small Group ({min_group_size:.1f}%)")
                else:
                    sig_flag = "SIG" if all_significant else "NOT SIG"
                    print(f"BIC: {bic:.2f} | Min %: {min_group_size:.1f}% | P-vals: {p_val_str} | {sig_flag}")
                
                # Save the best model that passes the 5% rule and is fully significant
                if bic < best_bic and min_group_size >= 5.0 and all_significant:
                    best_bic = bic
                    best_orders = orders_list
                    best_result = result
            else:
                print("FAILED to converge.")
                
    return best_orders, best_result



# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    raw_wide = load_cambridge_data()
    long_data = prep_trajectory_data(raw_wide).dropna(subset=['Time', 'Outcome'])

    # Let autotraj find the best model! (Max 2 groups, Max Order 2)
    winning_orders, winning_result = run_autotraj(long_data, max_groups=2, max_order=2)
    
    print("\n" + "*"*50)
    print("AUTOTRAJ SEARCH COMPLETE")
    print(f"THE WINNER IS: {len(winning_orders)}-Group Model with orders {winning_orders}")
    print("*"*50)
    
    # Print the full metrics for the winning model
    extract_and_print_metrics(winning_result, long_data, winning_orders)